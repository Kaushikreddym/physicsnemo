#!/usr/bin/env python
# generate.py (extended with Score-based Data Assimilation)
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
import datetime

import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path

import torch
import torch._dynamo
from torch.distributed import gather
import numpy as np
import nvtx
import netCDF4 as nc

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.experimental.models.diffusion.preconditioning import (
    tEDMPrecondSuperRes,
)
from physicsnemo.utils.patching import GridPatching2D
from physicsnemo import Module
from physicsnemo.utils.diffusion import deterministic_sampler, stochastic_sampler
from physicsnemo.utils.corrdiff import (
    NetCDFWriter,
    regression_step,
    diffusion_step,
)

from helpers.generate_helpers import (
    get_dataset_and_sampler,
    save_images,
)
from helpers.train_helpers import set_patch_shape
from datasets.dataset import register_dataset

# === SDA imports ===
# Your repo example used these wrappers; ensure sda package is in PYTHONPATH
from sda.score import GaussianScore_from_denoiser, VPSDE_from_denoiser, VPSDE

# ------------

def get_time_from_range(times_range, time_format="%Y-%m-%dT%H:%M:%S"):
    start_time = datetime.datetime.strptime(times_range[0], time_format)
    end_time = datetime.datetime.strptime(times_range[1], time_format)

    interval = datetime.timedelta(hours=1)
    if len(times_range) > 2:
        interval_arg = times_range[2]
        if isinstance(interval_arg, str):
            if interval_arg.endswith("h"):
                hours = float(interval_arg[:-1])
                interval = datetime.timedelta(hours=hours)
            elif interval_arg.endswith("d"):
                days = float(interval_arg[:-1])
                interval = datetime.timedelta(days=days)
            else:
                raise ValueError(
                    f"Unsupported interval format: '{interval_arg}'. Use 'h' or 'd'."
                )
        else:
            interval = datetime.timedelta(hours=float(interval_arg))

    times = []
    t = start_time
    while t <= end_time:
        times.append(t.strftime(time_format))
        t += interval

    return times


# -------------------
# SDA helper functions
# -------------------
def load_obs_and_mask(obs_path: str, mask_path: str):
    """
    Load observation dataset and station mask DataArray.
    - obs: xarray Dataset (variables named as output channels)
           indexed by 'DATE' or 'time' (function will try both).
    - mask: xarray DataArray (boolean) with dims (y,x) or (DATE,y,x). If (DATE,y,x), time selection is required later.
    Returns xarray objects (obs_ds, mask_da).
    """
    obs_ds = None
    mask_da = None
    if obs_path is not None and os.path.exists(obs_path):
        obs_ds = xr.open_dataset(to_absolute_path(obs_path))
    if mask_path is not None and os.path.exists(mask_path):
        mask_da = xr.open_dataarray(to_absolute_path(mask_path))
    return obs_ds, mask_da


def build_observation_operator_from_mask(mask_bool: np.ndarray):
    """
    Given mask_bool shape (H,W) where True indicates observed pixel,
    return:
      - A callable A(x) -> vector of observed entries (stacked channels),
      - index tuple to use for assignment back.
    Expect tensors shaped (B, C, H, W) or (C, H, W) for x.
    We'll flatten (C,H,W) -> vector in channel-major ordering.
    """
    # flatten indices for channel stacking later
    # mask_bool has shape (H,W)
    obs_idx = np.where(mask_bool.ravel())[0]  # positions within flattened H*W
    # We'll apply across channels by offsetting
    def A(x: torch.Tensor):
        # x: (..., C, H, W) or (C, H, W)
        single = False
        if x.ndim == 3:
            x = x[None, ...]  # make batch dim
            single = True
        B, C, H, W = x.shape
        flat = x.reshape(B, C, H * W)  # (B, C, HW)
        # select observed positions, for all channels
        obs_list = []
        for ch in range(C):
            obs_list.append(flat[:, ch, obs_idx])  # (B, len(obs_idx))
        # concatenate channels
        obs_vec = torch.cat(obs_list, dim=1)  # (B, C * len(obs_idx))
        if single:
            return obs_vec[0]
        return obs_vec

    # inverse mapping info (not strictly needed here, but provided for clarity)
    return A, obs_idx


def extract_obs_vector_for_time(obs_ds, mask_da, channel_names, time_key):
    """
    Given obs_ds (xarray), mask_da (xarray), list of channel_names,
    and a time_key (string or datetime), return:
      - obs_tensor (C, H, W) with NaNs where missing
      - mask_bool (H, W) boolean True where observations exist (station mask)
    Flexible selection: tries 'DATE' index, then 'time'.
    """
    if obs_ds is None:
        raise ValueError("No observation dataset provided for SDA.")
    # Select by DATE or time coordinate if available
    # Try DATE first
    selected = None
    if "DATE" in obs_ds.coords:
        idx = obs_ds.indexes["DATE"]
        try:
            pos = idx.get_indexer([time_key])
            if (pos < 0).any():
                # fallback: attempt string formatting/time parsing
                raise IndexError
            sel = slice(pos[0], pos[0] + 1)
            selected = obs_ds.isel(DATE=sel).isel(DATE=0)
        except Exception:
            # try using sel with time_key directly
            try:
                selected = obs_ds.sel(DATE=time_key, method="nearest")
            except Exception:
                raise IndexError(f"Time {time_key} not found in obs dataset (DATE).")
    elif "time" in obs_ds.coords:
        try:
            selected = obs_ds.sel(time=time_key, method="nearest")
        except Exception:
            raise IndexError(f"Time {time_key} not found in obs dataset (time).")
    else:
        # if dataset has a single time dimension, take first
        selected = obs_ds.isel({list(obs_ds.dims)[0]: 0})

    # Build observation array in channel order
    ch_arrays = []
    for ch in channel_names:
        if ch not in selected:
            # try variants: e.g., '10u' vs 'u10' - leave to user to ensure naming matches
            raise KeyError(f"Channel {ch} not found in observation dataset")
        arr = selected[ch].values  # (H, W)
        ch_arrays.append(arr)

    obs_np = np.stack(ch_arrays, axis=0).astype(np.float32)
    # mask_da: if it has time dim, pick the same time; else assume (H,W)
    if mask_da is None:
        # Build mask from finite obs
        mask_bool = np.isfinite(obs_np[0])
    else:
        if "DATE" in mask_da.coords:
            try:
                pos = mask_da.indexes["DATE"].get_indexer([time_key])
                if (pos < 0).any():
                    mask_slice = mask_da.isel(DATE=0)
                else:
                    mask_slice = mask_da.isel(DATE=pos[0])
            except Exception:
                mask_slice = mask_da.isel(DATE=0)
            mask_bool = mask_slice.values.astype(bool)
        elif "time" in mask_da.coords:
            try:
                mask_slice = mask_da.sel(time=time_key, method="nearest")
            except Exception:
                mask_slice = mask_da.isel(time=0)
            mask_bool = mask_slice.values.astype(bool)
        else:
            mask_bool = mask_da.values.astype(bool)

    return obs_np, mask_bool


# -------------------
# main hydra entrypoint
# -------------------
@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    logger = PythonLogger("generate")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate.log")

    seeds = list(np.arange(cfg.generation.num_ensembles))
    num_batches = (
        (len(seeds) - 1) // (cfg.generation.seed_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

    if dist.world_size > 1:
        torch.distributed.barrier()

    if cfg.generation.times_range and cfg.generation.times:
        raise ValueError("Either times_range or times must be provided, but not both")
    if cfg.generation.times_range:
        times = get_time_from_range(cfg.generation.times_range)
    else:
        times = cfg.generation.times

    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    register_dataset(cfg.dataset.type)
    logger0.info(f"Using dataset: {cfg.dataset.type}")

    has_lead_time = cfg.generation.get("has_lead_time", False)
    dataset, sampler = get_dataset_and_sampler(
        dataset_cfg=dataset_cfg, times=times, has_lead_time=has_lead_time
    )
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())

    if cfg.generation.patching:
        patch_shape_x = cfg.generation.patch_shape_x
        patch_shape_y = cfg.generation.patch_shape_y
    else:
        patch_shape_x, patch_shape_y = None, None
    patch_shape = (patch_shape_y, patch_shape_x)
    use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if use_patching:
        patching = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            boundary_pix=cfg.generation.boundary_pix,
            overlap_pix=cfg.generation.overlap_pix,
        )
        logger0.info("Patch-based generation enabled")
    else:
        patching = None
        logger0.info("Patch-based generation disabled")

    if cfg.generation.inference_mode == "regression":
        load_net_reg, load_net_res = True, False
    elif cfg.generation.inference_mode == "diffusion":
        load_net_reg, load_net_res = False, True
    elif cfg.generation.inference_mode == "all":
        load_net_reg, load_net_res = True, True
    else:
        raise ValueError(f"Invalid inference mode {cfg.generation.inference_mode}")

    if load_net_res:
        res_ckpt_filename = cfg.generation.io.res_ckpt_filename
        logger0.info(f'Loading residual network from "{res_ckpt_filename}"...')
        net_res = Module.from_checkpoint(
            to_absolute_path(res_ckpt_filename),
            override_args={
                "use_apex_gn": getattr(cfg.generation.perf, "use_apex_gn", False)
            },
        )
        net_res.profile_mode = getattr(cfg.generation.perf, "profile_mode", False)
        net_res.use_fp16 = getattr(cfg.generation.perf, "use_fp16", False)
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
        if hasattr(net_res, "amp_mode"):
            net_res.amp_mode = False
    else:
        net_res = None

    if load_net_reg:
        reg_ckpt_filename = cfg.generation.io.reg_ckpt_filename
        logger0.info(f'Loading network from "{reg_ckpt_filename}"...')
        net_reg = Module.from_checkpoint(
            to_absolute_path(reg_ckpt_filename),
            override_args={
                "use_apex_gn": getattr(cfg.generation.perf, "use_apex_gn", False)
            },
        )
        net_reg.profile_mode = getattr(cfg.generation.perf, "profile_mode", False)
        net_reg.use_fp16 = getattr(cfg.generation.perf, "use_fp16", False)
        net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
        if hasattr(net_reg, "amp_mode"):
            net_reg.amp_mode = False
    else:
        net_reg = None

    if cfg.generation.perf.use_torch_compile:
        torch._dynamo.config.cache_size_limit = 264
        torch._dynamo.reset()
        if net_res:
            net_res = torch.compile(net_res)
        if net_reg:
            net_reg = torch.compile(net_reg)

    if cfg.sampler.type == "deterministic":
        sampler_fn = partial(
            deterministic_sampler,
            num_steps=cfg.sampler.num_steps,
            solver=cfg.sampler.solver,
            patching=patching,
        )
    elif cfg.sampler.type == "stochastic":
        sampler_fn = partial(stochastic_sampler, patching=patching)
    else:
        raise ValueError(f"Unknown sampling method {cfg.sampler.type}")

    distribution = getattr(cfg.generation, "distribution", None)
    student_t_nu = getattr(cfg.generation, "student_t_nu", None)
    if distribution is not None and not cfg.generation.inference_mode in [
        "diffusion",
        "all",
    ]:
        raise ValueError(
            f"cfg.generation.distribution should only be specified for inference mode 'diffusion' or 'all'"
        )
    if distribution not in ["normal", "student_t", None]:
        raise ValueError(f"Invalid distribution: {distribution}.")
    if distribution == "student_t":
        if student_t_nu is None:
            raise ValueError(
                "student_t_nu must be provided in cfg.generation.student_t_nu for student_t distribution"
            )
        elif student_t_nu <= 2:
            raise ValueError(f"Expected nu > 2, but got {student_t_nu}.")
        if net_res and not isinstance(net_res, tEDMPrecondSuperRes):
            logger0.warning(
                f"Student-t distribution sampling is supposed to be used with tEDMPrecondSuperRes model, but got {type(net_res)}."
            )
    elif isinstance(net_res, tEDMPrecondSuperRes):
        logger0.warning(
            f"tEDMPrecondSuperRes model is supposed to be used with student-t distribution, but got {distribution}."
        )

    P_mean = getattr(cfg.generation, "P_mean", None)
    P_std = getattr(cfg.generation, "P_std", None)

    # Load observation & mask once (if SDA enabled)
    obs_ds = mask_da = None
    if getattr(cfg.generation, "sda", {}).get("enabled", False):
        obs_path = cfg.generation.sda.get("obs_path", None)
        mask_path = cfg.generation.sda.get("mask_path", None)
        obs_ds, mask_da = load_obs_and_mask(obs_path, mask_path)
        logger0.info(f"SDA enabled. obs: {obs_path}, mask: {mask_path}")

    # main generation function (wrapped inside loop below)
    def generate_fn():
        with nvtx.annotate("generate_fn", color="green"):
            diffusion_step_kwargs = {}
            if distribution is not None:
                diffusion_step_kwargs["distribution"] = distribution
            if student_t_nu is not None:
                diffusion_step_kwargs["nu"] = student_t_nu
            if P_mean is not None:
                diffusion_step_kwargs["P_mean"] = P_mean
            if P_std is not None:
                diffusion_step_kwargs["P_std"] = P_std

            img_lr = image_lr.to(memory_format=torch.channels_last)

            if net_reg:
                with nvtx.annotate("regression_model", color="yellow"):
                    image_reg = regression_step(
                        net=net_reg,
                        img_lr=img_lr,
                        latents_shape=(
                            sum(map(len, rank_batches)),
                            img_out_channels,
                            img_shape[0],
                            img_shape[1],
                        ),
                        lead_time_label=lead_time_label,
                    )

            if net_res:
                if cfg.generation.hr_mean_conditioning:
                    mean_hr = image_reg[0:1]
                else:
                    mean_hr = None

                with nvtx.annotate("diffusion model", color="purple"):
                    # If SDA is enabled, run score-based data assimilation instead of default diffusion_step
                    if getattr(cfg.generation, "sda", {}).get("enabled", False):
                        # Build y_star and operator A for the current time index
                        try:
                            # time string/key matching times list
                            # times list is from earlier, same ordering as dataset.time()
                            time_key = times[time_index]
                        except Exception:
                            time_key = None

                        # Convert dataset channel metadata to list of names for obs selection
                        channel_names = [c.name for c in dataset.output_channels()]

                        # Extract obs_np (C,H,W) and mask_bool (H,W)
                        try:
                            obs_np, mask_bool = extract_obs_vector_for_time(obs_ds, mask_da, channel_names, time_key)
                        except Exception as e:
                            logger0.warning(f"SDA obs extraction failed for time {time_key}: {e}")
                            # fall back to default diffusion_step
                            image_res = diffusion_step(
                                net=net_res,
                                sampler_fn=sampler_fn,
                                img_shape=img_shape,
                                img_out_channels=img_out_channels,
                                rank_batches=rank_batches,
                                img_lr=img_lr.expand(cfg.generation.seed_batch_size, -1, -1, -1).to(memory_format=torch.channels_last),
                                rank=dist.rank,
                                device=device,
                                mean_hr=mean_hr,
                                lead_time_label=lead_time_label,
                                **diffusion_step_kwargs,
                            )
                            # decide downstream behavior
                            return image_res

                        # convert to torch and move to device
                        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)  # (C,H,W)
                        mask_bool = np.asarray(mask_bool).astype(bool)  # (H,W)
                        # make mask for all channels (C,H,W)
                        # Create callable A and obs_vec y_star using flattening operator that accounts for channels
                        A_callable, obs_idx = build_observation_operator_from_mask(mask_bool)

                        # Build y_star as flattened vector across channels (C * n_obs)
                        # We assume obs_t uses same channel order as network output; flatten channel-wise
                        B = 1  # we operate single example at a time inside generate loop
                        C, H, W = obs_t.shape
                        flat = obs_t.reshape(C, H * W)
                        obs_blocks = [flat[ch, obs_idx] for ch in range(C)]
                        y_star_vec = torch.cat(obs_blocks, dim=0).to(device)  # (C * n_obs,)

                        # Now wrap denoiser into score + VPSDE
                        sde_std = cfg.generation.sda.get("std", 0.1)
                        sde_gamma = cfg.generation.sda.get("gamma", 0.001)
                        sde_steps = int(cfg.generation.sda.get("sde_steps", cfg.sampler.num_steps))
                        sde_corrections = int(cfg.generation.sda.get("sde_corrections", 2))
                        sde_tau = float(cfg.generation.sda.get("sde_tau", 0.3))

                        # Build score and VPSDE wrappers
                        # GaussianScore_from_denoiser expects y_star and A to be tensors/callables with proper shapes
                        # We construct score with y_star_vec (observations) and A callable that maps full field->obs vector
                        score = GaussianScore_from_denoiser(
                            y_star_vec,
                            A=A_callable,
                            std=sde_std,
                            gamma=sde_gamma,
                            sde=VPSDE_from_denoiser(net_res, shape=())  # shape unused by wrapper, kept for API parity
                        )

                        sde = VPSDE(
                            score,
                            shape=(1, img_out_channels, img_shape[0], img_shape[1])
                        ).cuda() if str(device).startswith("cuda") else VPSDE(
                            score,
                            shape=(1, img_out_channels, img_shape[0], img_shape[1])
                        )

                        # Run SDA sampling; result shape (C, H, W) or (1, C, H, W) depending on sde.sample
                        sample = sde.sample(steps=sde_steps, corrections=sde_corrections, tau=sde_tau, makefigs=False)
                        if isinstance(sample, torch.Tensor):
                            sample = sample.cpu()
                        else:
                            sample = torch.tensor(sample)

                        # sample may be (1,C,H,W) or (C,H,W) — normalize to (batch, C, H, W)
                        if sample.ndim == 3:
                            sample = sample.unsqueeze(0)

                        # sample contains assimilated residual (or full field depending on wrapper implementation).
                        # We'll treat sample as image_res for downstream composition
                        image_res = sample.to(device=device)
                    else:
                        # default diffusion path
                        image_res = diffusion_step(
                            net=net_res,
                            sampler_fn=sampler_fn,
                            img_shape=img_shape,
                            img_out_channels=img_out_channels,
                            rank_batches=rank_batches,
                            img_lr=img_lr.expand(cfg.generation.seed_batch_size, -1, -1, -1).to(memory_format=torch.channels_last),
                            rank=dist.rank,
                            device=device,
                            mean_hr=mean_hr,
                            lead_time_label=lead_time_label,
                            **diffusion_step_kwargs,
                        )
            # end net_res block

            if cfg.generation.inference_mode == "regression":
                image_out = image_reg
            elif cfg.generation.inference_mode == "diffusion":
                image_out = image_res
            else:
                image_out = image_reg + image_res

            # gather on rank 0
            if dist.world_size > 1:
                if dist.rank == 0:
                    gathered_tensors = [
                        torch.zeros_like(image_out, dtype=image_out.dtype, device=image_out.device)
                        for _ in range(dist.world_size)
                    ]
                else:
                    gathered_tensors = None

                torch.distributed.barrier()
                gather(image_out, gather_list=gathered_tensors if dist.rank == 0 else None, dst=0)

                if dist.rank == 0:
                    return torch.cat(gathered_tensors)
                else:
                    return None
            else:
                return image_out

    # generate images
    output_path = getattr(cfg.generation.io, "output_filename", "corrdiff_output.nc")
    logger0.info(f"Generating images, saving results to {output_path}...")
    batch_size = 1
    warmup_steps = min(len(times) - 1, 2)

    if dist.rank == 0:
        f = nc.Dataset(output_path, "w")
        f.cfg = str(cfg)

    torch_cuda_profiler = (
        torch.cuda.profiler.profile()
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    torch_nvtx_profiler = (
        torch.autograd.profiler.emit_nvtx()
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with torch_cuda_profiler:
        with torch_nvtx_profiler:
            data_loader = torch.utils.data.DataLoader(
                dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True
            )
            time_index = -1
            if dist.rank == 0:
                writer = NetCDFWriter(
                    f,
                    lat=dataset.latitude(),
                    lon=dataset.longitude(),
                    input_channels=dataset.input_channels(),
                    output_channels=dataset.output_channels(),
                    has_lead_time=has_lead_time,
                )
                if cfg.generation.perf.io_syncronous:
                    writer_executor = ThreadPoolExecutor(max_workers=cfg.generation.perf.num_writer_workers)
                    writer_threads = []

            use_cuda_timing = torch.cuda.is_available()
            if use_cuda_timing:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
            else:
                class DummyEvent:
                    def record(self): pass
                    def synchronize(self): pass
                    def elapsed_time(self, _): return 0
                start = end = DummyEvent()

            times = dataset.time()
            for index, (image_tar, image_lr, *lead_time_label) in enumerate(iter(data_loader)):
                time_index += 1
                if dist.rank == 0:
                    logger0.info(f"starting index: {time_index}")

                if time_index == warmup_steps:
                    start.record()

                if lead_time_label:
                    lead_time_label = lead_time_label[0].to(dist.device).contiguous()
                else:
                    lead_time_label = None

                image_lr = image_lr.to(device=device).to(torch.float32).to(memory_format=torch.channels_last)
                image_tar = image_tar.to(device=device).to(torch.float32)
                image_out = generate_fn()

                if dist.rank == 0:
                    batch_size = image_out.shape[0]
                    if cfg.generation.perf.io_syncronous:
                        writer_threads.append(
                            writer_executor.submit(
                                save_images,
                                writer,
                                dataset,
                                list(times),
                                image_out.cpu(),
                                image_tar.cpu(),
                                image_lr.cpu(),
                                time_index,
                                index,
                                has_lead_time,
                            )
                        )
                    else:
                        save_images(
                            writer, dataset, list(times), image_out.cpu(), image_tar.cpu(), image_lr.cpu(), time_index, index, has_lead_time
                        )
            end.record()
            end.synchronize()
            elapsed_time = (start.elapsed_time(end) / 1000.0 if use_cuda_timing else 0)
            timed_steps = time_index + 1 - warmup_steps
            if dist.rank == 0 and use_cuda_timing:
                average_time_per_batch_element = elapsed_time / timed_steps / batch_size
                logger.info(f"Total time to run {timed_steps} steps and {batch_size} members = {elapsed_time} s")
                logger.info(f"Average time per batch element = {average_time_per_batch_element} s")

            if dist.rank == 0 and cfg.generation.perf.io_syncronous:
                for thread in list(writer_threads):
                    thread.result()
                    writer_threads.remove(thread)
                writer_executor.shutdown()

    if dist.rank == 0:
        f.close()
    logger0.info("Generation Completed.")


if __name__ == "__main__":
    main()
