import os
import glob
import numpy as np
import xarray as xr
import torch
from typing import List, Tuple, Optional
import cv2
from datasets.base import ChannelMetadata, DownscalingDataset
import datetime
import cftime


class era5(DownscalingDataset):
    """
    ERA5 daily combined files stored as:
      image[channel, lat, lon]
    Each file corresponds to one day.
    """

    def __init__(
        self,
        data_path: str,
        train: bool = True,
        train_years: Tuple[int, int] = (1999, 2020),
        val_years: Tuple[int, int] = (2021, 2024),
        input_channels: Optional[List[str]] = None,
        output_channels: Optional[List[str]] = None,
        normalize: bool = True,
        stats_path: Optional[str] = None,
        patch_size: Optional[Tuple[int, int]] = (128, 128),
        center_latlon: Optional[Tuple[float, float]] = None,
    ):
        self.data_path = data_path
        self.normalize = normalize
        self.patch_size = patch_size
        self.center_latlon = center_latlon

        # ---------------- File List Filter by Year ----------------
        all_files = sorted(glob.glob(os.path.join(data_path, "E5pl00_1D_*.nc")))
        if not all_files:
            raise FileNotFoundError(f"No ERA5 files found in {data_path}")

        def get_year(filepath):
            basename = os.path.basename(filepath)
            return int(basename.split("_")[2].split("-")[0])

        start_year, end_year = train_years if train else val_years
        self.files = [f for f in all_files if start_year <= get_year(f) <= end_year]
        if not self.files:
            raise ValueError(f"No files found for {start_year}-{end_year}")

        # ---------------- Read Metadata ----------------
        with xr.open_dataset(self.files[0]) as ds:
            self.channels = list(ds.channel.values)
            lat = ds.image.lat.values
            lon = ds.image.lon.values

        lon2d, lat2d = np.meshgrid(lon, lat)
        self.lat_full = lat2d
        self.lon_full = lon2d

        self.top = None
        self.left = None

        # ---------------- Region-of-Interest Crop ----------------
        if self.patch_size:
            ph, pw = self.patch_size

            if self.center_latlon:
                self.top, self.left = self._get_center_indices(
                    self.lat_full, self.lon_full, *self.center_latlon, ph, pw
                )
            else:
                # default: use full grids (random crop later)
                self.top, self.left = 0, 0

            self.lat = self.lat_full[self.top:self.top + ph, self.left:self.left + pw]
            self.lon = self.lon_full[self.top:self.top + ph, self.left:self.left + pw]
        else:
            self.lat = self.lat_full
            self.lon = self.lon_full

        # ---------------- Channels ----------------
        self.input_channels_list = input_channels or self.channels
        self.output_channels_list = output_channels or self.channels

        # ---------------- Normalization stats ----------------
        if normalize and stats_path and os.path.exists(stats_path):
            import json
            with open(stats_path, "r") as f:
                stats = json.load(f)

            def get_stats(channel_list):
                mean = np.array([stats[ch]["mean"] for ch in channel_list], dtype=np.float32)[:, None, None]
                std = np.array([stats[ch]["std"] for ch in channel_list], dtype=np.float32)[:, None, None]
                return mean, std

            self.input_mean, self.input_std = get_stats(self.input_channels_list)
            self.output_mean, self.output_std = get_stats(self.output_channels_list)
        else:
            self.input_mean = self.input_std = 1.0
            self.output_mean = self.output_std = 1.0

        # ---------------- Time Metadata ----------------
        self.times = [self._extract_time_from_filename(f) for f in self.files]

    # ----------------- Filename Time Extraction -----------------
    def convert_datetime_to_cftime(self, time: datetime, cls=cftime.DatetimeGregorian):
        return cls(time.year, time.month, time.day)

    def _extract_time_from_filename(self, filename):
        date_str = os.path.basename(filename).split("_")[2]
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return self.convert_datetime_to_cftime(dt)

    # ----------------- Spatial Index Utility -----------------
    @staticmethod
    def _get_center_indices(lat2d, lon2d, center_lat, center_lon, ph, pw):
        dist = (lat2d - center_lat) ** 2 + (lon2d - center_lon) ** 2
        cy, cx = np.unravel_index(np.argmin(dist), dist.shape)
        top = max(0, min(cy - ph // 2, lat2d.shape[0] - ph))
        left = max(0, min(cx - pw // 2, lat2d.shape[1] - pw))
        return top, left

    # ----------------- Dataset API -----------------
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        ds = xr.open_dataset(file)
        arr = ds["image"].values.astype(np.float32)  # (1, C, H, W)
        ds.close()

        input_arr = arr[0, [self.channels.index(ch) for ch in self.input_channels_list]]
        output_arr = arr[0, [self.channels.index(ch) for ch in self.output_channels_list]]

        # Normalization
        input_arr = (input_arr - self.input_mean) / self.input_std
        output_arr = (output_arr - self.output_mean) / self.output_std

        ph, pw = self.patch_size

        if self.center_latlon:
            # ✅ fixed ROI crop
            top, left = self.top, self.left
        else:
            # ✅ random crop during training
            H, W = input_arr.shape[-2:]
            top = np.random.randint(0, H - ph + 1)
            left = np.random.randint(0, W - pw + 1)

        # Apply spatial crop
        input_arr = input_arr[:, top:top + ph, left:left + pw]
        output_arr = output_arr[:, top:top + ph, left:left + pw]

        # Low-res degraded LR input
        input_arr = self._create_lowres_(input_arr, factor=8)
        lead_time_label = 0  # static dataset for now

        return output_arr, input_arr, lead_time_label

    def longitude(self): return self.lon
    def latitude(self): return self.lat
    def time(self): return self.times
    def image_shape(self): return self.lat.shape  # ✅ real shape

    def input_channels(self): return [ChannelMetadata(name=n) for n in self.input_channels_list]
    def output_channels(self): return [ChannelMetadata(name=n) for n in self.output_channels_list]

    def info(self): return {
        "input_norm": (self.input_mean.squeeze(), self.input_std.squeeze()),
        "target_norm": (self.output_mean.squeeze(), self.output_std.squeeze()),
        "patch_size": self.patch_size,
        "center_crop": self.center_latlon is not None,
    }

    # ----------------- Low-res generator -----------------
    @staticmethod
    def _create_lowres_(x, factor=8):
        # reduce & upscale using bicubic
        x = x.transpose(1, 2, 0)
        x = x[::factor, ::factor]
        x = cv2.resize(x, (x.shape[1] * factor, x.shape[0] * factor), interpolation=cv2.INTER_CUBIC)
        return x.transpose(2, 0, 1)
