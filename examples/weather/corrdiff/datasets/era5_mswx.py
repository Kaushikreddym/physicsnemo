import os
import glob
import numpy as np
import xarray as xr
import torch
from typing import List, Tuple, Optional
import cv2
import datetime
import cftime
import json

from datasets.base import ChannelMetadata, DownscalingDataset


class era5_mswx(DownscalingDataset):
    """
    ERA5 → MSWX downscaling dataset (full-resolution, no patching).
    
    ✅ Returns full HR MSWX and full HR ERA5 (interpolated)
    ✅ Low-resolution version created internally using subsampling + bicubic resize
    ✅ Supports static channels: elevation, lsm (when included in input_channels)
    ✅ No cropping, no patch extraction, no center-latlon logic
    """

    # ----------------------------------------------------
    # 📆 Time helpers
    # ----------------------------------------------------
    def convert_datetime_to_cftime(self, time: datetime.datetime, cls=cftime.DatetimeGregorian):
        return cls(time.year, time.month, time.day, time.hour, time.minute, time.second)

    @staticmethod
    def _extract_date_from_filename(filename: str) -> datetime.datetime:
        base = os.path.basename(filename)
        if base.startswith("E5pl00"):   # ERA5
            parts = base.split("_")
            return datetime.datetime.strptime(parts[2], "%Y-%m-%d")
        else:                           # MSWX YYYYDOY.nc
            parts = base.split(".")
            return datetime.datetime.strptime(parts[0], "%Y%j")

    # ----------------------------------------------------
    # 🌍 Extent / cropping utilities
    # ----------------------------------------------------
    def _get_extent(self, filename):
        ds = xr.open_dataset(filename)
        lat_min, lat_max = float(np.min(ds.lat)), float(np.max(ds.lat))
        lon_min, lon_max = float(np.min(ds.lon)), float(np.max(ds.lon))
        self.era5_box = (lat_min, lat_max, lon_min, lon_max)
        return self.era5_box

    @staticmethod
    def _crop_box(ds, box):
        lat_min, lat_max, lon_min, lon_max = box
        return ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    @staticmethod
    def _fix_longitude(ds):
        """
        Detects if longitude is in 0–360 format and converts to -180–180.

        Parameters
        ----------
        ds : xarray.Dataset or xarray.DataArray

        Returns
        -------
        ds_fixed : xarray.Dataset
        """
        if "lon" not in ds.coords:
            return ds

        lon = ds["lon"].values

        # Check if longitudes are 0–360
        if lon.max() > 180:
            # Convert to -180–180
            lon_new = ((lon + 180) % 360) - 180

            ds = ds.assign_coords(lon=lon_new)

            # Sort longitudes so they increase monotonically
            ds = ds.sortby("lon")

        return ds
    # ----------------------------------------------------
    # ✅ Initialize dataset (fully cleaned)
    # ----------------------------------------------------
    def __init__(
        self,
        data_path: str,
        train: bool = True,
        train_years: Tuple[int, int] = (1999, 2020),
        val_years: Tuple[int, int] = (2021, 2024),
        input_channels: Optional[List[str]] = None,
        output_channels: Optional[List[str]] = None,
        static_channels: Optional[List[str]] = None,
        normalize: bool = True,
        stats_era5: Optional[str] = None,
        stats_mswx: Optional[str] = None,
    ):
        self.data_path = data_path
        self.normalize = normalize

        # -----------------------------------------
        # ERA5 files
        # -----------------------------------------
        era5_files = sorted(glob.glob(os.path.join(data_path, "era5", "combined", "E5pl00_1D_*.nc")))
        if not era5_files:
            raise FileNotFoundError(f"No ERA5 files found in {data_path}")

        era5_times = [self._extract_date_from_filename(f) for f in era5_files]

        # -----------------------------------------
        # MSWX files (per-channel)
        # -----------------------------------------
        mswx_files = {}
        mswx_times = {}

        for ch in output_channels:
            mswx_files[ch] = sorted(glob.glob(os.path.join(data_path, "mswx", ch, "*.nc")))
            if not mswx_files[ch]:
                raise FileNotFoundError(f"No MSWX files found for channel {ch}")

            mswx_times[ch] = [self._extract_date_from_filename(f) for f in mswx_files[ch]]

        self.era5_files = era5_files
        self.mswx_files = mswx_files

        # -----------------------------------------
        # Intersect dates
        # -----------------------------------------
        common = set(era5_times)
        for ch in output_channels:
            common = common & set(mswx_times[ch])

        self.common_times = sorted(list(common))

        if train:
            start_year, end_year = train_years
        else:
            start_year, end_year = val_years

        self.files = [
            f for f in mswx_files[output_channels[0]]
            if (dt := self._extract_date_from_filename(f)) in self.common_times
            and start_year <= dt.year <= end_year
        ]

        self.times = [self.convert_datetime_to_cftime(self._extract_date_from_filename(f)) for f in self.files]

        # -----------------------------------------
        # Compute lat/lon grid from MSWX (full grid)
        # -----------------------------------------
        self._get_extent(era5_files[0])

        # Define factor for UNet (number of downsampling layers)
        factor = 16

        with xr.open_dataset(self.files[0]) as ds:
            if ds.lat.values[0] > ds.lat.values[-1]:
                ds = ds.sortby("lat")

            # crop MSWX to ERA5 domain (full field)
            ds = self._crop_box(ds, self.era5_box)

            h, w = len(ds.lat), len(ds.lon)

            # crop so height and width are divisible by factor
            new_h = (h // factor) * factor
            new_w = (w // factor) * factor
            lat_1d = ds.lat.values[:new_h]
            lon_1d = ds.lon.values[:new_w]

        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        self.lat = lat2d
        self.lon = lon2d

        # -----------------------------------------
        # Static features: elevation + land–sea mask
        # -----------------------------------------
        with xr.open_dataset("/data01/FDS/muduchuru/GMTED/GMTED2010_15n015_00625deg.nc") as ds_elev:
            ds_elev = ds_elev.rename({"latitude": "lat", "longitude": "lon"})
            ds_elev = self._fix_longitude(ds_elev)
            ds_elev = ds_elev.interp(lat=lat_1d, lon=lon_1d)
            self.elev = ds_elev

        with xr.open_dataset("/beegfs/muduchuru/data/imerg/IMERG_land_sea_mask.nc") as ds_lsm:
            ds_lsm = self._fix_longitude(ds_lsm)
            ds_lsm = ds_lsm.interp(lat=lat_1d, lon=lon_1d)
            self.lsm = ds_lsm

        # -----------------------------------------
        # Channels
        # -----------------------------------------
        self.input_channels_list = input_channels
        self.output_channels_list = output_channels
        self.static_channels_list = static_channels

        # -----------------------------------------
        # Load normalization statistics
        # -----------------------------------------
        if stats_era5 is not None and os.path.exists(stats_era5):
            with open(stats_era5, "r") as f:
                stats = json.load(f)
            input_mean_list = [stats[ch]["mean"] for ch in self.input_channels_list]
            input_std_list = [stats[ch]["std"] for ch in self.input_channels_list]
        else:
            input_mean_list = [0.0] * len(self.input_channels_list)
            input_std_list = [1.0] * len(self.input_channels_list)

        # Add mean/std for static channels if present
        if self.static_channels_list is not None:
            for ch in self.static_channels_list:
                if ch == "elevation":
                    input_mean_list.append(self.elev["elevation"].values.mean())
                    input_std_list.append(self.elev["elevation"].values.std())
                elif ch == "lsm":
                    varname = "landseamask"
                    input_mean_list.append(self.lsm[varname].values.mean())
                    input_std_list.append(self.lsm[varname].values.std())

        self.input_mean = np.array(input_mean_list)[:, None, None]
        self.input_std = np.array(input_std_list)[:, None, None]

        if stats_mswx is not None and os.path.exists(stats_mswx):
            with open(stats_mswx, "r") as f:
                stats = json.load(f)
            self.output_mean = np.array([stats[ch]["mean"] for ch in self.output_channels_list])[:, None, None]
            self.output_std = np.array([stats[ch]["std"] for ch in self.output_channels_list])[:, None, None]
        else:
            self.output_mean = 0.0
            self.output_std = 1.0

    # ----------------------------------------------------
    # ✅ Data Access
    # ----------------------------------------------------
    def __len__(self):
        return len(self.files)

    def _get_era5(self, t):
        tstr = t.strftime("%Y-%m-%d")
        file_match = next(f for f in self.era5_files if tstr in f)
        ds = xr.open_dataset(file_match).isel(time=0)
        return ds

    def _get_mswx(self, t):
        tstr = t.strftime("%Y%j")
        datasets = []

        for ch in self.output_channels_list:
            file_match = next(f for f in self.mswx_files[ch] if tstr in f)
            with xr.open_dataset(file_match) as ds:
                if ds.lat.values[0] > ds.lat.values[-1]:
                    ds = ds.sortby("lat")
                ds = self._crop_box(ds, self.era5_box)
                var_name = list(ds.data_vars)[0]
                datasets.append(ds[var_name].load().isel(time=0))

        return xr.concat(datasets, dim="channel").assign_coords(channel=self.output_channels_list)
    def _crop_for_unet(self, arr: np.ndarray, factor: int = 16) -> np.ndarray:
        """
        Crop array so height and width are divisible by `factor` (default 16 for 4 downsampling layers).
        arr: np.ndarray [C, H, W]
        """
        _, h, w = arr.shape
        new_h = (h // factor) * factor
        new_w = (w // factor) * factor
        return arr[:, :new_h, :new_w]
    # ----------------------------------------------------
    # ✅ Main data loader (full fields)
    # ----------------------------------------------------

    def _get_center_indices(self, lats, lons, lat0, lon0, ph, pw):
        """Find top-left corner indices for a patch centered on (lat0, lon0)."""
        iy = np.argmin(np.abs(lats - lat0))
        ix = np.argmin(np.abs(lons - lon0))

        # Ensure patch fits inside the domain
        iy = np.clip(iy, ph // 2, len(lats) - ph // 2)
        ix = np.clip(ix, pw // 2, len(lons) - pw // 2)

        top = int(iy - ph // 2)
        left = int(ix - pw // 2)
        return top, left
    def __getitem__(self, idx):
        date = self.times[idx]

        # HR MSWX
        ds_mswx = self._get_mswx(date)
        arr_mswx = ds_mswx.values.astype(np.float32)

        # HR ERA5 interpolated to MSWX grid
        ds_era5 = self._get_era5(date).interp(lat=ds_mswx.lat, lon=ds_mswx.lon)
        arr_era5 = ds_era5["image"].sel(channel=self.input_channels_list).values.astype(np.float32)
        # import ipdb; ipdb.set_trace()
        # Crop for UNet compatibility
        arr_era5 = self._crop_for_unet(arr_era5)
        arr_mswx = self._crop_for_unet(arr_mswx)
        # Add static channels if requested
        static_channels = []

        if "elevation" in self.static_channels_list:
            static_channels.append(self.elev["elevation"].values.astype(np.float32))

        if "lsm" in self.static_channels_list:
            # Adjust variable name if different
            varname = list(self.lsm.data_vars)[0]
            static_channels.append(self.lsm[varname].values.astype(np.float32))

        if static_channels:
            arr_static = np.stack(static_channels, axis=0)
            arr_era5 = np.concatenate([arr_era5, arr_static], axis=0)

        # Normalize
        input_arr = self.normalize_input(arr_era5)
        output_arr = self.normalize_output(arr_mswx)
        # # --- 🔹 Cropping logic ---
        # if self.patch_size is not None:
        #     ph, pw = self.patch_size
        #     h, w = input_arr.shape[-2: ]

        #     if ph > h or pw > w:
        #         raise ValueError(f"Patch size {self.patch_size} larger than image {h, w}")

        #     if self.center_latlon is not None:
        #         lat0, lon0 = self.center_latlon
        #         top, left = self._get_center_indices(ds_mswx.lat.values, ds_mswx.lon.values, lat0, lon0, ph, pw)
        #     else:
        #         top = np.random.randint(0, h - ph + 1)
        #         left = np.random.randint(0, w - pw + 1)

        #     input_arr = input_arr[:, top:top + ph, left:left + pw]
        #     output_arr = output_arr[:, top:top + ph, left:left + pw]
        # Create LR version
        input_arr = self._create_lowres_(input_arr, factor=4)
        lead_time_label = 0

        return output_arr, input_arr, lead_time_label

    # ----------------------------------------------------
    # ✅ Normalization
    # ----------------------------------------------------
    def normalize_input(self, x):
        if self.normalize:
            return (x - self.input_mean) / self.input_std
        return x

    def normalize_output(self, x):
        if self.normalize:
            return (x - self.output_mean) / self.output_std
        return x

    # ----------------------------------------------------
    # ✅ Meta
    # ----------------------------------------------------
    def input_channels(self):
        channels = self.input_channels_list.copy()
        if hasattr(self, "static_channels_list") and self.static_channels_list is not None:
            channels += self.static_channels_list
        return [ChannelMetadata(name=n) for n in channels]

    def output_channels(self):
        return [ChannelMetadata(name=n) for n in self.output_channels_list]

    def time(self):
        return self.times

    def image_shape(self):
        """Return full image shape (H, W)."""
        return self.lat.shape

    def info(self):
        return {
            "input_normalization": (self.input_mean.squeeze(), self.input_std.squeeze()),
            "target_normalization": (self.output_mean.squeeze(), self.output_std.squeeze()),
        }
    def longitude(self) -> np.ndarray:
        return self.lon

    def latitude(self) -> np.ndarray:
        return self.lat

    # ----------------------------------------------------
    # ✅ Downscaling utility (LR creation)
    # ----------------------------------------------------
    @staticmethod
    def _create_lowres_(x, factor=4):
        x = x.transpose(1, 2, 0)  # CHW → HWC
        x = x[::factor, ::factor, :]
        x = cv2.resize(x, (x.shape[1] * factor, x.shape[0] * factor), interpolation=cv2.INTER_CUBIC)
        x = x.transpose(2, 0, 1)  # HWC → CHW
        return x
