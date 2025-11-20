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
import json

class era5_mswx(DownscalingDataset):
    """
    Dataset for downscaling using ERA5 (input) and MSWX (target).
    Each ERA5 file is a combined daily file (shape: channel, lat, lon).
    Each MSWX file corresponds to the same date (format: YYYYDOY.nc).
    """
    # Extract year from filenames and filter
    def get_year(self, filename):
        basename = os.path.basename(filename)
        parts = basename.split("_")
        dt = datetime.datetime.strptime(parts[2], "%Y-%m-%d")
        return dt.year
    def __init__(
        self,
        data_path: str,
        train: bool = True,
        train_years: Tuple[int, int] = (1999, 2020),
        val_years: Tuple[int, int] = (2021, 2024),
        input_channels: Optional[List[str]] = None,
        output_channels: Optional[List[str]] = None,
        normalize: bool = True,
        stats_era5: Optional[str] = None,
        stats_mswx: Optional[str] = None,
        patch_size: Optional[Tuple[int, int]] = (128,128),
        center_latlon: Optional[Tuple[float, float]] =None,
    ):
        self.data_path = data_path
        self.normalize = normalize
        self.patch_size = patch_size
        self.center_latlon = center_latlon
        # ----------------------------------------------------
        # fetch ERA5 files
        # ----------------------------------------------------
        era5_files = sorted(glob.glob(os.path.join(data_path, 'era5','combined', "E5pl00_1D_*.nc")))
        if not era5_files:
            raise FileNotFoundError(f"No combined ERA5 files found in {data_path}")
        else:
            era5_times = [self._extract_date_from_filename(f) for f in era5_files]
        # ----------------------------------------------------
        # fetch MSWX files
        # ----------------------------------------------------
        mswx_files = {}
        mswx_times = {}
        for ch in output_channels:
            mswx_files[ch] = sorted(glob.glob(os.path.join(data_path,'mswx',ch, "*.nc")))
            if not mswx_files[ch]:
                raise FileNotFoundError(f"No {ch}-MSWX files found in {data_path}")
            else:
                mswx_times[ch] = [self._extract_date_from_filename(f) for f in mswx_files[ch]]
        self.era5_files = era5_files
        self.mswx_files = mswx_files
        self.common_times = sorted(list(set(era5_times) & set.intersection(*[set(mswx_times[ch]) for ch in output_channels])))

        if train:
            start_year, end_year = train_years
        else:
            start_year, end_year = val_years
        
        self.files = [
            f for f in mswx_files[output_channels[0]] 
            if self._extract_date_from_filename(f) in self.common_times 
            and start_year <= self._extract_date_from_filename(f).year <= end_year
        ]
        self.times = [self.convert_datetime_to_cftime(self._extract_date_from_filename(f)) for f in self.files]
            
        # Compute and store ERA5 extent
        era5_window = self._get_extent(era5_files[0])
        with xr.open_dataset(self.files[0]) as ds:
            if ds.lat.values[0] > ds.lat.values[-1]:
                ds = ds.sortby("lat")
            ds = self._crop_box(ds, era5_window)
            lat_vals = ds.lat.values
            lon_vals = ds.lon.values

            ph, pw = self.patch_size if self.patch_size else (128, 128)
            if hasattr(self, "center_latlon") and self.center_latlon is not None:
                lat0, lon0 = self.center_latlon
                top, left = self._get_center_indices(lat_vals, lon_vals, lat0, lon0, ph, pw)
                lat_1d = lat_vals[top:top + ph]
                lon_1d = lon_vals[left:left + pw]
            else:
                lat_1d = lat_vals[:ph]
                lon_1d = lon_vals[:pw]

        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        self.lat = lat2d
        self.lon = lon2d

        with xr.open_dataset("/data01/FDS/muduchuru/GMTED/GMTED2010_15n015_00625deg.nc") as ds_elev:
            ds_elev = ds_elev.rename({"latitude":"lat", "longitude":"lon"})
            # ds_elev = self._crop_box(ds_elev,era5_window)
            ds_elev = ds_elev.interp(lat=lat_1d,lon=lon_1d)
        self.elev = ds_elev

        with xr.open_dataset("/beegfs/muduchuru/data/imerg/IMERG_land_sea_mask.nc") as ds_lsm:
            # ds_lsm = self._crop_box(ds_lsm,era5_window)
            ds_lsm = ds_lsm.interp(lat=lat_1d,lon=lon_1d)
        self.lsm = ds_lsm

        # ----------------------------------------------------
        # 🔹 Channel setup
        # ----------------------------------------------------
        self.input_channels_list = input_channels
        self.output_channels_list = output_channels

        # ----------------------------------------------------
        # 🔹 Load normalization stats
        # ----------------------------------------------------
        if stats_era5 is not None and os.path.exists(stats_era5):
            with open(stats_era5, "r") as f:
                stats = json.load(f)
                print("reading stats_era5")
            self.input_mean = np.array([stats[ch]["mean"] for ch in self.input_channels_list])[:, None, None].astype(np.float32)
            self.input_std = np.array([stats[ch]["std"] for ch in self.input_channels_list])[:, None, None].astype(np.float32)
        else:
            self.input_mean = 0.0
            self.input_std = 1.0
        if stats_mswx is not None and os.path.exists(stats_mswx):
            with open(stats_mswx, "r") as f:
                stats = json.load(f)
                print("reading stats_mswx")
            self.output_mean = np.array([stats[ch]["mean"] for ch in self.output_channels_list])[:, None, None].astype(np.float32)
            self.output_std = np.array([stats[ch]["std"] for ch in self.output_channels_list])[:, None, None].astype(np.float32)
        else:
            self.output_mean = 0.0
            self.output_std = 1.0

    # ----------------------------------------------------
    # 📆 Helpers for time handling
    # ----------------------------------------------------
    def convert_datetime_to_cftime(self, time: datetime, cls=cftime.DatetimeGregorian) -> cftime.DatetimeGregorian:
        return cls(time.year, time.month, time.day, time.hour, time.minute, time.second)

    @staticmethod
    def _extract_date_from_filename(filename: str) -> datetime.date:
        base = os.path.basename(filename)
        if base.startswith("E5pl00"):
            parts = base.split("_")
            return datetime.datetime.strptime(parts[2], "%Y-%m-%d")
        else:
            parts = base.split(".")
            return datetime.datetime.strptime(parts[0], "%Y%j")

    # ----------------------------------------------------
    # 🌍 Extent and cropping
    # ----------------------------------------------------
    def _get_extent(self, filename):
        ds = xr.open_dataset(filename)        
        lat_min, lat_max = float(np.min(ds.lat)), float(np.max(ds.lat))
        lon_min, lon_max = float(np.min(ds.lon)), float(np.max(ds.lon))
        self.era5_box = (lat_min, lat_max, lon_min, lon_max)
        return self.era5_box

    @staticmethod
    def _crop_box(ds: xr.Dataset | xr.DataArray, box: tuple[float, float, float, float]):
        lat_min, lat_max, lon_min, lon_max = box
        return ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # ----------------------------------------------------
    # 🧠 Core methods
    # ----------------------------------------------------
    def __len__(self):
        return len(self.files)
    def _get_era5(self, t):
        tstr = t.strftime("%Y-%m-%d")      
        file_match = next((f for f in self.era5_files if tstr in f), None)
        ds = xr.open_dataset(file_match).isel(time=0)

        return ds
    def _get_mswx(self, t):
        tstr = t.strftime("%Y%j")
        datasets = []

        for ch in self.output_channels_list:
            file_match = next((f for f in self.mswx_files[ch] if tstr in f), None)
            if file_match is None:
                raise FileNotFoundError(f"No MSWX file found for {ch} at {tstr}")

            with xr.open_dataset(file_match) as ds:
                # Flip latitude if descending
                if ds.lat.values[0] > ds.lat.values[-1]:
                    ds = ds.sortby("lat")
                ds = self._crop_box(ds, self.era5_box)
                var_name = list(ds.data_vars)[0]   # automatic selection
                datasets.append(ds[var_name].load().isel(time=0))

        # Stack along a new 'channel' dimension and assign channel names
        label = xr.concat(datasets, dim="channel").assign_coords(channel=self.output_channels_list)
        label.name = "label"
        return label
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
        # ---- ERA5 input ----
        date = self.times[idx]
        
        ds_mswx = self._get_mswx(date)        
        arr_mswx = ds_mswx.values.astype(np.float32)
    
        ds_era5 = self._get_era5(date).interp(lat=ds_mswx.lat,lon=ds_mswx.lon)
        arr_era5 = ds_era5["image"].sel(channel=self.input_channels_list).values.astype(np.float32)
        # ---- Normalize ----
        input_arr = self.normalize_input(arr_era5)
        output_arr = self.normalize_output(arr_mswx)

        # --- 🔹 Cropping logic ---
        if self.patch_size is not None:
            ph, pw = self.patch_size
            h, w = input_arr.shape[-2:]

            if ph > h or pw > w:
                raise ValueError(f"Patch size {self.patch_size} larger than image {h, w}")

            if self.center_latlon is not None:
                lat0, lon0 = self.center_latlon
                top, left = self._get_center_indices(ds_mswx.lat.values, ds_mswx.lon.values, lat0, lon0, ph, pw)
            else:
                top = np.random.randint(0, h - ph + 1)
                left = np.random.randint(0, w - pw + 1)

            input_arr = input_arr[:, top:top + ph, left:left + pw]
            output_arr = output_arr[:, top:top + ph, left:left + pw]

        lead_time_label = 0
        
        input_arr = self._create_lowres_(input_arr,4)

        return output_arr, input_arr, lead_time_label

    # ----------------------------------------------------
    # ⚙️ Normalization helpers
    # ----------------------------------------------------
    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        if self.normalize:
            return (x - self.input_mean) / self.input_std
        return x

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        if self.normalize:
            return x * self.input_std + self.input_mean
        return x

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        if self.normalize:
            return (x - self.output_mean) / self.output_std
        return x

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        if self.normalize:
            return x * self.output_std + self.output_mean
        return x

    # ----------------------------------------------------
    # 🧩 Miscellaneous
    # ----------------------------------------------------
    def longitude(self) -> np.ndarray:
        return self.lon

    def latitude(self) -> np.ndarray:
        return self.lat

    def input_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=n) for n in self.input_channels_list]

    def output_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=n) for n in self.output_channels_list]

    def time(self) -> List:
        return self.times

    def image_shape(self) -> Tuple[int, int]:
        return (128, 128)

    def info(self) -> dict:
        return {
            "input_normalization": (self.input_mean.squeeze(), self.input_std.squeeze()),
            "target_normalization": (self.output_mean.squeeze(), self.output_std.squeeze()),
        }

    # ----------------------------------------------------
    # 🔽 Downscaling utility
    # ----------------------------------------------------
    @staticmethod
    def _create_lowres_(x, factor=4):
        x = x.transpose(1, 2, 0)
        x = x[::factor, ::factor, :]  # subsample
        x = cv2.resize(x, (x.shape[1] * factor, x.shape[0] * factor), interpolation=cv2.INTER_CUBIC)
        x = x.transpose(2, 0, 1)
        return x
