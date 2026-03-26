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
from pathlib import Path

from datasets.base import ChannelMetadata, DownscalingDataset


class cmip6_to_10km(DownscalingDataset):
    """
    CMIP6 → 10km downscaling dataset using ERA5-trained model.
    
    ✅ Loads CMIP6 data from /data01/FDS/muduchuru/Atmos/CMIP6/data
    ✅ Interpolates CMIP6 to ERA5 100km grid
    ✅ Uses ERA5→MSWX trained model for downscaling
    ✅ Outputs at MSWX 10km resolution
    ✅ Supports static channels: elevation, lsm
    """

    def convert_datetime_to_cftime(self, time: datetime.datetime, cls=cftime.DatetimeGregorian):
        return cls(time.year, time.month, time.day, time.hour, time.minute, time.second)

    @staticmethod
    def _fix_longitude(ds):
        """Convert longitude from 0–360 format to -180–180."""
        if "lon" not in ds.coords:
            return ds

        lon = ds["lon"].values

        if lon.max() > 180:
            lon_new = ((lon + 180) % 360) - 180
            ds = ds.assign_coords(lon=lon_new)
            ds = ds.sortby("lon")

        return ds

    def _get_extent(self, ds):
        """Get spatial extent from ERA5 reference grid."""
        lat_min, lat_max = float(np.min(ds.lat)), float(np.max(ds.lat))
        lon_min, lon_max = float(np.min(ds.lon)), float(np.max(ds.lon))
        self.era5_box = (lat_min, lat_max, lon_min, lon_max)
        return self.era5_box

    @staticmethod
    def _crop_box(ds, box):
        lat_min, lat_max, lon_min, lon_max = box
        return ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    def __init__(
        self,
        cmip6_data_path: str,
        era5_reference_file: str,  # ERA5 file to get target grid
        mswx_reference_file: str,  # MSWX file to get output grid
        input_channels: Optional[List[str]] = None,
        output_channels: Optional[List[str]] = None,
        static_channels: Optional[List[str]] = None,
        normalize: bool = True,
        stats_era5: Optional[str] = None,
        stats_mswx: Optional[str] = None,
        patch_size: Optional[tuple] = (128, 128),
        center_latlon: Optional[tuple] = None,
        year_range: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize CMIP6 to 10km downscaling dataset.
        
        Parameters
        ----------
        cmip6_data_path : str
            Path to CMIP6 data directory 
            Expected structure: <path>/CMIP6/CMIP/<institution>/<model>/<experiment>/<variant>/day/<variable>/<grid>/<version>/*.nc
            Example: /data01/FDS/muduchuru/Atmos/CMIP6/data/CMIP6/CMIP/NOAA-GFDL/GFDL-ESM4/historical/r1i1p1f1/day/pr/gr1/v20190726/pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1_19300101-19491231.nc
        era5_reference_file : str
            Path to an ERA5 file to extract the 100km grid
        mswx_reference_file : str
            Path to an MSWX file to extract the 10km output grid
        input_channels : List[str]
            CMIP6 variable names to use as input (e.g., ['pr', 'tas'])
        output_channels : List[str]
            Target channel names (for compatibility)
        static_channels : List[str], optional
            Static features to include (elevation, lsm)
        normalize : bool
            Whether to apply normalization
        stats_era5 : str, optional
            Path to ERA5 normalization statistics JSON
        stats_mswx : str, optional
            Path to MSWX normalization statistics JSON
        patch_size : tuple, optional
            Patch size for training/inference
        center_latlon : tuple, optional
            Center coordinates for deterministic patching
        year_range : tuple, optional
            (start_year, end_year) to filter data
        """
        self.cmip6_data_path = cmip6_data_path
        self.normalize = normalize
        self.patch_size = patch_size
        self.center_latlon = center_latlon
        self.last_patch_lat = None
        self.last_patch_lon = None

        self.input_channels_list = input_channels
        self.output_channels_list = output_channels or input_channels
        self.static_channels_list = static_channels

        # -----------------------------------------
        # Load ERA5 reference grid (100km)
        # -----------------------------------------
        with xr.open_dataset(era5_reference_file) as ds_era5:
            # Extract ERA5 grid
            self.era5_lat = ds_era5.lat.values
            self.era5_lon = ds_era5.lon.values
            self._get_extent(ds_era5)

        # -----------------------------------------
        # Load MSWX reference grid (10km output)
        # -----------------------------------------
        factor = 16  # UNet compatibility

        with xr.open_dataset(mswx_reference_file) as ds_mswx:
            if ds_mswx.lat.values[0] > ds_mswx.lat.values[-1]:
                ds_mswx = ds_mswx.sortby("lat")

            ds_mswx = self._crop_box(ds_mswx, self.era5_box)

            h, w = len(ds_mswx.lat), len(ds_mswx.lon)
            new_h = (h // factor) * factor
            new_w = (w // factor) * factor
            
            lat_1d = ds_mswx.lat.values[:new_h]
            lon_1d = ds_mswx.lon.values[:new_w]

        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        self.lat = lat2d
        self.lon = lon2d

        # -----------------------------------------
        # Discover CMIP6 files
        # -----------------------------------------
        self.cmip6_files = self._discover_cmip6_files(cmip6_data_path, input_channels)
        
        # Filter by year range if provided
        if year_range is not None:
            start_year, end_year = year_range
            self.cmip6_files = [
                f for f in self.cmip6_files
                if start_year <= self._extract_year_from_cmip6(f) <= end_year
            ]

        if not self.cmip6_files:
            raise FileNotFoundError(f"No CMIP6 files found in {cmip6_data_path}")

        # Extract timestamps
        self.times = [self._extract_time_from_cmip6(f) for f in self.cmip6_files]

        # -----------------------------------------
        # Static features: elevation + land–sea mask
        # -----------------------------------------
        with xr.open_dataset("/data01/FDS/muduchuru/Land/GMTED/GMTED2010_15n015_00625deg.nc") as ds_elev:
            ds_elev = ds_elev.rename({"latitude": "lat", "longitude": "lon"})
            ds_elev = self._fix_longitude(ds_elev)
            ds_elev = ds_elev.interp(lat=lat_1d, lon=lon_1d)
            self.elev = ds_elev

        with xr.open_dataset("/data01/FDS/muduchuru/Atmos/IMERG/IMERG_land_sea_mask.nc") as ds_lsm:
            ds_lsm = self._fix_longitude(ds_lsm)
            ds_lsm = ds_lsm.interp(lat=lat_1d, lon=lon_1d)
            self.lsm = ds_lsm

        # -----------------------------------------
        # Load normalization statistics
        # -----------------------------------------
        self._setup_normalization(stats_era5, stats_mswx)

    def _discover_cmip6_files(self, data_path: str, channels: List[str]) -> List[str]:
        """
        Discover CMIP6 files for the specified channels.
        CMIP6 structure: data_path/CMIP6/CMIP/<institution>/<model>/<experiment>/<variant>/day/<variable>/<grid>/<version>/*.nc
        Returns a dictionary mapping (institution, model, experiment, variant, variable) to list of files
        """
        all_files = []
        
        # Recursively find all CMIP6 files for the specified variables
        for ch in channels:
            # Search pattern: data_path/**/day/<variable>/**/*.nc
            pattern = os.path.join(data_path, "**", "day", ch, "**", "*.nc")
            files = sorted(glob.glob(pattern, recursive=True))
            
            if not files:
                raise FileNotFoundError(f"No CMIP6 files found for variable {ch} in {data_path}")
            
            all_files.extend(files)
        
        # Sort by filename to ensure chronological order
        return sorted(all_files)

    def _extract_year_from_cmip6(self, filename: str) -> int:
        """
        Extract year from CMIP6 filename.
        Format: <var>_day_<model>_<experiment>_<variant>_<grid>_<start>-<end>.nc
        Example: pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1_19300101-19491231.nc
        """
        base = os.path.basename(filename)
        parts = base.split("_")
        
        # The date range is typically the second-to-last or last part before .nc
        for part in reversed(parts):
            if "-" in part and ".nc" not in part:
                # Extract start date: YYYYMMDD-YYYYMMDD
                date_str = part.split("-")[0]
                if len(date_str) >= 4:
                    try:
                        return int(date_str[:4])
                    except ValueError:
                        continue
            elif ".nc" in part and "-" in part:
                # Handle case where date is part of filename.nc
                date_str = part.split("-")[0]
                if len(date_str) >= 4:
                    try:
                        return int(date_str[:4])
                    except ValueError:
                        continue
        
        raise ValueError(f"Could not extract year from CMIP6 filename: {filename}")

    def _extract_time_from_cmip6(self, filename: str) -> cftime.DatetimeGregorian:
        """
        Extract time from CMIP6 file.
        Uses the actual time coordinate from the file, handling cftime calendars.
        """
        with xr.open_dataset(filename) as ds:
            if "time" in ds.coords and len(ds.time) > 0:
                time_val = ds.time.values[0]
                
                # Handle different cftime calendar types
                if isinstance(time_val, (cftime.DatetimeGregorian, 
                                        cftime.DatetimeProlepticGregorian,
                                        cftime.DatetimeNoLeap,
                                        cftime.Datetime360Day)):
                    # Convert to Gregorian for consistency
                    return cftime.DatetimeGregorian(
                        time_val.year, time_val.month, time_val.day,
                        time_val.hour, time_val.minute, time_val.second
                    )
                elif isinstance(time_val, np.datetime64):
                    dt = time_val.astype('datetime64[s]').astype(datetime.datetime)
                    return self.convert_datetime_to_cftime(dt)
        
        # Fallback: extract from filename
        year = self._extract_year_from_cmip6(filename)
        return cftime.DatetimeGregorian(year, 1, 1, 0, 0, 0)

    def _setup_normalization(self, stats_era5: Optional[str], stats_mswx: Optional[str]):
        """Setup normalization statistics using ERA5 and MSWX stats."""
        # Input normalization (ERA5 stats for CMIP6 input)
        if stats_era5 is not None and os.path.exists(stats_era5):
            with open(stats_era5, "r") as f:
                stats = json.load(f)
            input_mean_list = [stats[ch]["mean"] for ch in self.input_channels_list]
            input_std_list = [stats[ch]["std"] for ch in self.input_channels_list]
        else:
            input_mean_list = [0.0] * len(self.input_channels_list)
            input_std_list = [1.0] * len(self.input_channels_list)

        # Add static channel stats
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

        # Output normalization (MSWX stats for 10km output)
        if stats_mswx is not None and os.path.exists(stats_mswx):
            with open(stats_mswx, "r") as f:
                stats = json.load(f)
            self.output_mean = np.array([stats[ch]["mean"] for ch in self.output_channels_list])[:, None, None]
            self.output_std = np.array([stats[ch]["std"] for ch in self.output_channels_list])[:, None, None]
        else:
            self.output_mean = 0.0
            self.output_std = 1.0

    def _crop_for_unet(self, arr: np.ndarray, factor: int = 16) -> np.ndarray:
        """Crop array so dimensions are divisible by factor."""
        _, h, w = arr.shape
        new_h = (h // factor) * factor
        new_w = (w // factor) * factor
        return arr[:, :new_h, :new_w]

    def _get_cmip6_data(self, idx: int) -> np.ndarray:
        """
        Load CMIP6 data and interpolate to ERA5 grid.
        
        Returns
        -------
        arr : np.ndarray [C, H, W]
            CMIP6 data interpolated to ERA5 100km grid
        """
        filename = self.cmip6_files[idx]
        
        with xr.open_dataset(filename) as ds:
            # Fix longitude convention (0-360 to -180-180)
            ds = self._fix_longitude(ds)
            
            # Select only the channels we need
            data_arrays = []
            for ch in self.input_channels_list:
                if ch in ds.data_vars:
                    da = ds[ch]
                    
                    # Handle time dimension - take first timestep or daily average
                    if "time" in da.dims:
                        if len(da.time) == 1:
                            da = da.isel(time=0)
                        else:
                            # For files with multiple timesteps, take the first one
                            # In production, you might want to iterate over all timesteps
                            da = da.isel(time=0)
                    
                    data_arrays.append(da)
                else:
                    raise ValueError(f"Channel {ch} not found in CMIP6 file {filename}")
            
            # Stack channels
            if len(data_arrays) > 1:
                ds_stacked = xr.concat(data_arrays, dim="channel").assign_coords(channel=self.input_channels_list)
            else:
                ds_stacked = data_arrays[0].expand_dims("channel").assign_coords(channel=self.input_channels_list)
            
            # Interpolate to ERA5 grid
            ds_interp = ds_stacked.interp(lat=self.era5_lat, lon=self.era5_lon, method="linear")
            
            # Convert to numpy array [C, H, W]
            arr = ds_interp.values.astype(np.float32)
        
        return arr

    def _get_center_indices(self, lats, lons, lat0, lon0, ph, pw):
        """Find top-left corner indices for a patch centered on (lat0, lon0)."""
        iy = np.argmin(np.abs(lats - lat0))
        ix = np.argmin(np.abs(lons - lon0))

        iy = np.clip(iy, ph // 2, len(lats) - ph // 2)
        ix = np.clip(ix, pw // 2, len(lons) - pw // 2)

        top = int(iy - ph // 2)
        left = int(ix - pw // 2)
        return top, left

    def __len__(self):
        return len(self.cmip6_files)

    def __getitem__(self, idx):
        """
        Get CMIP6 data interpolated to ERA5 grid for downscaling.
        
        Returns
        -------
        output_arr : np.ndarray [C, H, W]
            Target (dummy - same as input for inference)
        input_arr : np.ndarray [C + static_channels, H, W]
            CMIP6 data + static channels, ready for model input
        """
        # Load CMIP6 interpolated to ERA5 grid
        arr_cmip6 = self._get_cmip6_data(idx)
        
        # Interpolate to MSWX grid (output resolution)
        ds_cmip6 = xr.DataArray(
            arr_cmip6,
            dims=["channel", "lat", "lon"],
            coords={"channel": self.input_channels_list, "lat": self.era5_lat, "lon": self.era5_lon}
        )
        
        ds_interp = ds_cmip6.interp(lat=self.lat[:, 0], lon=self.lon[0, :])
        arr_cmip6_hr = ds_interp.values.astype(np.float32)
        
        # Crop for UNet compatibility
        arr_cmip6_hr = self._crop_for_unet(arr_cmip6_hr)
        
        # Add static channels
        static_channels = []
        if self.static_channels_list:
            if "elevation" in self.static_channels_list:
                static_channels.append(self.elev["elevation"].values.astype(np.float32))
            if "lsm" in self.static_channels_list:
                varname = list(self.lsm.data_vars)[0]
                static_channels.append(self.lsm[varname].values.astype(np.float32))

        if static_channels:
            arr_static = np.stack(static_channels, axis=0)
            arr_static = self._crop_for_unet(arr_static)
            arr_cmip6_hr = np.concatenate([arr_cmip6_hr, arr_static], axis=0)

        # Normalize
        input_arr = self.normalize_input(arr_cmip6_hr)
        output_arr = input_arr.copy()  # Dummy target for inference
        
        # Apply patching if needed
        if self.patch_size is not None:
            ph, pw = self.patch_size
            h, w = input_arr.shape[-2:]

            if ph > h or pw > w:
                raise ValueError(f"Patch size {self.patch_size} larger than image {h, w}")

            if self.center_latlon is not None:
                lat0, lon0 = self.center_latlon
                top, left = self._get_center_indices(self.lat[:, 0], self.lon[0, :], lat0, lon0, ph, pw)
            else:
                top = np.random.randint(0, h - ph + 1)
                left = np.random.randint(0, w - pw + 1)

            input_arr = input_arr[:, top:top + ph, left:left + pw]
            output_arr = output_arr[:, top:top + ph, left:left + pw]
            
            self.last_patch_lat = self.lat[top:top + ph, left:left + pw]
            self.last_patch_lon = self.lon[top:top + ph, left:left + pw]
        else:
            self.last_patch_lat = self.lat
            self.last_patch_lon = self.lon
        
        # Create LR version (for model input)
        input_arr = self._create_lowres_(input_arr, factor=4)
        
        return output_arr, input_arr

    # ----------------------------------------------------
    # Normalization
    # ----------------------------------------------------
    def normalize_input(self, x):
        if self.normalize:
            return (x - self.input_mean) / self.input_std
        return x

    def normalize_output(self, x):
        if self.normalize:
            return (x - self.output_mean) / self.output_std
        return x

    def denormalize_input(self, x):
        if self.normalize:
            return x * self.input_std + self.input_mean
        return x

    def denormalize_output(self, x):
        if self.normalize:
            return x * self.output_std + self.output_mean
        return x

    # ----------------------------------------------------
    # Metadata
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
        return self.patch_size if self.patch_size is not None else self.lat.shape

    def info(self):
        return {
            "input_normalization": (self.input_mean.squeeze(), self.input_std.squeeze()),
            "target_normalization": (self.output_mean.squeeze(), self.output_std.squeeze()),
        }

    def longitude(self) -> np.ndarray:
        return self.last_patch_lon

    def latitude(self) -> np.ndarray:
        return self.last_patch_lat

    # ----------------------------------------------------
    # Downscaling utility
    # ----------------------------------------------------
    @staticmethod
    def _create_lowres_(x, factor=4):
        """Create low-resolution version by downsampling and upsampling."""
        c, h, w = x.shape
        x = x.transpose(1, 2, 0)  # CHW → HWC
        x = x[::factor, ::factor, :]
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
        x = x.transpose(2, 0, 1)  # HWC → CHW
        return x
