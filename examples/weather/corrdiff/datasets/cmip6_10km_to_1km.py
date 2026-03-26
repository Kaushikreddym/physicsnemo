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
import xesmf as xe

from datasets.base import ChannelMetadata, DownscalingDataset


def add_corners_curvilinear(ds):
    """Add corner coordinates for curvilinear grids (for xESMF)."""
    def _midpoint(arr):
        mid = (arr[:-1] + arr[1:]) / 2
        first = arr[0] - (mid[0] - arr[0])
        last = arr[-1] + (arr[-1] - mid[-1])
        return np.concatenate([[first], mid, [last]])

    lat = ds.lat.values
    lon = ds.lon.values

    lat_b = np.apply_along_axis(_midpoint, 1, lat)
    lat_b = np.apply_along_axis(_midpoint, 0, lat_b)

    lon_b = np.apply_along_axis(_midpoint, 1, lon)
    lon_b = np.apply_along_axis(_midpoint, 0, lon_b)

    return xr.Dataset(
        {
            "lat": (("y", "x"), lat),
            "lon": (("y", "x"), lon),
            "lat_b": (("y_b", "x_b"), lat_b),
            "lon_b": (("y_b", "x_b"), lon_b),
        }
    )


class cmip6_10km_to_1km(DownscalingDataset):
    """
    Stage 2: 10km → 1km downscaling using MSWX→DWD trained model.
    
    ✅ Takes 10km MSWX grid data (from Stage 1 output)
    ✅ Regrids to DWD 1km curvilinear grid
    ✅ Uses MSWX→DWD trained model for downscaling
    ✅ Supports static channels: elevation, lsm, dwd_mask, pos_embed
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

    @staticmethod
    def _crop_box(ds, box):
        lat_min, lat_max, lon_min, lon_max = box
        return ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    def _find_valid_spatial_bounds(self, data_2d: np.ndarray) -> Tuple[slice, slice]:
        """Find rows and columns that contain at least one non-NaN value."""
        valid_rows = ~np.all(np.isnan(data_2d), axis=1)
        valid_cols = ~np.all(np.isnan(data_2d), axis=0)
        
        row_indices = np.where(valid_rows)[0]
        col_indices = np.where(valid_cols)[0]
        
        if len(row_indices) == 0 or len(col_indices) == 0:
            raise ValueError("All data is NaN - cannot find valid spatial bounds")
        
        row_slice = slice(row_indices[0], row_indices[-1] + 1)
        col_slice = slice(col_indices[0], col_indices[-1] + 1)
        
        return row_slice, col_slice

    def __init__(
        self,
        stage1_output_path: str,  # Path to 10km data (Stage 1 outputs or processed CMIP6)
        mswx_reference_file: str,  # MSWX file to get 10km grid
        dwd_reference_file: str,   # DWD file to get 1km output grid
        input_channels: Optional[List[str]] = None,
        output_channels: Optional[List[str]] = None,
        static_channels: Optional[List[str]] = None,
        normalize: bool = True,
        stats_mswx: Optional[str] = None,
        stats_dwd: Optional[str] = None,
        patch_size: Optional[tuple] = (128, 128),
        center_latlon: Optional[tuple] = None,
        patch_index: Optional[int] = None,
        overlap_pix: int = 0,
        year_range: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize Stage 2 downscaling dataset (10km → 1km).
        
        Parameters
        ----------
        stage1_output_path : str
            Path to 10km resolution data files (output from Stage 1 model)
            Expected structure options:
            - Nested: <path>/<institution>/<model>/<experiment>/<variant>/<variable>/*.nc
            - Flat: <path>/<variable>/*.nc
            Files should contain Stage 1 model outputs at MSWX 10km resolution
        mswx_reference_file : str
            Path to MSWX file to extract the 10km input grid
        dwd_reference_file : str
            Path to DWD file to extract the 1km output grid
        input_channels : List[str]
            Channel names from 10km data
        output_channels : List[str]
            Target channel names (1km)
        static_channels : List[str], optional
            Static features (elevation, lsm, dwd_mask, pos_embed)
        normalize : bool
            Whether to apply normalization
        stats_mswx : str, optional
            Path to MSWX normalization statistics JSON (for input)
        stats_dwd : str, optional
            Path to DWD normalization statistics JSON (for output)
        patch_size : tuple, optional
            Patch size for training/inference
        center_latlon : tuple, optional
            Center coordinates for deterministic patching
        patch_index : int, optional
            Specific patch index for systematic patching
        overlap_pix : int
            Overlap pixels for patching
        year_range : tuple, optional
            (start_year, end_year) to filter data
        """
        self.stage1_output_path = stage1_output_path
        self.normalize = normalize
        self.patch_size = patch_size
        self.center_latlon = center_latlon
        self.patch_index = patch_index
        self.overlap_pix = overlap_pix
        self.factor = 16  # UNet compatibility

        self.input_channels_list = input_channels
        self.output_channels_list = output_channels or input_channels
        self.static_channels_list = static_channels

        # -----------------------------------------
        # Load MSWX reference grid (10km input)
        # -----------------------------------------
        with xr.open_dataset(mswx_reference_file) as ds_mswx:
            if ds_mswx.lat.values[0] > ds_mswx.lat.values[-1]:
                ds_mswx = ds_mswx.sortby("lat")
            
            # Get extent for cropping
            lat_min, lat_max = float(np.min(ds_mswx.lat)), float(np.max(ds_mswx.lat))
            lon_min, lon_max = float(np.min(ds_mswx.lon)), float(np.max(ds_mswx.lon))
            self.mswx_box = (lat_min, lat_max, lon_min, lon_max)
            
            self.mswx_lat_1d = ds_mswx.lat.values
            self.mswx_lon_1d = ds_mswx.lon.values

        lon2d, lat2d = np.meshgrid(self.mswx_lon_1d, self.mswx_lat_1d)
        self.mswx_lat = lat2d
        self.mswx_lon = lon2d

        # -----------------------------------------
        # Load DWD reference grid (1km output)
        # -----------------------------------------
        with xr.open_dataset(dwd_reference_file) as ds_dwd:
            var_name = list(ds_dwd.data_vars)[0]
            data_2d = ds_dwd[var_name].values
            
            # Find valid spatial bounds
            self.row_slice, self.col_slice = self._find_valid_spatial_bounds(data_2d)
            
            # Crop to valid bounds
            ds_dwd_cropped = ds_dwd.isel(y=self.row_slice, x=self.col_slice)
            
            # Create mask and get lat/lon
            self.data_mask = np.where(np.isnan(ds_dwd_cropped[var_name].values), 0.0, 1.0).astype(np.float32)
            self.ds_target = add_corners_curvilinear(ds_dwd_cropped)
            self.dwd_lat = ds_dwd_cropped.lat.values
            self.dwd_lon = ds_dwd_cropped.lon.values

        # -----------------------------------------
        # Setup regridder (MSWX 10km → DWD 1km)
        # -----------------------------------------
        ds_src = xr.Dataset({
            "lat": (["lat"], self.mswx_lat_1d),
            "lon": (["lon"], self.mswx_lon_1d)
        })
        
        weights_file = "mswx_to_dwd_bilinear_stage2.nc"
        reuse = Path(weights_file).exists()
        
        self.regridder = xe.Regridder(
            ds_src,
            self.ds_target,
            method="bilinear",
            periodic=False,
            reuse_weights=reuse,
            filename=weights_file
        )

        # -----------------------------------------
        # Discover Stage 1 output files
        # -----------------------------------------
        self.stage1_files = self._discover_stage1_files(stage1_output_path, input_channels)
        
        if year_range is not None:
            start_year, end_year = year_range
            self.stage1_files = [
                f for f in self.stage1_files
                if start_year <= self._extract_year_from_file(f) <= end_year
            ]

        if not self.stage1_files:
            raise FileNotFoundError(f"No Stage 1 output files found in {stage1_output_path}")

        self.times = [self._extract_time_from_file(f) for f in self.stage1_files]

        # -----------------------------------------
        # Static features for DWD grid
        # -----------------------------------------
        self._prepare_static_channels()

        # -----------------------------------------
        # Normalization
        # -----------------------------------------
        self._setup_normalization(stats_mswx, stats_dwd)

        # Last used lat/lon
        self.last_patch_lat = self.dwd_lat
        self.last_patch_lon = self.dwd_lon

    def _discover_stage1_files(self, data_path: str, channels: List[str]) -> List[str]:
        """
        Discover Stage 1 output files.
        Expected structure: 
        - Option 1: data_path/<institution>/<model>/<experiment>/<variant>/<variable>/*.nc
        - Option 2: data_path/<variable>/*.nc (flat structure)
        """
        all_files = []
        
        # Try recursive search first (for nested CMIP6-style structure)
        for ch in channels:
            pattern = os.path.join(data_path, "**", ch, "*.nc")
            files = sorted(glob.glob(pattern, recursive=True))
            
            if not files:
                # Try flat structure
                pattern = os.path.join(data_path, ch, "*.nc")
                files = sorted(glob.glob(pattern))
            
            if not files:
                raise FileNotFoundError(f"No files found for channel {ch} in {data_path}")
            
            all_files.extend(files)
        
        return sorted(all_files)

    def _extract_year_from_file(self, filename: str) -> int:
        """
        Extract year from filename.
        Handles both CMIP6 format and MSWX/DWD formats.
        """
        base = os.path.basename(filename)
        
        # Try CMIP6 format: <var>_<freq>_<model>_<exp>_<variant>_<grid>_<start>-<end>.nc
        parts = base.split("_")
        for part in reversed(parts):
            if "-" in part and ".nc" not in part:
                date_str = part.split("-")[0]
                if len(date_str) >= 4:
                    try:
                        return int(date_str[:4])
                    except ValueError:
                        continue
            elif ".nc" in part and "-" in part:
                date_str = part.split("-")[0]
                if len(date_str) >= 4:
                    try:
                        return int(date_str[:4])
                    except ValueError:
                        continue
        
        # Try MSWX/simple format (YYYYDOY.nc or YYYY-MM-DD.nc)
        try:
            return int(base[:4])
        except:
            pass
        
        raise ValueError(f"Could not extract year from filename: {filename}")

    def _extract_time_from_file(self, filename: str) -> cftime.DatetimeGregorian:
        """
        Extract time from file.
        Handles cftime calendars from CMIP6 and standard datetime formats.
        """
        try:
            with xr.open_dataset(filename) as ds:
                if "time" in ds.coords and len(ds.time) > 0:
                    time_val = ds.time.values[0]
                    
                    # Handle different cftime calendar types
                    if isinstance(time_val, (cftime.DatetimeGregorian,
                                            cftime.DatetimeProlepticGregorian,
                                            cftime.DatetimeNoLeap,
                                            cftime.Datetime360Day)):
                        return cftime.DatetimeGregorian(
                            time_val.year, time_val.month, time_val.day,
                            time_val.hour, time_val.minute, time_val.second
                        )
                    elif isinstance(time_val, np.datetime64):
                        dt = time_val.astype('datetime64[s]').astype(datetime.datetime)
                        return self.convert_datetime_to_cftime(dt)
        except:
            pass
        
        # Fallback: use year from filename
        year = self._extract_year_from_file(filename)
        return cftime.DatetimeGregorian(year, 1, 1, 0, 0, 0)

    def _prepare_static_channels(self):
        """Prepare static channels for DWD grid."""
        static_layers = []

        if "elevation" in (self.static_channels_list or []):
            # Use pre-regridded file or generate
            regridded_elev_file = "gmted_dwd_cropped_bilinear.nc"
            
            if Path(regridded_elev_file).exists():
                with xr.open_dataset(regridded_elev_file) as ds:
                    self.elev = ds
                    elev = ds['surface_altitude_maximum'].values.astype(np.float32)
                    static_layers.append(elev)
            else:
                print(f"Warning: {regridded_elev_file} not found. Elevation not available.")
                self.elev = None

        if "lsm" in (self.static_channels_list or []):
            ds_tgt = xr.Dataset({"lat": (["y", "x"], self.dwd_lat), "lon": (["y", "x"], self.dwd_lon)})
            with xr.open_dataset("/data01/FDS/muduchuru/Atmos/IMERG/IMERG_land_sea_mask.nc") as ds_lsm:
                ds_lsm = self._fix_longitude(ds_lsm)
                
                weights_file = "imerg_to_dwd_cropped_stage2.nc"
                reuse = Path(weights_file).exists()
                regridder_lsm = xe.Regridder(
                    ds_lsm, ds_tgt, method="nearest_s2d",
                    reuse_weights=reuse, filename=weights_file,
                    ignore_degenerate=True, unmapped_to_nan=True
                )
                self.lsm = regridder_lsm(ds_lsm)
                lsm = self.lsm["landseamask"].values.astype(np.float32)
                static_layers.append(lsm)

        if "dwd_mask" in (self.static_channels_list or []):
            static_layers.append(self.data_mask)

        if "pos_embed" in (self.static_channels_list or []):
            pos = self._positional_embedding(self.dwd_lat, self.dwd_lon)
            static_layers.extend([pos[0], pos[1]])

        self.static_data = np.stack(static_layers) if static_layers else None

    def _positional_embedding(self, lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
        """Generate 2-channel normalized positional embeddings."""
        lat_min, lat_max = lat2d.min(), lat2d.max()
        lon_min, lon_max = lon2d.min(), lon2d.max()

        lat_norm = 2 * (lat2d - lat_min) / (lat_max - lat_min) - 1
        lon_norm = 2 * (lon2d - lon_min) / (lon_max - lon_min) - 1

        pos = np.stack([lat_norm, lon_norm], axis=0)
        return pos.astype(np.float32)

    def _setup_normalization(self, stats_mswx: Optional[str], stats_dwd: Optional[str]):
        """Setup normalization statistics."""
        # Input normalization (MSWX 10km stats)
        if stats_mswx is not None and os.path.exists(stats_mswx):
            with open(stats_mswx, "r") as f:
                stats = json.load(f)
            input_mean_list = [stats[ch]["mean"] for ch in self.input_channels_list]
            input_std_list = [stats[ch]["std"] for ch in self.input_channels_list]
        else:
            input_mean_list = [0.0] * len(self.input_channels_list)
            input_std_list = [1.0] * len(self.input_channels_list)

        # Add static channel stats
        if self.static_channels_list:
            for ch in self.static_channels_list:
                if ch == "elevation" and self.elev is not None:
                    input_mean_list.append(self.elev['surface_altitude_maximum'].values.mean())
                    input_std_list.append(self.elev['surface_altitude_maximum'].values.std())
                elif ch == "lsm":
                    input_mean_list.append(self.lsm["landseamask"].values.mean())
                    input_std_list.append(self.lsm["landseamask"].values.std())
                elif ch == "dwd_mask":
                    input_mean_list.append(0)
                    input_std_list.append(1)
                elif ch == "pos_embed":
                    input_mean_list.extend([0, 0])
                    input_std_list.extend([1, 1])

        self.input_mean = np.array(input_mean_list)[:, None, None]
        self.input_std = np.array(input_std_list)[:, None, None]

        # Output normalization (DWD 1km stats)
        if stats_dwd is not None and os.path.exists(stats_dwd):
            with open(stats_dwd, "r") as f:
                stats = json.load(f)
            self.output_mean = np.array([stats[ch]["mean"] for ch in self.output_channels_list])[:, None, None]
            self.output_std = np.array([stats[ch]["std"] for ch in self.output_channels_list])[:, None, None]
        else:
            self.output_mean = 0.0
            self.output_std = 1.0

    def _get_stage1_data(self, idx: int) -> np.ndarray:
        """
        Load Stage 1 output data and regrid to DWD grid.
        
        Returns
        -------
        arr : np.ndarray [C, H_dwd, W_dwd]
            Stage 1 data regridded to DWD 1km grid
        """
        filename = self.stage1_files[idx]
        
        with xr.open_dataset(filename) as ds:
            ds = self._fix_longitude(ds)
            
            # Select channels
            data_arrays = []
            for ch in self.input_channels_list:
                if ch in ds.data_vars:
                    data_arrays.append(ds[ch])
                else:
                    raise ValueError(f"Channel {ch} not found in file {filename}")
            
            # Stack channels
            if len(data_arrays) > 1:
                ds_stacked = xr.concat(data_arrays, dim="channel").assign_coords(channel=self.input_channels_list)
            else:
                ds_stacked = data_arrays[0].expand_dims("channel").assign_coords(channel=self.input_channels_list)
            
            # Remove time dimension if present
            if "time" in ds_stacked.dims:
                ds_stacked = ds_stacked.isel(time=0)
            
            # Regrid to DWD grid
            regridded_list = []
            for i, ch in enumerate(self.input_channels_list):
                ds_ch = ds_stacked.isel(channel=i)
                ds_regridded = self.regridder(ds_ch)
                regridded_list.append(ds_regridded.values)
            
            arr = np.stack(regridded_list, axis=0).astype(np.float32)
        
        return arr

    def _get_center_indices(self, lats, lons, lat0, lon0, ph, pw):
        """Find top-left corner indices for a patch with bottom-left at (lat0, lon0)."""
        dist = np.sqrt((lats - lat0)**2 + (lons - lon0)**2)
        iy, ix = np.unravel_index(np.argmin(dist), lats.shape)

        top = int(iy - (ph - 1))
        left = int(ix)
        
        top = np.clip(top, 0, lats.shape[0] - ph)
        left = np.clip(left, 0, lats.shape[1] - pw)
        
        return top, left

    def get_patch_bounds_by_index(self, patch_index: int, ph: int, pw: int,
                                  overlap_pix: int = 0) -> Tuple[int, int, int, int]:
        """Calculate patch bounds for a given patch index."""
        h, w = self.dwd_lat.shape
        
        stride_y = ph - overlap_pix
        stride_x = pw - overlap_pix
        
        patches_per_row = (w + stride_x - 1) // stride_x
        
        patch_row = patch_index // patches_per_row
        patch_col = patch_index % patches_per_row
        
        top = patch_row * stride_y
        left = patch_col * stride_x
        
        top = min(top, h - ph)
        left = min(left, w - pw)
        
        bottom = top + ph
        right = left + pw
        
        return top, bottom, left, right

    def __len__(self):
        return len(self.stage1_files)

    def __getitem__(self, idx):
        """
        Get Stage 1 output data regridded and ready for Stage 2 model.
        
        Returns
        -------
        output_arr : np.ndarray [C, H, W]
            Target at 1km (dummy for inference)
        input_arr : np.ndarray [C + static, H, W]
            Stage 1 output regridded + static channels
        """
        # Load Stage 1 data regridded to DWD grid
        arr_stage1 = self._get_stage1_data(idx)
        
        # Add static channels
        if self.static_data is not None:
            arr_stage1 = np.concatenate([arr_stage1, self.static_data], axis=0)

        # Normalize
        input_arr = self.normalize_input(arr_stage1)
        output_arr = input_arr.copy()  # Dummy target for inference

        # Replace NaNs
        input_arr = np.nan_to_num(input_arr, nan=0.0)
        output_arr = np.nan_to_num(output_arr, nan=0.0)

        # Apply mask
        mask = self.data_mask[None, :, :]
        input_arr = input_arr * mask
        output_arr = output_arr * mask

        # Apply patching if needed
        if self.patch_size is not None:
            ph, pw = self.patch_size
            h, w = input_arr.shape[-2:]

            if ph > h or pw > w:
                raise ValueError(f"Patch size {self.patch_size} larger than image {h, w}")

            if self.patch_index is not None:
                top, bottom, left, right = self.get_patch_bounds_by_index(
                    self.patch_index, ph, pw, self.overlap_pix
                )
            elif self.center_latlon is not None:
                lat0, lon0 = self.center_latlon
                top, left = self._get_center_indices(self.dwd_lat, self.dwd_lon, lat0, lon0, ph, pw)
            else:
                top = np.random.randint(0, h - ph + 1)
                left = np.random.randint(0, w - pw + 1)

            input_arr = input_arr[:, top:top + ph, left:left + pw]
            output_arr = output_arr[:, top:top + ph, left:left + pw]
            
            self.last_patch_lat = self.dwd_lat[top:top + ph, left:left + pw]
            self.last_patch_lon = self.dwd_lon[top:top + ph, left:left + pw]
        else:
            self.last_patch_lat = self.dwd_lat
            self.last_patch_lon = self.dwd_lon

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
        if self.static_channels_list:
            for ch in self.static_channels_list:
                if ch == "pos_embed":
                    channels.extend(["pos_embed_lat", "pos_embed_lon"])
                else:
                    channels.append(ch)
        return [ChannelMetadata(name=n) for n in channels]

    def output_channels(self):
        return [ChannelMetadata(name=n) for n in self.output_channels_list]

    def time(self):
        return self.times

    def image_shape(self):
        if self.patch_size is not None:
            return tuple(self.patch_size)
        return self.dwd_lat.shape

    def info(self):
        return {
            "input_normalization": (self.input_mean.squeeze(), self.input_std.squeeze()),
            "target_normalization": (self.output_mean.squeeze(), self.output_std.squeeze()),
        }

    def longitude(self) -> np.ndarray:
        return self.last_patch_lon

    def latitude(self) -> np.ndarray:
        return self.last_patch_lat
