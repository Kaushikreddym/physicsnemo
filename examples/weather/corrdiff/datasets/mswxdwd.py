import os
import glob
import numpy as np
import xarray as xr
import torch
import cv2
import json
import xesmf as xe
from pathlib import Path
from typing import List, Tuple, Optional
import datetime
import cftime
import xesmf.util as xe_util

from datasets.base import ChannelMetadata, DownscalingDataset

def add_corners_curvilinear(ds):
    def _midpoint(arr):
        mid = (arr[:-1] + arr[1:]) / 2
        first = arr[0] - (mid[0] - arr[0])
        last  = arr[-1] + (arr[-1] - mid[-1])
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


def add_corners_regular(ds):
    """Adds bounds to a regular lat/lon grid for xESMF."""
    # xesmf.util.grid_2d can do this, or we can manually shift
    d_lat = np.abs(ds.lat[1] - ds.lat[0])
    d_lon = np.abs(ds.lon[1] - ds.lon[0])
    
    lat_b = np.concatenate([ds.lat - d_lat/2, [ds.lat[-1] + d_lat/2]])
    lon_b = np.concatenate([ds.lon - d_lon/2, [ds.lon[-1] + d_lon/2]])
    
    return ds.assign_coords(lat_b=lat_b, lon_b=lon_b)
class mswxdwd(DownscalingDataset):
    def __init__(
        self,
        data_path: str,
        train: bool = True,
        train_years: Tuple[int, int] = (1989, 2020),
        val_years: Tuple[int, int] = (2021, 2024),
        input_channels: Optional[List[str]] = None,
        output_channels: Optional[List[str]] = None,
        static_channels: Optional[List[str]] = None,
        normalize: bool = True,
        stats_dwd: Optional[str] = None,
        stats_mswx: Optional[str] = None,
        patch_size: Optional[Tuple[int, int]] = (128, 128),
        center_latlon: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.normalize = normalize
        self.patch_size = patch_size
        self.center_latlon = center_latlon
        self.factor = 16 # For UNet compatibility
        self.input_channels_list = input_channels
        self.output_channels_list = output_channels
        # 1. File Discovery
        self.mswx_files = {ch: sorted(glob.glob(os.path.join(data_path, "mswx", ch, "*.nc"))) for ch in input_channels}
        self.dwd_files = {ch: sorted(glob.glob(os.path.join(data_path, "hyras_daily", ch, "*.nc"))) for ch in output_channels}
        
        # 2. Time Intersection
        mswx_times = {self._extract_date_from_filename(f) for f in self.mswx_files[input_channels[0]]}
        dwd_times = {self._extract_date_from_filename(f) for f in self.dwd_files[output_channels[0]]}
        self.common_times = sorted(list(mswx_times & dwd_times))

        start_year, end_year = train_years if train else val_years
        self.valid_dates = [d for d in self.common_times if start_year <= d.year <= end_year]
        self.times = [self.convert_datetime_to_cftime(d) for d in self.valid_dates]
        
        self._get_extent(self.dwd_files[output_channels[0]][0])
        
        # 3. Setup Grid & Regridder
        # Load DWD Template (Target Curvilinear Grid)
        with xr.open_dataset(self.dwd_files[output_channels[0]][0]) as ds_target:
            # Crop to factor 16
            # h, w = ds_target.lat.shape
            # new_h, new_w = (h // 16) * 16, (w // 16) * 16
            # self.ds_target = ds_target.isel(y=slice(0, new_h), x=slice(0, new_w))
            
            # Add corners for conservative regridding
            var_name = output_channels[0]
            self.data_mask = np.where(np.isnan(ds_target[var_name].values), 0.0, 1.0).astype(np.float32)
            self.ds_target = add_corners_curvilinear(ds_target)
            # self.ds_target = xe_util.grid_2d(ds_target.lon, ds_target.lat)
            self.lat = ds_target.lat.values
            self.lon = ds_target.lon.values

        # 2. Prepare MSWX (Source) with corners
        with xr.open_dataset(self.mswx_files[input_channels[0]][0]) as ds_src:
            if ds_src.lat.values[0] > ds_src.lat.values[-1]:
                ds_src = ds_src.sortby("lat")
            
            ds_src_cropped = self._crop_box(ds_src, self.ds_box)
            # ds_src_cropped = add_corners_regular(ds_src_cropped)
            
            weights_file = "mswx_to_dwd_bilinear.nc"
            reuse = Path(weights_file).exists()

            # 3. Initialize Conservative Normed Regridder
            self.regridder = xe.Regridder(
                ds_src_cropped,
                self.ds_target,
                method="bilinear", # Changed from bilinear
                periodic=False,
                reuse_weights=reuse, 
                filename=weights_file
            )
        # 4. Load Static Channels (Regrid them to DWD grid)
        self.static_channels_list = static_channels or []
        self.static_data = self._prepare_static_channels()

        # 5. Normalization Logic
        self._setup_normalization(stats_dwd, stats_mswx, input_channels, output_channels)
    
    @staticmethod
    def _crop_box(ds, box):
        lat_min, lat_max, lon_min, lon_max = box
        return ds.sel(lat=slice(lat_min-1, lat_max+1), lon=slice(lon_min-1, lon_max+1))
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
    def _get_extent(self, filename):
        ds = xr.open_dataset(filename)
        lat_min, lat_max = float(np.min(ds.lat)), float(np.max(ds.lat))
        lon_min, lon_max = float(np.min(ds.lon)), float(np.max(ds.lon))
        self.ds_box = (lat_min, lat_max, lon_min, lon_max)
        return self.ds_box    
    def _positional_embedding(self, lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
        """
        Generate 2-channel normalized positional embeddings from 2D lat/lon arrays.

        Parameters
        ----------
        lat2d : np.ndarray (H, W)
            Latitude grid
        lon2d : np.ndarray (H, W)
            Longitude grid

        Returns
        -------
        pos : np.ndarray (2, H, W)
            Channels: [lat_norm, lon_norm], scaled to [-1, 1]
        """
        lat_min, lat_max = lat2d.min(), lat2d.max()
        lon_min, lon_max = lon2d.min(), lon2d.max()

        lat_norm = 2 * (lat2d - lat_min) / (lat_max - lat_min) - 1
        lon_norm = 2 * (lon2d - lon_min) / (lon_max - lon_min) - 1

        pos = np.stack([lat_norm, lon_norm], axis=0)
        return pos.astype(np.float32)
    def _prepare_static_channels(self):
        static_layers = []
        # Target dataset for xesmf
        ds_tgt = xr.Dataset({"lat": (["y", "x"], self.lat), "lon": (["y", "x"], self.lon)})

        if "elevation" in self.static_channels_list:
            with xr.open_dataset("/data01/FDS/muduchuru/Land/GMTED/GMTED2010_15n015_00625deg.nc") as ds:
                ds = ds.rename({"latitude": "lat", "longitude": "lon"})
                ds = self._fix_longitude(ds)
                
                weights_file = "gmted_to_dwd.nc"
                reuse = Path(weights_file).exists()
                regridder = xe.Regridder(ds, ds_tgt, method="bilinear", reuse_weights=reuse, filename=weights_file)
                self.elev = regridder(ds)
                elev = self.elev['elevation'].values.astype(np.float32)
                static_layers.append(elev)

        if "lsm" in self.static_channels_list:
            with xr.open_dataset("/data01/FDS/muduchuru/Atmos/IMERG/IMERG_land_sea_mask.nc") as ds:
                ds = self._fix_longitude(ds)
                
                weights_file = "imerg_to_dwd.nc"
                reuse = Path(weights_file).exists()
                regridder = xe.Regridder(ds, ds_tgt, method="nearest_s2d", reuse_weights=reuse, filename=weights_file, ignore_degenerate=True, unmapped_to_nan=True)
                self.lsm = regridder(ds)
                lsm = self.lsm["landseamask"].values.astype(np.float32)
                static_layers.append(lsm)
        if "dwd_mask" in self.static_channels_list:
            # mask_channel = self.data_mask[None, :, :]
            static_layers.append(self.data_mask)
        if 'pos_embed' in self.static_channels_list:
            pos = self._positional_embedding(self.lat, self.lon)
            static_layers.extend([pos[0], pos[1]])
        return np.stack(static_layers) if static_layers else None

    def _get_mswx(self, t):
        tstr = t.strftime("%Y%j")
        ds_list = []
        for ch in self.input_channels_list:
            file_match = next(f for f in self.mswx_files[ch] if tstr in f)
            with xr.open_dataset(file_match) as ds:
                if ds.lat.values[0] > ds.lat.values[-1]:
                    ds = ds.sortby("lat")
                ds = self._crop_box(ds, self.ds_box).load()
                varname = list(ds.data_vars)[0]
                ds_out = self.regridder(ds[[varname]])
                ds_regridded = ds_out[varname].isel(time=0)
                ds_list.append(ds_regridded)
        # Stack into [C, H, W]
        return np.stack([d.values for d in ds_list]).astype(np.float32)

    def _get_dwd(self, t):
        tstr = t.strftime("%Y-%m-%d")
        arrs = []
        for ch in self.output_channels_list:
            file_match = next(f for f in self.dwd_files[ch] if tstr in f)
            with xr.open_dataset(file_match) as ds:
                # Ensure it matches the cropped template size
                val = ds[ch].values
                arrs.append(val)
        return np.stack(arrs).astype(np.float32)

    def __getitem__(self, idx):
        date = self.valid_dates[idx]

        # 1. Get Data (MSWX is already regridded to DWD grid in _get_mswx)
        arr_mswx = self._get_mswx(date) 
        arr_dwd = self._get_dwd(date)

        # 3. Add Static Channels
        if self.static_data is not None:
            # Add mask itself as a static channel to help the model identify boundaries
            arr_mswx = np.concatenate([arr_mswx, self.static_data], axis=0)

        # 4. Normalization
        input_arr = self.normalize_input(arr_mswx)
        output_arr = self.normalize_output(arr_dwd)

        # Replace NaNs with 0
        input_arr = np.nan_to_num(input_arr, nan=0.0)
        output_arr = np.nan_to_num(output_arr, nan=0.0)

        # Apply spatial mask (0/1)
        mask = self.data_mask[None, :, :]  # add batch/channel dim if needed

        input_arr = input_arr * mask
        output_arr = output_arr * mask
        
        # --- 🔹 Cropping logic ---
        if self.patch_size is not None:
            ph, pw = self.patch_size
            h, w = input_arr.shape[-2:]

            if ph > h or pw > w:
                raise ValueError(f"Patch size {self.patch_size} larger than image {h, w}")

            if self.center_latlon is not None:
                lat0, lon0 = self.center_latlon
                top, left = self._get_center_indices(self.lat.values, self.lon.values, lat0, lon0, ph, pw)
            else:
                top = np.random.randint(0, h - ph + 1)
                left = np.random.randint(0, w - pw + 1)
            input_arr = input_arr[:, top:top + ph, left:left + pw]
            output_arr = output_arr[:, top:top + ph, left:left + pw]
            # Save lat/lon for this patch (slice the full 2D lat/lon grid)
            self.last_patch_lat = self.lat[top:top + ph, left:left + pw]
            self.last_patch_lon = self.lon[top:top + ph, left:left + pw]
        else:
            # No patching: last_patch_* point to the full-grid lat/lon
            self.last_patch_lat = self.lat
            self.last_patch_lon = self.lon
        
        # return torch.from_numpy(output_arr), torch.from_numpy(input_arr), 0
        return output_arr, input_arr
    
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
    def _apply_patch(self, in_arr, out_arr):
        ph, pw = self.patch_size
        _, h, w = in_arr.shape
        if self.center_latlon:
            top, left = self._get_center_indices(self.lat, self.lon, *self.center_latlon, ph, pw)
        else:
            top = np.random.randint(0, h - ph + 1)
            left = np.random.randint(0, w - pw + 1)
        return in_arr[:, top:top+ph, left:left+pw], out_arr[:, top:top+ph, left:left+pw]
    
    def convert_datetime_to_cftime(self, time: datetime.datetime, cls=cftime.DatetimeGregorian):
        return cls(time.year, time.month, time.day, time.hour, time.minute, time.second)

    @staticmethod
    def _extract_date_from_filename(filename: str) -> datetime.datetime:
        base = os.path.basename(filename)
        if "_de.nc" in base or "hyras" in filename: # DWD/HYRAS
            # Expecting format like 'pr_hyras_de_2000-01-01.nc'
            parts = base.replace(".nc", "").split("_")
            return datetime.datetime.strptime(parts[-3], "%Y-%m-%d")
        else: # MSWX
            parts = base.split(".")
            return datetime.datetime.strptime(parts[0], "%Y%j")

    def _setup_normalization(self, stats_dwd, stats_mswx, input_channels, output_channels):

        # -----------------------------------------
        # Load normalization statistics
        # -----------------------------------------
        if stats_dwd is not None and os.path.exists(stats_dwd):
            with open(stats_dwd, "r") as f:
                stats = json.load(f)
            input_mean_list = [stats[ch]["mean"] for ch in output_channels]
            input_std_list = [stats[ch]["std"] for ch in output_channels]
        else:
            input_mean_list = [0.0] * len(output_channels)
            input_std_list = [1.0] * len(output_channels)

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
                elif ch == "dwd_mask":
                    varname = "dwd_mask"
                    input_mean_list.append(0)
                    input_std_list.append(1)
                elif ch == "pos_embed":
                    input_mean_list.extend([0,0])
                    input_std_list.extend([1,1])

        self.input_mean = np.array(input_mean_list)[:, None, None]
        self.input_std = np.array(input_std_list)[:, None, None]

        if stats_mswx is not None and os.path.exists(stats_mswx):
            with open(stats_mswx, "r") as f:
                stats = json.load(f)
            self.output_mean = np.array([stats[ch]["mean"] for ch in input_channels])[:, None, None]
            self.output_std = np.array([stats[ch]["std"] for ch in input_channels])[:, None, None]
        else:
            self.output_mean = 0.0
            self.output_std = 1.0
    # ----------------------------------------------------
    # ✅ Data Access
    # ----------------------------------------------------
    def __len__(self):
        return len(self.valid_dates)

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
            for ch in self.static_channels_list:
                if ch == "pos_embed":
                    # pos_embed provides 2 channels: lat_norm and lon_norm
                    channels.extend(["pos_embed_lat", "pos_embed_lon"])
                else:
                    channels.append(ch)
        return [ChannelMetadata(name=n) for n in channels]

    def output_channels(self):
        return [ChannelMetadata(name=n) for n in self.output_channels_list]

    def time(self):
        return self.times

    def image_shape(self):
        """Return full image shape (H, W)."""
        return self.patch_size

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
    # ✅ Downscaling utility (LR creation)
    # ----------------------------------------------------
    @staticmethod
    def _create_lowres_(x, factor=4):
        x = x.transpose(1, 2, 0)  # CHW → HWC
        x = x[::factor, ::factor, :]
        x = cv2.resize(x, (x.shape[1] * factor, x.shape[0] * factor), interpolation=cv2.INTER_CUBIC)
        x = x.transpose(2, 0, 1)  # HWC → CHW
        return x
