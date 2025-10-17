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
    Dataset for combined ERA5 daily files with shape (channel, lat, lon).
    Each sample is a daily file containing all variables as channels.
    """

    def __init__(
        self,
        data_path: str,
        input_channels: Optional[List[str]] = None,
        output_channels: Optional[List[str]] = None,
        normalize: bool = True,
        stats_path: Optional[str] = None,
        patch_size: Optional[Tuple[int, int]] = (128,128),
    ):
        self.data_path = data_path
        self.normalize = normalize
        self.output_channel_list = output_channels
        self.input_channel_list = input_channels
        self.patch_size = patch_size
        # Find all combined files
        self.files = sorted(glob.glob(os.path.join(data_path, "E5pl00_1D_*.nc")))
        if not self.files:
            raise FileNotFoundError(f"No combined ERA5 files found in {data_path}")

        # Load channel names from first file
        with xr.open_dataset(self.files[0]) as ds:
            self.channels = list(ds.channel.values)
            lat_1d = ds.image.lat.values[:128]
            lon_1d = ds.image.lon.values[:128]
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        self.lat = lat2d
        self.lon = lon2d

        # Set input/output channels
        self.input_channels_list = input_channels or self.channels
        self.output_channels_list = output_channels or self.channels

        # Load stats if provided
        if stats_path is not None and os.path.exists(stats_path):
            import json
            with open(stats_path, "r") as f:
                stats = json.load(f)
            self.input_mean = np.array([stats[ch]["mean"] for ch in self.input_channels_list])[:, None, None].astype(np.float32)
            self.input_std = np.array([stats[ch]["std"] for ch in self.input_channels_list])[:, None, None].astype(np.float32)
            self.output_mean = np.array([stats[ch]["mean"] for ch in self.output_channels_list])[:, None, None].astype(np.float32)
            self.output_std = np.array([stats[ch]["std"] for ch in self.output_channels_list])[:, None, None].astype(np.float32)
        else:
            self.input_mean = 0.0
            self.input_std = 1.0
            self.output_mean = 0.0
            self.output_std = 1.0

        # Extract time from filenames
        self.times = [self._extract_time_from_filename(f) for f in self.files]
    def convert_datetime_to_cftime(self,
        time: datetime, cls=cftime.DatetimeGregorian
    ) -> cftime.DatetimeGregorian:
        """Convert a Python datetime object to a cftime DatetimeGregorian object."""
        return cls(time.year, time.month, time.day, time.hour, time.minute, time.second)

    def _extract_time_from_filename(self, filename):
        # Example: E5pl00_1D_2022-01-01_17var.nc
        basename = os.path.basename(filename)
        parts = basename.split("_")
        tdt = datetime.datetime.strptime(parts[2],"%Y-%m-%d")
        tdt_cf = self.convert_datetime_to_cftime(tdt)
        return tdt_cf  # '2022-01-01'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load file
        file = self.files[idx]
        ds = xr.open_dataset(file)
        arr = ds["image"].values.astype(np.float32)  # (channel, lat, lon)
        ds.close()

        # Select input/output channels
        input_idx = [self.channels.index(ch) for ch in self.input_channels_list]
        output_idx = [self.channels.index(ch) for ch in self.output_channels_list]
        input_arr = arr[0, input_idx]
        output_arr = arr[0, output_idx]

        # Normalize
        input_arr = self.normalize_input(input_arr)
        output_arr = self.normalize_output(output_arr)
        # --- 🔹 Random patching ---
        if self.patch_size is not None:
            ph, pw = self.patch_size
            h, w = input_arr.shape[-2:]  # 203, 210

            if ph > h or pw > w:
                raise ValueError(f"Patch size {self.patch_size} larger than image {h, w}")

            top = np.random.randint(0, h - ph + 1)
            left = np.random.randint(0, w - pw + 1)

            input_arr = input_arr[:, top:top + ph, left:left + pw]
            output_arr = output_arr[:, top:top + ph, left:left + pw]

        # Lead time label
        lead_time_label = 0

        input_arr = self._create_lowres_(input_arr,4)

        return output_arr, input_arr, lead_time_label


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
        # return (len(self.lat), len(self.lon))
        return (128, 128)

    def info(self) -> dict:
        return {
            "input_normalization": (self.input_mean.squeeze(), self.input_std.squeeze()),
            "target_normalization": (self.output_mean.squeeze(), self.output_std.squeeze()),
        }

    @staticmethod
    def _create_lowres_(x, factor=4):
        # downsample the high res imag
        x = x.transpose(1, 2, 0)
        x = x[::factor, ::factor, :]  # 8x8x3  #subsample
        # upsample with bicubic interpolation to bring the image to the nominal size
        x = cv2.resize(
            x, (x.shape[1] * factor, x.shape[0] * factor), interpolation=cv2.INTER_CUBIC
        )  # 32x32x3
        x = x.transpose(2, 0, 1)  # 3x32x32
        return x
