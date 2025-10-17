import xarray as xr
import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

data_path = "/data01/FDS/muduchuru/ERA5/europe/ERA5"

era5_surface_channels = {
    167: "tas",
    201: "tasmax",
    202: "tasmin",
    228: "pr",
}

era5_isobaric_channels = {
    130: "t",
    133: "q",
    138: "vo",
    157: "r",
}

def decode_time(time_values):
    """Convert time encoded as %Y%m%d.%f to pandas.DatetimeIndex."""
    base_dates = pd.to_datetime(time_values.astype(int).astype(str), format="%Y%m%d")
    return base_dates


# ------------------ PROCESSING FUNCTIONS ------------------

def process_isobaric_file(path, ch, out_root="/beegfs/muduchuru/data"):
    """Split an isobaric ERA5 file into per-day, per-level NetCDF files."""
    try:
        with xr.open_dataset(path) as ds:
            print(f"[ISOBARIC] Processing: {path}")

            # Decode time
            if str(ds.time.dtype).startswith("float"):
                times = decode_time(ds.time.values)
            else:
                times = pd.to_datetime(ds.time.values)
            ds = ds.assign_coords(time=("time", times))

            dirname, filename = os.path.split(path)
            prefix, _, _ = filename.partition("_1D_")

            for t in ds.time:
                for plev in ds.plev:
                    plev_val = int(plev.data) // 100
                    var_name = f"{era5_isobaric_channels[ch]}{plev_val}"
                    date_str = pd.to_datetime(t.values).strftime("%Y-%m-%d")

                    ds_day = ds.sel(time=t, plev=plev)
                    ds_day = ds_day.rename({f"var{ch}": var_name})

                    out_dir = os.path.join(out_root, "era5", "pl", var_name)
                    os.makedirs(out_dir, exist_ok=True)

                    out_path = os.path.join(out_dir, f"{prefix}_1D_{date_str}_{var_name}.nc")

                    ds_day.load()  # force compute before writing
                    ds_day.to_netcdf(out_path)
        return f"✅ Done: {path}"

    except Exception as e:
        return f"❌ Failed {path}: {e}"


def process_surface_file(path, ch, out_root="/beegfs/muduchuru/data"):
    """Split a surface ERA5 file into per-day NetCDF files."""
    try:
        with xr.open_dataset(path) as ds:
            print(f"[SURFACE] Processing: {path}")

            if str(ds.time.dtype).startswith("float"):
                times = decode_time(ds.time.values)
            else:
                times = pd.to_datetime(ds.time.values)
            ds = ds.assign_coords(time=("time", times))

            dirname, filename = os.path.split(path)
            prefix, _, _ = filename.partition("_1D_")

            for t in ds.time:
                date_str = pd.to_datetime(t.values).strftime("%Y-%m-%d")
                var_name = era5_surface_channels[ch]

                ds_day = ds.sel(time=t)
                ds_day = ds_day.rename({f"var{ch}": var_name})

                out_dir = os.path.join(out_root, "era5", "sf", var_name)
                os.makedirs(out_dir, exist_ok=True)

                out_path = os.path.join(out_dir, f"{prefix}_1D_{date_str}_{var_name}.nc")

                ds_day.load()
                ds_day.to_netcdf(out_path)
        return f"✅ Done: {path}"

    except Exception as e:
        return f"❌ Failed {path}: {e}"


# ------------------ GATHER PATHS ------------------

def collect_paths():
    surface = {
        ch: sorted(glob.glob(os.path.join(data_path, "sf", str(ch), "*.nc"), recursive=True))
        for ch in era5_surface_channels
    }
    isobaric = {
        ch: sorted(glob.glob(os.path.join(data_path, "pl", str(ch), "*.nc"), recursive=True))
        for ch in era5_isobaric_channels
    }
    return surface, isobaric


# ------------------ MAIN PARALLEL EXECUTION ------------------

def main(n_workers=8):
    era5_surface_paths, era5_isobaric_paths = collect_paths()
    tasks = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Queue isobaric files
        for ch, paths in era5_isobaric_paths.items():
            for path in paths:
                tasks.append(executor.submit(process_isobaric_file, path, ch))

        # Queue surface files
        for ch, paths in era5_surface_paths.items():
            for path in paths:
                tasks.append(executor.submit(process_surface_file, path, ch))

        for future in as_completed(tasks):
            print(future.result())


if __name__ == "__main__":
    main(n_workers=20)  # adjust to your available CPU cores
