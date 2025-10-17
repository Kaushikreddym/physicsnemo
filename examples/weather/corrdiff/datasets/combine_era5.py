import os
import glob
import re
import xarray as xr
import pandas as pd


def extract_dates(root="/beegfs/muduchuru/data/era5"):
    """Extract all available dates for each variable and return their intersection."""
    pattern = re.compile(r"_(\d{4}-\d{2}-\d{2})_")

    all_dates_per_var = []
    for sub in [os.path.join(root, "sf", "*"), os.path.join(root, "pl", "*")]:
        for var_dir in glob.glob(sub):
            files = glob.glob(os.path.join(var_dir, "*.nc"))
            dates = set()
            for f in files:
                m = pattern.search(os.path.basename(f))
                if m:
                    dates.add(m.group(1))
            if dates:
                all_dates_per_var.append(dates)

    common_dates = sorted(set.intersection(*all_dates_per_var))
    print(f"✅ Found {len(common_dates)} common dates across all variables.")
    return common_dates


def combine_era5_channels(date_str, root="/beegfs/muduchuru/data/era5"):
    """Combine all ERA5 variables for a given date into one dataset with a single variable 'image'."""
    subdirs = [os.path.join(root, "sf", "*"), os.path.join(root, "pl", "*")]
    files = []
    for sub in subdirs:
        files.extend(glob.glob(os.path.join(sub, f"*_{date_str}_*.nc")))

    if not files:
        raise FileNotFoundError(f"No ERA5 files found for {date_str}")

    datasets = []
    channel_names = []

    for f in sorted(files):
        try:
            ds = xr.open_dataset(f)
            var = list(ds.data_vars.keys())[0]
            da = ds[var].squeeze(drop=True)

            # Drop plev if present (we already encode it in the var name)
            if "plev" in da.coords:
                da = da.drop_vars("plev")

            da = da.expand_dims("channel").assign_coords(channel=[var])
            datasets.append(da)
            channel_names.append(var)
        except Exception as e:
            print(f"⚠️ Skipping {f}: {e}")

    # Combine into single 3D array
    image = xr.concat(datasets, dim="channel", coords="minimal", compat="override")
    image = image.sortby("lat", ascending=False)
    image.attrs["variables"] = ",".join(channel_names)

    # Add time coordinate (scalar)
    time_val = pd.to_datetime(date_str)
    image = image.expand_dims("time").assign_coords(time=[time_val])

    # Wrap into Dataset with one variable: 'image'
    ds_out = xr.Dataset({"image": image})
    ds_out["channel"] = image["channel"]

    print(f"✅ Combined {len(channel_names)} vars for {date_str}")
    return ds_out


def combine_all_common_dates(root="/beegfs/muduchuru/data/era5", out_dir=None):
    """Loop through all common dates and save combined files as E5pl00_1D_<date>_<nvar>var.nc"""
    if out_dir is None:
        out_dir = os.path.join(root, "combined")
    os.makedirs(out_dir, exist_ok=True)

    common_dates = extract_dates(root)

    for date_str in common_dates:
        try:
            ds_out = combine_era5_channels(date_str, root)
            nvars = len(ds_out.channel)
            out_filename = f"E5pl00_1D_{date_str}_{nvars}var.nc"
            out_path = os.path.join(out_dir, out_filename)

            ds_out.to_netcdf(out_path)
            print(f"💾 Saved {out_path}")
        except Exception as e:
            print(f"❌ Skipping {date_str}: {e}")


# === Run ===
combine_all_common_dates("/beegfs/muduchuru/data/era5")