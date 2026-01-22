import os
import glob
import xarray as xr
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = "/beegfs/muduchuru/data/hyras"
OUT_DIR = "/beegfs/muduchuru/data/hyras_daily"

VARIABLES = ["PR", "TAS", "TASMAX", "TASMIN", "HURS", 'RSDS']

def process_file(file, var):
    """Split a yearly file into daily files and save."""
    base_name = os.path.basename(file)
    year = base_name.split("_")[3]  # e.g., tas_hyras_1_2004_v6-0_de.nc → '2004'
    out_var_dir = os.path.join(OUT_DIR, var.lower())
    os.makedirs(out_var_dir, exist_ok=True)

    try:
        with xr.open_dataset(file, chunks="auto") as ds:
            varname = list(ds.data_vars)[0]
            time_dim = ds.time

            # if len(time_dim) <= 365:
            #     raise ValueError(f"Expected 365 days in {file}, found {len(time_dim)}")

            for t_idx in range(len(time_dim)):
                ds_day = ds.isel(time=t_idx)
                
                # Preserve attributes
                ds_day.attrs = ds.attrs
                ds_day[varname].attrs = ds[varname].attrs
                
                date = pd.to_datetime(ds_day.time.values)
                date_str = date.strftime("%Y-%m-%d")

                # Replace year in base_name with date_str
                out_file_name = base_name.replace(year, date_str)
                out_file = os.path.join(out_var_dir, out_file_name)

                ds_day.to_netcdf(out_file)
        return f"✅ Saved {file}"
    except Exception as e:
        return f"⚠️ Error in {file}: {e}"

if __name__ == "__main__":
    tasks = []
    with ProcessPoolExecutor() as executor:
        for var in VARIABLES:
            files = sorted(glob.glob(os.path.join(DATA_DIR, var, f"{var.lower()}_hyras_*_de.nc")))
            for file in files:
                tasks.append(executor.submit(process_file, file, var))

        for future in as_completed(tasks):
            print(future.result())
