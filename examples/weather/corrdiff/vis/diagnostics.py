"""
vis/diagnostics.py

Full-feature diagnostics for gridded model predictions vs truth (ERA5-style).
Features:
 - automatic spatial dim handling (y/x -> lat/lon)
 - ensemble handling (average or choose member)
 - flexible time intersection
 - denormalization support (stats JSON)
 - daily spatial metrics (rmse, mae, mape, bias, corr, acc)
 - monthly aggregation & plotting
 - spatial maps (truth/pred/error), time series, monthly bars
 - Taylor diagram and PDF report export

Usage example:
    from vis.diagnostics import variable_diagnostics
    out = variable_diagnostics(
        ds_truth, ds_pred,
        var="tas",
        stats=stats_dict,
        denormalize=True,
        ensemble="mean",   # or int index
        monthly_agg="mean",
        outdir="./diag_out",
        save_pdf=True
    )
"""

import os
import json
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Optional: use scipy for robust correlation if available
try:
    from scipy.stats import pearsonr
except Exception:
    pearsonr = None


# ---------------------- Utilities ----------------------
def _ensure_spatial_dims(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensure dataset uses dims 'lat' and 'lon'. Rename 'y'/'x' if needed.
    """
    rename_map = {}
    if "y" in ds.dims and "lat" not in ds.dims:
        rename_map["y"] = "lat"
    if "x" in ds.dims and "lon" not in ds.dims:
        rename_map["x"] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def _squeeze_ensemble(ds: xr.Dataset, ensemble: Optional[Any] = "mean") -> xr.Dataset:
    """
    Handle ensemble dimension:
      - ensemble='mean' -> return ensemble mean (dropped)
      - ensemble=int -> pick that member index
      - ensemble=None -> leave as-is
    """
    if "ensemble" in ds.dims:
        if ensemble is None:
            return ds
        if ensemble == "mean":
            return ds.mean(dim="ensemble")
        if isinstance(ensemble, int):
            return ds.isel(ensemble=ensemble).drop_vars([], errors="ignore")
        raise ValueError("ensemble must be 'mean', None, or integer index")
    return ds


def _align_time_flexible(ds_truth: xr.Dataset, ds_pred: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Intersect time coordinates and return aligned datasets (truth, pred).
    If times are identical, returns a copy. Assumes ds have 'time' coordinate.
    """
    t_truth = pd.to_datetime(ds_truth["time"].values)
    t_pred = pd.to_datetime(ds_pred["time"].values)
    common = np.intersect1d(t_truth, t_pred)
    if len(common) == 0:
        raise ValueError("No overlapping times between truth and prediction datasets.")
    # select common times in original order of truth if possible
    ds_truth_sel = ds_truth.sel(time=common)
    ds_pred_sel = ds_pred.sel(time=common)
    return ds_truth_sel, ds_pred_sel


def denormalize_ds(ds: xr.Dataset, stats: Dict[str, Dict[str, float]], vars: Optional[List[str]] = None) -> xr.Dataset:
    """
    Denormalize dataset variables using stats dict: stats[var] = {'mean':..., 'std':...}
    Operation: x * std + mean
    """
    ds_out = ds.copy(deep=True)
    var_list = vars or list(ds_out.data_vars)
    for v in var_list:
        if v in stats:
            m = float(stats[v]["mean"])
            s = float(stats[v]["std"])
            ds_out[v] = ds_out[v] * s + m
    return ds_out


# ---------------------- Metrics ----------------------
def spatial_mean_metrics_per_timestep(ds_truth: xr.Dataset, ds_pred: xr.Dataset, var: str,
                                      spatial_dims: Tuple[str, str] = ("lat", "lon"),
                                      eps_mape: float = 1e-6) -> pd.DataFrame:
    """
    Compute RMSE, MAE, MAPE (%), Bias (mean(pred-truth)), Corr (Pearson), ACC (Anomaly correlation)
    Returns a pandas.DataFrame indexed by time.
    Assumes ds_pred and ds_truth are aligned in time and spatial dims exist.
    """
    t = ds_truth[var]
    p = ds_pred[var]

    # Ensure spatial dims exist
    for d in spatial_dims:
        if d not in t.dims or d not in p.dims:
            raise ValueError(f"Missing spatial dim {d} in truth or pred")

    # If ensemble present in p, average if not already handled
    if "ensemble" in p.dims:
        p = p.mean(dim="ensemble")

    # Errors
    err = p - t
    sqerr = (err ** 2).mean(dim=spatial_dims)
    rmse = np.sqrt(sqerr)
    mae = np.abs(err).mean(dim=spatial_dims)
    bias = err.mean(dim=spatial_dims)

    # MAPE: avoid division by tiny values
    denom = xr.where(np.abs(t) < eps_mape, np.nan, t)
    mape_field = (np.abs(err / denom)) * 100.0
    mape = mape_field.mean(dim=spatial_dims)

    # Correlation and Anomaly Correlation across flattened spatial domain per timestep
    times = t["time"].values
    corr_list = []
    acc_list = []
    for ti in range(t.sizes["time"]):
        t_slice = t.isel(time=ti).values.ravel()
        p_slice = p.isel(time=ti).values.ravel()
        mask = np.isfinite(t_slice) & np.isfinite(p_slice)
        if mask.sum() < 3:
            corr_list.append(np.nan)
            acc_list.append(np.nan)
            continue
        # Pearson r
        if pearsonr:
            r = pearsonr(t_slice[mask], p_slice[mask])[0]
        else:
            r = np.corrcoef(t_slice[mask], p_slice[mask])[0, 1]
        corr_list.append(float(r))

        # Anomaly correlation: correlation of anomalies wrt climatology (here: temporal mean of truth)
        # compute anomaly fields using temporal mean across time dimension (use full t)
        t_clim = np.nanmean(t.values, axis=0).ravel()
        p_clim = np.nanmean(p.values, axis=0).ravel()
        t_anom = t_slice - t_clim
        p_anom = p_slice - t_clim  # note: ACC uses truth climatology as reference
        mask2 = np.isfinite(t_anom) & np.isfinite(p_anom)
        if mask2.sum() < 3:
            acc_list.append(np.nan)
        else:
            if pearsonr:
                acc = pearsonr(t_anom[mask2], p_anom[mask2])[0]
            else:
                acc = np.corrcoef(t_anom[mask2], p_anom[mask2])[0, 1]
            acc_list.append(float(acc))

    corr = xr.DataArray(corr_list, coords={"time": times}, dims=["time"])
    acc = xr.DataArray(acc_list, coords={"time": times}, dims=["time"])

    df = pd.DataFrame({
        "rmse": rmse.values.squeeze(),
        "mae": mae.values.squeeze(),
        "mape": mape.values.squeeze(),
        "bias": bias.values.squeeze(),
        "corr": corr.values.squeeze(),
        "acc": acc.values.squeeze()
    }, index=pd.to_datetime(times))

    return df


def monthly_aggregate(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    if how not in ("mean", "median"):
        raise ValueError("how must be 'mean' or 'median'")
    if how == "mean":
        return df.resample("M").mean()
    return df.resample("M").median()


# ---------------------- Plot helpers ----------------------
def _get_lat_lon_from_ds(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return 2D lon, lat arrays compatible with pcolormesh.
    Accepts 1D coords (lat, lon) or 2D (lat, lon).
    """
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("Dataset lacks 'lat' or 'lon' coordinates")
    lat = ds["lat"].values
    lon = ds["lon"].values
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lon2d, lat2d
    return lon, lat


def plot_spatial_triplet(ds_truth: xr.Dataset, ds_pred: xr.Dataset, var: str, time_idx: int = 0,
                         vmin=None, vmax=None, cmap=None, figsize=(15, 5), savepath: Optional[str] = None):
    lon, lat = _get_lat_lon_from_ds(ds_truth)
    da_t = ds_truth[var].isel(time=time_idx)
    da_p = ds_pred[var].isel(time=time_idx)
    da_err = da_p - da_t

    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(1, 3, figsize=figsize, subplot_kw=dict(projection=proj))
    for ax, da, ttl in zip(axs, [da_t, da_p, da_err], ["Truth", "Prediction", "Error"]):
        im = ax.pcolormesh(lon, lat, da.values, transform=ccrs.PlateCarree(),
                           vmin=vmin, vmax=vmax, cmap=cmap)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.set_title(f"{var} - {ttl}")
        plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def plot_metric_timeseries(df_metrics: pd.DataFrame, vars_to_plot: Optional[List[str]] = None,
                           figsize=(12, 4), savepath: Optional[str] = None):
    plt.figure(figsize=figsize)
    cols = vars_to_plot or df_metrics.columns.tolist()
    df_metrics[cols].plot(marker='.', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    return plt.gcf()


def plot_monthly_bar(df_monthly: pd.DataFrame, metric: str = "rmse", figsize=(10, 4), savepath: Optional[str] = None):
    plt.figure(figsize=figsize)
    df_monthly[metric].plot(kind="bar")
    plt.title(f"{metric.upper()} (monthly)")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    return plt.gcf()


# ---------------------- Taylor diagram ----------------------
# Reference: simple custom Taylor diagram implementation
def plot_taylor_diagram(obs: np.ndarray, pred: np.ndarray, ax=None, label_obs="Obs", label_pred="Model", savepath: Optional[str] = None):
    """
    obs, pred: 1D arrays (flattened spatial values across time or vector of means)
    This implementation plots std vs correlation with RMS contours.
    """
    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, polar=True)
    else:
        fig = ax.figure

    # compute stats
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 3:
        raise ValueError("Not enough valid points for Taylor diagram")

    obs = obs[mask]
    pred = pred[mask]
    s_obs = obs.std(ddof=0)
    s_pred = pred.std(ddof=0)
    corr = np.corrcoef(obs, pred)[0,1]

    # polar coordinates
    theta = np.arccos(corr)
    r = s_pred

    # plot obs point at angle 0
    ax.plot(0, s_obs, 'k*', markersize=10, label=label_obs)
    # plot model point
    ax.plot(theta, r, 'o', label=label_pred)

    # RMS contours
    maxstd = max(s_obs, s_pred) * 1.5
    rs = np.linspace(0, maxstd, 100)
    thetas = np.linspace(0, np.pi/2, 100)
    # draw circular grid
    ax.set_ylim(0, maxstd)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_title("Taylor Diagram (std vs corr)")

    ax.legend(loc='upper right')
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


# ---------------------- Report & main routine ----------------------
def variable_diagnostics(ds_truth: xr.Dataset,
                         ds_pred: xr.Dataset,
                         var: str,
                         stats: Optional[Dict[str, Dict[str, float]]] = None,
                         denormalize: bool = True,
                         ensemble: Optional[Any] = "mean",
                         monthly_agg: str = "mean",
                         outdir: str = "./diagnostics_out",
                         save_pdf: bool = True,
                         make_spatial_examples: bool = True) -> Dict[str, Any]:
    """
    Full diagnostics pipeline.
    Returns a dict with:
      - daily_metrics (pd.DataFrame)
      - monthly_metrics (pd.DataFrame)
      - figs (list of matplotlib.Figure)
      - pdf_path (if created)
    """
    os.makedirs(outdir, exist_ok=True)

    # Preprocess datasets
    ds_truth = _ensure_spatial_dims(ds_truth)
    ds_pred = _ensure_spatial_dims(ds_pred)

    # Squeeze ensemble if requested
    ds_pred = _squeeze_ensemble(ds_pred, ensemble=ensemble)
    # # Fix time coordinate dtype issues before diagnostics
    # ds_truth['time'] = xr.conventions.decode_cf(ds_truth).time.astype('datetime64[ns]')
    # ds_pred['time'] = xr.conventions.decode_cf(ds_pred).time.astype('datetime64[ns]')

    # # Optionally round because ERA5 often has micro-second drift
    # ds_truth['time'] = ds_truth['time'].dt.round('H')
    # ds_pred['time'] = ds_pred['time'].dt.round('H')

    # Assign coords from truth if pred lacks them
    if "lat" in ds_truth.coords and "lat" not in ds_pred.coords:
        ds_pred = ds_pred.assign_coords({"lat": ds_truth["lat"]})
    if "lon" in ds_truth.coords and "lon" not in ds_pred.coords:
        ds_pred = ds_pred.assign_coords({"lon": ds_truth["lon"]})

    # Denormalize if requested
    if denormalize and stats:
        ds_truth = denormalize_ds(ds_truth, stats, vars=[var]) if var in ds_truth else ds_truth
        ds_pred = denormalize_ds(ds_pred, stats, vars=[var]) if var in ds_pred else ds_pred

    # Align times (flexible)
    # ds_truth, ds_pred = _align_time_flexible(ds_truth, ds_pred)

    # Compute daily spatially-aggregated metrics
    daily_df = spatial_mean_metrics_per_timestep(ds_truth, ds_pred, var, spatial_dims=("lat", "lon"))
    monthly_df = monthly_aggregate(daily_df, how=monthly_agg)

    figs = []
    # Spatial examples: first, middle, last
    if make_spatial_examples:
        ntime = ds_truth.sizes["time"]
        idxs = [0, max(0, ntime//2), max(0, ntime-1)]
        for i in idxs:
            pth = os.path.join(outdir, f"{var}_spatial_{i}.png")
            fig = plot_spatial_triplet(ds_truth, ds_pred, var, time_idx=i, savepath=pth)
            figs.append(fig)

    # Time series figure
    pth_ts = os.path.join(outdir, f"{var}_daily_timeseries.png")
    fig_ts = plot_metric_timeseries(daily_df, savepath=pth_ts)
    figs.append(fig_ts)

    # Monthly bar
    pth_bar = os.path.join(outdir, f"{var}_monthly_{monthly_agg}.png")
    fig_bar = plot_monthly_bar(monthly_df, metric="rmse", savepath=pth_bar)
    figs.append(fig_bar)

    # Taylor diagram using flattened truth/pred over entire period
    try:
        obs_flat = ds_truth[var].values.ravel()
        pred_flat = ds_pred[var].values.ravel()
        pth_taylor = os.path.join(outdir, f"{var}_taylor.png")
        fig_taylor = plot_taylor_diagram(obs_flat, pred_flat, savepath=pth_taylor)
        figs.append(fig_taylor)
    except Exception as e:
        print(f"Taylor diagram failed: {e}")

    pdf_path = None
    if save_pdf:
        pdf_path = os.path.join(outdir, f"{var}_diagnostics_report.pdf")
        with PdfPages(pdf_path) as pdf:
            # Cover page
            fig_cover = plt.figure(figsize=(8.5, 11))
            fig_cover.clf()
            fig_cover.text(0.5, 0.6, f"Diagnostics report for {var}", ha="center", fontsize=18)
            fig_cover.text(0.5, 0.55, f"Number of timesteps: {ds_truth.sizes['time']}", ha="center")
            fig_cover.text(0.5, 0.5, f"Ensemble handling: {ensemble}", ha="center")
            fig_cover.text(0.5, 0.45, f"Denormalized: {denormalize and bool(stats)}", ha="center")
            pdf.savefig(fig_cover); plt.close(fig_cover)

            # Append all saved figures (spatial/time/monthly/taylor)
            for fig in figs:
                pdf.savefig(fig)
                plt.close(fig)

            # Append a summary table (daily metrics head)
            fig_tab = plt.figure(figsize=(8.5, 11))
            fig_tab.clf()
            txt = daily_df.head(20).to_string()
            fig_tab.text(0.01, 0.99, "Daily metrics (head)", va="top", fontsize=8, family="monospace")
            fig_tab.text(0.01, 0.95, txt, va="top", fontsize=6, family="monospace")
            pdf.savefig(fig_tab); plt.close(fig_tab)

        print(f"Saved PDF report to {pdf_path}")

    return {
        "daily_metrics": daily_df,
        "monthly_metrics": monthly_df,
        "figures": figs,
        "pdf_path": pdf_path,
        "outdir": outdir
    }


# ---------------------- CLI-like helper ----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run diagnostics on truth vs prediction netCDFs")
    parser.add_argument("--truth", type=str, required=True, help="truth netCDF file (or path to openable xarray dataset)")
    parser.add_argument("--pred", type=str, required=True, help="prediction netCDF file")
    parser.add_argument("--var", type=str, required=True, help="variable name to evaluate")
    parser.add_argument("--stats", type=str, default=None, help="json stats file for denormalization")
    parser.add_argument("--ensemble", type=str, default="mean", help="'mean' or integer index or 'none'")
    parser.add_argument("--outdir", type=str, default="./diagnostics_out", help="output directory")
    parser.add_argument("--no-pdf", action="store_true", help="do not save PDF")
    args = parser.parse_args()

    ds_truth = xr.open_dataset(args.truth, decode_times=True)
    ds_pred = xr.open_dataset(args.pred, decode_times=True, group="prediction") if "group=" in args.pred else xr.open_dataset(args.pred, decode_times=True)
    # allow user to pass 'none' to keep ensemble
    ensemble_arg = None if args.ensemble.lower() in ("none", "null") else (int(args.ensemble) if args.ensemble.isdigit() else args.ensemble)

    stats = None
    if args.stats:
        with open(args.stats, "r") as fh:
            stats = json.load(fh)

    variable_diagnostics(ds_truth, ds_pred, var=args.var, stats=stats, denormalize=bool(stats),
                         ensemble=ensemble_arg, outdir=args.outdir, save_pdf=not args.no_pdf)
