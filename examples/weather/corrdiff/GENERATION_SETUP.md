# Full Domain Generation Setup for MSWX-DWD

## Overview
This setup enables generating CorrDiff predictions over the entire domain using automatic tiling/patching.

## Key Changes

### 1. Configuration File (`conf/config_generate_mswxdwd.yaml`)
- **Switched to patched mode**: Uses `generation: patched` instead of `non_patched`
- **Disabled center cropping**: Set `center_latlon: null` for full domain coverage
- **Defined patch size**: `patch_shape_x: 256` and `patch_shape_y: 256`
  - Patches will automatically tile across entire domain with overlap
  - Adjust patch size based on GPU memory (larger = faster but more memory)

### 2. Submit Script (`submit/generate_mswxdwd.sh`)
- **Added training years**: Now includes years 1989-2020 (training) + 2021-2024 (validation)
- **Increased runtime**: Changed from 1 day to 3 days for full domain processing
- **Better naming**: Output files named `mswxdwd_corrdiff_fulldomain_YYYY.nc`
- **Error handling**: Checks exit status and reports success/failure per year
- **Explicit parameters**: Passes `center_latlon=null` and patch sizes via command line

### 3. How Tiling Works
The `GridPatching2D` utility in generate.py automatically:
1. Divides the full domain into overlapping patches (256x256)
2. Processes each patch separately
3. Blends overlapping regions to avoid artifacts
4. Reconstructs the full domain output

Overlap and boundary settings:
- `overlap_pix: 4` - pixels of overlap between patches
- `boundary_pix: 2` - boundary pixels cropped to reduce artifacts

## Usage

### Generate All Years (1989-2024)
```bash
sbatch submit/generate_mswxdwd.sh
```

### Generate Specific Years Only
Edit the script and modify:
```bash
# Change this line:
ALL_YEARS=("${TRAIN_YEARS[@]}" "${VAL_YEARS[@]}")

# To only specific years:
ALL_YEARS=(2021 2022 2023)
```

### Adjust Patch Size
For different GPU memory:
- **16 GB GPU**: Use 256x256 (current setting)
- **24 GB GPU**: Can use 384x384 or 512x512
- **40 GB GPU**: Can use 512x512 or larger

Edit in script:
```bash
++generation.patch_shape_x=384 \
++generation.patch_shape_y=384
```

Or edit in config file directly.

## Output Files
Generated files are saved as:
```
./generated/mswxdwd_corrdiff_fulldomain_1989.nc
./generated/mswxdwd_corrdiff_fulldomain_1990.nc
...
./generated/mswxdwd_corrdiff_fulldomain_2024.nc
```

Each NetCDF file contains:
- `input`: Low-resolution MSWX input
- `truth`: High-resolution DWD ground truth
- `prediction`: CorrDiff downscaled output (8 ensemble members)

## Performance Notes
- Full domain generation is significantly slower than patch-based
- Expected time per year: ~4-8 hours (depending on domain size and GPU)
- Uses 2 GPUs in parallel (adjust `--nproc-per-node=2` if needed)
- Total time for all 36 years: ~150-300 hours (6-12 days)

## Monitoring Progress
```bash
# Check SLURM output
tail -f submit/logs/jnb.job.JOBID.out

# Check for errors
tail -f submit/logs/jnb.job.JOBID.err

# Check which year is being processed
grep "Processing year" submit/logs/jnb.job.JOBID.out

# Check completed years
grep "Successfully completed" submit/logs/jnb.job.JOBID.out
```
