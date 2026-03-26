# Patch-based Generation for mswxdwd Dataset

This directory contains scripts for systematic patch-based generation that processes the entire domain by looping through each patch index and then combining the results.

## Files Created:

### 1. `generate_mswxdwd_patches.sh`
**SLURM job script for patch-based generation**
- Loops through all patch indices systematically
- Generates results for each patch individually  
- Combines patches into spatially coherent NetCDF files
- Handles both training and validation years

### 2. `config_generate_mswxdwd_patches.yaml`
**Configuration file for patch-based generation**
- Uses dataset-level patching with `patch_index` parameter
- Configured for non-overlapping 128x128 patches
- Uses `non_patched` generation mode since patching is handled at dataset level

### 3. `combine_patches.py` 
**Python script for spatial patch combination**
- Reconstructs full domain from individual patch files
- Places patches in correct spatial locations based on patch index
- Handles overlapping regions if needed
- Creates spatially coherent NetCDF output

### 4. `test_patch_indexing.py`
**Test script to verify patch functionality**
- Tests patch indexing before running full generation
- Validates that patches are created correctly
- Reports total number of patches and metadata

## Usage:

### Step 1: Test the patch indexing
```bash
cd /beegfs/muduchuru/codes/physicsnemo/examples/weather/corrdiff
python3 test_patch_indexing.py
```

### Step 2: Submit the patch generation job
```bash
sbatch submit/generate_mswxdwd_patches.sh
```

### Step 3: Check results
Individual patches will be stored in:
```
./generated/patches/YYYY/mswxdwd_patch_N_YYYY.nc
```

Combined files will be stored in:
```
./generated/combined/mswxdwd_combined_YYYY.nc
```

## Features:

- **Systematic Coverage**: Processes entire domain without gaps or overlaps
- **Memory Efficient**: Processes one patch at a time instead of full domain
- **Reproducible**: Deterministic patch ordering based on spatial grid
- **Scalable**: Easy to parallelize by running different patch ranges on different nodes
- **Compatible**: Works with existing physicsnemo GridPatching2D framework

## Configuration:

Current settings (can be modified in the scripts):
- **Patch size**: 128x128 pixels
- **Overlap**: 0 pixels (non-overlapping)
- **Domain**: 867x642 pixels (Germany region)
- **Total patches**: 42 patches per domain
- **Years**: Currently set to 1989-1990 for testing

To process different years, modify the `YEARS` array in `generate_mswxdwd_patches.sh`.