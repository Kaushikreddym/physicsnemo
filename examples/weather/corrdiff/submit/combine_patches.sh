#!/bin/bash
#SBATCH --job-name=combine_patches              # Job name
#SBATCH --output=submit/logs/combine_patches.%j.out  # STDOUT file
#SBATCH --error=submit/logs/combine_patches.%j.err   # STDERR file
#SBATCH --time=2-00:00:00                      # Runtime
#SBATCH --partition=gpu                    # Use highmem partition for large memory operations


# --- Load environment ---
source ~/.bashrc
conda activate sdba

# --- Move to project directory ---
cd /beegfs/muduchuru/codes/physicsnemo/examples/weather/corrdiff || exit 1

# Define years to combine (auto-detect from patches directory)
YEARS=(1989 1990 1991 1992 1993 2020 2021 2022 2023 2024)
# for year_dir in ./generated/patches/*/; do
#     if [ -d "${year_dir}" ]; then
#         year=$(basename "${year_dir}")
#         YEARS+=("${year}")
#     fi
# done

# Sort years
IFS=$'\n' YEARS=($(sort -n <<<"${YEARS[*]}"))
unset IFS

# Patch configuration
PATCH_SIZE=128
N_WORKERS=32  # Match CPU count for parallel processing

echo "=================================================="
echo "COMBINING PATCHES INTO FULL DOMAINS WITH DASK"
echo "=================================================="
echo "Years to combine: ${YEARS[@]}"
echo "Patch size: ${PATCH_SIZE}x${PATCH_SIZE}"
echo "Parallel workers: ${N_WORKERS}"
echo "Partition: highmem"
echo "Memory: 256GB"
echo "=================================================="

# Create output directory
mkdir -p ./generated/combined

for YEAR in "${YEARS[@]}"; do
    echo ""
    echo "========================================="
    echo "Combining patches for year: $YEAR"
    echo "========================================="
    
    # Check if patch directory exists
    PATCH_DIR="./generated/patches/${YEAR}"
    if [ ! -d "${PATCH_DIR}" ]; then
        echo "✗ ERROR: Patch directory ${PATCH_DIR} does not exist"
        continue
    fi
    
    # Count available patches
    PATCH_COUNT=$(ls ${PATCH_DIR}/mswxdwd_patch_*_${YEAR}.nc 2>/dev/null | wc -l)
    echo "Found ${PATCH_COUNT} patch files for year ${YEAR}"
    
    if [ $PATCH_COUNT -eq 0 ]; then
        echo "✗ No patches found for year ${YEAR}. Skipping..."
        continue
    fi
    
    # Set output file
    COMBINED_OUTPUT="./generated/combined/mswxdwd_combined_${YEAR}.nc"
    
    # Create pattern for patch files
    PATCH_PATTERN="${PATCH_DIR}/mswxdwd_patch_*_${YEAR}.nc"
    
    echo "Combining patches using spatial reconstruction..."
    echo "Input pattern: ${PATCH_PATTERN}"
    echo "Output file: ${COMBINED_OUTPUT}"
    
    # Use our custom Python script for spatial combination with Dask parallelization
    python -u combine_patches.py \
        --input-pattern "${PATCH_PATTERN}" \
        --output-file "${COMBINED_OUTPUT}" \
        --domain-height 867 \
        --domain-width 642 \
        --patch-height ${PATCH_SIZE} \
        --patch-width ${PATCH_SIZE} \
        --overlap-pix 2 \
        --n-workers ${N_WORKERS}
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully combined patches for year $YEAR"
        echo "  Output: ${COMBINED_OUTPUT}"
        echo "  Output size: $(du -h ${COMBINED_OUTPUT} 2>/dev/null | cut -f1 || echo "unknown")"
        
        # Verify the output file
        echo "  Verifying output file..."
        python3 -c "
import xarray as xr
try:
    ds = xr.open_dataset('${COMBINED_OUTPUT}')
    print(f'    Dimensions: {dict(ds.dims)}')
    print(f'    Variables: {list(ds.data_vars)}')
    print(f'    Time range: {ds.time.min().values} to {ds.time.max().values}')
    print('    ✓ File verification passed')
except Exception as e:
    print(f'    ✗ File verification failed: {e}')
    exit(1)
"
        
    else
        echo "✗ ERROR: Failed to combine patches for year $YEAR"
    fi
done

echo ""
echo "=================================================="
echo "Patch combination completed!"
echo "=================================================="
echo "Combined files stored in: ./generated/combined/"

# Summary
echo ""
echo "SUMMARY:"
for YEAR in "${YEARS[@]}"; do
    COMBINED_FILE="./generated/combined/mswxdwd_combined_${YEAR}.nc"
    if [ -f "${COMBINED_FILE}" ]; then
        SIZE=$(du -h "${COMBINED_FILE}" 2>/dev/null | cut -f1)
        echo "  ✓ ${YEAR}: ${COMBINED_FILE} (${SIZE})"
    else
        echo "  ✗ ${YEAR}: No combined file generated"
    fi
done
echo "=================================================="
