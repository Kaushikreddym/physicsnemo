#!/bin/bash
#SBATCH --job-name=generate_era5mswx             # Job name
#SBATCH --output=submit/logs/generate_era5mswx.%j.out  # STDOUT file
#SBATCH --error=submit/logs/generate_era5mswx.%j.err   # STDERR file
#SBATCH --time=3-00:00:00                # Runtime (D-HH:MM:SS) - increased for full domain
#SBATCH --ntasks=1                        # Single task
#SBATCH --cpus-per-task=40               # CPU cores per task
#SBATCH --partition=gpu                   # Partition
#SBATCH --mem=0                           # Use all available memory on node
##SBATCH --hint=nomultithread              # Optional: for CPU affinity
## SBATCH --nodelist=gpu001               # Uncomment if you need a specific node

# --- Load environment ---
source ~/.bashrc
conda activate diffusion

# --- Move to project directory ---
cd /beegfs/muduchuru/codes/physicsnemo/examples/weather/corrdiff || exit 1

# --- Optional: print GPU info ---
echo "Running on node: $(hostname)"
nvidia-smi

# --- Configuration ---
# Find the newest checkpoints
RES_CKPT=$(ls -t checkpoints_diffusion_era5mswx/checkpoints_diffusion/EDMPrecond*.mdlus 2>/dev/null | head -n 1)
REG_CKPT=$(ls -t checkpoints_regression_era5mswx/checkpoints_regression/UNet.*.mdlus 2>/dev/null | head -n 1)

if [ -z "$RES_CKPT" ]; then
    echo "ERROR: No diffusion checkpoint found in checkpoints_diffusion_era5mswx/"
    echo "Please train the diffusion model first using diffusion_era5mswx.sh"
    exit 1
fi

if [ -z "$REG_CKPT" ]; then
    echo "ERROR: No regression checkpoint found in checkpoints_regression_era5mswx/"
    echo "Please train the regression model first using regression_era5mswx.sh"
    exit 1
fi

echo "Using diffusion checkpoint: $RES_CKPT"
echo "Using regression checkpoint: $REG_CKPT"

# Define years to process - includes both training (1999-2020) and validation (2021-2024)
# Training years
TRAIN_YEARS=($(seq 1999 2020))
# Validation years
VAL_YEARS=(2021 2022 2023 2024)

# Combine all years
ALL_YEARS=("${TRAIN_YEARS[@]}" "${VAL_YEARS[@]}")

# Or specify only specific years to process
# ALL_YEARS=(2021 2022 2023 2024)

echo "=================================================="
echo "FULL DOMAIN GENERATION WITH TILING (PATCHED MODE)"
echo "=================================================="
echo "Patch size: 256x256 with overlap"
echo "Total years to process: ${#ALL_YEARS[@]}"
echo "=================================================="

# Loop through each year
for YEAR in "${ALL_YEARS[@]}"; do
    echo ""
    echo "========================================="
    echo "Processing year: $YEAR"
    echo "========================================="
    
    # Set start and end dates for the year
    START_DATE="${YEAR}-01-01T00:00:00"
    END_DATE="${YEAR}-12-31T00:00:00"
    
    # Set output filename with year
    OUTPUT_FILE="./generated/era5mswx_corrdiff_fulldomain_${YEAR}.nc"
    
    # Determine if this is a training year or validation year
    if [ $YEAR -ge 1999 ] && [ $YEAR -le 2020 ]; then
        TRAIN_FLAG="True"
        YEAR_TYPE="training"
    else
        TRAIN_FLAG="False"
        YEAR_TYPE="validation"
    fi
    
    echo "Year type: ${YEAR_TYPE} (dataset.train=${TRAIN_FLAG})"
    
    # Create output directory if it doesn't exist
    mkdir -p ./generated
    
    # Run generation for this year with full domain tiling using torchrun
    torchrun --nproc_per_node=2 generate.py --config-name=config_generate_era5mswx.yaml \
        generation.io.res_ckpt_filename=${RES_CKPT} \
        generation.io.reg_ckpt_filename=${REG_CKPT} \
        generation.times_range="[${START_DATE}, ${END_DATE}, 1d]" \
        generation.io.output_filename=${OUTPUT_FILE} \
        dataset.train=${TRAIN_FLAG}
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed generation for year $YEAR"
        echo "  Output: ${OUTPUT_FILE}"
    else
        echo "✗ ERROR: Generation failed for year $YEAR"
        echo "  Check logs for details"
    fi
    echo ""
done

echo "=================================================="
echo "All years completed!"
echo "=================================================="
