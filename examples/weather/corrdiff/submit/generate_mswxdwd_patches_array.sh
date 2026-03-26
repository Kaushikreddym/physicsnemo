#!/bin/bash
#SBATCH --job-name=gen_patches_array         # Job name
#SBATCH --output=submit/logs/gen_patches_%A_%a.out  # STDOUT file (%A=job_id, %a=array_index)
#SBATCH --error=submit/logs/gen_patches_%A_%a.err   # STDERR file
#SBATCH --time=2-00:00:00                    # Runtime per job
#SBATCH --ntasks=1                           # Single task per array job
#SBATCH --cpus-per-task=8                    # CPU cores per task
#SBATCH --partition=gpu                      # Partition
#SBATCH --mem=32G                            # Memory per job
#SBATCH --array=0-41%8                       # Array jobs for patches 0-41, max 8 concurrent

# --- Load environment ---
source ~/.bashrc
conda activate diffusion

# --- Move to project directory ---
cd /beegfs/muduchuru/codes/physicsnemo/examples/weather/corrdiff || exit 1

# --- Configuration ---
# Find the newest checkpoints
RES_CKPT=$(ls -t checkpoints_diffusion_mswxdwd/checkpoints_diffusion//EDMPrecond*.mdlus 2>/dev/null | head -n 1)
REG_CKPT=$(ls -t checkpoints_regression_mswxdwd/checkpoints_regression/UNet.*.mdlus 2>/dev/null | head -n 1)

if [ -z "$RES_CKPT" ]; then
    echo "ERROR: No diffusion checkpoint found in checkpoints_diffusion_mswxdwd/"
    exit 1
fi

if [ -z "$REG_CKPT" ]; then
    echo "ERROR: No regression checkpoint found in checkpoints_regression_mswxdwd/"
    exit 1
fi

echo "Using diffusion checkpoint: $RES_CKPT"
echo "Using regression checkpoint: $REG_CKPT"

# Get patch index from SLURM array task ID
PATCH_IDX=$SLURM_ARRAY_TASK_ID

# Define years to process - each array job will process one patch for all years
YEARS=(1999 2024)

echo "=================================================="
echo "PROCESSING PATCH ${PATCH_IDX} (Array Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})"
echo "=================================================="
echo "Running on node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Years to process: ${YEARS[@]}"

# Process this patch for each year
for YEAR in "${YEARS[@]}"; do
    echo ""
    echo "========================================="
    echo "Processing patch ${PATCH_IDX} for year: $YEAR"
    echo "========================================="
    
    # Set start and end dates for the year
    START_DATE="${YEAR}-01-01T00:00:00"
    END_DATE="${YEAR}-12-31T00:00:00"
    
    # Determine if this is a training year
    if [ $YEAR -ge 1989 ] && [ $YEAR -le 2020 ]; then
        TRAIN_FLAG="True"
        YEAR_TYPE="training"
    else
        TRAIN_FLAG="False"
        YEAR_TYPE="validation"
    fi
    
    echo "Year type: ${YEAR_TYPE} (dataset.train=${TRAIN_FLAG})"
    
    # Create output directories
    mkdir -p ./generated/patches/${YEAR}
    
    # Set output filename for this patch
    PATCH_OUTPUT="./generated/patches/${YEAR}/mswxdwd_patch_${PATCH_IDX}_${YEAR}.nc"
    
    echo "  Generating patch ${PATCH_IDX} for ${YEAR}..."
    echo "  Output: ${PATCH_OUTPUT}"
    
    # Run generation for this specific patch
    python generate.py --config-name=config_generate_mswxdwd_patches.yaml \
        generation.io.res_ckpt_filename=${RES_CKPT} \
        generation.io.reg_ckpt_filename=${REG_CKPT} \
        generation.times_range="[${START_DATE}, ${END_DATE}, 1d]" \
        ++generation.io.output_filename=${PATCH_OUTPUT} \
        dataset.train=${TRAIN_FLAG} \
        dataset.patch_index=${PATCH_IDX} \
        dataset.overlap_pix=0 \
        +distributed.enable=false
    
    if [ $? -eq 0 ]; then
        echo "    ✓ Patch ${PATCH_IDX} for ${YEAR} completed successfully"
        echo "    Output size: $(du -h ${PATCH_OUTPUT} 2>/dev/null | cut -f1 || echo "unknown")"
    else
        echo "    ✗ ERROR: Patch ${PATCH_IDX} for ${YEAR} failed"
        exit 1
    fi
done

echo ""
echo "=================================================="
echo "Patch ${PATCH_IDX} completed for all years!"
echo "=================================================="