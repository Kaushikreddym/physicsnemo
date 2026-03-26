#!/bin/bash
#SBATCH --job-name=generate_patches_mswxdwd     # Job name
#SBATCH --output=submit/logs/generate_patches_mswxdwd.%j.out  # STDOUT file
#SBATCH --error=submit/logs/generate_patches_mswxdwd.%j.err   # STDERR file
#SBATCH --time=4-00:00:00                # Runtime (D-HH:MM:SS) - increased for patch processing
#SBATCH --ntasks=1                        # Single task
#SBATCH --cpus-per-task=40               # CPU cores per task
#SBATCH --partition=gpu                   # Partition
##SBATCH --mem=0                           # Use all available memory on node

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

# Define years to process
YEARS=($(seq 2020 2024))  # All years from 1989 to 2024

# Patch configuration
PATCH_SIZE=128
TOTAL_PATCHES=42  # From our dataset analysis

# Parallel processing configuration
PARALLEL_PATCHES=6  # Process 2 patches simultaneously (conservative for shared GPUs)
MAX_GPU_MEMORY=32   # Total GPU memory in GB
ESTIMATED_MEMORY_PER_PATCH=6  # Increased estimate since we're using 1 GPU per process

echo "=================================================="
echo "PARALLEL PATCH-BY-PATCH GENERATION"
echo "=================================================="
echo "Patch size: ${PATCH_SIZE}x${PATCH_SIZE}"
echo "Total patches: ${TOTAL_PATCHES}"
echo "Parallel patches: ${PARALLEL_PATCHES}"
echo "Estimated GPU usage: $((PARALLEL_PATCHES * ESTIMATED_MEMORY_PER_PATCH))GB / ${MAX_GPU_MEMORY}GB"
echo "Years to process: ${YEARS[@]}"
echo "=================================================="

# Function to get total number of patches from dataset
get_total_patches() {
    python3 << EOF
import sys
sys.path.append('/beegfs/muduchuru/codes/physicsnemo/examples/weather/corrdiff')
from datasets.mswxdwd import mswxdwd

# Create a temporary loader to get total patches
loader = mswxdwd(
    data_path='/beegfs/muduchuru/data',
    input_channels=['pr', 'tas'],
    output_channels=['pr', 'tas'],
    patch_size=(${PATCH_SIZE}, ${PATCH_SIZE})
)

total = loader.get_total_patches(${PATCH_SIZE}, ${PATCH_SIZE})
print(total)
EOF
}

# Get actual total patches from dataset
echo "Calculating total patches from dataset..."
ACTUAL_TOTAL=$(get_total_patches)
echo "Dataset reports ${ACTUAL_TOTAL} total patches"

# Loop through each year
for YEAR in "${YEARS[@]}"; do
    echo ""
    echo "========================================="
    echo "Processing year: $YEAR"
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
    mkdir -p ./generated/combined
    
    # Array to store patch files for combining
    PATCH_FILES=()
    
    # Array to store background job PIDs
    PIDS=()
    RUNNING_JOBS=0
    
    # Function to process a single patch
    process_patch() {
        local PATCH_IDX=$1
        local PATCH_OUTPUT=$2
        local START_DATE=$3
        local END_DATE=$4
        local TRAIN_FLAG=$5
        
        echo "  Starting patch ${PATCH_IDX}/${ACTUAL_TOTAL}..."
        
        # Ensure conda environment is activated for this subprocess
        source ~/.bashrc
        conda activate diffusion
        
        # Set environment variables for single-node operation
        export WORLD_SIZE=1
        export RANK=0
        export LOCAL_RANK=0
        export MASTER_ADDR=localhost
        export MASTER_PORT=$((12345 + PATCH_IDX))  # Different port for each patch
        
        # Create log file for this specific patch
        PATCH_LOG="./generated/logs/patch_${PATCH_IDX}.log"
        mkdir -p ./generated/logs
        
        # Run generation for this specific patch with error logging
        torchrun --nproc-per-node=2 --master_port=$((12345 + PATCH_IDX)) generate.py --config-name=config_generate_mswxdwd_patches.yaml \
            generation.io.res_ckpt_filename=${RES_CKPT} \
            generation.io.reg_ckpt_filename=${REG_CKPT} \
            generation.times_range="[${START_DATE}, ${END_DATE}, 1d]" \
            ++generation.io.output_filename=${PATCH_OUTPUT} \
            dataset.train=${TRAIN_FLAG} \
            dataset.patch_index=${PATCH_IDX} \
            dataset.overlap_pix=10 \
            &> ${PATCH_LOG}
        
        local EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "    ✓ Patch ${PATCH_IDX} completed: ${PATCH_OUTPUT}"
        else
            echo "    ✗ ERROR: Patch ${PATCH_IDX} failed with exit code ${EXIT_CODE}"
            echo "    ✗ Check log: ${PATCH_LOG}"
        fi
        return $EXIT_CODE
    }
    
    # Wait for jobs to finish and collect successful patches
    wait_for_jobs() {
        for i in "${!PIDS[@]}"; do
            local PID=${PIDS[$i]}
            local PATCH_IDX=${PATCH_INDICES[$i]}
            local PATCH_OUTPUT=${PATCH_OUTPUTS[$i]}
            
            if wait $PID; then
                PATCH_FILES+=("${PATCH_OUTPUT}")
                echo "    ✓ Background job for patch ${PATCH_IDX} finished successfully"
            else
                echo "    ✗ Background job for patch ${PATCH_IDX} failed"
            fi
        done
        PIDS=()
        PATCH_INDICES=()
        PATCH_OUTPUTS=()
        RUNNING_JOBS=0
    }
    
    # Arrays to track job info
    PATCH_INDICES=()
    PATCH_OUTPUTS=()
    
    # Loop through each patch with parallel processing
    for PATCH_IDX in $(seq 0 $((ACTUAL_TOTAL - 1))); do
        # Set output filename for this patch
        PATCH_OUTPUT="./generated/patches/${YEAR}/mswxdwd_patch_${PATCH_IDX}_${YEAR}.nc"
        
        # Start background job for this patch
        process_patch ${PATCH_IDX} ${PATCH_OUTPUT} ${START_DATE} ${END_DATE} ${TRAIN_FLAG} &
        PID=$!
        
        # Store job info
        PIDS+=($PID)
        PATCH_INDICES+=(${PATCH_IDX})
        PATCH_OUTPUTS+=(${PATCH_OUTPUT})
        RUNNING_JOBS=$((RUNNING_JOBS + 1))
        
        echo "  Started background job (PID: $PID) for patch ${PATCH_IDX}"
        
        # If we've reached the parallel limit, wait for jobs to finish
        if [ $RUNNING_JOBS -ge $PARALLEL_PATCHES ]; then
            echo "  Waiting for ${RUNNING_JOBS} parallel jobs to complete..."
            wait_for_jobs
        fi
    done
    
    # Wait for any remaining jobs
    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "  Waiting for final ${#PIDS[@]} jobs to complete..."
        wait_for_jobs
    fi
    
    echo ""
    echo "✓ Completed generating ${#PATCH_FILES[@]} patches for year ${YEAR}"
    echo ""
done

echo "=================================================="
echo "Patch-by-patch generation completed!"
echo "=================================================="
echo "Individual patches stored in: ./generated/patches/"
echo "To combine patches into full domain files, submit:"
echo "  sbatch submit/combine_patches.sh"
echo "=================================================="
