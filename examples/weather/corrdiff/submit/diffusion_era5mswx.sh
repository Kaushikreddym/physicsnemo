#!/bin/bash
#SBATCH --job-name=diffusion_era5mswx             # Job name
#SBATCH --output=submit/logs/diffusion_era5mswx.%j.out  # STDOUT file
#SBATCH --error=submit/logs/diffusion_era5mswx.%j.err   # STDERR file
#SBATCH --time=3-00:00:00                # Runtime (D-HH:MM:SS)
#SBATCH --ntasks=1                        # Single task (torchrun will handle multiple GPUs)
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

# --- Create checkpoint directory ---
mkdir -p checkpoints_diffusion_era5mswx

# --- Find the newest UNet regression checkpoint ---
REGRESSION_CKPT=$(ls -t checkpoints_regression_era5mswx/checkpoints_regression/UNet.*.mdlus 2>/dev/null | head -n 1)

if [ -z "$REGRESSION_CKPT" ]; then
    echo "ERROR: No regression checkpoint found in checkpoints_regression_era5mswx/"
    echo "Please train the regression model first using regression_era5mswx.sh"
    exit 1
fi

echo "Using regression checkpoint: $REGRESSION_CKPT"

# --- Launch distributed training ---
torchrun --nproc-per-node=2 train.py --config-name=config_training_era5mswx_diffusion.yaml \
    ++training.hp.total_batch_size=16 \
    ++training.io.checkpoint_dir=checkpoints_diffusion_era5mswx \
    ++training.io.regression_checkpoint_path=${REGRESSION_CKPT}
