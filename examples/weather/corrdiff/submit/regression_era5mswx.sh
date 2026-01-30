#!/bin/bash
#SBATCH --job-name=regression_era5mswx             # Job name
#SBATCH --output=submit/logs/regression_era5mswx.%j.out  # STDOUT file
#SBATCH --error=submit/logs/regression_era5mswx.%j.err   # STDERR file
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
mkdir -p checkpoints_regression_era5mswx

# --- Launch distributed training ---
torchrun --nproc-per-node=2 train.py --config-name=config_training_era5mswx_regression.yaml \
    ++training.hp.total_batch_size=16 \
    ++training.io.checkpoint_dir=checkpoints_regression_era5mswx
