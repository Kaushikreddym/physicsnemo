#!/bin/bash
#SBATCH --job-name=hyras_to_daily         # Job name
#SBATCH --output=submit/logs/submit/hyras_to_daily.%j.out  # STDOUT file
#SBATCH --error=submit/logs/submit/hyras_to_daily.%j.err   # STDERR file
#SBATCH --time=1-00:00:00                 # Runtime (D-HH:MM:SS)
#SBATCH --ntasks=1                        # Single task
#SBATCH --cpus-per-task=32                # CPU cores per task for parallel processing
#SBATCH --partition=highmem               # High memory partition
#SBATCH --mem=100G                        # Memory allocation (adjust as needed)
##SBATCH --hint=nomultithread             # Optional: for CPU affinity

# --- Load environment ---
source ~/.bashrc
conda activate diffusion

# --- Move to project directory ---
cd /beegfs/muduchuru/codes/physicsnemo/examples/weather/corrdiff || exit 1

# --- Print system info ---
echo "Running on node: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# --- Run the Python script ---
python datasets/hyras_to_daily.py

echo "Job completed at: $(date)"
