#!/bin/bash

# Master script to submit parallel patch generation and combining jobs
# Usage: ./submit_parallel_patches.sh

cd /beegfs/muduchuru/codes/physicsnemo/examples/weather/corrdiff

echo "=================================================="
echo "SUBMITTING PARALLEL PATCH GENERATION JOBS"
echo "=================================================="

# Create logs directory if it doesn't exist
mkdir -p submit/logs

# Submit the array job for patch generation
echo "Submitting array job for patch generation..."
ARRAY_JOB_ID=$(sbatch --parsable submit/generate_mswxdwd_patches_array.sh)

if [ $? -eq 0 ]; then
    echo "✓ Array job submitted successfully"
    echo "  Job ID: $ARRAY_JOB_ID"
    echo "  Array range: 0-41 (42 patches total)"
    echo "  Max concurrent jobs: 8"
    echo ""
    
    # Submit the combination job that depends on the array job completion
    echo "Submitting combination job (depends on array job completion)..."
    COMBINE_JOB_ID=$(sbatch --parsable --dependency=afterok:$ARRAY_JOB_ID submit/combine_patches.sh)
    
    if [ $? -eq 0 ]; then
        echo "✓ Combination job submitted successfully"
        echo "  Job ID: $COMBINE_JOB_ID"
        echo "  Dependency: afterok:$ARRAY_JOB_ID"
        echo ""
        
        echo "=================================================="
        echo "JOB SUBMISSION SUMMARY"
        echo "=================================================="
        echo "Array Job (patch generation): $ARRAY_JOB_ID"
        echo "Combine Job (patch combining): $COMBINE_JOB_ID"
        echo ""
        echo "Monitor jobs with:"
        echo "  squeue -u \$USER"
        echo "  squeue -j $ARRAY_JOB_ID"
        echo "  squeue -j $COMBINE_JOB_ID"
        echo ""
        echo "Check logs in:"
        echo "  submit/logs/gen_patches_${ARRAY_JOB_ID}_*.out"
        echo "  submit/logs/combine_patches.${COMBINE_JOB_ID}.out"
        echo "=================================================="
        
    else
        echo "✗ ERROR: Failed to submit combination job"
        echo "You can manually submit it later with:"
        echo "  sbatch --dependency=afterok:$ARRAY_JOB_ID submit/combine_patches.sh"
        exit 1
    fi
else
    echo "✗ ERROR: Failed to submit array job"
    exit 1
fi