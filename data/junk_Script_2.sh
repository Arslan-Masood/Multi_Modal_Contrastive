#!/bin/bash
#SBATCH --job-name=measure_parquet
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/measure_parquet_%j.out
#SBATCH --time=01:00:00
#SBATCH --mem=150G

# Load necessary modules
module load mamba

# Activate your conda environment
source activate /scratch/work/masooda1/.conda_envs/Multi_Modal_Contrastive

# Run the Python script
python /scratch/work/masooda1/Multi_Modal_Contrastive/data/measure_parquet_read.py

# Print Slurm job statistics
echo "Slurm Job Statistics:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed