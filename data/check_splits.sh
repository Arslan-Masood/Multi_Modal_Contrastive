#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --mem=60G
#SBATCH --job-name=check_splits
#SBATCH --output=/scratch/work/masooda1/trial/check_splits_jump_lincs.out

VENV_PATH="/scratch/work/masooda1/.conda_envs/Multi_Modal_Contrastive"

echo "Activating conda environment: $VENV_PATH"
module load mamba
source activate "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi

echo "Running check_splits script..."
python /scratch/work/masooda1/Multi_Modal_Contrastive/mocop/check_splits.py

echo "Done!"
