#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH --job-name=LINCS_proc
#SBATCH --output=/scratch/work/masooda1/trials/LINCS_proc_%j.out

DATA_DIR="/scratch/cs/pml/AI_drug/molecular_representation_learning/LINCS/"
VENV_PATH="/scratch/work/masooda1/.conda_envs/cmappy_env"

echo "Data directory: $DATA_DIR"

echo "Activating conda environment: $VENV_PATH"
module load mamba
source activate "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi

echo "Running LINCS processing script..."
python /scratch/work/masooda1/Multi_Modal_Contrastive/data/process_LINCS_data.py

echo "Done!"