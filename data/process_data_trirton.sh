#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --mem=150G
#SBATCH --job-name=aggregate_data
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/aggregate_data.out

OUTPUT_DIR="/scratch/work/masooda1/datasets/Multi_Modal_Contrastive"
SPLITS_DIR="/scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data"
echo "Data directory: $OUTPUT_DIR"

VENV_PATH="/scratch/work/masooda1/.conda_envs/Multi_Modal_Contrastive"

echo "Activating conda environment: $VENV_PATH"
module load mamba
source activate "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi

echo "Running aggregation script..."
python /scratch/work/masooda1/Multi_Modal_Contrastive/data/_jump_aggregate.py -d ${OUTPUT_DIR} -o ${OUTPUT_DIR} --is_centered

echo "Running splits creation script..."
python /scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data_splits.py \
    ${OUTPUT_DIR}/centered.filtered.parquet \
    ${SPLITS_DIR}
echo "Done!"
