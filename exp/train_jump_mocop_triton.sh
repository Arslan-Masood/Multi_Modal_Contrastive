#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --mem=500G
#SBATCH --cpus-per-task=48
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=/scratch/work/masooda1/trial/temp_%A_%a.out

SAVE_DIR=$1
CONDA_ENV="mocop"
VENV_PATH="/scratch/work/masooda1/.conda_envs/Multi_Modal_Contrastive"

# Calculate SEED and SPLIT based on array task ID (they are the same in this case)
SEED=$SLURM_ARRAY_TASK_ID
SPLIT=$SEED

# Define the new output file name
NEW_OUTPUT_FILE="/scratch/work/masooda1/trained_model_pred/mocop_pretraining/jump-mocop_seed${SEED}_split${SPLIT}_${SLURM_ARRAY_JOB_ID}.out"

# Redirect all following output to the new file
exec 1> "${NEW_OUTPUT_FILE}"
exec 2>&1

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Running job for SEED=${SEED}, SPLIT=${SPLIT}"

echo 'Starting script execution...'
module load mamba
echo "Activating conda environment: ${VENV_PATH}"
source activate "${VENV_PATH}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

# Create a specific directory for this run
RUN_DIR="${SAVE_DIR}/jump_mocop_seed_${SEED}_split_${SPLIT}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

echo 'Running training script...'
srun python bin/train.py -cn jump_mocop.yml \
                    seed=${SEED} \
                    dataloaders.dataset.data_path=/scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data/cell_fetures_with_smiles.parquet \
                    dataloaders.splits.train=/scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data/jump-compound-split-${SPLIT}-train.csv \
                    dataloaders.splits.val=/scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data/jump-compound-split-${SPLIT}-val.csv \
                    dataloaders.splits.test=/scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data/jump-compound-split-${SPLIT}-test.csv \
                    trainer.callbacks.1.dirpath=${CHECKPOINT_DIR} \
                    trainer.logger.project=jump_mocop \
                    trainer.logger.save_dir=${RUN_DIR} \
                    trainer.logger.name=jump_mocop_seed_${SEED}_split_${SPLIT} \
                    trainer.logger.id=jump_mocop_seed_${SEED}_split_${SPLIT} \
                    trainer.logger.mode=offline

echo 'Training completed successfully.'

# Remove the temporary file
rm "/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/jump_mocop/temp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"