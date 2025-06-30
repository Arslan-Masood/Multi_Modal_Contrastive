#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --mem=500G
#SBATCH --partition=gpu-a100-80g
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=/scratch/work/masooda1/trial/temp_%A_%a.out

SAVE_DIR=$1
CONDA_ENV="mocop"
VENV_PATH="/scratch/work/masooda1/.conda_envs/Multi_Modal_Contrastive"

# Calculate SEED and SPLIT based on array task ID
SEED=$SLURM_ARRAY_TASK_ID
SPLIT=$SEED

# Define output file
NEW_OUTPUT_FILE="/scratch/work/masooda1/trained_model_pred/mocop_LINCS_pretraining/jump-mocop-lincs_seed${SEED}_split${SPLIT}_${SLURM_ARRAY_JOB_ID}.out"

# Redirect output
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

# Create run directory
RUN_DIR="${SAVE_DIR}/jump_mocop_lincs_seed_${SEED}_split_${SPLIT}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

echo 'Running training script...'
srun python bin/train.py -cn jump_mocop_LINCS.yml \
                    seed=${SEED} \
                    dataloaders.splits.train=/scratch/work/masooda1/Multi_Modal_Contrastive/data/LINCS/JUMP-LINCS-compound-split-${SPLIT}-train.csv \
                    dataloaders.splits.val=/scratch/work/masooda1/Multi_Modal_Contrastive/data/LINCS/JUMP-LINCS-compound-split-${SPLIT}-val.csv \
                    dataloaders.splits.test=/scratch/work/masooda1/Multi_Modal_Contrastive/data/jump/jump-compound-split-${SPLIT}-test.csv \
                    trainer.callbacks.1.dirpath=${CHECKPOINT_DIR} \
                    trainer.logger.project=jump_mocop_lincs \
                    trainer.logger.save_dir=${RUN_DIR} \
                    trainer.logger.name=jump_mocop_lincs_seed_${SEED}_split_${SPLIT} \
                    trainer.logger.id=jump_mocop_lincs_seed_${SEED}_split_${SPLIT} \
                    trainer.logger.mode=offline

echo 'Training completed successfully.'

# Remove temp file
rm "/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/jump_mocop/temp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"