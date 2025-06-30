#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-debug
#SBATCH --output=/scratch/work/masooda1/trained_model_pred/trial/chembl20-mocop-linear.out

SAVE_DIR=$1
CONDA_ENV="mocop"
VENV_PATH="/scratch/work/masooda1/.conda_envs/Multi_Modal_Contrastive"

FRAC=1
SEED=0
SPLIT=1

# Set the Weights & Biases API key
export WANDB_API_KEY="27edf9c66b032c03f72d30e923276b93aa736429"

echo 'Starting script execution...'
module load mamba
echo "Activating conda environment: ${VENV_PATH}"
source activate "${VENV_PATH}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

# Create a specific directory for this run
RUN_DIR="${SAVE_DIR}/chembl20_mocop_linear_frac${FRAC}_split${SPLIT}_seed${SEED}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

# Use the pretrained checkpoint for the chosen SEED
pretrained_ckpt_path="/scratch/work/masooda1/Multi_Modal_Contrastive/models/jump_mocop_seed_${SEED}_split_${SEED}/version_0/checkpoints/best-ckpt-remapped.ckpt"

echo "Using checkpoint: ${pretrained_ckpt_path}"
echo "Running for SPLIT=${SPLIT}"

echo 'Running training script...'
srun python /scratch/work/masooda1/Multi_Modal_Contrastive/bin/train.py -cn chembl20_mocop_linear.yml \
                    seed=${SEED} \
                    model._args_.0=${pretrained_ckpt_path} \
                    ++model.freeze=true \
                    dataloaders.splits.train=/scratch/work/masooda1/Multi_Modal_Contrastive/data/chembl20/chembl20-frac${FRAC}-split${SPLIT}-train.csv \
                    dataloaders.splits.val=/scratch/work/masooda1/Multi_Modal_Contrastive/data/chembl20/chembl20-split${SPLIT}-val.csv \
                    dataloaders.splits.test=/scratch/work/masooda1/Multi_Modal_Contrastive/data/chembl20/chembl20-split${SPLIT}-test.csv \
                    dataloaders.num_workers=4 \
                    trainer.callbacks.1.dirpath=${CHECKPOINT_DIR} \
                    trainer.logger.save_dir=${RUN_DIR} \
                    trainer.logger.project=chembl20_mocop_linear \
                    trainer.logger.name=frac${FRAC}_split${SPLIT}_seed${SEED} \
                    trainer.logger.id=frac${FRAC}_split${SPLIT}_seed${SEED}
echo 'Training completed. Running test script...'

# Assuming the best checkpoint is saved as best_ckpt.ckpt
BEST_CKPT="${CHECKPOINT_DIR}/best_ckpt.ckpt"
TEST_RESULTS_DIR="${SAVE_DIR}/test_results/chembl20_mocop_linear"
TEST_RESULTS_FILENAME="frac${FRAC}_split${SPLIT}_seed${SEED}"

mkdir -p "${TEST_RESULTS_DIR}"
srun python /scratch/work/masooda1/Multi_Modal_Contrastive/bin/test.py -cn chembl20_mocop_linear.yml \
                    seed=${SEED} \
                    dataloaders.splits.train=/scratch/work/masooda1/Multi_Modal_Contrastive/data/chembl20/chembl20-frac${FRAC}-split${SPLIT}-train.csv \
                    dataloaders.splits.val=/scratch/work/masooda1/Multi_Modal_Contrastive/data/chembl20/chembl20-split${SPLIT}-val.csv \
                    dataloaders.splits.test=/scratch/work/masooda1/Multi_Modal_Contrastive/data/chembl20/chembl20-split${SPLIT}-test.csv \
                    dataloaders.num_workers=4 \
                    trainer.logger.save_dir=${RUN_DIR} \
                    trainer.logger.project=chembl20_mocop_linear_test \
                    trainer.logger.name=frac${FRAC}_split${SPLIT}_seed${SEED} \
                    trainer.logger.id=frac${FRAC}_split${SPLIT}_seed${SEED} \
                    test_model.checkpoint_path=${BEST_CKPT} \
                    test_results_dir=${TEST_RESULTS_DIR} \
                    test_results_filename=${TEST_RESULTS_FILENAME}

echo 'Script execution completed successfully.'