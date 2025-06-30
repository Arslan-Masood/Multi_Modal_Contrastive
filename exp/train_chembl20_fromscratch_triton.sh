#!/bin/bash
#SBATCH --time=120:0:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --array=0-53
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/chembl20_fromscratch/temp_%A_%a.out

SAVE_DIR=$1
CONDA_ENV="mocop"
VENV_PATH="/scratch/work/masooda1/.conda_envs/Multi_Modal_Contrastive"

# Set the Weights & Biases API key
export WANDB_API_KEY="27edf9c66b032c03f72d30e923276b93aa736429"

# Define arrays for FRAC, SEED, and SPLIT
FRAC_ARRAY=(1 5 10 25 50 100)
SEED_ARRAY=(0 1 2)
SPLIT_ARRAY=(1 2 3)

# Calculate indices for FRAC, SEED, and SPLIT based on SLURM_ARRAY_TASK_ID
FRAC_INDEX=$((SLURM_ARRAY_TASK_ID / 9))
SEED_INDEX=$(((SLURM_ARRAY_TASK_ID % 9) / 3))
SPLIT_INDEX=$((SLURM_ARRAY_TASK_ID % 3))

# Get the actual values
FRAC=${FRAC_ARRAY[$FRAC_INDEX]}
SEED=${SEED_ARRAY[$SEED_INDEX]}
SPLIT=${SPLIT_ARRAY[$SPLIT_INDEX]}

# Define the new output file name
NEW_OUTPUT_FILE="/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/chembl20-fromscratch_frac${FRAC}_seed${SEED}_split${SPLIT}_${SLURM_ARRAY_JOB_ID}.out"

# Directory where test results should be
TEST_RESULTS_DIR="${SAVE_DIR}/test_results/chembl_20_from_scratch"
TEST_RESULT_FILE="${TEST_RESULTS_DIR}/frac${FRAC}_split${SPLIT}_seed${SEED}.json"

# Check if test result exists
if [ -f "$TEST_RESULT_FILE" ]; then
    echo "Test result already exists for FRAC=${FRAC}, SPLIT=${SPLIT}, SEED=${SEED}. Skipping."
    exit 0
fi

# Redirect all following output to the new file
exec 1> "${NEW_OUTPUT_FILE}"
exec 2>&1

echo "Running job for FRAC=${FRAC}, SEED=${SEED}, SPLIT=${SPLIT}"
module load mamba
echo "Activating conda environment: ${VENV_PATH}"
source activate "${VENV_PATH}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

# Create a specific directory for this run
RUN_DIR="${SAVE_DIR}/chembl20_fromscratch_frac${FRAC}_split${SPLIT}_seed${SEED}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

echo 'Running training script...'
srun python /scratch/work/masooda1/Multi_Modal_Contrastive/bin/train.py -cn chembl20_fromscratch.yml \
                    seed=${SEED} \
                    dataloaders.splits.train=/scratch/work/masooda1/Multi_Modal_Contrastive/data/chembl20/chembl20-frac${FRAC}-split${SPLIT}-train.csv \
                    dataloaders.splits.val=/scratch/work/masooda1/Multi_Modal_Contrastive/data/chembl20/chembl20-split${SPLIT}-val.csv \
                    dataloaders.splits.test=/scratch/work/masooda1/Multi_Modal_Contrastive/data/chembl20/chembl20-split${SPLIT}-test.csv \
                    dataloaders.num_workers=12 \
                    trainer.callbacks.1.dirpath=${CHECKPOINT_DIR} \
                    trainer.logger.save_dir=${RUN_DIR} \
                    trainer.logger.project=chembl20_fromscratch \
                    trainer.logger.name=frac${FRAC}_split${SPLIT}_seed${SEED} \
                    trainer.logger.id=frac${FRAC}_split${SPLIT}_seed${SEED}
echo 'Training completed. Running test script...'

# Assuming the best checkpoint is saved as best_ckpt.ckpt
BEST_CKPT="${CHECKPOINT_DIR}/best_ckpt.ckpt"
TEST_RESULTS_FILENAME="frac${FRAC}_split${SPLIT}_seed${SEED}"

mkdir -p "${TEST_RESULTS_DIR}"
srun python /scratch/work/masooda1/Multi_Modal_Contrastive/bin/test.py -cn chembl20_fromscratch.yml \
                    seed=${SEED} \
                    dataloaders.splits.train=/scratch/work/masooda1/Multi_Modal_Contrastive/data/chembl20/chembl20-frac${FRAC}-split${SPLIT}-train.csv \
                    dataloaders.splits.val=/scratch/work/masooda1/Multi_Modal_Contrastive/data/chembl20/chembl20-split${SPLIT}-val.csv \
                    dataloaders.splits.test=/scratch/work/masooda1/Multi_Modal_Contrastive/data/chembl20/chembl20-split${SPLIT}-test.csv \
                    dataloaders.num_workers=12 \
                    trainer.logger.save_dir=${RUN_DIR} \
                    trainer.logger.name=chembl20_fromscratch_test \
                    trainer.logger.id=frac${FRAC}_split${SPLIT}_seed${SEED} \
                    test_model.checkpoint_path=${BEST_CKPT} \
                    test_results_dir=${TEST_RESULTS_DIR} \
                    test_results_filename=${TEST_RESULTS_FILENAME}
echo 'Script execution completed successfully.'

# Remove the temporary file
rm "/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/temp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
