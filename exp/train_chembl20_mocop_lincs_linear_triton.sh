#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-17     # Modified: 6 fractions × 1 seed × 3 splits = 18 jobs
#SBATCH --output=/scratch/cs/pml/AI_drug/script_output/temp_%A_%a.out

SAVE_DIR=$1
CONDA_ENV="mocop"
VENV_PATH="/scratch/work/masooda1/.conda_envs/Multi_Modal_Contrastive"

# Define arrays
FRAC_ARRAY=(1 5 10 25 50 100)
SEED=2                    # Fixed seed where we have pretrained model
SPLIT_ARRAY=(1 2 3)

# Calculate indices based on SLURM_ARRAY_TASK_ID
FRAC_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SPLIT_INDEX=$((SLURM_ARRAY_TASK_ID % 3))

# Get the actual values
FRAC=${FRAC_ARRAY[$FRAC_INDEX]}
SPLIT=${SPLIT_ARRAY[$SPLIT_INDEX]}

# pretrained checkpoint path
pretrained_ckpt_path="/scratch/cs/pml/AI_drug/trained_model_pred/mocop_LINCS_pretraining/jump_mocop_lincs_seed_${SEED}_split_${SEED}/checkpoints/best-ckpt-remapped.ckpt"

# Check if pretrained checkpoint exists
if [ ! -f "$pretrained_ckpt_path" ]; then
    echo "Error: Pretrained checkpoint not found at ${pretrained_ckpt_path}"
    exit 1
fi

# Define the new output file name
NEW_OUTPUT_FILE="/scratch/cs/pml/AI_drug/trained_model_pred/mocop/chembl_20_pretrained_mocop_lincs_linear_prob/script_output/chembl20-mocop-lincs-linear-frac${FRAC}_seed${SEED}_split${SPLIT}_${SLURM_ARRAY_JOB_ID}.out"

# Redirect all following output to the new file
exec 1> "${NEW_OUTPUT_FILE}"
exec 2>&1

module load mamba
echo "Activating conda environment: ${VENV_PATH}"
source activate "${VENV_PATH}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

# Define test results path once at the top
TEST_RESULTS_DIR="${SAVE_DIR}/test_results/chembl20_mocop_lincs_linear"
TEST_RESULTS_FILENAME="frac${FRAC}_split${SPLIT}_seed${SEED}"
TEST_RESULTS_FILE="${TEST_RESULTS_DIR}/${TEST_RESULTS_FILENAME}.json"

# Check if test results already exist
if [ -f "${TEST_RESULTS_FILE}" ]; then
    echo "Test results file already exists at: ${TEST_RESULTS_FILE}"
    echo "Skipping this run as it was already completed successfully."
    exit 0
fi

# Add these before the training command
RUN_DIR="${SAVE_DIR}/chembl20_mocop_lincs_linear_frac${FRAC}_split${SPLIT}_seed${SEED}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

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
                    trainer.logger.project=chembl_20_pretrained_mocop_lincs_linear_prob_training \
                    trainer.logger.name=frac${FRAC}_split${SPLIT}_seed${SEED} \
                    trainer.logger.id=frac${FRAC}_split${SPLIT}_seed${SEED}

echo 'Training completed. Running test script...'

# Assuming the best checkpoint is saved as best_ckpt.ckpt
BEST_CKPT="${CHECKPOINT_DIR}/best_ckpt.ckpt"
mkdir -p "${TEST_RESULTS_DIR}"

srun python /scratch/work/masooda1/Multi_Modal_Contrastive/bin/test.py -cn chembl20_mocop_linear.yml \
                    seed=${SEED} \
                    dataloaders.splits.train=/scratch/work/masooda1/Multi_Modal_Contrastive/data/chembl20/chembl20-frac${FRAC}-split${SPLIT}-train.csv \
                    dataloaders.splits.val=/scratch/work/masooda1/Multi_Modal_Contrastive/data/chembl20/chembl20-split${SPLIT}-val.csv \
                    dataloaders.splits.test=/scratch/work/masooda1/Multi_Modal_Contrastive/data/chembl20/chembl20-split${SPLIT}-test.csv \
                    dataloaders.num_workers=4 \
                    trainer.logger.save_dir=${RUN_DIR} \
                    trainer.logger.project=chembl_20_pretrained_mocop_lincs_linear_prob_testing \
                    trainer.logger.name=frac${FRAC}_split${SPLIT}_seed${SEED} \
                    trainer.logger.id=frac${FRAC}_split${SPLIT}_seed${SEED} \
                    test_model.checkpoint_path=${BEST_CKPT} \
                    test_results_dir=${TEST_RESULTS_DIR} \
                    test_results_filename=${TEST_RESULTS_FILENAME}

echo 'Script execution completed successfully.'
# Remove the temporary file
rm "/scratch/cs/pml/AI_drug/script_output/temp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"