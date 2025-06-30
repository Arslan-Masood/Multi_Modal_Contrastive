#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --partition=gpu-v100-16g,gpu-v100-32g
#SBATCH --array=0-33
#SBATCH --output=/scratch/cs/pml/AI_drug/mocop/dummy/jump-mocop-lincs_all_cell_lines_iteration_2_%a.out

SAVE_DIR=$1
CONDA_ENV="mocop"
VENV_PATH="/scratch/work/masooda1/.conda_envs/Multi_Modal_Contrastive"
CONFIG_PATH="/scratch/work/masooda1/Multi_Modal_Contrastive/configs/jump_mocop_LINCS_all_cell_lines_dummy_data.yml"
TRAIN_SCRIPT="/scratch/work/masooda1/Multi_Modal_Contrastive/bin/train.py"

# Hyperparameter grid
NORM_TYPES=("null" "layernorm" "batchnorm")
USE_HIDDEN_BLOCKS=(False True)
N_HIDDEN_BLOCKS=(1 2 3 4 5)
USE_SKIP_CONNECTIONS=(True False)

# Build all combinations
COMBINATIONS=()
for norm in "${NORM_TYPES[@]}"; do
  for uhb in "${USE_HIDDEN_BLOCKS[@]}"; do
    if [ "$uhb" == "True" ]; then
      for nhb in "${N_HIDDEN_BLOCKS[@]}"; do
        for skip in "${USE_SKIP_CONNECTIONS[@]}"; do
          COMBINATIONS+=("$norm,$uhb,$nhb,$skip")
        done
      done
    else
      COMBINATIONS+=("$norm,$uhb,NA,NA")
    fi
  done
done

TOTAL_COMBINATIONS=${#COMBINATIONS[@]}

# Get this job's combination
IDX=$SLURM_ARRAY_TASK_ID
if [ $IDX -ge $TOTAL_COMBINATIONS ]; then
  echo "SLURM_ARRAY_TASK_ID $IDX exceeds total combinations $TOTAL_COMBINATIONS"
  exit 1
fi

IFS=',' read NORM_TYPE USE_HIDDEN_BLOCK N_HIDDEN_BLOCKS USE_SKIP_CONNECTION <<< "${COMBINATIONS[$IDX]}"

# Calculate SEED and SPLIT based on array task ID
SEED=0
SPLIT=0

# Set your Neptune project name below (format: workspace/project)
NEPTUNE_PROJECT="arslan-masood/Mocop" 


echo "Running job for SEED=${SEED}, SPLIT=${SPLIT}"
echo "Hyperparameters: norm_type=$NORM_TYPE, use_hidden_block=$USE_HIDDEN_BLOCK, n_hidden_blocks=$N_HIDDEN_BLOCKS, use_skip_connection=$USE_SKIP_CONNECTION"

echo 'Starting script execution...'
module load mamba
echo "Activating conda environment: ${VENV_PATH}"
source activate "${VENV_PATH}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

# Create run directory
RUN_DIR="${SAVE_DIR}/jump_mocop_lincs_all_cell_lines_seed_${SEED}_split_${SPLIT}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

# Prepare temp config
TMP_CONFIG="/tmp/hp_config_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yml"
cp $CONFIG_PATH $TMP_CONFIG

# Update encoder_b hyperparameters in YAML using sed
sed -i "s/^\( *norm_type:\).*/\1 \"$NORM_TYPE\"/" $TMP_CONFIG
sed -i "s/^\( *use_hidden_block:\).*/\1 $USE_HIDDEN_BLOCK/" $TMP_CONFIG

if [ "$USE_HIDDEN_BLOCK" == "True" ]; then
  sed -i "s/^\( *n_hidden_blocks:\).*/\1 $N_HIDDEN_BLOCKS/" $TMP_CONFIG
  sed -i "s/^\( *use_skip_connection:\).*/\1 $USE_SKIP_CONNECTION/" $TMP_CONFIG
else
  sed -i "s/^\( *n_hidden_blocks:\).*/\1 null/" $TMP_CONFIG
  sed -i "s/^\( *use_skip_connection:\).*/\1 null/" $TMP_CONFIG
fi

# Compose logger name
LOGGER_NAME="norm_${NORM_TYPE}_hiddenblock_${USE_HIDDEN_BLOCK}"
if [ "$USE_HIDDEN_BLOCK" == "True" ]; then
  LOGGER_NAME="${LOGGER_NAME}_nblocks_${N_HIDDEN_BLOCKS}_skip_${USE_SKIP_CONNECTION}"
else
  LOGGER_NAME="${LOGGER_NAME}_nblocks_None_skip_None"
fi

# Update logger name in config
sed -i "/^[ ]*logger:/,/^[^ ]/s/^\([ ]*name:\).*/\1 $LOGGER_NAME/" $TMP_CONFIG

echo "Logger name: $LOGGER_NAME"

echo 'Running training script...'
srun python $TRAIN_SCRIPT --config-path /tmp --config-name $(basename $TMP_CONFIG) \
    seed=${SEED} \
    dataloaders.splits.train=/scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/jump-compound-split-${SPLIT}-train.csv \
    dataloaders.splits.val=/scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/jump-compound-split-${SPLIT}-val.csv \
    dataloaders.splits.test=/scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/jump-compound-split-${SPLIT}-test.csv \
    trainer.callbacks.1.dirpath=${CHECKPOINT_DIR} \
    trainer.logger.project=${NEPTUNE_PROJECT}

# Clean up
rm $TMP_CONFIG

echo 'Training completed successfully.'

