#!/bin/bash
# remap_state_dict_mocop_interactive.sh

SAVE_DIR=$1
SEED=$2
CONDA_ENV="mocop"
VENV_PATH="/scratch/work/masooda1/.conda_envs/Multi_Modal_Contrastive"

SPLIT=${SEED}

echo 'Starting script execution...'
module load mamba
echo "Activating conda environment: ${VENV_PATH}"
source activate "${VENV_PATH}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

# Define checkpoint paths
CKPT_PATH=${SAVE_DIR}/jump_mocop_lincs_seed_${SEED}_split_${SPLIT}/checkpoints/best_ckpt.ckpt
OUTPUT_PATH=$(dirname $CKPT_PATH)/best-ckpt-remapped.ckpt

echo "Remapping state dict for SEED=${SEED}, SPLIT=${SPLIT}"
echo "Input checkpoint: ${CKPT_PATH}"
echo "Output path: ${OUTPUT_PATH}"

# Run remapping scripts (without srun since we're already in an interactive session)
echo "Running first remapping..."
python /scratch/work/masooda1/Multi_Modal_Contrastive/bin/remap_state_dict.py -i $CKPT_PATH -o $OUTPUT_PATH --map_from "encoder_a" --map_to "model"

echo "Running second remapping..."
python /scratch/work/masooda1/Multi_Modal_Contrastive/bin/remap_state_dict.py -i $OUTPUT_PATH -o $OUTPUT_PATH --map_from "model.fc_layers.0" --map_to "model.fc_layers.0.0"

echo "Remapping completed. Output saved to: ${OUTPUT_PATH}"
echo 'Script execution completed successfully.'