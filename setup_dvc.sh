#!/bin/bash

# Setup script for Data Version Control (DVC)
echo "Setting up DVC for Multi-Modal Contrastive Learning project..."

# Initialize DVC
dvc init

# Add data directories to DVC tracking
echo "Adding data directories to DVC..."
dvc add data/
dvc add datasets/
dvc add models/

# Add outputs to DVC
echo "Adding output directories to DVC..."
dvc add outputs/
dvc add wandb/

# Create DVC pipeline stages
echo "Creating DVC pipeline..."

# Stage 1: Data preprocessing
dvc stage add -n preprocess_data \
    -d data/ \
    -o data/processed/ \
    python data/preprocess.py

# Stage 2: Training
dvc stage add -n train \
    -d data/processed/ \
    -d configs/ \
    -d mocop/ \
    -o models/trained/ \
    -o outputs/metrics.json \
    python bin/train.py

# Stage 3: Evaluation
dvc stage add -n evaluate \
    -d models/trained/ \
    -d data/processed/ \
    -o outputs/results/ \
    python bin/test.py

echo "DVC setup complete!"
echo "Next steps:"
echo "1. Configure remote storage: dvc remote add -d myremote /path/to/storage"
echo "2. Push data to remote: dvc push"
echo "3. Run pipeline: dvc repro" 