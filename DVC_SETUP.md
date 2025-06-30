# Data Version Control (DVC) Setup Guide

This guide explains how to set up Data Version Control (DVC) for the Multi-Modal Molecular Representation Learning project.

## Why DVC for this project?

Your project deals with:
- **Large molecular datasets** (ChEMBL20, LINCS L1000)
- **Cell Painting morphological data** (potentially GBs of image features)
- **Model checkpoints** (hundreds of MBs)
- **Experiment outputs** and metrics
- **Multiple data modalities** that need to be synchronized

DVC provides:
- ✅ **Version control for large files** without bloating Git
- ✅ **Data pipeline management** and reproducibility
- ✅ **Remote storage** for datasets and models
- ✅ **Experiment tracking** and metric comparison
- ✅ **Collaboration** on data and models

## Installation

```bash
# Activate your conda environment
conda activate multi_modal_contrastive

# Install DVC with all backends
pip install dvc[all]

# Or install with specific storage backends
pip install dvc[s3,azure,gcs,ssh]
```

## Step 1: Initialize DVC

```bash
# Initialize DVC in your repository
dvc init

# This creates:
# - .dvc/ directory with DVC configuration
# - .dvcignore file
# - Updates .gitignore
```

## Step 2: Add Data to DVC Tracking

### Track large data directories
```bash
# Add molecular data
dvc add data/chembl20/
dvc add data/LINCS_U2OS/
dvc add data/LINCS_All_cell_lines/
dvc add data/jump/
dvc add data/jump_data/

# Add datasets
dvc add datasets/

# Add model outputs
dvc add models/
dvc add outputs/
dvc add wandb/
```

This creates `.dvc` files that Git tracks instead of the actual data.

## Step 3: Configure Remote Storage

### Option 1: Local/Network Storage
```bash
# Set up local remote storage
dvc remote add -d storage /path/to/shared/storage
dvc remote modify storage url /scratch/shared/dvc_storage
```

### Option 2: Cloud Storage (Recommended)

#### AWS S3
```bash
dvc remote add -d s3remote s3://your-bucket/dvc-storage
dvc remote modify s3remote region us-east-1
```

#### Google Cloud Storage
```bash
dvc remote add -d gcs gs://your-bucket/dvc-storage
```

#### Azure Blob Storage
```bash
dvc remote add -d azure azure://container/path
```

## Step 4: Create DVC Pipeline

Create `dvc.yaml` to define your ML pipeline:

```yaml
stages:
  data_download:
    cmd: bash data/download_data_triton.sh
    deps:
      - data/download_data_triton.sh
    outs:
      - data/raw/

  data_preprocessing:
    cmd: python data/process_LINCS_data.py
    deps:
      - data/process_LINCS_data.py
      - data/raw/
    outs:
      - data/processed/
    
  feature_extraction:
    cmd: python data/_jump_aggregate.py
    deps:
      - data/_jump_aggregate.py
      - data/processed/
    outs:
      - data/features/

  train_molecular_encoder:
    cmd: python bin/train.py -cn configs/multi_modal_config.yml
    deps:
      - bin/train.py
      - mocop/
      - configs/
      - data/features/
    outs:
      - models/molecular_encoder/
    metrics:
      - outputs/metrics.json

  train_multimodal:
    cmd: python bin/train.py -cn configs/multimodal_contrastive.yml
    deps:
      - bin/train.py
      - mocop/
      - configs/
      - data/features/
      - models/molecular_encoder/
    outs:
      - models/multimodal/
    metrics:
      - outputs/multimodal_metrics.json

  evaluate_chembl:
    cmd: python bin/test.py --config configs/chembl_eval.yml
    deps:
      - bin/test.py
      - models/multimodal/
      - data/chembl20/
    metrics:
      - outputs/chembl_results.json
```

## Step 5: Push Data to Remote

```bash
# Push all tracked data to remote storage
dvc push

# Push specific data
dvc push data/chembl20.dvc
```

## Step 6: Working with DVC

### Running the pipeline
```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro train_multimodal

# Force rerun (ignore cache)
dvc repro -f
```

### Pulling data (for collaborators)
```bash
# Pull all data
dvc pull

# Pull specific data
dvc pull data/chembl20.dvc
```

### Checking status
```bash
# Check pipeline status
dvc status

# Check data status
dvc data status
```

## Step 7: Experiment Tracking

DVC can track metrics and parameters:

```python
# In your training script
import json
import dvc.api

# Log metrics
metrics = {
    "train_loss": 0.234,
    "val_accuracy": 0.876,
    "chembl_performance": 0.823
}

with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f)

# Log parameters
params = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "embedding_dim": 512,
    "contrastive_temperature": 0.1
}

with open("params.yaml", "w") as f:
    yaml.dump(params, f)
```

### Compare experiments
```bash
# Show metrics across experiments
dvc metrics show

# Compare with specific commit
dvc metrics diff HEAD~1

# Plot metrics
dvc plots show outputs/metrics.json
```

## Step 8: Collaboration Workflow

### For team members:
```bash
# Clone repository
git clone https://github.com/Arslan-Masood/Multi_Modal_Contrastive.git
cd Multi_Modal_Contrastive

# Pull data
dvc pull

# Make changes and run experiments
dvc repro

# Push new data/models
dvc push

# Commit changes
git add .
git commit -m "Updated model with new hyperparameters"
git push
```

## DVC Commands Cheat Sheet

```bash
# Initialize
dvc init

# Track data
dvc add <file/directory>

# Configure remote
dvc remote add -d <name> <url>

# Push/Pull data
dvc push
dvc pull

# Run pipeline
dvc repro

# Check status
dvc status
dvc data status

# Metrics
dvc metrics show
dvc metrics diff

# Experiments
dvc exp run
dvc exp show
```

## Integration with Git

Your Git repository will contain:
- **Code files** (Python scripts, configs)
- **DVC files** (`.dvc` files pointing to data)
- **Pipeline definition** (`dvc.yaml`)
- **Parameters** (`params.yaml`)
- **Small metadata files**

The actual data lives in DVC remote storage.

## Benefits for Your Project

1. **Reproducibility**: Anyone can reproduce your results with `dvc repro`
2. **Collaboration**: Team members sync data with `dvc pull/push`
3. **Experiment Tracking**: Compare different model configurations
4. **Storage Efficiency**: Large files don't bloat your Git repository
5. **Pipeline Management**: Automatic dependency tracking and execution

## Best Practices

1. **Organize data by stages**: raw → processed → features → results
2. **Use meaningful stage names** in your pipeline
3. **Track important metrics** at each stage
4. **Regular pushes** to remote storage
5. **Document data sources** and preprocessing steps
6. **Use branches** for different experiments
7. **Tag important versions** for paper submissions

This setup will make your multi-modal molecular representation learning project much more manageable and reproducible! 