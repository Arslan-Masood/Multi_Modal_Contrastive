import pandas as pd
import numpy as np

# Read the parquet file
df = pd.read_parquet('/scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data/cell_fetures_with_smiles.parquet')

# Get columns that don't start with "Metadata_"
morph_cols = [c for c in df.columns if not c.startswith("Metadata_")]

# Print basic info
print(f"Total columns: {len(df.columns)}")
print(f"Number of morphological features (non-Metadata columns): {len(morph_cols)}")
print(f"Number of Metadata columns: {len(df.columns) - len(morph_cols)}")

# Check for NaN values
total_nan = df.isna().sum().sum()
print(f"\nTotal NaN values in dataset: {total_nan}")

# Get columns with NaN values
cols_with_nan = df.columns[df.isna().any()].tolist()
if cols_with_nan:
    print(f"\nColumns containing NaN values ({len(cols_with_nan)}):")
    for col in cols_with_nan:
        nan_count = df[col].isna().sum()
        print(f"{col}: {nan_count} NaN values")

# Print percentage of NaN values
total_elements = df.size
nan_percentage = (total_nan / total_elements) * 100
print(f"\nPercentage of NaN values: {nan_percentage:.4f}%")
