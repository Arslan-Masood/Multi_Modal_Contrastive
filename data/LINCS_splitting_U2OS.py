#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import logging
from cmapPy.pandasGEXpress.parse import parse
from typing import List, Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_lincs_splits(genomic_df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/val splits for LINCS data with 93/7 ratio.
    
    Args:
        genomic_df: DataFrame containing LINCS data
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df)
    """
    # Get unique SMILES from LINCS
    unique_smiles = genomic_df['Metadata_SMILES'].unique()
    total_smiles = len(unique_smiles)
    
    # Calculate split sizes for LINCS (93/7 split)
    train_size = int(0.93 * total_smiles)
    
    # Randomly shuffle and split LINCS data
    np.random.seed(seed)
    shuffled_smiles = np.random.permutation(unique_smiles)
    
    train_smiles = shuffled_smiles[:train_size]
    val_smiles = shuffled_smiles[train_size:]
    
    # Create DataFrames
    train_df = pd.DataFrame({'SMILES': train_smiles})
    val_df = pd.DataFrame({'SMILES': val_smiles})
    
    # Log split statistics
    logger.info("LINCS Split Statistics:")
    logger.info(f"- Total unique SMILES: {total_smiles}")
    logger.info(f"- Training SMILES: {len(train_smiles)} ({len(train_smiles)/total_smiles:.2%})")
    logger.info(f"- Validation SMILES: {len(val_smiles)} ({len(val_smiles)/total_smiles:.2%})")
    
    return train_df, val_df

def concatenate_with_jump(lincs_splits: Tuple[pd.DataFrame, pd.DataFrame], 
                         jump_data_path: str, 
                         seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Concatenate LINCS splits with JUMP data.
    
    Args:
        lincs_splits: Tuple of (train_df, val_df) for LINCS data
        jump_data_path: Path to JUMP data directory
        seed: Split seed number
        
    Returns:
        Tuple of (combined_train_df, combined_val_df)
    """
    lincs_train_df, lincs_val_df = lincs_splits
    
    # Load JUMP data
    logger.info(f"Loading JUMP data from {jump_data_path}")
    jump_train = pd.read_csv(os.path.join(jump_data_path, f'jump-compound-split-{seed}-train.csv'))
    jump_val = pd.read_csv(os.path.join(jump_data_path, f'jump-compound-split-{seed}-val.csv'))
    
    # Create combined datasets
    combined_train = pd.concat([
        jump_train,
        lincs_train_df
    ]).drop_duplicates(subset=['SMILES'])
    
    combined_val = pd.concat([
        jump_val,
        lincs_val_df
    ]).drop_duplicates(subset=['SMILES'])
    
    # Log concatenation statistics
    logger.info("\nConcatenation Statistics:")
    logger.info(f"JUMP Training SMILES: {len(jump_train)}")
    logger.info(f"JUMP Validation SMILES: {len(jump_val)}")
    logger.info(f"Combined Training SMILES: {len(combined_train)}")
    logger.info(f"Combined Validation SMILES: {len(combined_val)}")
    
    return combined_train, combined_val

def save_splits(splits: Dict[str, pd.DataFrame], output_dir: str, seed: int):
    """
    Save all splits to files.
    
    Args:
        splits: Dictionary containing all splits to save
        output_dir: Directory to save splits
        seed: Split seed number
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    file_paths = {
        'lincs_train': f"LINCS-compound-split-{seed}-train.csv",
        'lincs_val': f"LINCS-compound-split-{seed}-val.csv",
        'combined_train': f"JUMP-LINCS-compound-split-{seed}-train.csv",
        'combined_val': f"JUMP-LINCS-compound-split-{seed}-val.csv"
    }
    
    # Save each split
    for split_name, df in splits.items():
        file_path = os.path.join(output_dir, file_paths[split_name])
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {split_name} to: {file_path}")

def process_splits(genomic_data_path: str, jump_data_path: str, output_dir: str, seed: int):
    """
    Main function to process one split: create LINCS splits, concatenate with JUMP, and save.
    
    Args:
        genomic_data_path: Path to processed LINCS data
        jump_data_path: Path to JUMP data directory
        output_dir: Directory to save splits
        seed: Split seed number
    """
    logger.info(f"\nProcessing split with seed {seed}")
    
    # Load LINCS data
    logger.info(f"Loading LINCS data from {genomic_data_path}")
    genomic_df = pd.read_parquet(genomic_data_path)
    
    # Create LINCS splits
    lincs_train_df, lincs_val_df = create_lincs_splits(genomic_df, seed)
    
    # Concatenate with JUMP data
    combined_train_df, combined_val_df = concatenate_with_jump(
        (lincs_train_df, lincs_val_df),
        jump_data_path,
        seed
    )
    
    # Save all splits
    splits = {
        'lincs_train': lincs_train_df,
        'lincs_val': lincs_val_df,
        'combined_train': combined_train_df,
        'combined_val': combined_val_df
    }
    save_splits(splits, output_dir, seed)

def main():
    # Define paths
    genomic_data_path = "/scratch/cs/pml/AI_drug/molecular_representation_learning/LINCS/landmark_cmp_data_U2OS_single_dose_time.parquet"
    jump_data_path = "/scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data"
    output_dir = "/scratch/work/masooda1/Multi_Modal_Contrastive/data/LINCS"
    
    # Process each split
    for seed in [0, 1, 2]:
        process_splits(genomic_data_path, jump_data_path, output_dir, seed)

if __name__ == "__main__":
    main()