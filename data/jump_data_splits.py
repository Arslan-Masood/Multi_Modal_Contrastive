from rdkit import Chem
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse

def inchi2smiles(inchi_str: str):
    try:
        mol = Chem.inchi.MolFromInchi(inchi_str, sanitize=True, removeHs=True)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except:
        return None

def count_atoms(smiles: str) -> int:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return mol.GetNumAtoms()
    except:
        return 0

def convert_inchi_to_smiles(df):
    # Convert InChI to SMILES and create new column
    print("Converting InChI to SMILES...")
    df['Metadata_SMILES'] = df['Metadata_InChI'].apply(inchi2smiles)
    
    # Count successful conversions
    failed = df['Metadata_SMILES'].isna().sum()
    successful = df['Metadata_SMILES'].notna().sum()
    total = len(df)
    
    print(f"Conversion statistics:")
    print(f"Total InChIs: {total}")
    print(f"Successful conversions: {successful} ({successful/total*100:.2f}%)")
    print(f"Failed conversions: {failed} ({failed/total*100:.2f}%)")
    
    # Remove rows with failed conversions
    df_clean = df.dropna(subset=['Metadata_SMILES'])
    print(f"\nShape after removing failed conversions: {df_clean.shape}")
    
    # Filter out molecules with more than 250 atoms
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean['Metadata_SMILES'].apply(count_atoms) <= 250]
    removed_count = initial_count - len(df_clean)
    print(f"Removed {removed_count} molecules with more than 250 atoms.")
    print(f"\nShape after removing large molecules: {df_clean.shape}")
    
    # Reorder columns to put Metadata_SMILES first
    cols = df_clean.columns.tolist()
    cols.remove('Metadata_SMILES')
    new_cols = ['Metadata_SMILES'] + cols
    df_clean = df_clean[new_cols]
    
    return df_clean

def create_splits(df, seed=42):
    """
    Create train (90%), validation (5%), and test (5%) splits
    """
    # First split: 90% train, 10% temp
    train_df, temp_df = train_test_split(df, test_size=0.1, random_state=seed)
    
    # Second split: Split temp into two equal parts (5% each for val and test)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)
    
    print(f"Split sizes:")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df, output_dir, split_num):
    """
    Save the splits to CSV files with only SMILES column
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and rename SMILES column for each split
    train_smiles = train_df[['Metadata_SMILES']].rename(columns={'Metadata_SMILES': 'SMILES'})
    val_smiles = val_df[['Metadata_SMILES']].rename(columns={'Metadata_SMILES': 'SMILES'})
    test_smiles = test_df[['Metadata_SMILES']].rename(columns={'Metadata_SMILES': 'SMILES'})
    
    # Save splits
    train_path = os.path.join(output_dir, f'jump-compound-split-{split_num}-train.csv')
    val_path = os.path.join(output_dir, f'jump-compound-split-{split_num}-val.csv')
    test_path = os.path.join(output_dir, f'jump-compound-split-{split_num}-test.csv')
    
    train_smiles.to_csv(train_path, index=False)
    val_smiles.to_csv(val_path, index=False)
    test_smiles.to_csv(test_path, index=False)
    
    print(f"\nSaved splits to:")
    print(f"Train: {train_path}")
    print(f"Val:   {val_path}")
    print(f"Test:  {test_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert InChI to SMILES and create train/val/test splits')
    parser.add_argument('input', help='Input parquet file path')
    parser.add_argument('output', help='Output directory for splits')
    args = parser.parse_args()
    
    # Read the parquet file
    print(f"Reading parquet file: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Initial shape: {df.shape}")
    
    # Convert InChI to SMILES and remove failed conversions
    df_clean = convert_inchi_to_smiles(df)
    
    # Save the cleaned dataframe with all data
    clean_file = os.path.join(args.output, 'cell_fetures_with_smiles.parquet')
    df_clean.to_parquet(clean_file)
    print(f"\nSaved cleaned dataframe with SMILES to: {clean_file}")
    
    # Get unique SMILES
    unique_smiles_df = df_clean[['Metadata_SMILES']].drop_duplicates()
    print(f"\nNumber of unique SMILES: {len(unique_smiles_df)}")
    
    # Create splits for 3 different random seeds using unique SMILES
    for split_num in range(3):
        print(f"\nCreating split {split_num}")
        train_df, val_df, test_df = create_splits(unique_smiles_df, seed=split_num)
        save_splits(train_df, val_df, test_df, args.output, split_num)

if __name__ == "__main__":
    main()
