from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your data
morph_data = pd.read_parquet("/scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/cell_fetures_with_smiles_2000.parquet")
controls = pd.read_csv("/scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data/jump_controls.csv")

# filter postive, negative controls
morph_data = morph_data[~morph_data.Metadata_InChIKey.isin(controls.Metadata_InChIKey_pos_neg_control)].reset_index(drop = True)

# Separate SMILES (if any) and features
# Assuming that morphological features start from column index 7
meta_data = morph_data.iloc[:, :7]   # e.g., SMILES or other non-feature columns
features = morph_data.iloc[:, 7:]    # numerical features to normalize

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the features
normalized_features = scaler.fit_transform(features)

# Convert back to DataFrame if needed
normalized_df = pd.DataFrame(normalized_features, columns=features.columns)

# Optionally, recombine with meta data
normalized_morph_data = pd.concat([meta_data.reset_index(drop=True), normalized_df], axis=1)
normalized_morph_data.to_parquet("/scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/normalized_cell_fetures_with_smiles_2000.parquet")