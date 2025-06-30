import sys 
sys.path.insert(1, '/scratch/work/masooda1/Multi_Modal_Contrastive/mocop')

from typing import Dict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from rdkit import Chem

from featurizer.smiles_transformation import (inchi2smiles, smiles2fp,
                                              smiles2graph)

class SupervisedGraphDataset(Dataset):
    def __init__(
        self, data_path, cmpd_col="smiles", cmpd_col_is_inchikey=False, pad_length=0
    ):
        if "parquet" in data_path:
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)

        self.df = self.df.set_index(cmpd_col)
        if cmpd_col_is_inchikey:
            self.df.index = [inchi2smiles(s) for s in self.df.index]
        self.df = self.df[[c for c in self.df.columns if not c.startswith("Metadata")]]
        self.unique_smiles = self.df.index
        self.pad_length = pad_length

    def __len__(self):
        return len(self.unique_smiles)

    def _pad(self, adj_mat, node_feat, atom_vec):
        p = self.pad_length - len(atom_vec)
        if p >= 0:
            adj_mat = F.pad(adj_mat, (0, p, 0, p), "constant", 0)
            node_feat = F.pad(node_feat, (0, 0, 0, p), "constant", 0)
            atom_vec = F.pad(atom_vec, (0, 0, 0, p), "constant", 0)
        return adj_mat, node_feat, atom_vec

    def __getitem__(self, index):
        smiles = self.unique_smiles[index]
        adj_mat, node_feat = smiles2graph(smiles)
        adj_mat = torch.FloatTensor(adj_mat)
        node_feat = torch.FloatTensor(node_feat)
        atom_vec = torch.ones(len(node_feat), 1)
        cmpd_feat = self._pad(adj_mat, node_feat, atom_vec)

        labels = self.df.loc[smiles]

        if len(labels.shape) > 1 and len(labels) > 1:
            labels = labels.sample(1).iloc[0]

        labels = torch.FloatTensor(labels.values)
        return {
            "inputs": {"x_a": [torch.FloatTensor(f) for f in cmpd_feat]},
            "labels": labels,
        }


class SupervisedGraphDatasetJUMP(SupervisedGraphDataset):
    def __init__(self, *args, **kwargs):
        super(SupervisedGraphDataset, self).__init__(*args, **kwargs)
        self.unique_smiles = self.df.index.unique()


class DualInputDatasetJUMP(Dataset):
    def __init__(self, data_path):
        if "parquet" in data_path:
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)

        self.smiles_col = "Metadata_SMILES"
        if self.smiles_col not in self.df.columns:
            self.df[self.smiles_col] = [
                inchi2smiles(s) if s is not None else None
                for s in self.df["Metadata_InChI"]
            ]

        self.unique_smiles = [
            s for s in self.df[self.smiles_col].unique() if s is not None
        ]

        self.morph_col = [c for c in self.df.columns if not c.startswith("Metadata_")]
        self.smiles2mask = {}

    def _create_index(self):
        smiles = self.df[self.smiles_col].values
        return {s: np.argwhere(smiles == s).reshape(-1) for s in smiles}

    def __len__(self):
        return len(self.unique_smiles)

    def __getitem__(self, index):
        smiles = self.unique_smiles[index]
        cmpd_feat = smiles2fp(smiles)

        df = self.df[self.df[self.smiles_col] == smiles]
        morph_feat = df.sample(1)[self.morph_col].values.astype(float).flatten()

        labels = torch.Tensor([-1])
        return {
            "inputs": {
                "x_a": torch.FloatTensor(cmpd_feat),
                "x_b": torch.FloatTensor(morph_feat),
            },
            "labels": labels,
        }


class DualInputGraphDatasetJUMP(DualInputDatasetJUMP):
    def __init__(self, pad_length, *args, **kwargs):
        super(DualInputGraphDatasetJUMP, self).__init__(*args, **kwargs)
        self.pad_length = pad_length

    def _pad(self, adj_mat, node_feat, atom_vec):
        p = self.pad_length - len(atom_vec)
        if p >= 0:
            adj_mat = F.pad(adj_mat, (0, p, 0, p), "constant", 0)
            node_feat = F.pad(node_feat, (0, 0, 0, p), "constant", 0)
            atom_vec = F.pad(atom_vec, (0, 0, 0, p), "constant", 0)
        return adj_mat, node_feat, atom_vec

    def __getitem__(self, index):
        smiles = self.unique_smiles[index]
        adj_mat, node_feat = smiles2graph(smiles)
        adj_mat = torch.FloatTensor(adj_mat)
        node_feat = torch.FloatTensor(node_feat)
        atom_vec = torch.ones(len(node_feat), 1)
        cmpd_feat = self._pad(adj_mat, node_feat, atom_vec)

        try:
            mask = self.smiles2mask[smiles]
        except KeyError:
            mask = self.df[self.smiles_col] == smiles
            self.smiles2mask[smiles] = mask
        df = self.df[mask]
        morph_feat = df.sample(1)[self.morph_col].values.astype(float).flatten()
        labels = torch.Tensor([-1])
        return {
            "inputs": {
                "x_a": [torch.FloatTensor(f) for f in cmpd_feat],
                "x_b": torch.FloatTensor(morph_feat),
            },
            "labels": labels,
        }


class TripleInputGraphDatasetJUMP(DualInputGraphDatasetJUMP):
    def __init__(self, data_path, genomic_data_path, pad_length, *args, **kwargs):
        # Define SMILES column name first
        self.smiles_col = "Metadata_SMILES"
        
        # Load cell data
        if "parquet" in data_path:
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)

        # Load genomic data
        self.genomic_df = pd.read_parquet(genomic_data_path)

        # Ensure SMILES column exists in both datasets
        for df in [self.df, self.genomic_df]:
            if self.smiles_col not in df.columns:
                df[self.smiles_col] = [
                    inchi2smiles(s) if s is not None else None
                    for s in df["Metadata_InChI"]
                ]
        
        # Get unique SMILES from each dataset (excluding None)
        cell_smiles = set([s for s in self.df[self.smiles_col].unique() if s is not None])
        genomic_smiles = set([s for s in self.genomic_df[self.smiles_col].unique() if s is not None])
        
        # Print initial statistics
        print("\nSMILES Statistics:")
        print(f"Cell data unique SMILES: {len(cell_smiles)}")
        print(f"Genomic data unique SMILES: {len(genomic_smiles)}")
        print(f"Common SMILES (intersection): {len(cell_smiles.intersection(genomic_smiles))}")
        
        # Combine all unique SMILES (excluding None)
        self.unique_smiles = list(cell_smiles.union(genomic_smiles))
        
        # Print final dataset composition
        print(f"\nFinal Dataset:")
        print(f"Total unique SMILES: {len(self.unique_smiles)}")
        
        # Set remaining attributes
        self.pad_length = pad_length
        self.genomic_cols = [c for c in self.genomic_df.columns if not c.startswith("Metadata_")]
        self.morph_cols = [c for c in self.df.columns if not c.startswith("Metadata_")]

    def _pad(self, adj_mat, node_feat, atom_vec):
        p = self.pad_length - len(atom_vec)
        if p >= 0:
            adj_mat = F.pad(adj_mat, (0, p, 0, p), "constant", 0)
            node_feat = F.pad(node_feat, (0, 0, 0, p), "constant", 0)
            atom_vec = F.pad(atom_vec, (0, 0, 0, p), "constant", 0)
        return adj_mat, node_feat, atom_vec

    def __len__(self):
        return len(self.unique_smiles)

    def __getitem__(self, index):
        smiles = self.unique_smiles[index]
        # Get graph features (x_a)
        adj_mat, node_feat = smiles2graph(smiles)
        adj_mat = torch.FloatTensor(adj_mat)
        node_feat = torch.FloatTensor(node_feat)
        atom_vec = torch.ones(len(node_feat), 1)
        cmpd_feat = self._pad(adj_mat, node_feat, atom_vec)
        
        # Get morphological features (x_b)
        morph_mask = self.df[self.smiles_col] == smiles
        if morph_mask.any():
            morph_feat = self.df[morph_mask].sample(1)[self.morph_cols].values.astype(float).flatten()
        else:
            morph_feat = -1 * np.ones(len(self.morph_cols))
            
        # Get genomic features (x_c)
        genomic_mask = self.genomic_df[self.smiles_col] == smiles
        if genomic_mask.any():
            genomic_feat = self.genomic_df[genomic_mask].sample(1)[self.genomic_cols].values.astype(float).flatten()
        else:
            genomic_feat = -1 * np.ones(len(self.genomic_cols))
        
        return {
            "inputs": {
                "x_a": [torch.FloatTensor(f) for f in cmpd_feat],
                "x_b": torch.FloatTensor(morph_feat),
                "x_c": torch.FloatTensor(genomic_feat)
            },
            "labels": torch.Tensor([-1])
        }


class CellLineTripleInputGraphDatasetJUMP(DualInputGraphDatasetJUMP):
    """Dataset class for handling molecular data with cell line-specific genomic features.
    
    This class processes three types of data:
    1. Molecular graph features (x_a): Structural information about compounds
    2. Morphological features (x_b): Cell morphology measurements
    3. Genomic features (x_c): Gene expression data for different cell lines
    
    Each compound (SMILES) can have:
    - Morphological data only
    - Genomic data only
    - Both morphological and genomic data
    - Genomic data for multiple cell lines
    """

    def __init__(self, data_path, genomic_data_path, pad_length, *args, **kwargs):
        """Initialize the dataset.
        
        Args:
            data_path (str): Path to morphological data file (CSV/Parquet)
            genomic_data_path (str): Path to genomic data file (Parquet)
            pad_length (int): Maximum length for padding molecular graphs
        """
        # Define column names for data identification
        self.smiles_col = "Metadata_SMILES"
        self.cell_line_col = "Metadata_cell_iname"
        
        # Load data files
        if "parquet" in data_path:
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)
        self.genomic_df = pd.read_parquet(genomic_data_path)
        
        # Convert InChI to SMILES if needed
        for df in [self.df, self.genomic_df]:
            if self.smiles_col not in df.columns:
                df[self.smiles_col] = [
                    inchi2smiles(s) if s is not None else None
                    for s in df["Metadata_InChI"]
                ]
        
        # Get unique SMILES from each dataset (excluding None)
        cell_smiles = set([s for s in self.df[self.smiles_col].unique() if s is not None])
        genomic_smiles = set([s for s in self.genomic_df[self.smiles_col].unique() if s is not None])
        
        # Create ordered list: genomic SMILES first, then morphological-only SMILES
        self.unique_smiles = (
            list(genomic_smiles) +  # First: all SMILES with genomic data
            list(cell_smiles - genomic_smiles)  # Then: SMILES with only morphological data
        )
        
        # Print dataset composition statistics
        print("\nDataset Statistics:")
        print(f"Cell data unique SMILES: {len(cell_smiles)}")
        print(f"Genomic data unique SMILES: {len(genomic_smiles)}")
        print(f"Common SMILES (intersection): {len(cell_smiles.intersection(genomic_smiles))}")
        print(f"Morphological-only SMILES: {len(cell_smiles - genomic_smiles)}")
        print(f"Total unique SMILES: {len(self.unique_smiles)}")
        print(f"\nSMILES order guarantee:")
        print(f"- First {len(genomic_smiles)} SMILES have genomic data")
        print(f"- Last {len(cell_smiles - genomic_smiles)} SMILES have only morphological data")
        
        # Modify cell line indexing to start from 1 (0 will be padding)
        self.unique_cell_lines = sorted(list(self.genomic_df[self.cell_line_col].unique()))
        self.cell_line_to_idx = {cell: idx + 1 for idx, cell in enumerate(self.unique_cell_lines)}
        print(f"Number of unique cell lines: {len(self.unique_cell_lines)}")

         # Create mappings for dose levels and time points (1-based indexing, 0 for padding)
        self.unique_doses = sorted(list(self.genomic_df['Metadata_Dose_Level'].unique()))
        self.dose_to_idx = {dose: idx + 1 for idx, dose in enumerate(self.unique_doses)}
        print(f"Dose levels: {self.unique_doses}")
        print(f"Dose mapping: {self.dose_to_idx}")  # e.g., {2:1, 5:2, 7:3, 8:4}

        self.unique_times = sorted(list(self.genomic_df['Metadata_pert_time'].unique()))
        self.time_to_idx = {time: idx + 1 for idx, time in enumerate(self.unique_times)}
        print(f"Time points: {self.unique_times}")
        print(f"Time mapping: {self.time_to_idx}")  # e.g., {6:1, 24:2}
        
        # Set parameters for feature extraction
        self.pad_length = pad_length
        self.genomic_cols = [c for c in self.genomic_df.columns if not c.startswith("Metadata_")]
        self.morph_cols = [c for c in self.df.columns if not c.startswith("Metadata_")]
        
        # Print feature dimensions
        print(f"\nFeature Dimensions:")
        print(f"Morphological features: {len(self.morph_cols)}")
        print(f"Genomic features: {len(self.genomic_cols)}")


    def __getitem__(self, index):
        # Create a new, isolated random number generator for this specific item fetch.
        # ensuring samples are different each time.
        rng = np.random.default_rng()
        
        smiles = self.unique_smiles[index]
        
        # Get graph and morph features
        adj_mat, node_feat = smiles2graph(smiles)
        adj_mat = torch.FloatTensor(adj_mat)
        node_feat = torch.FloatTensor(node_feat)
        atom_vec = torch.ones(len(node_feat), 1)
        cmpd_feat = self._pad(adj_mat, node_feat, atom_vec)
        
        # sample 1 replicate for each drug perturbation, passing the dedicated generator
        morph_mask = self.df[self.smiles_col] == smiles
        morph_feat = (self.df[morph_mask].sample(1, random_state=rng)[self.morph_cols].values.astype(float).flatten() 
                     if morph_mask.any() else -1 * np.ones(len(self.morph_cols)))
        
        # Initialize tensors for all cell lines
        n_cell_lines = len(self.cell_line_to_idx)
        n_features = len(self.genomic_cols)
        
        # Initialize with padding values (0)
        cell_features = -1 * np.ones((n_cell_lines, n_features))
        cell_indices = np.zeros(n_cell_lines)  # 0 indicates missing/padding
        doses = np.zeros(n_cell_lines)  # 0 indicates padding
        times = np.zeros(n_cell_lines)  # 0 indicates padding
        
        # Fill in available data
        smiles_mask = self.genomic_df[self.smiles_col] == smiles
        if smiles_mask.any():
            for cell_line in self.genomic_df[smiles_mask][self.cell_line_col].unique():
                cell_idx = self.cell_line_to_idx[cell_line]
                array_idx = cell_idx - 1
                mask = smiles_mask & (self.genomic_df[self.cell_line_col] == cell_line)
                if mask.any():
                    # Pass the dedicated generator to the sampling method here as well
                    sampled_row = self.genomic_df[mask].sample(1, random_state=rng)
                    cell_features[array_idx] = sampled_row[self.genomic_cols].values
                    cell_indices[array_idx] = cell_idx
                    
                    # Convert actual values to indices
                    dose_value = sampled_row['Metadata_Dose_Level'].values[0]
                    time_value = sampled_row['Metadata_pert_time'].values[0]
                    doses[array_idx] = self.dose_to_idx[dose_value]  # e.g., 2 -> 1, 5 -> 2, etc.
                    times[array_idx] = self.time_to_idx[time_value]  # e.g., 6 -> 1, 24 -> 2
        
        return {
            "inputs": {
                "x_a": [torch.FloatTensor(f) for f in cmpd_feat],
                "x_b": torch.FloatTensor(morph_feat),
                "x_c": torch.FloatTensor(cell_features),  # [n_cell_lines, n_features]
                "cell_indices": torch.LongTensor(cell_indices),  # [n_cell_lines], 0 = padding
                "doses": torch.FloatTensor(doses),  # Add doses for each cell line
                "times": torch.FloatTensor(times)   # Add times for each cell line
            },
            "labels": torch.Tensor([-1])
        }

    def collate_fn(self, batch):
        """Collate function for DataLoader.
        
        Handles batching of variable-length cell line data by:
        1. Padding to maximum number of possible cell lines
        2. Using 0 to indicate padding/missing data
        
        Args:
            batch (list): List of items from __getitem__
            
        Returns:
            dict: Contains:
                - inputs: Dict with batched x_a, x_b, x_c features, cell_indices, doses, and times
                - labels: Batched labels
        """
        # Stack molecular features
        adj_mats = torch.stack([item["inputs"]["x_a"][0] for item in batch])
        node_feats = torch.stack([item["inputs"]["x_a"][1] for item in batch])
        atom_vecs = torch.stack([item["inputs"]["x_a"][2] for item in batch])
        x_a_batch = [adj_mats, node_feats, atom_vecs]
        
        # Stack morphological features and labels
        x_b_batch = torch.stack([item["inputs"]["x_b"] for item in batch])
        labels_batch = torch.stack([item["labels"] for item in batch])
        
        # Stack genomic features and cell indices
        x_c_batch = torch.stack([item["inputs"]["x_c"] for item in batch])
        cell_indices_batch = torch.stack([item["inputs"]["cell_indices"] for item in batch])
        
        # Stack doses and times
        doses_batch = torch.stack([item["inputs"]["doses"] for item in batch])
        times_batch = torch.stack([item["inputs"]["times"] for item in batch])
        
        return {
            "inputs": {
                "x_a": x_a_batch,
                "x_b": x_b_batch,
                "x_c": x_c_batch,
                "cell_indices": cell_indices_batch,
                "doses": doses_batch,  # Add batched doses
                "times": times_batch   # Add batched times
            },
            "labels": labels_batch
        }

def _split_data(dataset: Dataset, splits: Dict[str, str]) -> Dict[str, Dataset]:
    if splits is None:
        unique_smiles = dataset.unique_smiles
        total_smiles = len(unique_smiles)
        train_idx = np.random.choice(
            total_smiles, size=int(0.9 * total_smiles), replace=False
        )
        val_idx = [i for i in range(total_smiles) if i not in train_idx]
        return {
            "train": Subset(dataset, train_idx),
            "val": Subset(dataset, val_idx),
            "test": Subset(dataset, val_idx),
        }

    assert "train" in splits and "val" in splits
    split_dataset = {}
    for k, v in splits.items():
        print(f"Split {k}: {v}")
        df_split = pd.read_csv(v)
        if "index" in df_split.columns:
            idx = df_split["index"].values
        else:
            split_smiles = df_split["SMILES"].unique()
            idx = [
                i
                for i, smiles in enumerate(dataset.unique_smiles)
                if smiles in split_smiles
            ]
        split_dataset[k] = Subset(dataset, idx)
    return split_dataset