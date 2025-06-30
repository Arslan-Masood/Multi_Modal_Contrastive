from typing import Any, Optional, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

class Adjacency:
    def __init__(self, mol: Chem.Mol):
        self.num_atoms = mol.GetNumAtoms()
        self.adj_mat = self._get_adjacency_matrix(mol)

    def _get_adjacency_matrix(self, mol: Chem.Mol) -> np.ndarray:
        return Chem.GetAdjacencyMatrix(mol)

    @property
    def id_adj_mat(self) -> np.ndarray:
        return self.adj_mat + np.eye(self.num_atoms)

    @property
    def norm_id_adj_mat(self) -> np.ndarray:
        id_adj_mat = self.id_adj_mat
        return self._normalize_adj_mat(id_adj_mat)

    @property
    def norm_laplacian_mat(self) -> np.ndarray:
        norm_adj_mat = self._normalize_adj_mat(self.adj_mat)
        return np.eye(self.num_atoms) - norm_adj_mat

    @staticmethod
    def _normalize_adj_mat(adj_mat: Union[list, np.ndarray]) -> np.ndarray:
        if isinstance(adj_mat, list):
            adj_mat = np.array(adj_mat)
        num_nodes = len(adj_mat)
        node_deg = np.sum(adj_mat, axis=0)
        node_deg = 1 / np.sqrt(node_deg)
        node_deg = np.eye(num_nodes) * node_deg
        return np.matmul(np.matmul(node_deg, adj_mat), node_deg)

class Nodes:
    def __init__(self, mol: Chem.Mol):
        self.node_feat = self._get_node_features(mol)

    def _get_node_features(self, mol: Chem.Mol) -> np.ndarray:
        return np.array([self._featurize_atom(a) for a in mol.GetAtoms()])

    @staticmethod
    def _featurize_atom(atom: Chem.Atom) -> np.ndarray:
        results = (
            Nodes.one_of_k_encoding(
                atom.GetSymbol(),
                ["C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na",
                 "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb",
                 "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li",
                 "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt",
                 "Hg", "Pb", "Unknown"],
                unk=True
            )
            + Nodes.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            + Nodes.one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6], unk=True)
            + [atom.GetFormalCharge()]
            + Nodes.one_of_k_encoding(
                atom.GetHybridization(),
                [Chem.rdchem.HybridizationType.SP, 
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3, 
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2,
                 Chem.rdchem.HybridizationType.UNSPECIFIED],
                unk=True
            )
            + [atom.GetIsAromatic()]
            + Nodes.one_of_k_encoding(atom.GetNumExplicitHs(), [0, 1, 2, 3, 4], unk=True)
        )
        return np.array(results).reshape(-1)

    @staticmethod
    def one_of_k_encoding(x: Any, allowable_set: list, unk: bool = False) -> list:
        if x not in allowable_set:
            if unk:
                x = allowable_set[-1]
            else:
                raise Exception(f"input {x} not in allowable set {allowable_set}")
        return [x == s for s in allowable_set]

class MolGraph(Adjacency, Nodes):
    def __init__(self, smiles: str, explicit_H: Optional[str] = None):
        self.mol = Chem.MolFromSmiles(smiles)
        self._process_mol(self.mol, explicit_H=explicit_H)

        Adjacency.__init__(self, self.mol)
        Nodes.__init__(self, self.mol)

    @staticmethod
    def _process_mol(mol: Chem.Mol, explicit_H: Optional[str] = None) -> Chem.Mol:
        if explicit_H == "all":
            mol = Chem.AddHs(mol)
        elif explicit_H == "polar":
            mol = Chem.AddHs(mol, onlyOnAtoms=[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in [7, 8, 15, 16]])
        elif explicit_H is None:
            mol = Chem.RemoveHs(mol)
        else:
            raise ValueError("Invalid value for explicit_H")
        
        Chem.Kekulize(mol)
        AllChem.Compute2DCoords(mol)
        return mol
