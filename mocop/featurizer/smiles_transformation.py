import functools
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, inchi
from openeye import oechem, oegraphsim

from featurizer.molgraph_rdkit import MolGraph


@functools.lru_cache(maxsize=None)
def smiles2fp(
    smiles: str,
    numbits=1024,
    minradius=0,
    maxradius=2,
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, 
        maxradius, 
        nBits=numbits, 
        useFeatures=False,
        useChirality=True,
    )
    
    return np.array(fp)


@functools.lru_cache(maxsize=None)
def smiles2graph(smiles, adj_type="norm_id_adj_mat", explicit_H_node=None, **kwargs):
    mol = MolGraph(smiles, explicit_H_node)
    adj_mat = getattr(mol, adj_type)
    node_feat = mol.node_feat
    return adj_mat, node_feat


@functools.lru_cache(maxsize=None)
def smiles2inchikey(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return Chem.MolToInchiKey(mol)


@functools.lru_cache(maxsize=None)

def inchi2smiles(inchi_str: str):
    mol = Chem.inchi.MolFromInchi(inchi_str)
    if mol is None:
        raise ValueError(f"Invalid InChI string: {inchi_str}")
    return Chem.MolToSmiles(mol)
