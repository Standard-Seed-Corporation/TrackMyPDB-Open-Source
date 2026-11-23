"""Physicochemical descriptors and 2D depictions (RDKit)."""

from __future__ import annotations

from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski, rdDepictor, rdMolDescriptors

from .similarity import InvalidSmilesError, mol_from_smiles


def calculate_properties(smiles: str) -> dict:
    """Core drug-likeness descriptors for a molecule.

    Returns molecular weight (Da), Crippen logP, H-bond donors/acceptors,
    topological polar surface area (Å^2), rotatable bonds, heavy-atom count,
    the molecular formula and the number of Lipinski rule-of-five violations.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise InvalidSmilesError(f"Invalid SMILES: {smiles!r}")
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular_Weight": round(mw, 2),
        "LogP": round(logp, 2),
        "HBD": hbd,
        "HBA": hba,
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "Rotatable_Bonds": Lipinski.NumRotatableBonds(mol),
        "Heavy_Atoms": mol.GetNumHeavyAtoms(),
        "Lipinski_Violations": int(violations),
    }


def draw_molecule(smiles: str, size: tuple[int, int] = (400, 400)):
    """2D depiction of a molecule as a PIL image."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise InvalidSmilesError(f"Invalid SMILES: {smiles!r}")
    rdDepictor.Compute2DCoords(mol)
    return Draw.MolToImage(mol, size=size)
