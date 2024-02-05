from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Torsions
from molSimplify.Classes.mol3D import mol3D


# Nice explanation of fingerprints here:
# https://www.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf
# Nice tutorial here:
# https://www.rdkit.org/docs/GettingStartedInPython.html
# Morgan paper: JCIM 50:742-54 (2010).

# This script takes in a mol3D object and gets an ECFP.
# This assumes that you can get the smiles from the mol3D,
# which can be obtained by the bound method "get_smiles"
# of the mol3D class. RDKit has functionality for
# drawing the necessary bits --> DrawMorganBit
# @param --> mol3D class of molecule, returns fingerprint
def get_morgan(mol, morgan_radius=4):
    if isinstance(mol, mol3D):
        smiles = mol.get_smiles(use_mol2=True)
    elif isinstance(mol, str):
        smiles = mol
    else:
        raise ValueError("First argument not a smiles string or mol3D")
    m = Chem.MolFromSmiles(smiles)
    # empty bit dictionary that gets populated
    # Morgan FPs should be compared with tanimoto similarity
    # By default, Morgan FPs have 2048 bits.
    bit_vector = {}
    morgan_fingerprint_vector = AllChem.GetMorganFingerprintAsBitVect(m, radius=morgan_radius, bitInfo=bit_vector)
    return bit_vector


def get_substructure_smiles(mol, atomID, radius):
    if isinstance(mol, mol3D):
        smiles = mol.get_smiles(use_mol2=True)
    elif isinstance(mol, str):
        smiles = mol
    else:
        raise ValueError("First argument not a smiles string or mol3D")
    m = Chem.MolFromSmiles(smiles)
    if radius > 0:
        environment_morgan = Chem.FindAtomEnvironmentOfRadiusN(m, radius, atomID)
        atoms_to_use = []
        for b in environment_morgan:
            atoms_to_use.append(m.GetBondWithIdx(b).GetBeginAtomIdx())
            atoms_to_use.append(m.GetBondWithIdx(b).GetEndAtomIdx())
        atoms_to_use = list(set(atoms_to_use))
    else:
        atoms_to_use = [atomID]
        environment_morgan = None
    smiles_1 = Chem.MolFragmentToSmiles(m, atoms_to_use, bondsToUse=environment_morgan,
                                        allHsExplicit=True, allBondsExplicit=True, rootedAtAtom=atomID)
    return smiles_1

# bit_vector_benzene = get_morgan(mol='c1ccccc1')
# smi = get_substructure_smiles('c1ccccc1',list(bit_vector_benzene.values())[0][0][0],list(bit_vector_benzene.values())[0][0][1])
# print(smi) # this tells you the substructures in the bit
# bit_vector_pyridine = get_morgan(mol='c1ncccc1')
# below you can see what pyridine and benzene have in common
# print(set(bit_vector.keys()).intersection(bit_vector_py.keys()))
