from openbabel import openbabel, pybel
from molSimplify.Classes.mol3D import mol3D

def constrained_forcefield_optimization(mol, fixed_atom_indices, max_steps=250, ff_name='mmff94'):
    """
    Perform forcefield optimization with some atoms fixed.

    Parameters:
    - mol: pybel.Molecule object
    - fixed_atom_indices: list of 0-based atom indices to fix
    - max_steps: max number of optimization steps
    - ff_name: 'mmff94' or 'uff'

    Returns:
    - optimized_coords: numpy array of shape (N, 3)
    """
    ff = openbabel.OBForceField.FindForceField(ff_name)
    if ff is None:
        raise RuntimeError(f"Forcefield '{ff_name}' not found.")

    obmol = mol.OBMol

    if not ff.Setup(obmol):
        raise RuntimeError("Failed to set up forcefield on molecule.")

    # Create constraints object and fix atoms (1-based indexing!)
    constraints = openbabel.OBFFConstraints()
    for idx in fixed_atom_indices:
        constraints.AddAtomConstraint(idx + 1)
    ff.SetConstraints(constraints)

    ff.ConjugateGradients(max_steps)
    ff.GetCoordinates(obmol)

    # Extract optimized coordinates
    optimized_coords = np.array([
        [atom.GetX(), atom.GetY(), atom.GetZ()]
        for atom in openbabel.OBMolAtomIter(mol.OBMol)
    ])
    
    return optimized_coords
    
def bond_order_from_str(bo_str):
    """
    Convert bond order representation to OpenBabel-compatible values.
    - '1', '2', '3' → integer bond orders
    - 'ar', 'am' → use custom order 5 with aromatic flag
    """
    if isinstance(bo_str, int):
        return bo_str, False

    bo_str = str(bo_str).lower()
    if bo_str == '1':
        return 1, False
    elif bo_str == '2':
        return 2, False
    elif bo_str == '3':
        return 3, False
    elif bo_str in {'ar', 'am'}:
        return 5, True  # use 5 as a distinctive order for aromatic bonds
    else:
        raise ValueError(f"Unknown bond order string: {bo_str}")

def replace_bonds(obmol, bo_dict):
    """
    Replaces all bonds in the OBMol object based on a bond order dictionary.

    Parameters:
    - obmol: OpenBabel OBMol object
    - bo_dict: dict with keys as (atom1_idx, atom2_idx) and values as bond orders ('1', '2', 'ar', etc.)
               Indices are 0-based.
    """
    # Remove existing bonds
    bonds_to_remove = [bond for bond in openbabel.OBMolBondIter(obmol)]
    for bond in bonds_to_remove:
        obmol.DeleteBond(bond)

    # Add new bonds
    for (a1, a2), bo_str in bo_dict.items():
        atom1_idx = a1 + 1  # OBMol uses 1-based indices
        atom2_idx = a2 + 1

        bond_order, aromatic = bond_order_from_str(bo_str)
        obmol.AddBond(atom1_idx, atom2_idx, bond_order)

        # Retrieve the bond and set aromatic if needed
        bond = obmol.GetBond(atom1_idx, atom2_idx)
        if bond is not None and aromatic:
            bond.SetAromatic(True)

    # Recalculate connectivity and bonding
    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()

def get_all_bonds(obmol):
    """
    Returns a list of bonds in the OBMol.
    Each bond is represented as a tuple: (atom1_index, atom2_index, bond_order)

    atom indices are 1-based in OBMol.
    """
    bonds = []
    for bond in openbabel.OBMolBondIter(obmol):
        a1 = bond.GetBeginAtomIdx()  # 1-based atom index
        a2 = bond.GetEndAtomIdx()
        order = bond.GetBondOrder()
        bonds.append((a1, a2, order))
    return bonds

def get_all_atoms(obmol):
    """
    Returns a list of atoms in the OBMol.
    Each atom is represented as a dictionary with info: index, atomic number, coordinates.

    Atom indices are 1-based in OBMol.
    """
    atoms = []
    for atom in openbabel.OBMolAtomIter(obmol):
        idx = atom.GetIdx()  # 1-based atom index
        atomic_num = atom.GetAtomicNum()
        x, y, z = atom.GetX(), atom.GetY(), atom.GetZ()
        element = atom.GetType()
        atoms.append({
            "index": idx,
            "element": element,
            "atomic_num": atomic_num,
            "coords": (x, y, z)
        })
    return atoms
