from openbabel import openbabel, pybel
from molSimplify.Classes.mol3D import mol3D
import numpy as np

def constrained_forcefield_optimization(
    mol,
    fixed_atom_indices=None,
    max_steps=250,
    ff_name='mmff94',
    return_per_atom_nonbonded=False,   # deprecated path (delete-one-atom)
    return_vdw_energy=False,
    *,
    return_per_atom_ff_force=False,    # NEW: preferred embedding
    fd_delta=1e-3,                     # Å, finite-difference step
    isolate_vdw=False                  # True -> zero partial charges to emphasize vdW
):
    from openbabel import openbabel
    import numpy as np

    ff = openbabel.OBForceField.FindForceField(ff_name)
    if ff is None:
        raise RuntimeError(f"Forcefield '{ff_name}' not found.")

    obmol = mol.OBMol
    obmol.FindRingAtomsAndBonds()
    openbabel.OBAtomTyper().AssignTypes(obmol)

    if isolate_vdw:
        for a in openbabel.OBMolAtomIter(obmol):
            a.SetPartialCharge(0.0)

    if not ff.Setup(obmol):
        raise RuntimeError("Failed to set up forcefield on molecule.")

    if fixed_atom_indices:
        constraints = openbabel.OBFFConstraints()
        for idx in fixed_atom_indices:
            constraints.AddAtomConstraint(int(idx) + 1)  # 1-based
        ff.SetConstraints(constraints)

    ff.ConjugateGradients(max_steps)
    ff.GetCoordinates(obmol)

    optimized_coords = np.array([[a.GetX(), a.GetY(), a.GetZ()]
                                 for a in openbabel.OBMolAtomIter(obmol)], dtype=float)

    # Early exit (original behavior)
    if not (return_per_atom_nonbonded or return_vdw_energy or return_per_atom_ff_force):
        return optimized_coords

    results = [optimized_coords]

    if return_per_atom_nonbonded:
        # keep for backwards-compat, but it's noisy; prefer return_per_atom_ff_force
        base_energy = ff.Energy()
        per_atom_nonbonded = []
        for i in range(obmol.NumAtoms()):
            tempmol = openbabel.OBMol(obmol)
            tempmol.DeleteAtom(tempmol.GetAtom(i + 1))
            fftemp = openbabel.OBForceField.FindForceField(ff_name)
            if not fftemp.Setup(tempmol):
                raise RuntimeError("Failed to set up temporary forcefield.")
            e = fftemp.Energy()
            per_atom_nonbonded.append(base_energy - e)
        results.append(per_atom_nonbonded)

    if return_per_atom_ff_force:
        # finite-difference gradient per atom -> |∇E| as steric pressure
        N = optimized_coords.shape[0]
        force_mag = np.zeros(N, dtype=float)

        def _apply_coords(xyz):
            for k, atom in enumerate(openbabel.OBMolAtomIter(obmol)):
                atom.SetVector(float(xyz[k,0]), float(xyz[k,1]), float(xyz[k,2]))
            ff.SetCoordinates(obmol)

        _apply_coords(optimized_coords)

        fixed_set = set(int(i) for i in (fixed_atom_indices or []))
        for i in range(N):
            # you can skip fixed atoms if you want: if i in fixed_set: continue
            gi = np.zeros(3, dtype=float)
            for d in range(3):
                x_f = optimized_coords.copy(); x_f[i, d] += fd_delta
                _apply_coords(x_f); Ef = ff.Energy()
                x_b = optimized_coords.copy(); x_b[i, d] -= fd_delta
                _apply_coords(x_b); Eb = ff.Energy()
                gi[d] = (Ef - Eb) / (2.0 * fd_delta)
            force_mag[i] = float(np.linalg.norm(gi))
        # restore minimized coords
        _apply_coords(optimized_coords)
        results.append(force_mag)

    if return_vdw_energy:
        results.append(ff.GetVDWEnergy())

    return tuple(results) if len(results) > 1 else results[0]



def bond_order_from_str(bo_str):
    """
    Convert bond order representation to OpenBabel-compatible values.
    - '1', '2', '3' ? integer bond orders
    - 'ar', 'am' ? use custom order 5 with aromatic flag
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

def get_bond_dict(obmol):
    """
    Returns a dictionary with bond atom index pairs as keys and bond order as values.

    Parameters:
        obmol: an openbabel.OBMol object

    Returns:
        bond_dict: dict with keys (i, j) and values as bond order (float)
    """
    bond_dict = {}
    for bond in openbabel.OBMolBondIter(obmol):
        a1 = bond.GetBeginAtomIdx() - 1  # convert from 1-based to 0-based
        a2 = bond.GetEndAtomIdx() - 1
        order = bond.GetBondOrder()
        bond_dict[tuple(sorted((a1, a2)))] = order
    return bond_dict
