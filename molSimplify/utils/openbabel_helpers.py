from openbabel import openbabel, pybel
from molSimplify.Classes.mol3D import mol3D
import numpy as np

def count_aromatic(obmol):
    nb = 0
    for bond in openbabel.OBMolBondIter(obmol):
        if bond.IsAromatic():
            nb += 1
    return nb


def constrained_forcefield_optimization_bu(
    mol,
    fixed_atom_indices=None,
    max_steps=250,
    ff_name='mmff94',
    return_per_atom_nonbonded=False,   # deprecated path (delete-one-atom)
    return_vdw_energy=False,
    *,
    return_per_atom_ff_force=False,    # preferred embedding
    fd_delta=1e-3,                     # Å, finite-difference step
    isolate_vdw=False                  # True -> zero partial charges to emphasize vdW
):
    """
    Run an OpenBabel forcefield optimization on mol.OBMol, optionally freezing atoms.

    Haptics-aware behavior:
      - If mol has _haptic_groups_global and caller requested an unconstrained relax
        (fixed_atom_indices is None/empty), we automatically freeze metal atoms only.
      - Rotor search is skipped when metals or haptics are present (prevents "peeling").
    """
    from openbabel import openbabel
    import numpy as np

    print("aromatic BEFORE FF:", count_aromatic(mol.OBMol))

    # --- hard assert: if bo_dict says aromatic but OBMol isn't, fix it right here ---
    if hasattr(mol, "bo_dict"):
        n_ar_tokens = sum(1 for v in mol.bo_dict.values() if str(v).lower() in ("ar", "am"))
        if n_ar_tokens > 0 and count_aromatic(mol.OBMol) == 0:
            replace_bonds(mol.OBMol, mol.bo_dict)
            mol.OBMol.FindRingAtomsAndBonds()
            # optional: print once
            # print("reapplied aromaticity; aromatic bonds now:", count_aromatic(mol.OBMol))



    ff = openbabel.OBForceField.FindForceField(ff_name)
    if ff is None:
        raise RuntimeError(f"Forcefield '{ff_name}' not found.")

    obmol = mol.OBMol
    obmol.FindRingAtomsAndBonds()
    #openbabel.OBAtomTyper().AssignTypes(obmol)

    if isolate_vdw:
        for a in openbabel.OBMolAtomIter(obmol):
            a.SetPartialCharge(0.0)

    if not ff.Setup(obmol):
        raise RuntimeError("Failed to set up forcefield on molecule.")

    # -------------------- haptics-aware defaults --------------------
    groups = getattr(mol, "_haptic_groups_global", None) or []
    has_haptics = bool(groups)

    # Detect metals (prefer mol3D metal finder; fall back to OB atomic numbers)
    try:
        metals = mol.findMetal(transition_metals_only=False) or []
    except Exception:
        tm_atomic_nums = {
            21,22,23,24,25,26,27,28,29,30,
            39,40,41,42,43,44,45,46,47,48,
            57,72,73,74,75,76,77,78,79,80
        }
        metals = []
        for i, a in enumerate(openbabel.OBMolAtomIter(obmol)):
            if a.GetAtomicNum() in tm_atomic_nums:
                metals.append(i)
    has_metal = bool(metals)

    # Normalize fixed list
    effective_fixed = None
    if fixed_atom_indices:
        effective_fixed = [int(i) for i in fixed_atom_indices]
    else:
        effective_fixed = None

    # If caller asked for "fully free" but haptics exist, freeze metal(s) only.
    # This prevents η^n ligands from drifting / breaking under FF.
    if (effective_fixed is None) and has_haptics and has_metal:
        effective_fixed = sorted(set(int(i) for i in metals))

    # -------------------- constraints (always set; avoids sticky constraints) --------------------
    constraints = openbabel.OBFFConstraints()
    if effective_fixed:
        for idx in effective_fixed:
            constraints.AddAtomConstraint(int(idx) + 1)  # OB is 1-based
    ff.SetConstraints(constraints)

    # -------------------- optimization schedule --------------------
    # SD warmup helps escape large clashes; CG finishes near minimum.
    sd_steps = min(500, max_steps // 4 if max_steps >= 4 else max_steps)
    cg_steps = max(1, max_steps - sd_steps)

    ff.SteepestDescent(sd_steps)

    # Rotor search can unstick torsions, but it's risky for TM complexes / haptics.
    if (not has_haptics) and (not has_metal):
        try:
            ff.WeightedRotorSearch(50, 25)
        except Exception:
            pass

    ff.ConjugateGradients(cg_steps)
    ff.GetCoordinates(obmol)

    optimized_coords = np.array(
        [[a.GetX(), a.GetY(), a.GetZ()] for a in openbabel.OBMolAtomIter(obmol)],
        dtype=float
    )

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
                atom.SetVector(float(xyz[k, 0]), float(xyz[k, 1]), float(xyz[k, 2]))
            ff.SetCoordinates(obmol)

        _apply_coords(optimized_coords)

        fixed_set = set(int(i) for i in (effective_fixed or []))
        for i in range(N):
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


    print("aromatic AFTER FF:", count_aromatic(mol.OBMol))

    return tuple(results) if len(results) > 1 else results[0]

def constrained_forcefield_optimization(
    mol,
    fixed_atom_indices=None,
    max_steps=250,
    ff_name="mmff94",
    return_per_atom_nonbonded=False,   # deprecated path (delete-one-atom)
    return_vdw_energy=False,
    *,
    return_per_atom_ff_force=False,    # preferred embedding
    fd_delta=1e-3,                     # Å, finite-difference step
    isolate_vdw=False,                 # True -> zero partial charges to emphasize vdW
    freeze_aromatic_rings=True,        # NEW: freeze aromatic ring atoms (from bo_dict)
    fallback_forcefields=("uff", "ghemical"),  # NEW: auto fallback if ff_name fails
    verbose_ff=False                   # NEW: extra prints for debugging
):
    """
    Run an OpenBabel forcefield optimization on mol.OBMol, optionally freezing atoms.

    Haptics-aware behavior:
      - If mol has _haptic_groups_global and caller requested an unconstrained relax
        (fixed_atom_indices is None/empty), we automatically freeze metal atoms only.
      - Rotor search is skipped when metals or haptics are present (prevents "peeling").

    NEW robustness:
      - Re-assert aromaticity from mol.bo_dict if OBMol has 0 aromatic bonds.
      - Freeze aromatic ring atoms (optional) to keep rings planar/rigid for steric screening.
      - Auto fallback to UFF/Ghemical if requested FF cannot Setup() (MMFF94 often fails on TMs).
    """
    from openbabel import openbabel
    import numpy as np

    # -------------------- small local helpers --------------------
    def _count_aromatic(obmol):
        nb = 0
        for b in openbabel.OBMolBondIter(obmol):
            try:
                if b.IsAromatic():
                    nb += 1
            except Exception:
                pass
        return nb

    def _aromatic_atom_indices_from_bodict(bo_dict):
        ar_atoms = set()
        for (i, j), bo in bo_dict.items():
            s = str(bo).strip().lower()
            if s in ("ar", "am", "5", "1.5"):
                ar_atoms.add(int(i))
                ar_atoms.add(int(j))
        return sorted(ar_atoms)

    def _find_forcefield(name_list):
        for nm in name_list:
            ff = openbabel.OBForceField.FindForceField(nm)
            if ff is not None:
                return nm, ff
        return None, None

    # -------------------- sanity: aromaticity present in OBMol --------------------
    obmol = mol.OBMol

    # If bo_dict has aromatic tokens but OBMol has none, reapply bonds+aromatic flags
    if hasattr(mol, "bo_dict"):
        bo = mol.bo_dict
        n_ar_tokens = sum(1 for v in bo.values() if str(v).strip().lower() in ("ar", "am", "5", "1.5"))
        if n_ar_tokens > 0 and _count_aromatic(obmol) == 0:
            if verbose_ff:
                print(f"[FF] Reapplying aromaticity from bo_dict (ar_edges={n_ar_tokens})")
            replace_bonds(obmol, bo)
            if hasattr(obmol, "FindRingAtomsAndBonds"):
                obmol.FindRingAtomsAndBonds()
            if verbose_ff:
                print("[FF] aromatic after reapply:", _count_aromatic(obmol))

    # Some OB builds benefit from ring finding before typing/setup
    if hasattr(obmol, "FindRingAtomsAndBonds"):
        obmol.FindRingAtomsAndBonds()

    # -------------------- choose a forcefield (with fallback) --------------------
    ff_try_order = (ff_name,) + tuple(fallback_forcefields or ())
    chosen_name, ff = _find_forcefield(ff_try_order)
    if ff is None:
        raise RuntimeError(f"No usable forcefield found. Tried: {ff_try_order}")

    if isolate_vdw:
        for a in openbabel.OBMolAtomIter(obmol):
            a.SetPartialCharge(0.0)

    if not ff.Setup(obmol):
        # Try fallbacks if the chosen one failed setup
        if verbose_ff:
            print(f"[FF] ff.Setup('{chosen_name}') failed; trying fallbacks…")
        ok = False
        for nm in tuple(fallback_forcefields or ()):
            ff2 = openbabel.OBForceField.FindForceField(nm)
            if ff2 is not None and ff2.Setup(obmol):
                ff = ff2
                chosen_name = nm
                ok = True
                break
        if not ok:
            raise RuntimeError(f"Failed to set up forcefield on molecule. Tried: {ff_try_order}")

    if verbose_ff:
        print(f"[FF] Using forcefield: {chosen_name}")

    # -------------------- haptics-aware defaults --------------------
    groups = getattr(mol, "_haptic_groups_global", None) or []
    has_haptics = bool(groups)

    # Detect metals (prefer mol3D metal finder; fall back to OB atomic numbers)
    try:
        metals = mol.findMetal(transition_metals_only=False) or []
    except Exception:
        tm_atomic_nums = {
            21,22,23,24,25,26,27,28,29,30,
            39,40,41,42,43,44,45,46,47,48,
            57,72,73,74,75,76,77,78,79,80
        }
        metals = []
        for i, a in enumerate(openbabel.OBMolAtomIter(obmol)):
            if a.GetAtomicNum() in tm_atomic_nums:
                metals.append(i)
    has_metal = bool(metals)

    # Normalize fixed list
    if fixed_atom_indices:
        effective_fixed = [int(i) for i in fixed_atom_indices]
    else:
        effective_fixed = None

    # If caller asked for "fully free" but haptics exist, freeze metal(s) only.
    if (effective_fixed is None) and has_haptics and has_metal:
        effective_fixed = sorted(set(int(i) for i in metals))

    # NEW: optionally freeze aromatic ring atoms to keep rings planar/rigid
    if freeze_aromatic_rings and hasattr(mol, "bo_dict"):
        ar_atoms = _aromatic_atom_indices_from_bodict(mol.bo_dict)
        if ar_atoms:
            if effective_fixed is None:
                effective_fixed = []
            effective_fixed = sorted(set(effective_fixed) | set(ar_atoms))
            if verbose_ff:
                print(f"[FF] Freezing aromatic atoms: {len(ar_atoms)}")

    # -------------------- constraints (always set; avoids sticky constraints) --------------------
    constraints = openbabel.OBFFConstraints()
    if effective_fixed:
        for idx in effective_fixed:
            constraints.AddAtomConstraint(int(idx) + 1)  # OB is 1-based
    ff.SetConstraints(constraints)

    # -------------------- optimization schedule --------------------
    sd_steps = min(500, max_steps // 4 if max_steps >= 4 else max_steps)
    cg_steps = max(1, max_steps - sd_steps)

    ff.SteepestDescent(sd_steps)

    # Rotor search can unstick torsions, but it's risky for TM complexes / haptics.
    if (not has_haptics) and (not has_metal):
        try:
            ff.WeightedRotorSearch(50, 25)
        except Exception:
            pass

    ff.ConjugateGradients(cg_steps)
    ff.GetCoordinates(obmol)

    optimized_coords = np.array(
        [[a.GetX(), a.GetY(), a.GetZ()] for a in openbabel.OBMolAtomIter(obmol)],
        dtype=float
    )

    # Early exit
    if not (return_per_atom_nonbonded or return_vdw_energy or return_per_atom_ff_force):
        return optimized_coords

    results = [optimized_coords]

    if return_per_atom_nonbonded:
        # backwards-compat (delete-one-atom)
        base_energy = ff.Energy()
        per_atom_nonbonded = []
        for i in range(obmol.NumAtoms()):
            tempmol = openbabel.OBMol(obmol)
            tempmol.DeleteAtom(tempmol.GetAtom(i + 1))
            fftemp = openbabel.OBForceField.FindForceField(chosen_name)
            if not fftemp or not fftemp.Setup(tempmol):
                # try fallbacks here too
                ok = False
                for nm in tuple(fallback_forcefields or ()):
                    ff2 = openbabel.OBForceField.FindForceField(nm)
                    if ff2 is not None and ff2.Setup(tempmol):
                        fftemp = ff2
                        ok = True
                        break
                if not ok:
                    raise RuntimeError("Failed to set up temporary forcefield.")
            e = fftemp.Energy()
            per_atom_nonbonded.append(base_energy - e)
        results.append(per_atom_nonbonded)

    if return_per_atom_ff_force:
        # finite-difference gradient magnitude per atom -> "steric pressure"
        N = optimized_coords.shape[0]
        force_mag = np.zeros(N, dtype=float)

        def _apply_coords(xyz):
            for k, atom in enumerate(openbabel.OBMolAtomIter(obmol)):
                atom.SetVector(float(xyz[k, 0]), float(xyz[k, 1]), float(xyz[k, 2]))
            ff.SetCoordinates(obmol)

        _apply_coords(optimized_coords)

        fixed_set = set(int(i) for i in (effective_fixed or []))

        for i in range(N):
            # If atom is constrained/frozen, report 0 force (prevents misleading "pressure")
            if i in fixed_set:
                force_mag[i] = 0.0
                continue

            gi = np.zeros(3, dtype=float)
            for d in range(3):
                x_f = optimized_coords.copy()
                x_f[i, d] += fd_delta
                _apply_coords(x_f)
                Ef = ff.Energy()

                x_b = optimized_coords.copy()
                x_b[i, d] -= fd_delta
                _apply_coords(x_b)
                Eb = ff.Energy()

                gi[d] = (Ef - Eb) / (2.0 * fd_delta)

            force_mag[i] = float(np.linalg.norm(gi))

        # restore minimized coords
        _apply_coords(optimized_coords)
        results.append(force_mag)

    if return_vdw_energy:
        # Some FFs may not implement VDWEnergy; guard it.
        vdw = None
        try:
            vdw = ff.GetVDWEnergy()
        except Exception:
            pass
        results.append(vdw)

    return tuple(results) if len(results) > 1 else results[0]



def bond_order_from_str_bu(bo_str):
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

def bond_order_from_str(bo):
    # numeric tokens can encode aromatic (common in pipelines)
    if isinstance(bo, (int, float)):
        v = float(bo)
        if abs(v - 5.0) < 1e-6 or abs(v - 1.5) < 1e-6:
            return 1, True      # aromatic -> order 1 + aromatic flag
        return int(round(v)), False

    s = str(bo).strip().lower()
    if s in {"ar", "am", "5", "1.5"}:
        return 1, True
    if s in {"1", "2", "3"}:
        return int(s), False
    raise ValueError(f"Unknown bond token: {bo!r}")


def replace_bonds_bu(obmol, bo_dict):
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
    #obmol.ConnectTheDots()
    #obmol.PerceiveBondOrders()


from openbabel import openbabel

_AR_TOKENS = {"ar", "am", "aro", "aromatic"}

def _is_aromatic_token(bo):
    # normalize strings
    if isinstance(bo, str):
        s = bo.strip().lower()
        return s in _AR_TOKENS or s == "5" or s == "5.0" or s == "1.5"
    # numeric conventions sometimes used for aromatic
    if isinstance(bo, (int, float)):
        return (bo == 5) or (abs(float(bo) - 1.5) < 1e-6)
    return False

def _numeric_bond_order(bo):
    # for OBMol we keep aromatic as single + aromatic flag
    if _is_aromatic_token(bo):
        return 1
    if isinstance(bo, str):
        s = bo.strip()
        if s.isdigit():
            return int(s)
        try:
            return int(float(s))
        except Exception:
            pass
    if isinstance(bo, (int, float)):
        return int(round(float(bo)))
    raise ValueError(f"Unrecognized bond order token: {bo!r}")

def replace_bonds(obmol, bo_dict):
    for bond in list(openbabel.OBMolBondIter(obmol)):
        obmol.DeleteBond(bond)

    aromatic_atoms = set()

    for (a1, a2), bo in bo_dict.items():
        i, j = a1 + 1, a2 + 1
        order, is_ar = bond_order_from_str(bo)
        obmol.AddBond(i, j, order)
        if is_ar:
            b = obmol.GetBond(i, j)
            if b:
                b.SetAromatic(True)
            aromatic_atoms.add(i); aromatic_atoms.add(j)

    for idx in aromatic_atoms:
        a = obmol.GetAtom(idx)
        if a:
            a.SetAromatic(True)

    obmol.FindRingAtomsAndBonds()


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
