import time
import copy
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from collections import defaultdict
from ast import literal_eval
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree as KDTree  # fast nearest-neighbor
from openbabel import openbabel, pybel

# molSimplify imports (local project utilities)
from molSimplify.Classes.atom3D import atom3D
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.voxelgrid import VoxelGrid, plot_voxels
from molSimplify.Scripts.enhanced_structgen_functionality import *
from molSimplify.utils.openbabel_helpers import *
from molSimplify.Scripts.io import lig_load
from molSimplify.Classes.globalvars import vdwrad
from molSimplify.Scripts.io import getlicores, lig_load_safe, parse_bracketed_list

from typing import Any, List, Dict, Tuple, Union, Optional

# Silence Open Babel warnings (less noisy logs)
openbabel.obErrorLog.SetOutputLevel(0)

# ----------------------------- small utility helpers ----------------------------- #

def _metal_indices_from_elements(elements):
    """
    Return indices of atoms that are transition metals (basic 3d/4d/5d set).
    """
    METALS = {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"
    }
    return [i for i, sym in enumerate(elements) if sym in METALS]


def _nearest_metal_axes(donor_indices, sample_coords, complex_coords, complex_elements):
    """
    For each donor index, return a unit vector pointing 'outward' = donor -> nearest metal.
    Falls back to +x if no metals or degenerate geometry.

    Parameters
    ----------
    donor_indices : Iterable[int]
        Indices (into sample_coords) for donor atoms (or donor representatives).
    sample_coords : (N, 3) float
        Coordinates of the ligand in its current frame (same indexing as donor_indices).
    complex_coords : (M, 3) float
        Coordinates of the current complex (used to locate nearest metal centers).
    complex_elements : List[str]
        Elements of the current complex (length M).

    Returns
    -------
    dict[int, np.ndarray]
        Map donor index -> (3,) unit vector, oriented donor -> nearest metal.
    """
    metal_ids = _metal_indices_from_elements(complex_elements)
    axes = {}
    if not metal_ids:
        unit = np.array([1.0, 0.0, 0.0])
        for d in donor_indices:
            axes[d] = unit
        return axes

    metal_coords = complex_coords[np.array(metal_ids)]
    tree = KDTree(metal_coords)

    for d in donor_indices:
        dpos = sample_coords[d]
        _, idx = tree.query(dpos)
        u = dpos - metal_coords[idx]
        n = np.linalg.norm(u)
        axes[d] = (u / n) if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    return axes

def load_in_ligand(inp):
    lig3D, emsg = lig_load_safe(inp)
    if lig3D.bo_dict == {}:
        lig3D.convert2OBMol()
        lig3D.populateBOMatrix(bonddict=True)
    return lig3D, emsg

def _get_catoms(userligand, emsg, usercatoms):
    if emsg == 'dict':
        licores = getlicores()
        # Case 1: Ligand is in the dictionary ligands.dict.
        if userligand in list(licores.keys()):
            print(f'Loading ligand from dictionary: {userligand}')
            dbentry = licores[userligand]
        # Case 2: max(text_similarities) > 0.6
        # It is likely the user made a typo in inputting a ligand that is in ligands.dict
        else:
            text_similarities = [difflib.SequenceMatcher(None, userligand, i).ratio() for i in list(licores.keys())]
            max_similarity = max(text_similarities)
            index_max = text_similarities.index(max_similarity)
            desired_ligand = list(licores.keys())[index_max]
            print(f'Ligand was not in dictionary, but the sequence is very similar to a ligand that is: {desired_ligand}')
            print(f'Loading ligand from dictionary: {desired_ligand}')
            dbentry = licores[desired_ligand]  # Loading the typo ligand.
        catoms = dbentry[2]
        corrected_catoms = parse_bracketed_list(catoms)
    else:
        assert usercatoms != None, "Ligand must either be in ligand dictionary or user must pass in coordinating atom indices"
        catoms = usercatoms
        try:
            corrected_catoms = parse_bracketed_list(catoms)
            assert type(corrected_catoms) == list
        except:
            print("Coordinating atom indices are not in a readable format, please refer to the tutorial on passing in coordinating atoms")
    return corrected_catoms

def _check_list_lengths(userligand_list, usercatoms_list, occupancy_list, isomer_list):
    length_check = []
    for l in [userligand_list, usercatoms_list, occupancy_list, isomer_list]:
        if l != None:
            length_check.append(len(l))
    assert len(list(set(length_check)))==1, "All passed in lists must be the same length!"

def _create_ligand_tuple(userligand, usercatoms=None, occupancy=1, isomer=None):
    lig3D, emsg = load_in_ligand(userligand)
    catoms = _get_catoms(userligand, emsg, usercatoms)
    return (lig3D,catoms,occupancy,isomer)

def create_ligand_list(userligand_list, usercatoms_list=None, occupancy_list=None, isomer_list=None):
    _check_list_lengths(userligand_list, usercatoms_list, occupancy_list, isomer_list)

    ret_ligand_list = []
    for i in range(0,len(userligand_list)):
        userligand = userligand_list[i]
        try:
            usercatoms = usercatoms_list[i]
        except:
            usercatoms = None
        try:
            occupancy = occupancy_list[i]
        except:
            occupancy = None
        try:
            isomer = isomer_list[i]
        except:
            isomer = None
        ligand_tuple = _create_ligand_tuple(userligand, usercatoms, occupancy, isomer)
        ret_ligand_list.append(ligand_tuple)
    return ret_ligand_list

def generate_complex(
    ligand_list,
    *,
    metals: str = "Fe",
    voxel_size: float = 0.5,
    vdw_scale: float = 0.8,
    clash_weight: float = 10.0,
    nudge_alpha: float = 0.1,
    geometry: str = "octahedral",
    coords = None,
    max_steps: int = 500,
    ff_name: str = "UFF",

    vis_save_dir = None,
    vis_stride: int = 1,
    vis_view: Tuple[float, float] = (22, -60),
    vis_prefix: str = "kabsch",
    e_d: float = 10.0,
    fixed_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,

    manual: bool = False,
    manual_list = None,
    smart_generation: bool = True,
    verbose: bool = False,

    orientation_weight: float = 6.0,   # key knob
    orientation_k_neighbors: int = 4,
    orientation_hinge: float = 0.5,
    orientation_cap: float = 1.0,

    # existing multibond controls (already args)
    multibond_haptics: bool = True,
    multibond_bond_order: int = 1,
    multibond_prefer_nearest_metal: bool = True,

    # sterics report
    run_sterics: bool = True,
):
    """
    Build an complex from a list of ligands.

    Parameters mirror prior tuned constants; defaults preserve earlier behavior.
    If fixed_bounds is None, it is computed from e_d as:
        (-e_d, e_d, -e_d, e_d, -e_d, e_d)
    """
    # Compute fixed bounds if not provided (preserves old behavior)
    if fixed_bounds is None:
        fixed_bounds = (-1*e_d, 1*e_d, -1*e_d, 1*e_d, -1*e_d, 1*e_d)

    core3D, metals_structures = initialize_core(metals, coords, geometry)
    metals_structures_copy = copy.deepcopy(metals_structures)

    # Accumulate ALL ligands' haptic groups (GLOBAL indices) so we can re-apply anytime
    all_haptic_groups_global = []

    iteration = 0
    for ligand in ligand_list:
        donor_indices = list(ligand[1])   # may include lists (haptics) or ints
        occupancy = ligand[2]
        isomer = ligand[3]
        times = occupancy

        while times > 0:
            lig3D = mol3D()
            lig3D.copymol3D(ligand[0])

            metals_structures_copy = copy.deepcopy(metals_structures)
            complex_coordinates, complex_elements = get_all_coords_and_elements(core3D)

            VG = VoxelGrid(voxel_size=voxel_size, vdw_scale=vdw_scale)
            VG.add_atoms(elements=complex_elements, coords=complex_coordinates, group="complex")

            # ligand quick info (grouping handled later)
            ligand_donor_coords, ligand_all_coords = get_ligand_coordinates(
                lig3D,
                [d if not is_iterable_but_not_str(d) else d[0] for d in donor_indices]
            )
            _, ligand_elements = get_all_coords_and_elements(lig3D)

            # denticity = number of donor groups (Cp counts as 1)
            donor_groups, _, _ = parse_donor_spec_make_virtuals(donor_indices, ligand_all_coords)
            denticity = len(donor_groups)

            structure = get_next_structure(metals_structures_copy, denticity)
            if manual and iteration < len(manual_list):
                valid_subsets = [manual_list[iteration]]
            else:
                valid_subsets = get_valid_isomer_subsets(structure, isomer=isomer, denticity=denticity)

            if verbose:
                print(f"Valid subsets: {valid_subsets}")
                print(structure)

            # group-aware Kabsch
            best_subset, best_aligned_coords, best_rmsd, placement_attempts, best_perm_idx = clash_aware_kabsch(
                ligand_all_coords,
                donor_indices,                                  
                structure['backbone_coords'],
                valid_subsets,
                ligand_all_coords,
                complex_coordinates,
                complex_elements,
                ligand_elements,
                VoxelGrid,
                voxel_size=voxel_size,
                vdw_scale=vdw_scale,
                clash_weight=clash_weight,
                orientation_weight=orientation_weight,
                orientation_k_neighbors=orientation_k_neighbors,
                orientation_hinge=orientation_hinge,
                orientation_cap=orientation_cap,
                vis_save_dir=vis_save_dir,
                vis_stride=vis_stride,
                vis_view=vis_view,
                vis_prefix=vis_prefix,
                fixed_bounds=fixed_bounds,
            )

            # mark sites occupied
            if verbose:
                print(best_subset)

            assert best_subset != None, "Check total number of coordination sites in tested geometry, as well as total number of coordinating atoms"
            structure['occupied_mask'][np.array(best_subset)] = True

            # group→site mapping (not strictly needed for bonding)
            donor_to_site, site_indices_in_group_order = map_donors_to_sites_haptic_aware(
                donor_groups=donor_groups,
                best_perm_group_ids=best_perm_idx,
                chosen_subset=best_subset
            )

            # nudge using group representatives
            donor_reps = [g[0] for g in donor_groups]
            target_coords_in_rep_order = structure['backbone_coords'][np.array(site_indices_in_group_order)]
            new_coords = nudge_ligand_coords(
                ligand_coords=best_aligned_coords,
                donor_indices=donor_reps,
                target_sites_coords=target_coords_in_rep_order,
                alpha=nudge_alpha,
                verbose=False
            )

            # update ligand coordinates
            lig3D_copy = set_new_coords(lig3D, new_coords)
            ligand_all_coordinates, ligand_elements = get_all_coords_and_elements(lig3D_copy)

            # ---- Combine into complex; capture mapping of THIS ligand local→global ----
            core_count_before = len(core3D.atoms)
            core3D = core3D.roland_combine(lig3D_copy, donor_reps)
            core_count_after = len(core3D.atoms)

            local2global = map_ligand_local_to_core_indices_by_range(
                core_count_before, core_count_after, ligand_len=len(ligand_all_coordinates)
            )
            if local2global is None:
                # geometric fallback (exact coords, pre-optimization)
                local2global = map_ligand_local_to_core_indices_geometric(
                    core3D, ligand_all_coordinates, tol=0.05
                )

            # ---------- Re-apply ALL haptic bonds immediately (pre-replace_bonds) ----------
            if multibond_haptics:
                # convert THIS ligand's haptic groups to global indices
                donor_groups_now, _, _ = parse_donor_spec_make_virtuals(donor_indices, ligand_all_coordinates)
                donor_groups_global = to_global_groups(donor_groups_now, local2global)
                # keep only haptic ones (len >= 2)
                new_haptics = [g for g in donor_groups_global if len(g) >= 2]
                # extend accumulator
                all_haptic_groups_global.extend(new_haptics)
                # re-add ALL so far BEFORE bonds get rebuilt from bo_dict
                add_haptic_multibonds_to_metal_for_core(
                    core3D,
                    all_haptic_groups_global,
                    bond_order=multibond_bond_order,
                    prefer_nearest_metal=multibond_prefer_nearest_metal,
                )
            # ---------------------------------------------------------------------------------

            # clean bonds & optimize
            core3D.convert2OBMol(force_clean=True)
            replace_bonds(core3D.OBMol, core3D.bo_dict)

            optimized_coords = constrained_forcefield_optimization(
                core3D,
                get_all_bonded_atoms_bonded_to_metal(core3D),
                max_steps=max_steps,
                ff_name=ff_name
            )
            core3D = set_new_coords(core3D, optimized_coords)
            metals_structures = copy.deepcopy(metals_structures_copy)

            # smart generation / repair
            if smart_generation:
                overlap, same_order = check_badjob(core3D)
                if verbose:
                    print(f"Overlap: {overlap}, Ordering: {same_order}\n")

                piercings = detect_ring_piercing(core3D, angstrom_threshold=2.3, edge_buffer=0.15, inplane_pad=0.35)
                keep_piercings = [p for p in piercings if 0 not in p]
                if len(keep_piercings) != 0:
                    new_coords2, moved_atoms = correct_ring_piercings(core3D, keep_piercings)
                    core3D = set_new_coords(core3D, new_coords2)
                    core3D.convert2OBMol(force_clean=True)
                    replace_bonds(core3D.OBMol, core3D.bo_dict)

                    optimized_coords = constrained_forcefield_optimization(
                        core3D,
                        get_all_bonded_atoms_bonded_to_metal(core3D) + moved_atoms,
                        max_steps=250,
                        ff_name=ff_name
                    )
                    core3D = set_new_coords(core3D, optimized_coords)
                    core3D.convert2OBMol(force_clean=True)
                    replace_bonds(core3D.OBMol, core3D.bo_dict)

                    optimized_coords = constrained_forcefield_optimization(
                        core3D,
                        get_all_bonded_atoms_bonded_to_metal(core3D),
                        max_steps=max_steps,
                        ff_name=ff_name
                    )
                    core3D = set_new_coords(core3D, optimized_coords)
                    metals_structures = copy.deepcopy(metals_structures_copy)

                if overlap is True:
                    optimized_coords = constrained_forcefield_optimization(
                        core3D,
                        get_all_bonded_atoms_bonded_to_metal(core3D),
                        max_steps=250,
                        ff_name='GAFF'
                    )
                    core3D = set_new_coords(core3D, optimized_coords)
                    core3D.convert2OBMol(force_clean=True)
                    replace_bonds(core3D.OBMol, core3D.bo_dict)

                    optimized_coords = constrained_forcefield_optimization(
                        core3D,
                        get_all_bonded_atoms_bonded_to_metal(core3D),
                        max_steps=max_steps,
                        ff_name=ff_name
                    )
                    core3D = set_new_coords(core3D, optimized_coords)

            # OPTIONAL: add η^n bonds again after optimization (idempotent; keeps visuals consistent)
            if multibond_haptics and all_haptic_groups_global:
                add_haptic_multibonds_to_metal_for_core(
                    core3D,
                    all_haptic_groups_global,
                    bond_order=multibond_bond_order,
                    prefer_nearest_metal=multibond_prefer_nearest_metal,
                )
            reapply_all_haptics_and_sync(core3D, bond_order=1, prefer_nearest_metal=True)

            times -= 1
            iteration += 1

    # Final safety re-apply (idempotent)
    if multibond_haptics and all_haptic_groups_global:
        add_haptic_multibonds_to_metal_for_core(
            core3D,
            all_haptic_groups_global,
            bond_order=multibond_bond_order,
            prefer_nearest_metal=multibond_prefer_nearest_metal,
        )
        reapply_all_haptics_and_sync(core3D, bond_order=1, prefer_nearest_metal=True)

    clashes = None
    severity = None
    # get sterics report
    if run_sterics:
        clashes, severity, fig = run_sterics_check(core3D, max_steps, ff_name)

    return core3D, clashes, severity, fig

def run_sterics_check(core3D, max_steps, ff_name):
    optimized_coords, per_atom_ff_force = constrained_forcefield_optimization(
        core3D,
        max_steps=max_steps,
        ff_name=ff_name,
        return_per_atom_ff_force=True,
        fd_delta=1e-3,
        isolate_vdw=True,  # optional: emphasize sterics
    )

    elements = [at.sym for at in core3D.atoms]
    tree = KDTree(optimized_coords)

    clashes, severity = check_sterics_with_ff_embedding(
        tree=tree,
        coords=optimized_coords,
        elements=elements,
        vdw_radii=vdwrad,
        bo_dict=core3D.bo_dict,
        per_atom_ff_force=per_atom_ff_force,
        energy_source='ff_force',
        energy_weighted=True,
        pair_energy_threshold=0.5,
        scale=1.0,
        clearance_heavy=0.20,
        clearance_HX=0.30,
        clearance_HH=0.35,
        exclude_hops=(1,2,3),
    )

    # NOTE: pass clashes (list of pairs) to steric_pairs
    fig = visualize_molecule(
        optimized_coords,
        bond_dict=core3D.bo_dict,
        steric_pairs=clashes,
        severity_scores=severity,
        severity_threshold=0.05
    )
    return clashes, severity, fig

def visualize_molecule(coords, bond_dict=None, steric_pairs=None, severity_scores=None, severity_threshold=0.05):
    """
    Visualize atoms, bonds, and steric clashes with optional severity coloring.

    Parameters:
        coords (np.ndarray): Nx3
        bond_dict (dict): {(i, j): bond_order}
        steric_pairs (list of (i, j)): clash pairs
        severity_scores (dict): (i, j) -> severity (float), optional
    """
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    atom_scatter = ax.scatter(xs, ys, zs, color='blue', s=60, label='Atoms')

    legend_handles = [atom_scatter]
    legend_labels = ['Atoms']

    # Bonds
    if bond_dict:
        for i, j in bond_dict:
            p1, p2 = coords[i], coords[j]
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color='black', lw=2, alpha=1.0, zorder=1
            )
        # Add clean legend handle separately
        dummy_bond, = ax.plot([], [], [], color='black', lw=2)
        legend_handles.append(dummy_bond)
        legend_labels.append('Bonds')

    # Clashes
    if steric_pairs:
        severity_scores = severity_scores or {}
        max_sev = max(severity_scores.values(), default=0.1)
        norm = mcolors.Normalize(vmin=0.0, vmax=max_sev)
        cmap = plt.get_cmap('Reds')

        for i, j in steric_pairs:
            severity = severity_scores.get((i, j), 0.0)
            if severity < severity_threshold:
                continue  # skip weak clashes

            p1, p2 = coords[i], coords[j]
            color = cmap(norm(severity))
            lw = 1 + 4 * norm(severity)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=color, lw=lw, zorder=2)

        # Add dummy for legend only if we actually drew any
        if any(severity_scores.get((i, j), 0.0) >= severity_threshold for (i, j) in steric_pairs):
            dummy_clash, = ax.plot([], [], [], color=cmap(1.0), lw=3)
            legend_handles.append(dummy_clash)
            legend_labels.append('Steric Clashes (severity ≥ {:.2f} Å)'.format(severity_threshold))

    ax.legend(legend_handles, legend_labels)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Molecular Structure with Steric Clash Severity')
    plt.tight_layout()
    return fig



# ----------------------------- CLI entrypoint ----------------------------- #
if __name__ == "__main__":
    import argparse, json, os

    def _maybe_none(x: str):
        return None if x.strip().lower() in {"none", "null", ""} else x

    def _parse_usercatoms(s: str):
        """
        Accepts:
          - "None" / "none" / ""  -> None
          - JSON or Python-like lists, e.g. "[[0,1,2,3,4,5]]" or "[0,1]"
        """
        s = s.strip()
        if s.lower() in {"", "none", "null"}:
            return None
        try:
            # First try JSON
            return json.loads(s)
        except Exception:
            # Fallback to Python literal (since upstream uses parse_bracketed_list anyway)
            from ast import literal_eval
            return literal_eval(s)

    parser = argparse.ArgumentParser(
        description="Build a coordination complex from ligands (wrapper around generate_complex)."
    )

    # Repeated flags collect into lists in positional order
    parser.add_argument("--ligand", dest="ligands", action="append", required=True,
                        help="Ligand identifier or SMILES (repeat flag for multiple ligands).")
    parser.add_argument("--usercatoms", dest="usercatoms", action="append", default=None,
                        help='Coordinating atom indices per ligand (e.g. "[[0,1,2]]"). '
                             'Use "None" to auto-detect / dictionary.')
    parser.add_argument("--occupancy", dest="occupancies", action="append", type=int, default=None,
                        help="Occupancy per ligand (repeat to match --ligand count).")
    parser.add_argument("--isomer", dest="isomers", action="append", default=None,
                        help='Isomer tag per ligand (or "None").')

    # Common build knobs (pass-through to generate_complex)
    parser.add_argument("--metal", default="Fe")
    parser.add_argument("--geometry", default="octahedral")
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--vdw-scale", type=float, default=0.8)
    parser.add_argument("--clash-weight", type=float, default=10.0)
    parser.add_argument("--nudge-alpha", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--ff-name", default="UFF")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--smart-generation", action="store_true", default=True)
    parser.add_argument("--no-smart-generation", dest="smart_generation", action="store_false")

    # Orientation terms
    parser.add_argument("--orientation-weight", type=float, default=6.0)
    parser.add_argument("--orientation-k-neighbors", type=int, default=4)
    parser.add_argument("--orientation-hinge", type=float, default=0.5)
    parser.add_argument("--orientation-cap", type=float, default=1.0)

    # Haptic multi-bond behavior
    parser.add_argument("--multibond-haptics", action="store_true", default=True)
    parser.add_argument("--no-multibond-haptics", dest="multibond_haptics", action="store_false")
    parser.add_argument("--multibond-bond-order", type=int, default=1)
    parser.add_argument("--multibond-prefer-nearest-metal", action="store_true", default=True)
    parser.add_argument("--no-multibond-prefer-nearest-metal", dest="multibond_prefer_nearest_metal", action="store_false")

    # Visualization/output
    parser.add_argument("--vis-save-dir", default=None)
    parser.add_argument("--vis-stride", type=int, default=1)
    parser.add_argument("--vis-view", default="22,-60",
                        help="Comma-separated elevation,azimuth (e.g., '22,-60').")
    parser.add_argument("--vis-prefix", default="kabsch")

    parser.add_argument("--out", default=None, help="Path to write the final molecule (e.g., complex.mol2).")
    parser.add_argument("--out-format", default=None,
                        help="Open Babel output format (e.g., mol2, sdf, xyz). "
                             "If omitted, inferred from --out extension.")

    args = parser.parse_args()

    # Normalize lists: usercatoms/occupancies/isomers can be missing; create_ligand_list handles None
    ligands: List[str] = args.ligands
    usercatoms_list: Optional[List[Any]]
    if args.usercatoms is not None:
        usercatoms_list = [_parse_usercatoms(_maybe_none(s)) for s in args.usercatoms]
    else:
        usercatoms_list = None

    occupancies = args.occupancies if args.occupancies is not None else None
    isomers = [ _maybe_none(s) for s in args.isomers ] if args.isomers is not None else None

    # Sanity: lengths (create_ligand_list will assert too)
    if usercatoms_list is not None and len(usercatoms_list) != len(ligands):
        parser.error("--usercatoms count must match --ligand count.")
    if occupancies is not None and len(occupancies) != len(ligands):
        parser.error("--occupancy count must match --ligand count.")
    if isomers is not None and len(isomers) != len(ligands):
        parser.error("--isomer count must match --ligand count.")

    # Build ligand tuples via existing helper
    ligand_list = create_ligand_list(
        userligand_list=ligands,
        usercatoms_list=usercatoms_list,
        occupancy_list=occupancies,
        isomer_list=isomers
    )

    # Parse vis view tuple
    try:
        elev, azim = [float(x) for x in args.vis_view.split(",")]
        vis_view = (elev, azim)
    except Exception:
        vis_view = (22, -60)

    # Run build
    mol = generate_complex(
        ligand_list,
        metals=args.metal,
        voxel_size=args.voxel_size,
        vdw_scale=args.vdw_scale,
        clash_weight=args.clash_weight,
        nudge_alpha=args.nudge_alpha,
        geometry=args.geometry,
        coords=None,
        max_steps=args.max_steps,
        ff_name=args.ff_name,
        vis_save_dir=args.vis_save_dir,
        vis_stride=args.vis_stride,
        vis_view=vis_view,
        vis_prefix=args.vis_prefix,
        manual=False,
        manual_list=None,
        smart_generation=args.smart_generation,
        verbose=args.verbose,
        orientation_weight=args.orientation_weight,
        orientation_k_neighbors=args.orientation_k_neighbors,
        orientation_hinge=args.orientation_hinge,
        orientation_cap=args.orientation_cap,
        multibond_haptics=args.multibond_haptics,
        multibond_bond_order=args.multibond_bond_order,
        multibond_prefer_nearest_metal=args.multibond_prefer_nearest_metal,
    )

    # Optional write-out using Open Babel via pybel
    if args.out:
        out_path = args.out
        out_fmt = args.out_format or os.path.splitext(out_path)[1].lstrip(".").lower() or "mol2"
        try:
            # Ensure OBMol exists and write
            mol.convert2OBMol(force_clean=True)
            ob_mol = mol.OBMol
            pb_mol = pybel.Molecule(ob_mol)
            pb_mol.write(out_fmt, out_path, overwrite=True)
            print(f"[ok] Wrote complex to {out_path} (format={out_fmt})")
        except Exception as e:
            print(f"[warn] Could not write file ({e}). Returning object only.")
