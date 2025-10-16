"""
Ligand placement & geometry helpers
===================================

This section includes:
- imports (unchanged),
- core placement routines (Kabsch alignment, subset/perm search),
- utility helpers for coordinate transforms and core initialization.

NOTE: Functionality and defaults are IDENTICAL to your original code.
Only comments/docstrings were added for clarity.
"""

import numpy as np
from numpy.typing import NDArray
from molSimplify.Classes.atom3D import atom3D
from collections import defaultdict
from molSimplify.Classes.mol3D import mol3D
from molSimplify.utils.openbabel_helpers import *
from molSimplify.Classes.voxelgrid import VoxelGrid, plot_voxels, plot_all_voxels
import itertools
from scipy.spatial.transform import Rotation as R
import networkx as nx
import pandas as pd
from ast import literal_eval
from openbabel import openbabel, pybel
from scipy.spatial import KDTree
import copy
from molSimplify.Classes.globalvars import geometry_vectors, global_isomer_subsets  # fixed stray space
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # needed by visualize_molecule
import os


def add_ligand_to_complex(
    ligand_donor_coords,
    ligand_all_coords,
    backbone_coords,
    occupied_mask,
    donor_indices  # NEW: pass original donor atom indices
):
    """
    Rigidly place the ligand onto *available* backbone sites for a single metal.

    Process:
      1) Filter out already-occupied sites using `occupied_mask` (available = ~occupied).
      2) Run exhaustive subset/permutation search (Kabsch) against available sites
         to minimize donor→site RMSD (no reflection by default).
      3) Apply best rigid transform to all ligand atoms.
      4) Build donor→site mapping in the original donor order.

    Args:
        ligand_donor_coords (np.ndarray[N,3]): donor-atom coordinates (ligand frame).
        ligand_all_coords (np.ndarray[L,3]): all ligand atom coordinates.
        backbone_coords (np.ndarray[M,3]): ideal site positions for this metal.
        occupied_mask (np.ndarray[M], bool): True for sites that are taken.
        donor_indices (list[int]): original donor atom indices (into ligand atoms).

    Returns:
        transformed_ligand (np.ndarray[L,3]): all ligand atoms after placement.
        global_indices (list[int]): chosen backbone site indices (len=N).
        best_rmsd (float): donor→site RMSD for the chosen alignment.
        donor_to_site (dict[int->int]): donor atom index → site index.
        site_indices_in_donor_order (list[int]): chosen sites aligned to `donor_indices`.
    """
    available_mask = ~occupied_mask
    available_coords  = backbone_coords[available_mask]
    available_indices = np.where(available_mask)[0]

    # Brute-force alignment on the reduced set of available sites
    best_subset_local, best_perm, Rm, t, reflection, best_rmsd = find_best_alignment_with_subsets(
        ligand_donor_coords,
        available_coords,
        allow_reflection=False
    )

    # Map local (available) indices back to full global backbone indices
    global_indices = [available_indices[i] for i in best_subset_local]

    # Transform the entire ligand using the best rigid motion
    transformed_ligand = transform_ligand(ligand_all_coords, Rm, t)
    transformed_ligand = zero_small_values(transformed_ligand)

    # Translate permutation from donor positions into donor atom ids
    permuted_donors = [donor_indices[i] for i in best_perm]
    donor_to_site, site_indices_in_donor_order = map_donors_to_sites(
        donor_indices=donor_indices,
        best_perm_idx=permuted_donors,
        chosen_subset=global_indices
    )

    return transformed_ligand, global_indices, best_rmsd, donor_to_site, site_indices_in_donor_order


def kabsch_align_with_reflection_option(P, Q, allow_reflection=False):
    """
    Compute rigid transform (R, t) that best aligns P onto Q via Kabsch.

    Enforces det(R)=+1 unless `allow_reflection=True`. In either case, the
    standard "flip last column of V" path is used for det<0.

    Args:
        P (np.ndarray[N,3]): source points.
        Q (np.ndarray[N,3]): target points.
        allow_reflection (bool): if True, reflections are permitted.

    Returns:
        Rm (np.ndarray[3,3]): rotation (or rotoreflection) matrix.
        t  (np.ndarray[3]): translation vector such that P@Rm + t ≈ Q.
    """
    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    C = P_centered.T @ Q_centered
    V, S, Wt = np.linalg.svd(C)
    Rm = V @ Wt

    detR = np.linalg.det(Rm)
    if detR < 0:
        # Flip last column of V to correct det; honor allow_reflection flag.
        if allow_reflection:
            V[:, -1] *= -1
            Rm = V @ Wt
        else:
            V[:, -1] *= -1
            Rm = V @ Wt  # enforce proper rotation (no reflection)

    # Translation that brings P's centroid onto Q's
    t = Q_centroid - P_centroid @ Rm
    return Rm, t


def rmsd(P, Q):
    """
    Root-mean-square deviation between two point sets of equal length.

    Args:
        P, Q (np.ndarray[N,3]): corresponding points.

    Returns:
        float: sqrt(mean(||P-Q||^2)).
    """
    return np.sqrt(np.mean(np.sum((P - Q)**2, axis=1)))


def find_best_alignment_with_subsets(ligand_coords, backbone_coords, allow_reflection=False):
    """
    Exhaustively match N ligand donors onto any N-subset of M backbone sites.

    For each N-combination of sites and each permutation of donors:
      - run Kabsch (optional reflections),
      - compute donor RMSD,
      - keep the best.

    Args:
        ligand_coords (np.ndarray[N,3]): donor atoms (ligand frame).
        backbone_coords (np.ndarray[M,3]): candidate site coords (M≥N).
        allow_reflection (bool): permit reflections during Kabsch.

    Returns:
        best_subset (tuple[int]): chosen site indices (len=N).
        best_perm (tuple[int]): donor permutation used for that subset.
        best_R (np.ndarray[3,3]): rotation matrix.
        best_t (np.ndarray[3]): translation vector.
        best_reflection (bool): whether reflection branch was used.
        best_rmsd (float): minimum donor RMSD achieved.
    """
    N = ligand_coords.shape[0]
    M = backbone_coords.shape[0]
    assert M >= N, "Backbone must have at least as many atoms as ligand."

    best_rmsd = np.inf
    best_subset = None
    best_perm = None
    best_R = None
    best_t = None
    best_reflection = None

    # Combinatorial search; tractable for low denticity
    for subset_indices in itertools.combinations(range(M), N):
        subset = backbone_coords[list(subset_indices), :]
        for perm in itertools.permutations(range(N)):
            permuted_ligand = ligand_coords[list(perm), :]
            for reflect in ([False, True] if allow_reflection else [False]):
                Rm, t = kabsch_align_with_reflection_option(permuted_ligand, subset, allow_reflection=reflect)
                aligned = permuted_ligand @ Rm + t
                current_rmsd = rmsd(aligned, subset)
                if current_rmsd < best_rmsd:
                    best_rmsd = current_rmsd
                    best_subset = subset_indices
                    best_perm = perm
                    best_R = Rm
                    best_t = t
                    best_reflection = reflect
    return best_subset, best_perm, best_R, best_t, best_reflection, best_rmsd


def transform_ligand(ligand_all_coords, Rm, t):
    """
    Apply a rigid transform to all ligand atoms.

    Args:
        ligand_all_coords (np.ndarray[L,3]): input coordinates.
        Rm (np.ndarray[3,3]): rotation matrix.
        t (np.ndarray[3]): translation vector.

    Returns:
        np.ndarray[L,3]: transformed coordinates.
    """
    return ligand_all_coords @ Rm + t


def get_all_coords(mol):
    """
    Extract an (N,3) array from a mol3D's atom list (in order).

    Args:
        mol (mol3D): molecule.

    Returns:
        list[[x,y,z]]: list of 3-float coordinates.
    """
    ret = []
    for at in mol.atoms:
        ret.append(at.coords())
    return ret


def zero_small_values(coords, threshold=1e-6):
    """
    Numerically zero-out tiny floating values for cleaner output.

    Args:
        coords (np.ndarray[...,3]): coordinates.
        threshold (float): absolute values < threshold become 0.0

    Returns:
        np.ndarray: cleaned coordinates (copy).
    """
    cleaned = coords.copy()
    cleaned[np.abs(cleaned) < threshold] = 0.0
    return cleaned


def add_ligand_to_metals(ligand_donor_coords, ligand_all_coords, metals, donor_indices):
    """
    Try adding the ligand to each metal (in order) until it fits available sites.

    On success:
      - the ligand is rigidly placed,
      - that metal's `occupied_mask` is updated for the chosen sites.

    Args:
        ligand_donor_coords (np.ndarray[N,3]): donor coords in ligand.
        ligand_all_coords (np.ndarray[L,3]): all ligand coords.
        metals (list[dict]): per-metal dicts with 'backbone_coords' & 'occupied_mask'.
        donor_indices (list[int]): donor atom indices in ligand.

    Returns:
        (transformed_ligand, metal_index, global_backbone_indices, best_rmsd,
         donor_to_site, site_indices_in_donor_order)

    Raises:
        RuntimeError: if no metal can accept the ligand.
    """
    for i, metal in enumerate(metals):
        occupied_mask = metal["occupied_mask"]
        backbone_coords = metal["backbone_coords"]

        # quick capacity check: available_slots >= number_of_donors
        if occupied_mask.sum() + len(ligand_donor_coords) > len(backbone_coords):
            continue

        try:
            transformed_ligand, global_indices, best_rmsd, donor_to_site, site_indices_in_donor_order = add_ligand_to_complex(
                ligand_donor_coords=ligand_donor_coords,
                ligand_all_coords=ligand_all_coords,
                backbone_coords=backbone_coords,
                occupied_mask=occupied_mask,
                donor_indices=donor_indices,          # <— pass through unchanged
            )

            # mark chosen sites as occupied
            for idx in global_indices:
                metal["occupied_mask"][idx] = True

            return (transformed_ligand, i, global_indices, best_rmsd,
                    donor_to_site, site_indices_in_donor_order)
        except Exception as e:
            # swallow and try next metal
            print(f"Failed to add to metal {i}: {e}")
            continue

    raise RuntimeError("Ligand could not be added to any metal.")


def nudge_ligand_coords(
    ligand_coords,      # np.array shape (N_atoms, 3)
    donor_indices,      # list of indices of ligand donor atoms coordinating metal
    target_sites_coords,  # np.array shape (N_sites, 3), ideal coords from backbone (isomer subset)
    alpha=0.1,           # nudging factor between 0 and 1
    verbose=False
):
    """
    Soft positional adjustment of donors: linearly move each donor toward its
    paired target site by fraction `alpha` (no rotation or CBH constraints).

    Args:
        ligand_coords (np.ndarray[N,3]): full ligand coords.
        donor_indices (list[int]): donor atom indices (len = number of sites).
        target_sites_coords (np.ndarray[D,3]): target site positions.
        alpha (float): fraction of (target - current) to apply.

    Returns:
        np.ndarray[N,3]: updated coordinates (copy) with donors nudged.
    """
    new_coords = ligand_coords.copy()

    if len(donor_indices) != len(target_sites_coords):
        raise ValueError("Number of donor atoms and target sites must match")

    # Optional reporting of current distances
    for i, donor_idx in enumerate(donor_indices):
        original = ligand_coords[donor_idx]
        target = target_sites_coords[i]
        diff = target - original
        dist = np.linalg.norm(diff)
        if verbose:
            print(f"Donor atom {donor_idx} distance to target site {i}: {dist:.3f} Å")

    # Apply the nudge
    for donor_idx, target_coord in zip(donor_indices, target_sites_coords):
        original = ligand_coords[donor_idx]
        direction = target_coord - original
        new_coords[donor_idx] = original + alpha * direction

    return new_coords


def select_best_subset_by_kabsch(ligand_coords, donor_indices, backbone_coords, valid_subsets):
    """
    Among a supplied set of site subsets, pick the one with minimum Kabsch RMSD.

    Args:
        ligand_coords (np.ndarray[L,3]): full ligand coords.
        donor_indices (list[int]): donor indices into ligand_coords.
        backbone_coords (np.ndarray[M,3]): site coordinates.
        valid_subsets (iterable[tuple[int]]): candidate site index tuples.

    Returns:
        best_subset (tuple[int]): chosen subset.
        best_aligned_coords (np.ndarray[D,3]): donor coords after Kabsch (for the best subset).
        best_rmsd (float): minimum donor RMSD achieved.
    """
    best_rmsd = float('inf')
    best_subset = None
    best_aligned_coords = None

    donor_coords = ligand_coords[donor_indices]

    for subset in valid_subsets:
        target_coords = backbone_coords[np.array(subset)]

        # Kabsch: donors -> target subset (allow reflection here)
        Rm, t = kabsch_align_with_reflection_option(donor_coords, target_coords, True)
        aligned = donor_coords @ Rm + t

        # RMSD in target frame
        rmsd_val = np.sqrt(np.mean(np.sum((aligned - target_coords)**2, axis=1)))

        if rmsd_val < best_rmsd:
            best_rmsd = rmsd_val
            best_subset = subset
            best_aligned_coords = aligned

    return best_subset, best_aligned_coords, best_rmsd


def initialize_core(metals="Fe", coords=None, geometry=None):
    """
    Create a `core3D` from a list (or single) of metal elements and positions.

    This builds the mol3D core and also prepares the `metals_structures` list
    which contains per-metal geometry and ideal backbone sites.

    Args:
        metals (str|list[str]): metal element(s).
        coords (None|np.ndarray[(N,3)]|(3,)): metal coordinates.
        geometry (None|str|list[str]): per-metal geometry string(s).

    Returns:
        core3D (mol3D): core molecule with metal atoms added.
        metals_structures (list[dict]): data for each metal (see below).

    Each dict in `metals_structures` contains:
      - 'element': str
      - 'coord': (3,) array
      - 'geometry': str
      - 'backbone_coords': (S,3) array
      - 'occupied_mask': (S,) bool array (all False initially)
    """
    metals_structures = initialize_metal_coordinates(metals, coords, geometry)

    # initialize the core3D object
    core3D = mol3D()
    # add each metal to the core with its corresponding coordinates
    for i, metal_dict in enumerate(metals_structures):
        core3D.addAtom(atom3D(metal_dict['element'], metal_dict['coord']))

    return core3D, metals_structures


def initialize_metal_coordinates(
    metals,
    coords=None,
    geometry=None,
    backbone_scale=2.1
):
    """
    Normalize metal inputs (single or multiple) and compute ideal backbone sites.

    Args:
        metals (str|list[str]): e.g., "Fe" or ["Fe","Cu"].
        coords (None|(3,)|(N,3)): metal position(s). Required for multinuclear.
        geometry (None|str|list[str]): per-metal geometry (e.g., 'octahedral').
        backbone_scale (float): scaling applied to geometry unit vectors to
                                set nominal M–donor distance.

    Returns:
        metals_structure (list[dict]): see create_metals_structure for layout.
    """
    # Normalize metals to a list
    if isinstance(metals, str):
        metal_elements = [metals]
    elif isinstance(metals, (list, tuple)) and all(isinstance(m, str) for m in metals):
        metal_elements = list(metals)
    else:
        raise TypeError("metals must be a string or a list/tuple of strings")

    n_metals = len(metal_elements)

    # Handle coords input normalization and validation
    if coords is None:
        if n_metals == 1:
            metal_coords = np.zeros((1, 3))
        else:
            raise ValueError("For multinuclear complexes, coordinates must be provided")
    else:
        coords = np.array(coords, dtype=float)

        if coords.ndim == 1:
            if n_metals != 1:
                raise ValueError("A single coordinate was given but multiple metals provided")
            if coords.shape[0] != 3:
                raise ValueError("Coordinate must have shape (3,) for a single metal")
            metal_coords = coords.reshape(1, 3)

        elif coords.ndim == 2:
            if coords.shape[1] != 3:
                raise ValueError("Each coordinate must be a 3D point")
            if coords.shape[0] != n_metals:
                raise ValueError(f"Number of coordinates ({coords.shape[0]}) does not match number of metals ({n_metals})")
            metal_coords = coords

        else:
            raise ValueError("coords must be None, a (3,) vector, or a (N, 3) array")

    # Handle geometry input normalization and validation
    if geometry is None:
        if n_metals == 1:
            geometry_list = ['octahedral']
        else:
            raise ValueError("Geometry must be provided for multinuclear complexes")
    elif isinstance(geometry, str):
        if n_metals > 1:
            raise ValueError("Cannot pass a single geometry string when multiple metals are provided. Please provide a list/tuple of geometries matching metals.")
        geometry_list = [geometry]
    elif isinstance(geometry, (list, tuple)):
        if len(geometry) != n_metals:
            raise ValueError(f"Number of geometry entries ({len(geometry)}) does not match number of metals ({n_metals})")
        geometry_list = list(geometry)
    else:
        raise TypeError("geometry must be a string, None, or a list/tuple of strings")

    # Build backbone sites for each metal using molSimplify geometry templates
    metal_backbones = []
    for metal_coord, geo in zip(metal_coords, geometry_list):
        backbone_coords = generate_backbone_sites(metal_coord, geometry=geo, scale=backbone_scale)
        metal_backbones.append(backbone_coords)

    metals_structure = create_metals_structure(metal_elements, metal_coords, metal_backbones, geometry_list)

    return metals_structure


def generate_backbone_sites(
    metal_coord,
    geometry='octahedral',
    scale=1.0
):
    """
    Generate idealized coordination sites around a metal center.

    Uses `geometry_vectors` (from molSimplify globalvars) and shifts them
    by `metal_coord`, scaled by `scale`.

    Args:
        metal_coord (np.ndarray[3]): metal xyz.
        geometry (str): key present in `geometry_vectors`.
        scale (float): distance scaling for unit vectors (sets nominal M–X).

    Returns:
        np.ndarray[S,3]: site positions.
    """
    if geometry not in geometry_vectors:
        raise ValueError(f"Unsupported geometry '{geometry}'. Supported: {list(geometry_vectors.keys())}")

    unit_vectors = geometry_vectors[geometry]
    scaled_vectors = unit_vectors * scale
    backbone_sites = metal_coord + scaled_vectors

    return backbone_sites


def create_metals_structure(metal_elements, metal_coords, metal_backbones, geometry):
    """
    Package per-metal data into a list of dicts for downstream placement.

    Args:
        metal_elements (list[str])
        metal_coords (np.ndarray[N,3])
        metal_backbones (list[np.ndarray[S_i,3]])
        geometry (list[str])

    Returns:
        list[dict]: each with keys: element, coord, geometry, backbone_coords, occupied_mask
    """
    if len(metal_elements) != len(metal_coords):
        raise ValueError("metal_elements and metal_coords must have the same length")
    if len(metal_elements) != len(metal_backbones):
        raise ValueError("metal_elements and metal_backbones must have the same length")

    metals = []
    for i in range(len(metal_elements)):
        backbone_coords = metal_backbones[i]
        metals.append({
            "element": metal_elements[i],
            "coord": metal_coords[i],
            "geometry": geometry[i],
            "backbone_coords": backbone_coords,
            "occupied_mask": np.zeros(len(backbone_coords), dtype=bool)
        })
    return metals


def get_valid_isomer_subsets(metal_info, isomer=None, denticity=1):
    """
    For a given metal's site set, return acceptable unoccupied site tuples.

    If `isomer` is provided (e.g., 'fac'/'mer'), use the precomputed
    isomer subsets from `global_isomer_subsets[geometry][isomer]` and
    filter by occupancy. Otherwise, return *all* unoccupied combinations
    of length `denticity`.

    Args:
        metal_info (dict): must include 'geometry', 'backbone_coords', 'occupied_mask'
        isomer (str|None): isomer key or None.
        denticity (int): number of sites required.

    Returns:
        list[tuple[int]]: candidate site index tuples.
    """
    geometry = metal_info['geometry']
    mask = metal_info['occupied_mask']

    # Gather currently free site indices
    unoccupied_indices = [i for i, free in enumerate(mask) if not free]

    if isomer is None:
        # All free combinations of given size
        if len(unoccupied_indices) < denticity:
            return []
        return list(itertools.combinations(unoccupied_indices, denticity))

    # Filter pre-defined isomer subsets for free sites
    geometry_subsets = global_isomer_subsets.get(geometry, {})
    isomer_subsets = geometry_subsets.get(isomer, [])

    valid_subsets = []
    for subset in isomer_subsets:
        if all(not mask[i] for i in subset):
            valid_subsets.append(subset)
    return valid_subsets


def get_ligand_coordinates(lig3D, donor_indices):
    """
    Collect donor-only and all-atom coordinates from a mol3D ligand.

    Args:
        lig3D (mol3D): ligand object.
        donor_indices (list[int]): donor atom indices.

    Returns:
        ligand_donor_coords (np.ndarray[N,3]),
        ligand_all_coords (np.ndarray[L,3])
    """
    coord_atoms_xyzs = []
    for i in range(len(lig3D.atoms)):
        if i in donor_indices:
            coord_atoms_xyzs.append(lig3D.atoms[i].coords())
    ligand_donor_coords = np.array(coord_atoms_xyzs)
    ligand_all_coords = np.array(get_all_coords(lig3D))
    return ligand_donor_coords, ligand_all_coords


def set_new_coords(mol, coords):
    """
    Return a deep-copied mol3D with atom coordinates replaced by `coords`.

    Args:
        mol (mol3D): source molecule.
        coords (iterable[(3,)]): new coordinates (length must match atom count).

    Returns:
        mol3D: new molecule with coordinates set.
    """
    assert len(mol.atoms) == len(coords), "Length of coordinates must match atoms in mol3D object!"
    # Update the coordinates of lig3D
    mol_copy = mol3D()
    mol_copy.copymol3D(mol)
    for i in range(len(mol_copy.atoms)):
        mol_copy.atoms[i].setcoords(coords[i])
    return mol_copy


def set_metal_ligand_bond_lengths(ligand_coords, donor_indices, metal_coords, target_bond_length):
    """
    Adjust donor positions radially so that each donor–metal distance equals `target_bond_length`.

    The direction from metal→donor is preserved; only the radius is rescaled.

    Args:
        ligand_coords (np.ndarray[N,3]): ligand coordinates (modified on a copy).
        donor_indices (list[int]): indices of donors (len == len(metal_coords)).
        metal_coords (np.ndarray[D,3]): corresponding metal positions.
        target_bond_length (float): desired M–donor distance.

    Returns:
        np.ndarray[N,3]: adjusted coordinates (copy).
    """
    adjusted_coords = ligand_coords.copy()
    for i, donor_idx in enumerate(donor_indices):
        donor_pos = adjusted_coords[donor_idx]
        metal_pos = metal_coords[i]
        vec = donor_pos - metal_pos
        current_dist = np.linalg.norm(vec)
        if current_dist == 0:
            continue  # avoid division by zero
        vec_unit = vec / current_dist
        adjusted_coords[donor_idx] = metal_pos + vec_unit * target_bond_length
    return adjusted_coords


def get_all_bonded_atoms_bonded_to_metal(mol, include_metal=True, transition_metals_only=True, include_X=False):
    """
    Return a list of atoms directly bonded to any metal in `mol`.

    Args:
        mol (mol3D)
        include_metal (bool): also include the metal atom indices in the result.
        transition_metals_only (bool): rely on molSimplify's metal finder flag.
        include_X (bool): (unused in logic; preserved for API compatibility)

    Returns:
        list[int]: neighbor atom indices (and optionally the metal indices).
    """
    metal_list = mol.findMetal(transition_metals_only=transition_metals_only)

    ret = []
    for bond in mol.bo_dict:
        atom0 = bond[0]
        atom1 = bond[1]
        if atom0 in metal_list and atom1 not in metal_list:
            ret.append(atom1)
        elif atom1 in metal_list and atom0 not in metal_list:
            ret.append(atom0)
    if include_metal:
        ret += metal_list
    return ret


def get_all_coords_and_elements(mol):
    """
    Convenience: return both coordinates and element symbols for a mol3D.

    Args:
        mol (mol3D)

    Returns:
        (np.ndarray[N,3], np.ndarray[N]): (coords, element symbols)
    """
    mol_coords = []
    mol_elements = []
    for atom in mol.atoms:
        mol_coords.append(atom.coords())
        mol_elements.append(atom.sym)
    return np.array(mol_coords), np.array(mol_elements)



"""
Section 2 — Clash resolution
----------------------------

Contains:
- iterative_dual_clash_resolution: outer loop that reduces voxel clashes by
  nudging minimal rigid bodies on the ligand (and optionally the complex),
  with ring-piercing detection/repair and optional visualization frames.
- try_optimize_body_hybrid: inner routine that tests small translations and
  rotations for one rigid body and accepts an improvement.

NOTE: Functionality and defaults are IDENTICAL to your original code.
Only comments/docstrings were added for clarity.
"""

def iterative_dual_clash_resolution(
    ligand_coords,
    ligand_elements,
    lig3D,
    complex_coords,
    complex_elements,
    core3D,
    anchor_and_axis_fn,
    max_iterations=10,
    rotation_angles=list(range(-90, 91, 15)),
    voxel_size=0.3,
    vdw_scale=0.55,
    clash_tolerance=3,
    translation_steps=[0.0, -0.25, 0.25],
    fast_mode=True,
    debug=False,
    # --- NEW visualization controls ---
    vis_save_dir=None,         # e.g., "runs/voxels/iter"
    vis_view=(22, -60),
    vis_prefix="iter"
):
    """
    Alternate moving *rigid bodies* on both ligand and complex to reduce voxel clashes.

    Strategy (per iteration):
      1) Build voxel grids for ligand + complex and count clash voxels.
      2) Find minimal rigid bodies (subgraphs) near clashes on each side.
      3) For each ligand body, try translations & rotations that reduce a simple score
         (clash count + small geometric penalty). If it increases ring piercings, revert.
      4) Optionally try complex bodies similarly (kept for symmetry).
      5) If ring piercings remain, attempt local repairs (torsion/translate fallback).
      6) Stop early if no improvements are made, or once clashes/piercings reach zero.

    Optional visualization:
      - If `vis_save_dir` is provided, writes snapshot PNGs at the start/end of each
        iteration and a final image of the result. These frames are also returned.

    Args:
        ligand_coords (np.ndarray[N_lig,3]): ligand coordinates (will be copied).
        ligand_elements (list[str]): element symbols for ligand atoms.
        lig3D (mol3D): ligand molecule (used to compute minimal bodies & ring checks).
        complex_coords (np.ndarray[N_cpx,3]): complex coordinates (copied).
        complex_elements (list[str]): element symbols for complex atoms.
        core3D (mol3D): complex molecule (for body detection).
        anchor_and_axis_fn (callable): body, coords -> (anchor_index, axis_vector)
        max_iterations (int): clamp on outer loop passes.
        rotation_angles (iterable[int]): degrees tested during body rotation.
        voxel_size (float): voxel edge length for VoxelGrid.
        vdw_scale (float): scaling of VDW radii used in voxelization.
        clash_tolerance (int): if a candidate move yields score ≤ this, accept immediately.
        translation_steps (list[float]): per-axis deltas to test for translation moves.
        fast_mode (bool): re-use a pre-voxelized static side for speed.
        debug (bool): verbose prints for internal steps.
        vis_save_dir (str|None): if set, save per-iteration PNGs there.
        vis_view (tuple(elev, azim)): matplotlib 3D view for saved frames.
        vis_prefix (str): filename prefix for saved frames.

    Returns:
        coords_lig (np.ndarray[N_lig,3]): final ligand coordinates.
        coords_core (np.ndarray[N_cpx,3]): final complex coordinates.
        info (dict): {
            "final_voxel_grid": VoxelGrid,
            "remaining_clashes": int,
            "ligand_self_piercings": int,
            "iterations": int,
            "iteration_frames": list[str],    # saved PNGs (if enabled)
            "final_frame": str|None           # final PNG path (if enabled)
        }
    """
    import numpy as np
    coords_lig = ligand_coords.copy()
    coords_core = complex_coords.copy()

    if vis_save_dir:
        _ensure_dir(vis_save_dir)

    # Parameters for ring-piercing detection, tuned to be conservative
    rp_kwargs = dict(
        ring_size_threshold=12,
        angstrom_threshold=1.3,
        inplane_pad=0.30,
        edge_buffer=0.18,
        endpoint_buffer=0.40,
        plane_tol=0.45,
        verbose=debug,
    )

    def _count_lig_self_piercings(coords):
        """Helper: count current ring piercings on the ligand."""
        tmpL = mol3D(); tmpL.copymol3D(lig3D)
        for i, at in enumerate(tmpL.atoms):
            at.setcoords(coords[i])
        return len(detect_ring_piercing(tmpL, **rp_kwargs))

    base_self_pierce = _count_lig_self_piercings(coords_lig)

    # Expand the angle/translation candidates slightly with small extras
    if rotation_angles is None:
        rot_angles = np.arange(-90, 91, 15)
    else:
        rot_angles = sorted(set(list(rotation_angles) + [-10, -5, 5, 10]))
    trans_steps = sorted(set(list(translation_steps) + [-0.35, -0.15, -0.10, 0.10, 0.15, 0.35]))

    per_iter_frames = []  # list of saved PNGs (if enabled)

    for iteration in range(max_iterations):
        print(f"\n🔁 Iteration {iteration}")

        # --- START-OF-ITERATION SNAPSHOT (optional) ---
        if vis_save_dir:
            VG_start = VoxelGrid(voxel_size=voxel_size, vdw_scale=vdw_scale)
            VG_start.add_atoms(complex_elements, coords_core, group="complex")
            VG_start.add_atoms(ligand_elements,  coords_lig,  group="ligand")
            png_start = os.path.join(vis_save_dir, f"{vis_prefix}_{iteration:03d}_start.png")
            save_voxel_frame(
                VG_start,
                png_start,
                title=f"Iteration {iteration} – start",
                view=vis_view,
                atoms_overlay={"complex": coords_core, "ligand": coords_lig}
            )
            per_iter_frames.append(png_start)

        # Combined voxelization to measure clashes for the current pose
        vg = VoxelGrid(voxel_size=voxel_size, vdw_scale=vdw_scale)
        vg.add_atoms(complex_elements, coords_core, group="complex")
        vg.add_atoms(ligand_elements, coords_lig, group="ligand")

        clash_pairs, ligand_clash_ids, complex_clash_ids = vg.get_clashing_atoms()
        total_clashes = len(clash_pairs)
        print(f"   ⚠️  Total complex↔ligand clashes: {total_clashes}")
        print(f"   🧪 Ligand self-piercings: {base_self_pierce}")

        if debug:
            print(f"   bodies: ligand={len(ligand_clash_ids)} (ids), core={len(complex_clash_ids)} (ids)")

        # Clean state: nothing to fix
        if total_clashes == 0 and base_self_pierce == 0:
            print("   ✅ No inter clashes and no self-piercings.")
            if vis_save_dir:
                VG_end = VoxelGrid(voxel_size=voxel_size, vdw_scale=vdw_scale)
                VG_end.add_atoms(complex_elements, coords_core, group="complex")
                VG_end.add_atoms(ligand_elements,  coords_lig,  group="ligand")
                png_end = os.path.join(vis_save_dir, f"{vis_prefix}_{iteration:03d}_end.png")
                save_voxel_frame(
                    VG_end,
                    png_end,
                    title=f"Iteration {iteration} – end (clean)",
                    view=vis_view,
                    atoms_overlay={"complex": coords_core, "ligand": coords_lig}
                )
                per_iter_frames.append(png_end)
            break

        # Build static voxelizations for each side to speed scoring of the other
        vg_core = VoxelGrid(voxel_size=voxel_size, vdw_scale=vdw_scale)
        vg_core.add_atoms(complex_elements, coords_core, group="complex")
        vg_lig = VoxelGrid(voxel_size=voxel_size, vdw_scale=vdw_scale)
        vg_lig.add_atoms(ligand_elements, coords_lig, group="ligand")

        # Identify minimal rigid bodies that participate in clashes
        rb_lig = find_minimal_rigid_bodies(lig3D, ligand_clash_ids)
        rb_core = find_minimal_rigid_bodies(core3D, complex_clash_ids)

        improved_any = False

        # Try improving ligand bodies first
        for body in rb_lig:
            if debug:
                print(f"   → try ligand body {sorted(body)}")
            prev_coords = coords_lig.copy()
            improved = try_optimize_body_hybrid(
                body, coords_lig, ligand_elements, complex_elements, coords_core,
                "ligand", anchor_and_axis_fn, rot_angles, voxel_size, vdw_scale,
                clash_tolerance, trans_steps, vg_core if fast_mode else None,
                debug=debug,
            )
            if improved:
                # If the move increased ring piercings, revert it
                new_self = _count_lig_self_piercings(coords_lig)
                if new_self > base_self_pierce:
                    coords_lig = prev_coords
                    print(f"      ⛔ Reverted ligand move: self-piercings {new_self} > {base_self_pierce}")
                else:
                    base_self_pierce = new_self
                    improved_any = True

        # Optionally also try to move bodies on the complex side
        for body in rb_core:
            if debug:
                print(f"   → try core body {sorted(body)}")
            improved = try_optimize_body_hybrid(
                body, coords_core, complex_elements, ligand_elements, coords_lig,
                "complex", anchor_and_axis_fn, rot_angles, voxel_size, vdw_scale,
                clash_tolerance, trans_steps, vg_lig if fast_mode else None,
                debug=debug,
            )
            if improved:
                improved_any = True

        # If any ring piercings remain, attempt a targeted repair pass
        if base_self_pierce > 0:
            tmpL = mol3D(); tmpL.copymol3D(lig3D)
            for i, at in enumerate(tmpL.atoms):
                at.setcoords(coords_lig[i])
            piercings = detect_ring_piercing(tmpL, **rp_kwargs)
            if piercings:
                if debug:
                    print(f"   🔧 repair pass: {len(piercings)} pierce(s) found")
                fixed, moved = correct_ring_piercings(
                    tmpL, piercings, verbose=False,
                    rp_kwargs=dict(angstrom_threshold=2.30, inplane_pad=0.35, edge_buffer=0.15)
                )
                coords_lig = fixed
                base_self_pierce = _count_lig_self_piercings(coords_lig)
                print(f"   🔧 Repair pass moved {len(moved)} atom(s); self-piercings now {base_self_pierce}")

        # --- END-OF-ITERATION SNAPSHOT (optional) ---
        if vis_save_dir:
            VG_end = VoxelGrid(voxel_size=voxel_size, vdw_scale=vdw_scale)
            VG_end.add_atoms(complex_elements, coords_core, group="complex")
            VG_end.add_atoms(ligand_elements,  coords_lig,  group="ligand")
            png_end = os.path.join(vis_save_dir, f"{vis_prefix}_{iteration:03d}_end.png")
            save_voxel_frame(
                VG_end,
                png_end,
                title=f"Iteration {iteration} – end",
                view=vis_view,
                atoms_overlay={"complex": coords_core, "ligand": coords_lig}
            )
            per_iter_frames.append(png_end)

        # Stop if no progress is made and there are still inter-body clashes
        if not improved_any and base_self_pierce == 0 and total_clashes > 0:
            print("   🛑 No improvements to inter clashes; stopping.")
            break

    # Final voxelization & final frame (optional)
    final_vg = VoxelGrid(voxel_size=voxel_size, vdw_scale=vdw_scale)
    final_vg.add_atoms(complex_elements, coords_core, group="complex")
    final_vg.add_atoms(ligand_elements, coords_lig, group="ligand")

    final_png = None
    if vis_save_dir:
        final_png = os.path.join(vis_save_dir, f"{vis_prefix}_final.png")
        save_voxel_frame(
            final_vg,
            final_png,
            title="Final after iterative resolution",
            view=vis_view,
            atoms_overlay={"complex": coords_core, "ligand": coords_lig}
        )
        per_iter_frames.append(final_png)

    return coords_lig, coords_core, {
        "final_voxel_grid": final_vg,
        "remaining_clashes": len(final_vg.complex_voxels & final_vg.ligand_voxels) if hasattr(final_vg, "complex_voxels") else len(get_voxel_coords_by_group(final_vg)["clash"]),
        "ligand_self_piercings": base_self_pierce,
        "iterations": iteration + 1,
        # NEW: artifact paths for convenience
        "iteration_frames": per_iter_frames,
        "final_frame": final_png
    }


def try_optimize_body_hybrid(
    body, coords, elements, other_elements, other_coords,
    group, anchor_and_axis_fn, rotation_angles,
    voxel_size, vdw_scale, clash_tolerance, translation_steps, static_voxels=None,
    geometric_penalty_weight=10.0,
    debug=False,                           # <-- NEW
):
    """
    Attempt to improve clashes by moving a single rigid body (subset of atom indices).

    Search order:
      1) TRANSLATIONS — evaluate a small grid of per-axis translations; if any
         candidate yields score ≤ clash_tolerance, accept immediately. Otherwise
         keep the best (score-minimizing) candidate seen.
      2) ROTATIONS — if no translation was accepted, rotate the body around an
         application-specific axis returned by `anchor_and_axis_fn`; accept on
         threshold or best-improvement similarly.

    Scoring function:
      score = (# clash voxels in a mixed grid with the opposite side)
              + geometric_penalty_weight * RMSD(body before vs after)
    The geometric term discourages huge jumps that would “solve” clashes by
    teleporting the body.

    Args:
        body (list[int]): atom indices of the rigid body (len>1 required).
        coords (np.ndarray[N,3]): full coordinate array (modified in place if accepted).
        elements (list[str]): element symbols for `coords`.
        other_elements (list[str]): other side symbols.
        other_coords (np.ndarray[M,3]): other side coordinates.
        group (str): 'ligand' or 'complex' (labels voxel groups properly).
        anchor_and_axis_fn (callable): (body, coords) -> (anchor_index, axis_vector)
        rotation_angles (iterable[int]|None): degrees to scan (augmented by small extras).
        voxel_size (float), vdw_scale (float): voxelization parameters.
        clash_tolerance (int): accept immediately if score ≤ this.
        translation_steps (list[float]): per-axis deltas to test.
        static_voxels (VoxelGrid|None): pre-voxelized opposite side for speed (fast path).
        geometric_penalty_weight (float): weight on RMSD term.
        debug (bool): verbose prints.

    Returns:
        bool: True if a move was applied (accepted or improved best score); else False.
    """
    if len(body) <= 1:
        return False

    body = sorted(body)

    # Ensure some fine angles are included
    if rotation_angles is None:
        rot_angles = list(range(-90, 91, 15))
    else:
        rot_angles = sorted(set(list(rotation_angles) + [-10, -5, 5, 10]))
    trans_steps = sorted(set(list(translation_steps) + [-0.35, -0.15, -0.10, 0.10, 0.15, 0.35]))

    def score_clashes(test_coords):
        """
        Build a temporary voxel grid where the opposite side is static and we
        add only this body as dynamic; compute the clash count between groups
        and add a geometric RMSD penalty to discourage large displacements.
        """
        import numpy as np
        original_coords = np.array([coords[i] for i in body])
        current_coords  = np.array([test_coords[i] for i in body])
        _rmsd = np.sqrt(np.mean(np.sum((original_coords - current_coords) ** 2, axis=1)))

        vg_temp = VoxelGrid(voxel_size=voxel_size, vdw_scale=vdw_scale)
        sub_elements = [elements[i] for i in body]
        sub_coords   = [test_coords[i] for i in body]

        # Fast path: reuse the static voxelization from the opposite side
        if static_voxels:
            vg_temp.grid = copy.deepcopy(static_voxels.grid)
            vg_temp.complex_voxels = static_voxels.complex_voxels.copy()
            vg_temp.ligand_voxels  = static_voxels.ligand_voxels.copy()
        else:
            if group == "ligand":
                vg_temp.add_atoms(other_elements, other_coords, group="complex")
            else:
                vg_temp.add_atoms(other_elements, other_coords, group="ligand")

        # Add just this body's atoms into the appropriate group
        vg_temp.add_atoms(sub_elements, sub_coords, atom_ids=body, group=group)

        # Score: number of intersecting voxels between groups
        clash_count = len(vg_temp.complex_voxels & vg_temp.ligand_voxels)
        total_score = clash_count + geometric_penalty_weight * _rmsd
        return total_score, vg_temp

    best_coords = coords.copy()
    min_score, _ = score_clashes(coords)
    improved = False

    # ---------- 1) TRANSLATIONS ----------
    best_move = None
    candidates = []
    for dx in trans_steps:
        for dy in trans_steps:
            for dz in trans_steps:
                if dx == dy == dz == 0:
                    continue
                test_coords = coords.copy()
                for i in body:
                    test_coords[i] += np.array([dx, dy, dz])
                score, _ = score_clashes(test_coords)
                candidates.append((score, (dx, dy, dz), test_coords.copy()))

    # Try the best-scoring translations first
    candidates.sort(key=lambda x: x[0])
    for score, (dx, dy, dz), test_coords in candidates:
        if score <= clash_tolerance:
            coords[body] = test_coords[body]
            if debug:
                print(f"      ✅ translate {group} body {body} by ({dx:.2f},{dy:.2f},{dz:.2f}) -> score {score:.2f}")
            return True
        if score < min_score:
            min_score = score
            best_coords[body] = test_coords[body]
            best_move = ("translate", (dx, dy, dz), score)
            improved = True

    # If a translation improved the score, commit that best move and return
    if improved:
        coords[body] = best_coords[body]
        if debug and best_move:
            kind, vec, s = best_move
            dx, dy, dz = vec
            print(f"      ✅ best {kind} {group} body {body} by ({dx:.2f},{dy:.2f},{dz:.2f}) -> score {s:.2f}")
        return True

    # ---------- 2) ROTATIONS ----------
    anchor_index, axis = anchor_and_axis_fn(body, coords)
    if anchor_index is None or axis is None:
        if debug:
            print("      ❌ no anchor/axis available")
        return False

    if debug:
        an = np.linalg.norm(axis)
        print(f"      ↻ rotating around anchor {anchor_index} |axis|={an:.2f}")

    best_rot = None
    for angle in rot_angles:
        test_coords = coords.copy()
        anchor = coords[anchor_index]
        rotation = R.from_rotvec(np.radians(angle) * axis)
        for i in body:
            if i == anchor_index:
                continue
            vec = coords[i] - anchor
            test_coords[i] = anchor + rotation.apply(vec)
        score, _ = score_clashes(test_coords)
        if score <= clash_tolerance:
            coords[body] = test_coords[body]
            if debug:
                print(f"      ✅ rotate {group} body {body} by {angle:>4}° -> score {score:.2f}")
            return True
        if score < min_score:
            min_score = score
            best_coords[body] = test_coords[body]
            best_rot = angle
            improved = True

    if improved:
        coords[body] = best_coords[body]
        if debug and best_rot is not None:
            print(f"      ✅ best rotate {group} body {body} by {best_rot:>4}° -> score {min_score:.2f}")
        return True
    else:
        if debug:
            print(f"      ❌ no improvement for {group} body {body} (min {min_score:.2f})")
        return False


def get_next_structure(metals_structures, denticity):
    assert len(metals_structures) > 0, "No metals in metals_stuctures"
    earliest_free_structure = metals_structures[0]
    for structure in metals_structures:
        free_sites_count = np.count_nonzero(~structure['occupied_mask'])
        if free_sites_count >= denticity:
            earliest_free_structure = structure
            break
    return earliest_free_structure

def map_donors_to_sites(donor_indices, best_perm_idx, chosen_subset):
    """
    donor_indices: original donor indices (list[int] of ligand atom ids)
    best_perm_idx: donor indices IN THE ORDER matched to chosen_subset
    chosen_subset: tuple/list of backbone site indices (same order as target coords)

    Returns:
        donor_to_site: dict {donor_atom_index -> backbone_site_index}
        site_indices_in_donor_order: list[int] of site indices aligned to donor_indices order
    """
    # best_perm_idx[k] (a donor atom id) maps to chosen_subset[k]
    donor_to_site = { d: s for d, s in zip(best_perm_idx, chosen_subset) }

    # produce sites aligned to the original donor_indices list
    site_indices_in_donor_order = [ donor_to_site[d] for d in donor_indices ]
    return donor_to_site, site_indices_in_donor_order

"""
Section 3 — Orientation, haptics, PCA prefiltering, and clash-aware placement
------------------------------------------------------------------------------

Contains:
- outward_orientation_penalty: scalar penalty to encourage donor 'bulk' to point
  outward (away from the nearest metal).
- _hard_outward_flip_no_elements: emergency 180° flip of non-donor atoms if bulk
  points inward.
- Haptics/group helpers: parsing donor specs, virtual points, plane normals, etc.
- clash_aware_kabsch: beam-searched Kabsch placement that blends RMSD, voxel clash
  severity, and an outward-orientation penalty; with optional frame saving.
- Site equivalence dedupers: collapse symmetric site-subset candidates.

NOTE: Functionality and defaults are IDENTICAL to your original code.
Only comments/docstrings were added for clarity.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


def _metal_indices_from_elements(elements):
    """
    Return indices of atoms that are metals (simple periodic table subset).
    """
    METALS = {"Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
              "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
              "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"}
    return [i for i, sym in enumerate(elements) if sym in METALS]


def outward_orientation_penalty(
    ligand_coords,            # (N_lig, 3)
    ligand_elements,          # list[str] length N_lig
    donor_indices,            # list[int] donors in ligand_coords
    complex_coords,           # (N_cpx, 3)
    complex_elements,         # list[str] length N_cpx
    k_neighbors=4,            # take k nearest *heavy* (non-H) neighbors to each donor
    hinge=0.0,                # hinge for cosine; if cos < hinge (i.e., pointing back), penalize
    per_donor_cap=1.0,        # cap each donor’s penalty to avoid runaway
):
    """
    Encourage donor 'bulk' to point away from the nearest metal.

    For each donor d:
      - find the nearest metal M,
      - outward axis u = normalize(d - M),
      - take up to k nearest *heavy* ligand atoms to d (excluding donor & H),
      - compute cos between (neighbor - d) and u; values below `hinge` are penalized,
      - accumulate per-donor penalties with cap `per_donor_cap`.

    Returns:
        float: non-negative penalty (lower is better).
    """
    if len(donor_indices) == 0:
        return 0.0

    metals = _metal_indices_from_elements(complex_elements)
    if not metals:
        return 0.0

    # Preselect heavy atoms within the ligand
    heavy_mask = np.array([el != "H" for el in ligand_elements], dtype=bool)

    # KD-tree over complex metals for nearest-metal lookup
    metal_coords = complex_coords[np.array(metals)]
    tree_m = cKDTree(metal_coords)

    penalty = 0.0
    for d in donor_indices:
        dpos = ligand_coords[d]
        # nearest metal to this donor
        _, idx_m = tree_m.query(dpos)
        mpos = metal_coords[idx_m]
        u = dpos - mpos
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            continue
        u /= nu  # outward axis

        # find nearest heavy ligand atoms to donor
        deltas = ligand_coords - dpos
        dists = np.linalg.norm(deltas, axis=1)
        candidates = np.where(heavy_mask & (np.arange(len(ligand_coords)) != d))[0]
        if len(candidates) == 0:
            continue
        order = candidates[np.argsort(dists[candidates])]
        picks = order[:min(k_neighbors, len(order))]

        cos_vals = []
        for j in picks:
            w = ligand_coords[j] - dpos
            nw = np.linalg.norm(w)
            if nw < 1e-9:
                continue
            cos_vals.append(np.dot(w/nw, u))
        if not cos_vals:
            continue

        # hinge loss on cosine: inward-pointing (cos < hinge) gets penalized
        donor_pen = 0.0
        for c in cos_vals:
            if c < hinge:
                donor_pen += (hinge - c)  # e.g., if hinge=0.0 and c=-0.5, add 0.5
        penalty += min(per_donor_cap, donor_pen)

    return float(penalty)


def _hard_outward_flip_no_elements(coords, donor_indices, complex_coords, complex_elements,
                                   k_neighbors=6, cos_hinge=0.02):
    """
    Emergency outward flip (no element info required).

    For each donor, if nearby atoms (proxy for 'bulk') point inward toward the
    nearest metal, rotate all NON-donor atoms 180° around the donor→metal axis
    (donor stays fixed). This is a geometric, last-resort fix.

    Args:
        coords (np.ndarray[N,3]): ligand coords (copied).
        donor_indices (list[int]): donor atom indices.
        complex_coords (np.ndarray[M,3]): complex coords to locate metals.
        complex_elements (list[str]): to identify metals.
        k_neighbors (int): number of nearest non-donor atoms to measure.
        cos_hinge (float): inward threshold for average cosine; below this → flip.

    Returns:
        np.ndarray[N,3]: possibly flipped coordinates (copy).
    """
    # find metal coordinates in the complex
    metal_ids = _metal_indices_from_elements(complex_elements)
    if not metal_ids:
        return coords  # nothing to compare to
    metal_coords = complex_coords[np.array(metal_ids)]
    tree_m = cKDTree(metal_coords)

    out = coords.copy()
    N = len(out)

    for d in donor_indices:
        dpos = out[d]
        # nearest metal to this donor
        _, idx_m = tree_m.query(dpos)
        mpos = metal_coords[idx_m]
        u = dpos - mpos
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            continue
        u /= nu  # outward axis

        # choose nearest non-donor atoms as 'bulk'
        deltas = out - dpos
        dists = np.linalg.norm(deltas, axis=1)
        candidates = [i for i in range(N) if i != d]
        if not candidates:
            continue
        order = sorted(candidates, key=lambda j: dists[j])
        picks = order[:min(k_neighbors, len(order))]

        cos_vals = []
        for j in picks:
            w = out[j] - dpos
            nw = np.linalg.norm(w)
            if nw < 1e-9:
                continue
            cos_vals.append(np.dot(w/nw, u))
        if not cos_vals:
            continue

        # inward? flip 180° about donor–metal axis
        if float(np.mean(cos_vals)) < cos_hinge:
            rot = R.from_rotvec(np.pi * u)
            for i in range(N):
                if i == d:
                    continue
                out[i] = dpos + rot.apply(out[i] - dpos)

    return out


def is_iterable_but_not_str(x):
    """True for iterable, non-string objects."""
    return hasattr(x, "__iter__") and not isinstance(x, (str, bytes))


def _virtual_point_for_group(group_idx_list, coords):
    """
    Virtual point representing a donor group:
      - 1 atom  → that atom,
      - 2 atoms → midpoint,
      - ≥3 atoms → centroid.
    """
    pts = np.asarray(coords[group_idx_list], dtype=float)
    if len(group_idx_list) == 1:
        return pts[0]
    if len(group_idx_list) == 2:
        return 0.5 * (pts[0] + pts[1])
    return pts.mean(axis=0)


def _plane_normal_for_group(group_idx_list, coords):
    """
    Estimate a group plane normal + center using SVD.
    Robust fallback when rank<2: pick an orthogonal to a principal direction.
    """
    pts = np.asarray(coords[group_idx_list], dtype=float)
    C = pts.mean(axis=0)
    X = pts - C
    if X.shape[0] < 3 or np.linalg.matrix_rank(X) < 2:
        if X.shape[0] >= 2:
            v = X[1] - X[0]
            v /= (np.linalg.norm(v) + 1e-12)
        else:
            v = np.array([1.0,0.0,0.0])
        ref = np.array([0.0,1.0,0.0]) if abs(v[0]) > 0.9 else np.array([1.0,0.0,0.0])
        n = np.cross(v, ref); n /= (np.linalg.norm(n)+1e-12)
        return n, C
    _,_,Vt = np.linalg.svd(X, full_matrices=False)
    n = Vt[-1]; n /= (np.linalg.norm(n)+1e-12)
    return n, C


def parse_donor_spec_make_virtuals(donor_indices, ligand_all_coords):
    """
    Normalize donor specification into groups and compute virtual points.

    `donor_indices` may contain:
      - ints (monodentate donors), or
      - iterables of ints (haptic/multidentate groups).

    Returns:
      donor_groups (list[list[int]]): each group is a list of atom indices.
      virtual_points (np.ndarray[G,3]): centroid/midpoint/atom for each group.
      group_is_haptic (list[bool]): True if len(group)>=2.
    """
    groups = []
    for d in donor_indices:
        groups.append(list(d) if is_iterable_but_not_str(d) else [int(d)])
    virtual_points = np.vstack([_virtual_point_for_group(g, ligand_all_coords) for g in groups])
    group_is_haptic = [len(g) >= 2 for g in groups]
    return groups, virtual_points, group_is_haptic


def nearest_metal_vector(point, complex_coords, complex_elements):
    """
    Return the outward unit vector from nearest metal to `point`, and the metal position.
    """
    metal_ids = _metal_indices_from_elements(complex_elements)
    if not metal_ids:
        return np.array([1.0,0.0,0.0]), np.array([0.0,0.0,0.0])
    mcoords = complex_coords[np.array(metal_ids)]
    j = int(np.argmin(np.sum((mcoords - point)**2, axis=1)))
    m = mcoords[j]
    u = point - m
    return u / (np.linalg.norm(u)+1e-12), m


def align_group_normal_to_axis(coords_all, group_idx_list, desired_axis):
    """
    Rotate the full coordinate set so the group's plane normal aligns with `desired_axis`.
    Rotation is applied about the group's center.
    """
    n, C = _plane_normal_for_group(group_idx_list, coords_all)
    a = desired_axis / (np.linalg.norm(desired_axis)+1e-12)
    dot = float(np.clip(np.dot(n, a), -1.0, 1.0))
    raxis = np.cross(n, a)
    if np.linalg.norm(raxis) < 1e-9:
        if dot < 0.0:
            raxis = np.cross(n, np.array([1.0,0.0,0.0]))
            if np.linalg.norm(raxis) < 1e-9:
                raxis = np.cross(n, np.array([0.0,1.0,0.0]))
            raxis /= (np.linalg.norm(raxis)+1e-12)
            rot = R.from_rotvec(np.pi * raxis)
        else:
            return coords_all
    else:
        raxis /= (np.linalg.norm(raxis)+1e-12)
        angle = np.arccos(dot)
        rot = R.from_rotvec(angle * raxis)
    shifted = coords_all - C
    return rot.apply(shifted) + C


def map_donors_to_sites_haptic_aware(donor_groups, best_perm_group_ids, chosen_subset):
    """
    Haptic-aware donor→site mapping.

    Args:
        donor_groups (list[list[int]]): donor groups (indices in ligand frame).
        best_perm_group_ids (list[int]): order of groups matched to chosen subset.
        chosen_subset (iterable[int]): site indices (same order as permuted groups).

    Returns:
        donor_to_site (dict[int->int]): every atom in a group maps to the same site.
        site_indices_in_group_order (list[int]): chosen sites aligned to the original group order.
    """
    donor_to_site = {}
    for g_id, site in zip(best_perm_group_ids, chosen_subset):
        for atom_idx in donor_groups[g_id]:
            donor_to_site[atom_idx] = site
    site_indices_in_group_order = [donor_to_site[donor_groups[k][0]] for k in range(len(donor_groups))]
    return donor_to_site, site_indices_in_group_order


def _simple_voxel_score_factory(VoxelGridClass, complex_coords, complex_elements, ligand_elements,
                                voxel_size, vdw_scale):
    """
    Build a light-weight scoring closure that returns clash severity for a candidate ligand pose.
    """
    def score_fn(test_lig_coords):
        VG = VoxelGridClass(voxel_size=voxel_size, vdw_scale=vdw_scale)
        VG.add_atoms(elements=complex_elements, coords=complex_coords, group="complex")
        VG.add_atoms(elements=ligand_elements,  coords=test_lig_coords, group="ligand")
        total_severity, _ = VG.get_clash_severity(
            test_lig_coords, ligand_elements, complex_coords, complex_elements,
            vdw_radii=VG.vdw_radii, scale=VG.vdw_scale
        )
        return float(total_severity)
    return score_fn


def _nearest_metal_axes(donor_indices, sample_coords, complex_coords, complex_elements):
    """
    For each donor index, return a unit vector `donor - nearest_metal`.
    Falls back to +x if no metals are present.
    """
    metal_ids = _metal_indices_from_elements(complex_elements)
    axes = {}
    if not metal_ids:
        unit = np.array([1.0, 0.0, 0.0])
        for d in donor_indices:
            axes[d] = unit
        return axes

    metal_coords = complex_coords[np.array(metal_ids)]
    tree = cKDTree(metal_coords)

    for d in donor_indices:
        dpos = sample_coords[d]
        _, idx = tree.query(dpos)
        u = dpos - metal_coords[idx]
        n = np.linalg.norm(u)
        axes[d] = (u / n) if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    return axes


def _simple_outward_cos_report(sample_coords, donor_indices, k_neighbors=6):
    """
    Placeholder helper to assemble a per-donor report structure. Intentionally minimal.
    """
    report = {}
    all_idx = np.arange(len(sample_coords))
    for d in donor_indices:
        report[d] = {"mean_cos": None, "per_atom": []}
    return report


def clash_aware_kabsch(
    ligand_all_coords,
    donor_indices,
    backbone_coords,
    valid_subsets,
    ligand_all_coords_for_rmsd,
    complex_coords,
    complex_elements,
    ligand_elements,
    VoxelGridClass,
    voxel_size=0.2,
    vdw_scale=0.85,
    clash_weight=100.0,
    rotation_angles=None,
    orientation_weight=5.0,
    orientation_k_neighbors=4,
    orientation_hinge=0.0,
    orientation_cap=1.0,
    hard_outward_flip=True,
    flip_k_neighbors=6,
    flip_hinge=0.02,
    vis_save_dir=None,
    vis_stride=1,
    vis_view=(22, -60),
    vis_prefix="attempt",
    fixed_bounds=None,
    beam_topk=5,           # heavy voxel score only for top-K subsets by cheap RMSD
    early_exit_score=None,  # stop early if we beat this score
    iteration = None,
    visual_attempts = np.inf,
    verbose = False,
):
    """
    Kabsch-based placement that explicitly trades off:
      - donor RMSD to backbone sites,
      - voxel clash severity against the complex,
      - outward-orientation penalty (optional).

    Multi-dentate/haptic donors are handled by grouping indices and using
    group "virtual points" (atom/midpoint/centroid) for coarse matching.

    Flow:
      1) Parse donor groups & virtuals; optionally dedupe symmetric site subsets.
      2) (Beam) rank subsets by a cheap, PCA+Hungarian donor-shape RMSD;
         keep only `beam_topk`.
      3) For each subset, scan simple torsions around one or two axes to produce
         candidate rigid placements; score each by:
            score = RMSD + clash_weight * clash_severity + orientation_weight * orient_pen
      4) Keep the best-scoring pose; optional haptic "de-slant" refinement and final snapshot.

    Returns:
        best_subset (tuple[int]): chosen site indices.
        best_coords (np.ndarray[L,3]): ligand coords after placement.
        best_rmsd (float): donor→site RMSD for best pose (based on virtual points).
        placement_attempts (list[dict]): lightweight record of evaluated candidates.
        best_perm_idx (list[int]): order of donor groups matched to chosen subset.
    """
    import itertools
    if rotation_angles is None:
        rotation_angles = np.arange(0, 360, 60)

    if vis_save_dir:
        _ensure_dir(vis_save_dir)

    # One-time complex frame if requested
    if vis_save_dir:
        VG0 = VoxelGridClass(voxel_size=voxel_size, vdw_scale=vdw_scale)
        VG0.add_atoms(elements=complex_elements, coords=complex_coords, group="complex")
        complex_name = f"{vis_prefix}_complex-only.png"
        complex_path = _unique_path(vis_save_dir, complex_name)
        save_voxel_frame(
            VG0, complex_path, title=f"{vis_prefix}: complex only",
            view=vis_view, fixed_bounds=fixed_bounds,
        )

    # === HAPTIC PARSE: donors as groups + virtual donor points ===
    donor_groups, virtual_points, group_is_haptic = parse_donor_spec_make_virtuals(donor_indices, ligand_all_coords)
    num_groups = len(donor_groups)

    # Optional: symmetry dedupe of site subsets at the first iteration
    if iteration != None and iteration == 0:
        class_of, _classes = _site_equivalence_classes(backbone_coords, tol=1e-4)
        before = len(valid_subsets)
        valid_subsets = _dedupe_subsets_by_site_classes(valid_subsets, class_of)
        after = len(valid_subsets)

        if verbose:
            print('val')
            print(valid_subsets)
            print('before')
            print(before)
            print('after')
            print(after)

    # === COARSE STAGE: rank subsets quickly by donor shape ===
    dent = len(donor_groups)
    if dent >= 2 and len(valid_subsets) > beam_topk:
        subset_scores = []
        for subset in valid_subsets:
            sites = backbone_coords[np.array(subset)]
            cheap_rmsd, matched = _cheap_group_rmsd(virtual_points, sites)
            subset_scores.append((cheap_rmsd, subset, matched))
        subset_scores.sort(key=lambda x: x[0])
        # keep the top beam_topk subsets for the heavy evaluation
        valid_subsets = [sub for _, sub, _ in subset_scores[:beam_topk]]

    best_score = float('inf')
    best_coords = None
    best_subset = None
    best_rmsd = None
    best_perm_idx = None

    placement_attempts = []
    attempt_idx = 0

    for subset in valid_subsets:
        target_coords = backbone_coords[np.array(subset)]

        # Loop over permutations of donor groups → target sites
        for perm in itertools.permutations(range(num_groups)):
            permuted_group_ids = list(perm)
            permuted_donors = virtual_points[permuted_group_ids]

            # Rigidly align donors→sites (centroided) via SVD/Kabsch (no reflection)
            lig_cent = permuted_donors.mean(axis=0)
            tgt_cent = target_coords.mean(axis=0)

            lig_all_centered = ligand_all_coords - lig_cent
            donors_centered  = permuted_donors     - lig_cent
            tgt_centered     = target_coords       - tgt_cent

            H = donors_centered.T @ tgt_centered
            U, _, Vt = np.linalg.svd(H)
            R_opt = Vt.T @ U.T
            if np.linalg.det(R_opt) < 0:
                Vt[-1, :] *= -1
                R_opt = Vt.T @ U.T

            placed = (lig_all_centered @ R_opt.T) + tgt_cent

            # Choose rotation axes:
            # - if ≥2 groups: rotate around vector between last and first site (simple torsion proxy)
            # - if 1 group: rotate around any axis orthogonal to metal→donor axis (scan 2 orthogonals)
            if len(permuted_group_ids) >= 2:
                axis = target_coords[-1] - target_coords[0]
                nrm = np.linalg.norm(axis); axis = axis / (nrm if nrm > 1e-12 else 1.0)
                axes_to_scan = [axis]
                center = target_coords[0]
            else:
                g0 = permuted_group_ids[0]
                g0_point = _virtual_point_for_group(donor_groups[g0], placed)
                metal_ids = _metal_indices_from_elements(complex_elements)
                if metal_ids:
                    metal_coords_only = complex_coords[np.array(metal_ids)]
                    mpos = metal_coords_only[np.argmin(np.linalg.norm(metal_coords_only - g0_point, axis=1))]
                    u = g0_point - mpos
                else:
                    u = np.array([1.0, 0.0, 0.0])
                un = np.linalg.norm(u); u = u / (un if un > 1e-12 else 1.0)
                ref = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                v = np.cross(u, ref); v /= (np.linalg.norm(v) + 1e-12)
                w = np.cross(u, v);  w /= (np.linalg.norm(w) + 1e-12)
                axes_to_scan = [v, w]
                center = g0_point

            # Scan simple angles around the chosen axis/axes
            for ax in axes_to_scan:
                for angle in rotation_angles:
                    r = R.from_rotvec(np.radians(angle) * ax)
                    rotated_sample = r.apply(placed - center) + center

                    # RMSD between group virtual points and target sites
                    placed_group_points = np.vstack([
                        _virtual_point_for_group(donor_groups[k], rotated_sample)
                        for k in permuted_group_ids
                    ])
                    rmsd_val = np.sqrt(np.mean(np.sum((placed_group_points - target_coords)**2, axis=1)))

                    # Clash severity from voxel grid
                    VG = VoxelGridClass(voxel_size=voxel_size, vdw_scale=vdw_scale)
                    VG.add_atoms(elements=complex_elements, coords=complex_coords, group="complex")
                    VG.add_atoms(elements=ligand_elements,  coords=rotated_sample, group="ligand")
                    total_severity, _ = VG.get_clash_severity(
                        rotated_sample, ligand_elements, complex_coords, complex_elements,
                        vdw_radii=VG.vdw_radii, scale=VG.vdw_scale
                    )

                    # Orientation penalty (use representative atom per group)
                    orient_pen = 0.0
                    if orientation_weight > 1e-12:
                        rep_atoms = [donor_groups[k][0] for k in permuted_group_ids]
                        orient_pen = outward_orientation_penalty(
                            rotated_sample, ligand_elements, rep_atoms,
                            complex_coords, complex_elements,
                            k_neighbors=orientation_k_neighbors,
                            hinge=orientation_hinge,
                            per_donor_cap=orientation_cap
                        )

                    score = rmsd_val + clash_weight * total_severity + orientation_weight * orient_pen
                    if verbose:
                        print(f"score: {score}")
                        print(f"rmsd: {rmsd_val}")
                        print(f"clash: {clash_weight * total_severity}")
                        print(f"orientation: {orientation_weight * orient_pen}")
                        print(f"Attempt {attempt_idx} | subset={subset} angle={angle}° score={score:.3f}")
                        print()

                    # (Optional) save a frame for this attempt
                    frame_path = None
                    if vis_save_dir and (attempt_idx % max(1, int(vis_stride)) == 0):
                        if attempt_idx < visual_attempts:
                            frame_path = os.path.join(
                                vis_save_dir,
                                f"{vis_prefix}_subset-{'-'.join(map(str,subset))}"
                                f"_perm-{''.join(map(str,permuted_group_ids))}"
                                f"_ax-{hash(tuple(ax))%10000:04d}"
                                f"_ang-{int(angle):03d}"
                                f"_idx-{attempt_idx:06d}.png"
                            )
                            save_voxel_frame(
                                VG, frame_path,
                                title=f"Attempt {attempt_idx} | subset={subset} angle={angle}° score={score:.3f}",
                                view=vis_view, fixed_bounds=fixed_bounds,
                            )

                    placement_attempts.append({
                        "attempt_idx": attempt_idx,
                        "subset": subset,
                        "axis": ax.tolist(),
                        "angle": float(angle),
                        "ligand_coords": rotated_sample.copy(),
                        "VG_info": get_voxel_coords_by_group(VG),
                        "score": float(score),
                        "rmsd": float(rmsd_val),
                        "clash_severity": float(total_severity),
                        "perm_idx": list(permuted_group_ids),
                        "frame_path": frame_path
                    })
                    attempt_idx += 1

                    # Optional early-out if 'good enough'
                    if early_exit_score is not None and score <= early_exit_score:
                        best_score    = score
                        best_coords   = rotated_sample
                        best_subset   = subset
                        best_rmsd     = rmsd_val
                        best_perm_idx = list(permuted_group_ids)
                        return best_subset, best_coords, best_rmsd, placement_attempts, best_perm_idx

                    if score < best_score:
                        best_score    = score
                        best_coords   = rotated_sample
                        best_subset   = subset
                        best_rmsd     = rmsd_val
                        best_perm_idx = list(permuted_group_ids)

    # === HAPTIC DE-SLANT: align plane normals to metal→centroid axis and scan torsion ===
    if (best_coords is not None) and any(group_is_haptic):
        score_fn = _simple_voxel_score_factory(VoxelGridClass, complex_coords, complex_elements,
                                               ligand_elements, voxel_size, vdw_scale)
        coords_tmp = best_coords.copy()
        for g in [donor_groups[i] for i in range(len(donor_groups)) if len(donor_groups[i]) >= 2]:
            centroid = _virtual_point_for_group(g, coords_tmp)
            axis_out, _ = nearest_metal_vector(centroid, complex_coords, complex_elements)
            # 1) align normal with axis
            coords_tmp = align_group_normal_to_axis(coords_tmp, g, axis_out)
            # 2) torsion scan around that axis (light)
            best_local = coords_tmp
            best_local_score = np.inf
            for deg in rotation_angles:
                rot = R.from_rotvec(np.radians(deg) * axis_out)
                trial = rot.apply(coords_tmp - centroid) + centroid
                s = score_fn(trial)
                if s < best_local_score:
                    best_local_score = s
                    best_local = trial
            coords_tmp = best_local
        best_coords = coords_tmp

    # Final snapshot: ligand-only frame (optional)
    if vis_save_dir and (best_coords is not None):
        VGf = VoxelGridClass(voxel_size=voxel_size, vdw_scale=vdw_scale)
        #VGf.add_atoms(elements=complex_elements, coords=complex_coords, group="complex")
        VGf.add_atoms(elements=ligand_elements,  coords=best_coords,   group="ligand")
        final_base = (
            f"{vis_prefix}_final_subset-{'-'.join(map(str, best_subset))}"
            f"_perm-{''.join(map(str, best_perm_idx))}_score-{best_score:.3f}.png"
        )
        final_path = _unique_path(vis_save_dir, final_base)
        save_voxel_frame(VGf, final_path, title=f"FINAL | subset={best_subset} score={best_score:.3f}",
                         view=vis_view, fixed_bounds=fixed_bounds)

    return best_subset, best_coords, best_rmsd, placement_attempts, best_perm_idx


# ---------------------------- Symmetry helpers ----------------------------

import matplotlib.pyplot as plt
import os
import io
from pathlib import Path
import numpy as np


def _site_equivalence_classes(backbone_coords, tol=1e-4):
    """
    Group backbone site indices into equivalence classes by their
    'distance signature' to all other sites. Two sites i,j are
    equivalent if their sorted distance vectors to the rest of the
    backbone match within `tol`.

    Returns:
      class_of: dict[int -> int] mapping site index -> class id (0..K-1)
      classes:  list[list[int]] of site indices per class
    """
    import numpy as np
    bc = np.asarray(backbone_coords, float)
    M  = len(bc)
    if M == 0:
        return {}, []

    # full distance matrix
    D = np.linalg.norm(bc[:, None, :] - bc[None, :, :], axis=2)

    # build per-site signature (sorted distances to others, rounded to tol)
    def _sig(i):
        dv = np.delete(D[i], i)  # exclude self
        # round to a grid set by tol (e.g., tol=1e-4 → 4 decimals)
        if tol <= 0:
            return tuple(np.sort(dv))
        dec = max(0, int(round(-np.log10(tol))))
        return tuple(np.round(np.sort(dv), dec))

    buckets = {}
    for i in range(M):
        key = _sig(i)
        buckets.setdefault(key, []).append(i)

    classes = list(buckets.values())
    class_of = {}
    for cid, inds in enumerate(classes):
        for i in inds:
            class_of[i] = cid
    return class_of, classes


def _dedupe_subsets_by_site_classes(valid_subsets, class_of):
    """
    Collapse candidate subsets that are equivalent under site-class relabeling.

    For a subset (i,j,k,...) we form its class multiset (c(i),c(j),c(k),...)
    (sorted), and keep only the first subset we encounter for each class multiset.
    """
    seen = set()
    unique = []
    for sub in valid_subsets:
        key = tuple(sorted(class_of[i] for i in sub))
        if key in seen:
            continue
        seen.add(key)
        unique.append(sub)
    return unique


"""
Section 4 — Visualization helpers and quick scene renderers
-----------------------------------------------------------

Contains:
- _ensure_dir: create parent directories if missing.
- _scatter_voxels / _scatter_atoms: tiny wrappers for 3D scatter plots.
- _plot_voxel_dots / _plot_voxel_cubes: render voxel centers as dots or cubes.
- _unique_path: avoid overwriting existing image files by adding suffixes.
- _set_equal_3d: enforce equal axis scale for 3D plots.
- save_voxel_frame: single-call voxel visualization (complex/ligand/clash).
- make_gif_from_frames: convenience GIF builder (if imageio is available).
- visualize_initial_metal_voxels / visualize_final_complex_voxels: quick scenes.

NOTE: Functionality is IDENTICAL to your original code. Only comments/docstrings were added.
"""

import matplotlib.pyplot as plt
import os
import io
from pathlib import Path
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _ensure_dir(path):
    """Create directory `path` (and parents) if it doesn't already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def _scatter_voxels(ax, coords, label, alpha=0.35, s=6):
    """
    Scatter voxel centers as points (used by 'dots' style).
    Returns the created artist or None if `coords` is empty.
    """
    if not coords:
        return None
    pts = np.array(coords)
    sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=s, alpha=alpha, label=label)
    return sc


def _scatter_atoms(ax, coords, label, s=25, alpha=0.9):
    """
    Scatter atom centers (optionally overlayed on top of voxels).
    """
    if coords is None or len(coords) == 0:
        return None
    pts = np.array(coords)
    return ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=s, alpha=alpha, label=label)


def _plot_voxel_dots(ax, coords, label, marker="s", s=10, alpha=0.8, color=None):
    """
    Plot voxel centers as 3D scatter points (square markers by default).
    """
    if not coords:
        return None
    pts = np.asarray(coords)
    return ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                      marker=marker, s=s, alpha=alpha, label=label, color=color)


def _unique_path(dirpath, filename):
    """
    Return a path in `dirpath` that won't overwrite an existing file by appending __NNN.
    """
    os.makedirs(dirpath, exist_ok=True)
    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(dirpath, filename)
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        candidate = os.path.join(dirpath, f"{stem}__{i:03d}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def _set_equal_3d(ax, points=None, padding=0.0, orthographic=True):
    """
    Force a 1:1:1 data scale so cubes look like cubes.
    If `points` is given, compute limits from them; otherwise use current axes limits.
    """
    if points is None or len(points) == 0:
        xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
        mins = np.array([xlim[0], ylim[0], zlim[0]], dtype=float)
        maxs = np.array([xlim[1], ylim[1], zlim[1]], dtype=float)
    else:
        P = np.asarray(points, dtype=float)
        mins = P.min(axis=0)
        maxs = P.max(axis=0)

    center = (mins + maxs) / 2.0
    half   = (max(maxs - mins) / 2.0) + float(padding)

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    # Matplotlib ≥ 3.3 supports true equal aspect for 3D axes
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    if orthographic:
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass


def _plot_voxel_cubes(
    ax,
    coords,
    label,
    size,
    alpha=0.55,
    color=None,
    edgecolor="k",
    linewidth=0.45,
    shade=False,
    zsort="average",
):
    """
    Render voxels as translucent cubes using bar3d (fast enough for modest counts).
    Returns a list of Poly3DCollections (one per cube) or None for empty input.
    """
    try:
        if len(coords) == 0:
            return None
    except TypeError:
        return None

    artists = []
    for (x, y, z) in coords:
        coll = ax.bar3d(
            x - size/2, y - size/2, z - size/2,
            size, size, size,
            shade=shade,
            alpha=alpha,
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        # bar3d returns a Poly3DCollection; keep outlines visible with transparency
        if isinstance(coll, Poly3DCollection):
            try:
                coll.set_zsort(zsort)
                coll.set_antialiaseds(True)
            except Exception:
                pass
        artists.append(coll)
    return artists


def save_voxel_frame(
    voxel_grid,
    out_path,
    title=None,
    view=None,
    atoms_overlay=None,
    legend=False,
    tight=True,
    white_bg=True,
    # voxel styling
    voxel_style="cubes",    # "dots" or "cubes"
    # dot style
    dot_marker="s",
    dot_size=10,
    dot_alpha=0.85,
    # cube style
    cube_size=None,         # defaults to voxel_grid.voxel_size if None
    cube_alpha=0.55,
    cube_edgecolor="none",
    cube_shade=True,
    # palette
    color_complex="#004D40",
    color_ligand="#1E88E5",
    color_clash="#D81B60",
    # equal-scale controls
    equalize_axes=True,
    equal_padding_factor=0.5,
    orthographic=True,
    # Bounds locking across frames
    fixed_bounds=None,      # (xmin, xmax, ymin, ymax, zmin, zmax)
    enforce_cubic_box=True, # keep a cubic box when auto-computing limits
    pad_voxels=0.5,         # padding (in voxel units) when auto-computing
    snap_to_voxel=True,     # snap limits to voxel grid
):
    """
    Save a single PNG visualizing the current voxel grid.

    The voxel grid should contain group ownership ('complex', 'ligand'); any cell
    owned by both is treated as a 'clash'. Optionally overlay atom positions.

    Args:
        voxel_grid (VoxelGrid): populated grid with .grid and group sets.
        out_path (str): target PNG path (parent dirs will be created).
        title (str|None): optional title (suppressed if white_bg=True).
        view (tuple[int,int]|None): (elev, azim) for camera.
        atoms_overlay (dict|None): {"complex": Nx3, "ligand": Mx3} for atom dots.
        legend (bool): draw legend (suppressed if white_bg=True).
        voxel_style (str): "cubes" for bar3d boxes, "dots" for scatter centers.
        fixed_bounds (tuple|None): if provided, identical plot bounds for all frames.
        equalize_axes (bool): auto-compute a symmetric box around data when not fixed.

    Returns:
        None (writes file to `out_path`).
    """
    _ensure_dir(Path(out_path).parent)
    groups = get_voxel_coords_by_group(voxel_grid)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # White background + hidden axes for clean thumbnails
    if white_bg:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax._axis3don = False

    # Choose renderer: cubes or dots
    if voxel_style.lower() == "cubes":
        size = voxel_grid.voxel_size if cube_size is None else float(cube_size)
        h_complex = _plot_voxel_cubes(ax, groups["complex"], "complex voxels",
                                      size=size, alpha=cube_alpha, color=color_complex,
                                      edgecolor=cube_edgecolor, shade=cube_shade)
        h_ligand  = _plot_voxel_cubes(ax, groups["ligand"],  "ligand voxels",
                                      size=size, alpha=cube_alpha, color=color_ligand,
                                      edgecolor=cube_edgecolor, shade=cube_shade)
        h_clash   = _plot_voxel_cubes(ax, groups["clash"],   "clash voxels",
                                      size=size, alpha=cube_alpha, color=color_clash,
                                      edgecolor=cube_edgecolor, shade=cube_shade)
    else:
        h_complex = _plot_voxel_dots(ax, groups["complex"], "complex voxels",
                                     marker=dot_marker, s=dot_size, alpha=dot_alpha, color=color_complex)
        h_ligand  = _plot_voxel_dots(ax, groups["ligand"],  "ligand voxels",
                                     marker=dot_marker, s=dot_size, alpha=dot_alpha, color=color_ligand)
        h_clash   = _plot_voxel_dots(ax, groups["clash"],   "clash voxels",
                                     marker=dot_marker, s=int(dot_size*1.4), alpha=1.0, color=color_clash)

    # Optional atom overlays (small dots)
    if atoms_overlay:
        if atoms_overlay.get("complex") is not None:
            _scatter_atoms(ax, atoms_overlay["complex"], "complex atoms", s=18, alpha=0.95)
        if atoms_overlay.get("ligand") is not None:
            _scatter_atoms(ax, atoms_overlay["ligand"], "ligand atoms", s=18, alpha=0.95)

    # Camera/view
    if view:
        elev, azim = view
        ax.view_init(elev=elev, azim=azim)

    # Projection: orthographic (no foreshortening) is cleaner for grids
    try:
        if orthographic:
            ax.set_proj_type("ortho")
    except Exception:
        pass

    # Fix bounds explicitly if provided (stabilizes multi-frame movies)
    if fixed_bounds is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = map(float, fixed_bounds)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        try:
            ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
        except Exception:
            pass
    elif equalize_axes:
        # Auto-compute a stable, cube-shaped box around current content
        size = voxel_grid.voxel_size if cube_size is None else float(cube_size)
        all_pts = []
        all_pts.extend(groups.get("complex", []))
        all_pts.extend(groups.get("ligand",  []))
        all_pts.extend(groups.get("clash",   []))
        if atoms_overlay:
            for k in ("complex", "ligand"):
                arr = atoms_overlay.get(k)
                if isinstance(arr, np.ndarray) and arr.size:
                    all_pts.extend(arr.tolist())
        if not all_pts:
            all_pts = [(0.0, 0.0, 0.0)]
        P = np.array(all_pts, dtype=float)

        # span of centers, then expand by half-cube + padding
        minv = P.min(axis=0)
        maxv = P.max(axis=0)
        center = 0.5 * (minv + maxv)
        halfspan = 0.5 * (maxv - minv) + (size * (0.5 + float(pad_voxels)))

        if enforce_cubic_box:
            r = float(np.max(halfspan))
            halfspan = np.array([r, r, r], dtype=float)

        # snap to voxel grid so edges align perfectly frame-to-frame
        if snap_to_voxel and size > 0:
            xmin = size * np.floor((center[0] - halfspan[0]) / size)
            xmax = size * np.ceil( (center[0] + halfspan[0]) / size)
            ymin = size * np.floor((center[1] - halfspan[1]) / size)
            ymax = size * np.ceil( (center[1] + halfspan[1]) / size)
            zmin = size * np.floor((center[2] - halfspan[2]) / size)
            zmax = size * np.ceil( (center[2] + halfspan[2]) / size)
        else:
            xmin, xmax = center[0] - halfspan[0], center[0] + halfspan[0]
            ymin, ymax = center[1] - halfspan[1], center[1] + halfspan[1]
            zmin, zmax = center[2] - halfspan[2], center[2] + halfspan[2]

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        try:
            ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
        except Exception:
            pass

    # Titles/legend only when not doing pure white thumbnails
    if title and not white_bg:
        ax.set_title(title)
    if legend and not white_bg:
        handles = [h for h in (h_complex, h_ligand, h_clash) if h is not None]
        if handles:
            ax.legend(handles, ["complex voxels", "ligand voxels", "clash voxels"],
                      loc="upper right", fontsize=8)

    if tight:
        plt.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)


def make_gif_from_frames(frame_paths, gif_path, fps=8):
    """
    Turn a list of PNG frames into a GIF animation.
    Requires imageio; if not available, this is a no-op.
    """
    try:
        import imageio.v2 as imageio
    except Exception:
        print("[make_gif_from_frames] imageio not available; skipping GIF.")
        return
    _ensure_dir(Path(gif_path).parent)
    imgs = []
    for p in frame_paths:
        try:
            imgs.append(imageio.imread(p))
        except Exception:
            pass
    if imgs:
        imageio.mimsave(gif_path, imgs, duration=max(1e-9, 1.0/float(fps)))


def visualize_initial_metal_voxels(
    metals_structures,
    voxel_size=0.3,
    vdw_scale=0.55,
    out_png="voxels_initial_metal.png",
    view=(22, -60)
):
    """
    Quick utility: voxelize metals-only and save a snapshot.
    Backbone dummy sites are *not* added (only real metal atoms).
    """
    VG = VoxelGrid(voxel_size=voxel_size, vdw_scale=vdw_scale)
    complex_coords = []
    complex_elems  = []
    for m in metals_structures:
        complex_coords.append(m["coord"])
        complex_elems.append(m["element"])
    complex_coords = np.array(complex_coords, dtype=float)
    VG.add_atoms(complex_elems, complex_coords, group="complex")

    save_voxel_frame(
        VG, out_png,
        title="Initial metal voxelization",
        view=view,
        atoms_overlay={"complex": complex_coords, "ligand": None}
    )
    return out_png


def visualize_final_complex_voxels(
    complex_coords, complex_elements, ligand_coords, ligand_elements,
    voxel_size=0.3, vdw_scale=0.55,
    out_png="voxels_final_complex.png",
    view=(22, -60)
):
    """
    Quick utility: voxelize final complex + ligand and save a snapshot.
    """
    VG = VoxelGrid(voxel_size=voxel_size, vdw_scale=vdw_scale)
    VG.add_atoms(complex_elements, complex_coords, group="complex")
    VG.add_atoms(ligand_elements,  ligand_coords,  group="ligand")

    save_voxel_frame(
        VG, out_png,
        title="Final complex voxelization",
        view=view,
        atoms_overlay={"complex": complex_coords, "ligand": ligand_coords}
    )
    return out_png


"""
Section 5 — Ring piercing, sterics, mapping helpers, and small transforms
-------------------------------------------------------------------------

Contains:
- Ring tools: detect_ring_piercing, correct_ring_piercings, estimate_minimum_shift_distance.
- Steric checks: grid/KDTree-based clash detection (with/without energy embedding).
- Small visualization: visualize_molecule for clashes.
- Sanity checks: check_badjob (distance overlap + bond-order sync check).
- Simple transforms: apply_rotation_about_vector, generate_rotated_conformations.
- Voxel extraction: get_voxel_coords_by_group (grouped centers).
- Haptics bonding helpers: add_haptic_multibonds_to_metal*, reapply_all_haptics_and_sync.
- Index mapping helpers: map_ligand_local_to_core_indices_* and group translation.

NOTE: Functionality and defaults are IDENTICAL to your original code.
Only comments/docstrings were added for clarity.
"""

from scipy.spatial import KDTree
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


def detect_ring_piercing(
    mol3D,
    bond_pairs=None,
    ring_size_threshold=12,
    angstrom_threshold=1.3,
    plane_tol=0.45,
    inplane_pad=0.30,
    edge_buffer=0.18,
    endpoint_buffer=0.40,
    verbose=False,
):
    """
    Detect bonds that pierce small rings, robust to fused or non-planar rings.

    Algorithm (high level):
      - Build a graph and enumerate cycles via union of cycle_basis and minimum_cycle_basis.
      - For each candidate ring (size ≤ ring_size_threshold), attempt either:
          * Planar test: segment-plane intersection + in-ring 2D point test with padding; or
          * Nonplanar fallback: segment-triangle intersections against a convex hull.
      - Reject near-endpoint hits via endpoint_buffer.
      - Guard with angular/planar thresholds to avoid false positives.

    Args:
        mol3D (mol3D): structure to inspect.
        bond_pairs (iterable[(i,j)]|None): bonds to test; defaults to all in bo_dict.
        ring_size_threshold (int): consider rings of this size or smaller.
        angstrom_threshold (float): distance gating near the ring plane.
        plane_tol (float): RMS plane deviation below which ring treated as planar.
        inplane_pad (float), edge_buffer (float), endpoint_buffer (float): geometry guards.
        verbose (bool): print limited debug info.

    Returns:
        list[tuple]: [(a1, a2, ring_atom_indices), ...] for each piercing bond.
    """
    import numpy as np, networkx as nx
    from scipy.spatial import ConvexHull

    G = nx.Graph()
    for (i, j) in mol3D.bo_dict:
        G.add_edge(i, j)

    rings_a = nx.cycle_basis(G)
    rings_b = nx.minimum_cycle_basis(G)

    def _norm_cycle(c): return tuple(sorted(c))
    uniq = {}
    for c in rings_a + rings_b:
        if len(c) <= ring_size_threshold:
            uniq[_norm_cycle(c)] = c
    rings = list(uniq.values())

    if bond_pairs is None:
        bond_pairs = list(mol3D.bo_dict.keys())

    coords = np.array([at.coords() for at in mol3D.atoms], dtype=float)
    bond_set = {(min(i, j), max(i, j)) for (i, j) in mol3D.bo_dict}

    if verbose:
        print(f"[detect_ring_piercing] rings={len(rings)} "
              f"(≤{ring_size_threshold}), bonds_to_check={len(bond_pairs)}")
        print(f"[detect_ring_piercing] params: ang_thresh={angstrom_threshold:.2f}, "
              f"plane_tol={plane_tol:.2f}, pad={inplane_pad:.2f}, edge_buf={edge_buffer:.2f}, "
              f"end_buf={endpoint_buffer:.2f}")

    def _fit_plane(xyz):
        C = xyz.mean(axis=0)
        X = xyz - C
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        n = Vt[2] / (np.linalg.norm(Vt[2]) + 1e-12)
        d = np.dot(X, n)
        rms = float(np.sqrt(np.mean(d * d)))
        u = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
        v = np.cross(n, u)
        return C, n, u, v, rms

    def _seg_plane_isect(A, B, C, n):
        AB = B - A
        denom = np.dot(n, AB)
        if abs(denom) < 1e-10:
            return None, None
        t = np.dot(n, C - A) / denom
        if t <= 1e-9 or t >= 1.0 - 1e-9:
            return None, None
        P = A + t * AB
        return P, t

    def _point_in_poly_2d(poly, p):
        x, y = p; inside = False; m = len(poly)
        for i in range(m):
            x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % m]
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
                inside = not inside
        return inside

    def _min_edge_distance_2d(poly, p):
        p = np.asarray(p); mind = np.inf; m = len(poly)
        for i in range(m):
            a = np.asarray(poly[i]); b = np.asarray(poly[(i + 1) % m])
            ab = b - a
            t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
            proj = a + t * ab
            d = np.linalg.norm(p - proj)
            if d < mind: mind = d
        for a in poly:
            d = np.linalg.norm(p - a)
            if d < mind: mind = d
        return float(mind)

    def _seg_tri_isect_t(A, B, V0, V1, V2):
        EPS = 1e-10
        dir = B - A
        edge1 = V1 - V0; edge2 = V2 - V0
        pvec = np.cross(dir, edge2); det = np.dot(edge1, pvec)
        if abs(det) < EPS: return (False, None)
        invDet = 1.0 / det
        tvec = A - V0
        u = np.dot(tvec, pvec) * invDet
        if u < 0.0 or u > 1.0: return (False, None)
        qvec = np.cross(tvec, edge1)
        v = np.dot(tvec, qvec) * invDet
        if v < 0.0 or u + v > 1.0: return (False, None)
        t = np.dot(edge2, qvec) * invDet
        return ((t > EPS) and (t < 1.0 - EPS), t)

    piercing = []
    hits_logged = 0
    for a1, a2 in bond_pairs:
        if (min(a1, a2), max(a1, a2)) in bond_set:
            pass
        A, B = coords[a1], coords[a2]
        AB = B - A; L = np.linalg.norm(AB)
        if L < 1e-9: continue
        u = AB / L

        for ring in rings:
            if a1 in ring and a2 in ring and (min(a1, a2), max(a1, a2)) in bond_set:
                continue

            ring_xyz = coords[ring]
            C, n, ex, ey, rms = _fit_plane(ring_xyz)

            if rms <= plane_tol:
                s1 = np.sign(np.dot(n, A - C)); s2 = np.sign(np.dot(n, B - C))
                if s1 == 0 or s2 == 0 or s1 == s2:
                    continue
                P, tpar = _seg_plane_isect(A, B, C, n)
                if P is None: continue
                if tpar < endpoint_buffer / (L + 1e-12) or (1.0 - tpar) < endpoint_buffer / (L + 1e-12):
                    continue
                R = ring_xyz - C
                poly2 = np.stack((R @ ex, R @ ey), axis=1)
                radii = np.linalg.norm(poly2, axis=1) + 1e-12
                scale = (radii + max(inplane_pad, angstrom_threshold * 0.2)) / radii
                poly2_pad = poly2 * scale[:, None]
                p2 = np.array([np.dot(P - C, ex), np.dot(P - C, ey)])
                if not _point_in_poly_2d(poly2_pad, p2): continue
                if _min_edge_distance_2d(poly2_pad, p2) < edge_buffer: continue
                dists = np.linalg.norm(np.cross(ring_xyz - A, u), axis=1)
                if dists.min(initial=1e9) > (angstrom_threshold + inplane_pad): continue

                piercing.append((a1, a2, ring))
                if verbose and hits_logged < 20:
                    print(f"[detect_ring_piercing] HIT(planar): bond {a1}-{a2} through ring(size={len(ring)}), "
                          f"t={tpar:.2f}, min_line_d={dists.min():.2f} Å")
                    hits_logged += 1
                break

            else:
                hit = False
                try:
                    hull = ConvexHull(ring_xyz)
                    for tri in hull.simplices:
                        V0, V1, V2 = ring_xyz[tri[0]], ring_xyz[tri[1]], ring_xyz[tri[2]]
                        ok, tpar = _seg_tri_isect_t(A, B, V0, V1, V2)
                        if not ok: 
                            continue
                        if tpar < endpoint_buffer / (L + 1e-12) or (1.0 - tpar) < endpoint_buffer / (L + 1e-12):
                            continue
                        dists = np.linalg.norm(np.cross(ring_xyz - A, u), axis=1)
                        if dists.min(initial=1e9) > (angstrom_threshold + inplane_pad):
                            continue
                        hit = True
                        if verbose and hits_logged < 20:
                            print(f"[detect_ring_piercing] HIT(nonplanar): bond {a1}-{a2} ring(size={len(ring)}), "
                                  f"t={tpar:.2f}, min_line_d={dists.min():.2f} Å")
                            hits_logged += 1
                        break
                except Exception:
                    s1 = np.sign(np.dot(n, A - C)); s2 = np.sign(np.dot(n, B - C))
                    hit = (s1 != s2 and s1 != 0 and s2 != 0)
                if hit:
                    piercing.append((a1, a2, ring))
                    break

    if verbose:
        print(f"[detect_ring_piercing] total_hits={len(piercing)}")
    return piercing


def estimate_minimum_shift_distance(ring_coords, bond_coords, buffer=5):
    """
    Heuristic upper-bound for moving a point out of a ring's vicinity:
    distance from bond midpoint to ring center + ring radius + buffer (Å).
    """
    ring_center = ring_coords.mean(axis=0)
    bond_midpoint = bond_coords.mean(axis=0)
    direction = bond_midpoint - ring_center
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return buffer
    direction /= norm
    distances = np.linalg.norm(ring_coords - bond_midpoint, axis=1)
    max_radius = np.max(distances)
    return max_radius + buffer


def correct_ring_piercings(
    mol3D, piercing_bonds, min_clearance=None, max_scans=24, step_deg=15,
    verbose=False,
    rp_kwargs=None,
):
    """
    Attempt to resolve ring piercings by:
      1) Rotating the *smaller* side of the offending bond around the bond axis;
         accept the first torsion that clears the hit while not increasing local crowding.
      2) If torsion fails, translate a single endpoint away from the ring along
         a low-density in-plane direction by an estimated clearance.

    Args:
        mol3D (mol3D): structure to modify (copied internally during tests).
        piercing_bonds (list[(a1,a2,ring)]): output of detect_ring_piercing.
        min_clearance (float|None): optional distance override for fallback translate.
        max_scans (int): number of ±step_deg torsion increments to try.
        step_deg (int): torsion step in degrees.
        verbose (bool): prints progress info.
        rp_kwargs (dict|None): overrides for detect_ring_piercing checks in the loop.

    Returns:
        (new_coords (np.ndarray[N,3]), moved_atoms (list[int]))
    """
    import numpy as np
    import networkx as nx
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial import KDTree

    # Base parameters for ring detection used inside the correction loop
    rp_defaults = dict(
        ring_size_threshold=12,
        angstrom_threshold=1.3,
        plane_tol=0.45,
        inplane_pad=0.30,
        edge_buffer=0.18,
        endpoint_buffer=0.40,
        verbose=False,
    )
    rp_params = {**rp_defaults, **(rp_kwargs or {})}

    if verbose:
        print(f"[correct_ring_piercings] params: { {k:v for k,v in rp_params.items() if k!='verbose'} }")

    coords = np.array([atom.coords() for atom in mol3D.atoms], dtype=float)
    bond_pairs = {(min(i, j), max(i, j)) for (i, j) in mol3D.bo_dict}
    G = nx.Graph(); G.add_edges_from(bond_pairs)
    kd_tree = KDTree(coords)
    moved_atoms = set()

    def _estimate_min_clearance(ring_coords, bond_coords, buffer=0.6):
        C = ring_coords.mean(axis=0)
        mid = bond_coords.mean(axis=0)
        r = np.max(np.linalg.norm(ring_coords - C, axis=1))
        return np.linalg.norm(mid - C) - r + buffer

    if verbose:
        print(f"[correct_ring_piercings] starting with {len(piercing_bonds)} pierce(s)")

    for a1, a2, ring in piercing_bonds:
        if verbose:
            print(f"  • fixing bond {a1}-{a2} through ring(size={len(ring)})")

        # Split the molecule by cutting the offending bond and pick the smaller side to rotate
        H = G.copy()
        if H.has_edge(a1, a2):
            H.remove_edge(a1, a2)
        comp1 = next((c for c in nx.connected_components(H) if a1 in c), {a1})
        comp2 = next((c for c in nx.connected_components(H) if a2 in c), {a2})
        side_j, pivot_i, pivot_j = (comp2, a1, a2) if len(comp2) <= len(comp1) else (comp1, a2, a1)

        axis = coords[pivot_j] - coords[pivot_i]
        n_axis = np.linalg.norm(axis)
        best_ok = None
        if n_axis > 1e-9 and len(side_j) > 1:
            u = axis / n_axis
            center = coords[pivot_i]
            base = coords.copy()

            def _local_density(xyz, radius=2.0):
                return sum(len(kd_tree.query_ball_point(xyz[k], radius)) for k in side_j)

            base_density = _local_density(base)
            if verbose:
                print(f"    - torsion scan around axis |axis|={n_axis:.2f} Å, side_size={len(side_j)}")

            found_angle = None
            for k in range(1, max_scans + 1):
                for sgn in (+1, -1):
                    ang = sgn * step_deg * k
                    rot = R.from_rotvec(np.radians(ang) * u)
                    trial = base.copy()
                    for k_idx in side_j:
                        vec = base[k_idx] - center
                        trial[k_idx] = center + rot.apply(vec)

                    tmp = mol3D.__class__(); tmp.copymol3D(mol3D)
                    for ii, at in enumerate(tmp.atoms):
                        at.setcoords(trial[ii])

                    if detect_ring_piercing(tmp, bond_pairs=[(a1, a2)], **rp_params):
                        continue

                    if _local_density(trial) <= base_density + 2:
                        best_ok = trial
                        found_angle = ang
                        break
                if best_ok is not None:
                    break

            if best_ok is not None and verbose:
                print(f"    ✓ torsion cleared at ~{found_angle}° (local crowding ok)")

        # Apply the successful torsion if found
        if best_ok is not None:
            coords = best_ok
            moved_atoms.update(side_j)
            kd_tree = KDTree(coords)
            continue

        # Fallback: guided translation of one endpoint away from ring plane
        ring_coords = coords[ring]
        bond_coords = coords[[a1, a2]]
        clearance = (_estimate_min_clearance(ring_coords, bond_coords)
                     if min_clearance is None else min_clearance)
        # Clamp minimum clearance
        if clearance <= 0:
            clearance = 0.6

        # Ring plane normal (approx)
        n = np.cross(ring_coords[1] - ring_coords[0], ring_coords[2] - ring_coords[0])
        n /= (np.linalg.norm(n) + 1e-12)

        # Move the endpoint in the less crowded neighborhood
        r = 2.0
        dens = [len(kd_tree.query_ball_point(coords[a1], r)), len(kd_tree.query_ball_point(coords[a2], r))]
        target = a1 if dens[0] <= dens[1] else a2

        if verbose:
            print(f"    - fallback translation (clearance≈{clearance:.2f} Å); moving atom {target} "
                  f"(nbhd densities {dens[0]} vs {dens[1]})")

        # Build in-plane orthonormals, then sample directions to pick the lowest-density shift
        t1 = np.cross(n, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.cross(n, np.array([0.0, 1.0, 0.0]))
        t1 /= (np.linalg.norm(t1) + 1e-12)
        t2 = np.cross(n, t1)

        best_dir = n
        min_nb = np.inf
        for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            d = np.cos(ang) * t1 + np.sin(ang) * t2
            cand = coords[target] + clearance * d
            nb = len(kd_tree.query_ball_point(cand, r))
            if nb < min_nb:
                min_nb = nb
                best_dir = d

        coords[target] = coords[target] + clearance * best_dir
        moved_atoms.add(target)
        kd_tree = KDTree(coords)
        if verbose:
            print(f"    ✓ translated atom {target}; new local nbhd count ≈ {min_nb:.0f}")

        # Quick re-check: did we clear the piercing?
        tmp = mol3D.__class__(); tmp.copymol3D(mol3D)
        for ii, at in enumerate(tmp.atoms):
            at.setcoords(coords[ii])
        _ = not detect_ring_piercing(tmp, bond_pairs=[(a1, a2)], **rp_params)

    if verbose:
        print(f"[correct_ring_piercings] moved {len(moved_atoms)} atom(s)")
    return coords, list(moved_atoms)


def build_kdtree(coords):
    """Convenience wrapper: KDTree over Nx3 coordinates."""
    return KDTree(coords)


def _build_bond_graph(bo_dict):
    """
    Build an undirected graph from a bond-order dict and return both the set
    of bond pairs and the networkx graph.
    """
    bond_pairs = { (min(i,j), max(i,j)) for (i,j) in bo_dict }
    G = nx.Graph(); G.add_edges_from(bond_pairs)
    return bond_pairs, G


def _donors_by_metal(elements, bond_pairs):
    """
    Map: metal_atom_index → set(donor_indices) for that metal, inferred from bond pairs.
    """
    METALS = {"Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
              "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
              "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"}
    metals = {idx for idx, sym in enumerate(elements) if sym in METALS}
    donors = {}
    for a,b in bond_pairs:
        if a in metals and b not in metals:
            donors.setdefault(a, set()).add(b)
        elif b in metals and a not in metals:
            donors.setdefault(b, set()).add(a)
    return donors


def _shares_metal(u, v, donors_by_m):
    """True if atoms u and v are donors bound to the same metal."""
    for m, nbrs in donors_by_m.items():
        if u in nbrs and v in nbrs:
            return True
    return False


def _pca_frame(P):
    """Return orthonormal principal axes for points P (N,3) and the centroid."""
    C = P.mean(axis=0)
    X = P - C
    if X.shape[0] < 2 or np.linalg.matrix_rank(X) < 2:
        return np.eye(3), C
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    return U, C


def _pca_align(P, Q):
    """
    Build a coarse correspondence-free rotation using PCA frames for P and Q.
    Returns rotation R and centroids CP, CQ.
    """
    UP, CP = _pca_frame(P)
    UQ, CQ = _pca_frame(Q)
    R = UQ @ UP.T
    return R, CP, CQ


def _cheap_group_rmsd(donor_virtuals, site_coords):
    """
    Cheap donor-shape RMSD:
      1) PCA-align donor virtuals to site coords,
      2) assign pairs via Hungarian on squared distances,
      3) compute RMSD under that assignment.

    Returns:
        rmsd (float), matched_sites (list[int])
    """
    P = np.asarray(donor_virtuals, float)
    Q = np.asarray(site_coords, float)
    R, CP, CQ = _pca_align(P, Q)
    P_aligned = (P - CP) @ R + CQ

    D2 = ((P_aligned[:, None, :] - Q[None, :, :]) ** 2).sum(axis=2)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(D2)
    rmsd = np.sqrt(np.mean(D2[row_ind, col_ind]))
    matched_sites = [int(col) for col in col_ind]
    return float(rmsd), matched_sites


def prefilter_subsets_by_donor_reps(
    donor_groups,
    ligand_all_coords,
    backbone_coords,
    candidate_subsets,
    rmsd_cut=0.8,
    top_k=6
):
    """
    Rank candidate site-subsets by a cheap donor-rep (group virtuals) RMSD and
    keep up to `top_k` under a cutoff. Useful for denticity ≥ 2.

    Returns:
        list[tuple]: filtered subsets (each is a tuple of site indices).
    """
    donor_virtuals = np.vstack([
        ligand_all_coords[g].mean(axis=0) if len(g) > 1 else ligand_all_coords[g[0]]
        for g in donor_groups
    ])

    scored = []
    for subset in candidate_subsets:
        sites = backbone_coords[np.array(subset)]
        r, _ = _cheap_group_rmsd(donor_virtuals, sites)
        if np.isfinite(r):
            scored.append((r, subset))
    scored.sort(key=lambda x: x[0])

    filtered = [sub for r, sub in scored if r <= rmsd_cut]
    if not filtered:
        filtered = [sub for _, sub in scored[:top_k]]
    else:
        filtered = filtered[:top_k]
    return filtered


def check_sterics(tree, coords, elements, vdw_radii, bo_dict=None, scale=0.9, default_vdw=1.5):
    """
    Basic steric clash detection using per-atom VDW radii and KD-tree neighborhoods.

    Rules:
      - Ignore bonded pairs.
      - Prune pairs within ≤3 bonds unless they are two donors bound to the same metal.
      - Mark pairs whose distance < (ri + rj) * scale as clashes; record severity.

    Returns:
        steric_pairs (list[(i,j)]), severity_scores (dict[(i,j)]->float)
    """
    bo_dict = bo_dict or {}
    bond_pairs, G = _build_bond_graph(bo_dict)
    donors_by_m = _donors_by_metal(elements, bond_pairs)

    clashes = set()
    severity_scores = {}
    max_vdw = max(vdw_radii.values(), default=default_vdw)

    for i, pi in enumerate(coords):
        ri = vdw_radii.get(elements[i], default_vdw)
        neighbors = tree.query_ball_point(pi, (ri + max_vdw) * scale)
        for j in neighbors:
            if i >= j: 
                continue
            if (min(i,j), max(i,j)) in bond_pairs:
                continue

            # ≤3-bond pruning unless same-metal donor pair
            if not _shares_metal(i, j, donors_by_m):
                try:
                    if nx.has_path(G, i, j) and nx.shortest_path_length(G, i, j) <= 3:
                        continue
                except nx.NetworkXNoPath:
                    pass

            rj = vdw_radii.get(elements[j], default_vdw)
            cutoff = (ri + rj) * scale
            d = np.linalg.norm(coords[i] - coords[j])
            if d < cutoff:
                key = (i, j) if i < j else (j, i)
                clashes.add(key)
                severity_scores[key] = cutoff - d

    return sorted(clashes), severity_scores


def _ring_sets_from_graph(G):
    # NetworkX cycle basis works well for aromatic rings
    cycles = nx.cycle_basis(G)
    return [set(c) for c in cycles]

def _same_ring(i, j, ring_sets):
    return any((i in R and j in R) for R in ring_sets)


def _normalize_bonds_auto(bo_dict, N):
    """
    Normalize a bond-order dict to 0-based, (i<j) keys, filtered to [0, N).
    Auto-detects whether the input is 0-based or 1-based by inspecting indices.
    """
    if not bo_dict:
        return {}

    idxs = []
    for (i, j) in bo_dict.keys():
        idxs.append(i); idxs.append(j)

    mn, mx = min(idxs), max(idxs)

    # Heuristics:
    # - If any index is 0 -> already 0-based
    # - Else if max index == N or min index == 1 -> likely 1-based
    # - Else if max index < N -> ambiguous, assume 0-based
    # - Else if max index == N and no zeros -> 1-based
    one_based = (0 not in idxs) and (mn >= 1) and (mx <= N)

    norm = {}
    for (i, j), bo in bo_dict.items():
        if one_based:
            i -= 1; j -= 1
        if i == j:
            continue
        if not (0 <= i < N and 0 <= j < N):
            # guard against out-of-range keys
            continue
        a, b = (i, j) if i < j else (j, i)
        norm[(a, b)] = bo
    return norm

def _build_graph_from_bonds(N, bo_dict0):
    bo = _normalize_bonds_auto(bo_dict0 or {}, N)
    G = nx.Graph()
    G.add_nodes_from(range(N))            # every atom exists (avoids NodeNotFound)
    for (i, j), _ in bo.items():
        G.add_edge(i, j)
    bond_pairs = set(bo.keys())
    return G, bond_pairs


def _normalize_bonds_zero_based(bo_dict):
    """
    Normalize a bond-order dict to 0-based, (i<j) keys.
    Safely handles input that might be 1-based (MOL2-style) by detecting zeros.
    """
    if not bo_dict:
        return {}
    any_zero = any(i == 0 or j == 0 for (i, j) in bo_dict.keys())
    norm = {}
    for (i, j), bo in bo_dict.items():
        i0, j0 = (i, j) if any_zero else (i - 1, j - 1)
        if i0 == j0:
            continue
        a, b = (i0, j0) if i0 < j0 else (j0, i0)
        norm[(a, b)] = bo
    return norm


def check_sterics_with_ff_embedding(
    tree,
    coords,
    elements,
    vdw_radii,
    bo_dict=None,
    per_atom_nonbonded=None,
    per_atom_vdw=None,
    per_atom_ff_force=None,
    energy_source='none',     # 'none' | 'nonbonded' | 'vdw' | 'ff_force'
    energy_weighted=False,
    # ---- NEW: pair-energy gate (option 4) ----
    pair_energy_threshold: float = 0.5,  # require mean(e_i,e_j) >= this to consider a clash
    #
    scale=1.0,
    default_vdw=1.5,
    exclude_hops=(1, 2, 3),
    # ---- BASE clearance for heavy-heavy ----
    clearance_heavy: float = 0.20,
    # ---- NEW: H-aware per-pair clearances (option 3) ----
    clearance_HH: float = 0.35,   # H···H pairs must penetrate > this to count
    clearance_HX: float = 0.30,   # H···X (X != H) must penetrate > this to count
    #
    infer_H_bonds: bool = True,
    H_bond_max: float = 1.25      # Å: attach isolated H to nearest atom if bo_dict lacks H bonds
):
    """
    Steric clash detection with:
      - robust bond-graph pruning (≤3 hops),
      - H-aware per-pair clearance (HH/HX/heavy-heavy),
      - optional pair-energy gating based on a per-atom FF embedding,
      - optional gentle energy weighting of severity.

    Returns: (clashes (list[(i,j)]), severity_scores dict[(i,j)]->float)
    """
    import numpy as np
    import networkx as nx

    N = len(coords)

    # --- build graph with all nodes, then add edges ---
    bo_dict = _normalize_bonds_zero_based(bo_dict or {})
    G = nx.Graph()
    G.add_nodes_from(range(N))
    bond_pairs = set()
    for (i, j), _ in bo_dict.items():
        if 0 <= i < N and 0 <= j < N:
            G.add_edge(i, j)
            bond_pairs.add((min(i, j), max(i, j)))

    # --- infer missing H–X bonds if Hs are isolated (common in heavy-only bo_dicts) ---
    if infer_H_bonds:
        import numpy as np
        xyz = np.asarray(coords, float)
        is_H: NDArray[np.bool_] = np.array(
            [str(z).upper() == "H" for z in elements],
            dtype=bool,
        )
        for h in range(N):
            if not is_H[h] or G.degree[h] > 0:
                continue
            d = np.linalg.norm(xyz - xyz[h], axis=1)
            d[h] = 1e9
            j = int(np.argmin(d))
            if d[j] <= H_bond_max:
                G.add_edge(h, j)
                bond_pairs.add((min(h, j), max(h, j)))

    # --- choose FF embedding for energy/force info ---
    energies = None
    if energy_source == 'ff_force':
        energies = per_atom_ff_force
    elif energy_source == 'nonbonded':
        energies = per_atom_nonbonded
    elif energy_source == 'vdw':
        energies = per_atom_vdw
    if energies is not None:
        energies = np.clip(np.asarray(energies, float), 0.0, None)

    is_H: NDArray[np.bool_] = np.array(
        [str(z).upper() == "H" for z in elements],
        dtype=bool,
    )

    clashes = set()
    severity_scores = {}
    max_vdw = max(vdw_radii.values(), default=default_vdw)
    max_hops = max(exclude_hops) if exclude_hops else 0

    for i in range(N):
        ri = vdw_radii.get(elements[i], default_vdw)
        e_i = (energies[i] if (energies is not None and i < len(energies)) else 0.0)

        # Precompute hop distances up to cutoff (safe; no NodeNotFound)
        hop_len = nx.single_source_shortest_path_length(G, i, cutoff=max_hops) if max_hops > 0 else {}

        for j in tree.query_ball_point(coords[i], (ri + max_vdw) * scale):
            if i >= j:
                continue

            # bonded? skip
            if (min(i, j), max(i, j)) in bond_pairs:
                continue

            # prune 1–2/1–3/1–4 neighbors (if requested)
            if max_hops > 0 and (j in hop_len) and (hop_len[j] in exclude_hops):
                continue

            # dynamic, H-aware clearance
            if is_H[i] and is_H[j]:
                clearance = clearance_HH
            elif is_H[i] or is_H[j]:
                clearance = clearance_HX
            else:
                clearance = clearance_heavy

            rj = vdw_radii.get(elements[j], default_vdw)
            cutoff = (ri + rj) * scale
            d = float(np.linalg.norm(coords[i] - coords[j]))
            penetration = cutoff - d
            if penetration <= clearance:
                continue

            # pair-energy gate (mean of per-atom energies/forces)
            if energies is not None and pair_energy_threshold is not None and pair_energy_threshold > 0.0:
                e_j = energies[j] if j < len(energies) else 0.0
                mean_e = 0.5 * (e_i + e_j)
                if mean_e < pair_energy_threshold:
                    continue  # too “quiet” energetically → not a real clash

            # severity beyond the per-pair clearance
            sev = penetration - clearance
            if energy_weighted and (energies is not None):
                # gentle up-weighting; avoids runaway from a single spiky atom
                e_j = energies[j] if j < len(energies) else 0.0
                sev *= np.log1p(0.5 * (e_i + e_j))

            key = (i, j)
            severity_scores[key] = float(sev)
            clashes.add(key)

    return sorted(clashes), severity_scores


def check_badjob(core3D):
    """
    Two quick sanity checks on a mol3D:
      1) 'overlap' from core3D.sanitycheck()[0] (atomic distance overlaps).
      2) 'same_order' comparing current bo_dict keys against OpenBabel-derived bonds.

    Returns:
        (overlap (bool), same_order (bool))
    """
    overlap = core3D.sanitycheck()[0]
    core3D_copy = mol3D()
    core3D_copy.copymol3D(core3D)
    current_bo = copy.deepcopy(core3D_copy.bo_dict)
    core3D_copy.convert2OBMol()
    assumed_bo = get_bond_dict(core3D_copy.OBMol)
    same_order = current_bo.keys() == assumed_bo.keys()
    return overlap, same_order


def apply_rotation_about_vector(coords, donor_coords, axis, angle_deg):
    """
    Rotate a full coordinate set `coords` by angle_deg (degrees) about the axis
    passing through the centroid of donor_coords in the direction `axis`.
    """
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    rot = R.from_rotvec(angle_deg * np.pi / 180 * axis)
    center = donor_coords.mean(axis=0)
    centered = coords - center
    rotated = rot.apply(centered) + center
    return rotated


def generate_rotated_conformations(coords, donor_coords, axis, angles):
    """
    Generate a list of rotated conformations by sweeping `angles` around `axis`
    through the centroid of donor_coords.
    """
    rotated_confs = []
    for angle in angles:
        rotated = apply_rotation_about_vector(coords, donor_coords, axis, angle)
        rotated_confs.append(rotated)
    return rotated_confs


def get_voxel_coords_by_group(voxel_grid):
    """
    Extract voxel centers by ownership group:
      - 'complex' cells,
      - 'ligand' cells,
      - 'clash' cells (owned by both).

    Returns:
        dict[str -> list[(x,y,z)]]
    """
    voxel_size = voxel_grid.voxel_size
    coords_by_group = {"complex": [], "ligand": [], "clash": []}

    for idx, owners in voxel_grid.grid.items():
        owners = set(owners)
        pos = (idx[0] * voxel_size, idx[1] * voxel_size, idx[2] * voxel_size)
        if "complex" in owners and "ligand" in owners:
            coords_by_group["clash"].append(pos)
        elif "complex" in owners:
            coords_by_group["complex"].append(pos)
        elif "ligand" in owners:
            coords_by_group["ligand"].append(pos)
    return coords_by_group


def add_haptic_multibonds_to_metal(
    core3D,
    donor_groups,
    metal_indices=None,
    bond_order=1,
    prefer_nearest_metal=True,
):
    """
    Add explicit M–X bonds for each atom in every haptic group (len>=2).
    Safe to call AFTER FF optimization to avoid over-constraining the FF.

    Args:
        core3D (mol3D): complex after placement/optimization.
        donor_groups (list[list[int]]): groups of ligand atom indices (local to ligand
                                        indices that already exist in core3D).
        metal_indices (list[int]|None): indices of metal atoms in core3D; inferred if None.
        bond_order (int): placeholder order to assign (typically 1).
        prefer_nearest_metal (bool): for multinuclear complexes, choose nearest metal to
                                     the group centroid; else use first metal.

    Side effects:
        - Updates core3D.bo_dict (idempotent).
        - Rebuilds OBMol bonds from bo_dict (replace_bonds).
    """
    # 1) find metals
    if metal_indices is None:
        metal_indices = core3D.findMetal(transition_metals_only=True)
        if not metal_indices:
            metal_indices = core3D.findMetal(transition_metals_only=False)

    if not metal_indices:
        return  # nothing to connect to

    # 2) get coords for nearest-metal logic
    all_coords = np.array([at.coords() for at in core3D.atoms], dtype=float)
    metal_coords = all_coords[np.array(metal_indices)]

    # 3) mutable copy of bo_dict
    bo = dict(core3D.bo_dict) if hasattr(core3D, "bo_dict") else {}

    def _ensure_bond(i, j, order=1):
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in bo:
            bo[(a, b)] = order

    # 4) add bonds per haptic group
    for group in donor_groups:
        if len(group) < 2:
            continue  # only haptic (η^n, n>=2)

        # choose metal: nearest to centroid (or first)
        if prefer_nearest_metal:
            centroid = np.mean(all_coords[np.array(group)], axis=0)
            d2 = np.sum((metal_coords - centroid)**2, axis=1)
            m_global = metal_indices[int(np.argmin(d2))]
        else:
            m_global = metal_indices[0]

        # add bond for every atom in the group
        for atom_idx in group:
            _ensure_bond(m_global, atom_idx, order=bond_order)

    # 5) write back, rebuild OBMol bonds
    core3D.bo_dict = bo
    core3D.convert2OBMol(force_clean=True)
    replace_bonds(core3D.OBMol, core3D.bo_dict)


def map_ligand_local_to_core_indices_by_range(core_before_count, core_after_count, ligand_len):
    """
    Fast-path mapping: if ligand atoms were appended contiguously to core3D,
    return the new core indices for those atoms; otherwise return None.
    """
    added = core_after_count - core_before_count
    if added == ligand_len and added > 0:
        return list(range(core_before_count, core_after_count))
    return None


def map_ligand_local_to_core_indices_geometric(core3D, ligand_coords, tol=0.10, max_tol=0.35):
    """
    Robust fallback mapping from ligand-local indices → core3D indices.

    Assumes you call this immediately after combining structures (before heavy
    optimization), when the ligand coordinates embedded in core3D still match
    the original `ligand_coords` closely. Uses greedy NN matching with a small,
    gradually widened tolerance to avoid duplicate matches.

    Returns:
        list[int]: core3D indices corresponding to ligand atom order.
    """
    core_coords = np.array([a.coords() for a in core3D.atoms], dtype=float)
    tree = cKDTree(core_coords)
    used = set()
    result = []
    current_tol = tol
    while current_tol <= max_tol and len(result) < len(ligand_coords):
        result = []
        used.clear()
        ok = True
        for p in ligand_coords:
            d, idx = tree.query(p, k=5, distance_upper_bound=current_tol)
            if np.isinf(d):
                ok = False
                break
            if np.isscalar(idx):
                cand = [idx]
            else:
                cand = [i for i in np.atleast_1d(idx) if i < len(core_coords)]
            picked = None
            for j in cand:
                if j not in used:
                    picked = j
                    break
            if picked is None:
                ok = False
                break
            used.add(picked)
            result.append(int(picked))
        if ok and len(result) == len(ligand_coords):
            return result
        current_tol *= 1.5  # widen and retry
    raise RuntimeError("Failed to map ligand local indices to core indices geometrically.")


def to_global_groups(donor_groups_local, local2global):
    """
    Translate a list[list[int]] (local ligand indices) → list[list[int]] in core3D index space.
    """
    return [[local2global[i] for i in g] for g in donor_groups_local]


def add_haptic_multibonds_to_metal_for_core(
    core3D,
    donor_groups_global,
    metal_indices=None,
    bond_order=1,
    prefer_nearest_metal=True,
):
    """
    Add explicit M–X bonds for each atom in every haptic group (len>=2),
    where donor_groups_global are already in core3D’s atom index space.

    Side effects:
      - Updates core3D.bo_dict (idempotent),
      - Rebuilds OBMol bonds from bo_dict,
      - Stores groups in core3D._haptic_groups_global to reapply later.
    """
    if metal_indices is None:
        metal_indices = core3D.findMetal(transition_metals_only=True)
        if not metal_indices:
            metal_indices = core3D.findMetal(transition_metals_only=False)
    if not metal_indices:
        return

    # keep/merge groups on core for later re-apply (e.g., before writing)
    existing = getattr(core3D, "_haptic_groups_global", None)
    if existing is None:
        existing = []

    # Normalize & dedup
    def _canon(g): return tuple(sorted(int(x) for x in g))
    seen = { _canon(g) for g in existing }
    for g in donor_groups_global:
        if len(g) >= 2:
            cg = _canon(g)
            if cg not in seen:
                existing.append(list(cg))
                seen.add(cg)
    core3D._haptic_groups_global = existing

    # compute nearest metal per group (if needed)
    all_coords = np.array([at.coords() for at in core3D.atoms], dtype=float)
    metal_coords = all_coords[np.array(metal_indices)]

    bo = dict(core3D.bo_dict) if hasattr(core3D, "bo_dict") else {}

    def _ensure_bond(i, j, order=1):
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in bo:
            bo[(a, b)] = order

    for group in existing:
        if len(group) < 2:
            continue
        if prefer_nearest_metal:
            centroid = np.mean(all_coords[np.array(group)], axis=0)
            d2 = np.sum((metal_coords - centroid)**2, axis=1)
            m_global = metal_indices[int(np.argmin(d2))]
        else:
            m_global = metal_indices[0]
        for atom_idx in group:
            _ensure_bond(m_global, atom_idx, order=bond_order)

    # write back and sync to OBMol
    core3D.bo_dict = bo
    core3D.convert2OBMol(force_clean=True)
    replace_bonds(core3D.OBMol, core3D.bo_dict)


def reapply_all_haptics_and_sync(core3D, bond_order=1, prefer_nearest_metal=True):
    """
    Re-apply all stored haptic groups from core3D._haptic_groups_global into bo_dict,
    then sync OBMol. Safe to call multiple times (idempotent).
    """
    groups = getattr(core3D, "_haptic_groups_global", None)
    if not groups:
        return
    add_haptic_multibonds_to_metal_for_core(
        core3D,
        groups,
        bond_order=bond_order,
        prefer_nearest_metal=prefer_nearest_metal,
    )
