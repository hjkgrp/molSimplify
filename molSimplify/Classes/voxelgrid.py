import numpy as np
from molSimplify.Classes.globalvars import vdwrad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

class VoxelGrid:
    def __init__(self, voxel_size=0.5, vdw_scale=0.9):
        """
        voxel_size: � per voxel
        vdw_scale: shrink factor for van der Waals radii (0.9 = 90% size)
        """
        self.voxel_size = voxel_size
        self.vdw_scale = vdw_scale
        self.grid = {}  # (i, j, k) -> set of {"complex", "ligand"}
        self.complex_voxels = set()
        self.ligand_voxels = set()
        self.atom_map = {}  # (i, j, k) -> list of atom ids
        self.vdw_radii = vdwrad

    def _to_voxel_index(self, coord):
        return tuple((np.floor(coord / self.voxel_size)).astype(int))

    def _get_voxel_sphere_indices(self, center, radius):
        r_voxels = int(np.ceil(radius / self.voxel_size))
        center_idx = self._to_voxel_index(center)
        sphere_indices = []

        for dx in range(-r_voxels, r_voxels + 1):
            for dy in range(-r_voxels, r_voxels + 1):
                for dz in range(-r_voxels, r_voxels + 1):
                    offset = np.array([dx, dy, dz])
                    point = (np.array(center_idx) + offset) * self.voxel_size
                    if np.linalg.norm(point - center) <= radius:
                        sphere_indices.append(tuple(center_idx + offset))

        return sphere_indices

    def add_atom(self, element, coord, atom_id=None, group="complex"):
        base_radius = self.vdw_radii.get(element, 1.5)
        radius = base_radius * self.vdw_scale
        voxel_indices = self._get_voxel_sphere_indices(np.array(coord), radius)

        for idx in voxel_indices:
            self.grid.setdefault(idx, set()).add(group)
            if group == "complex":
                self.complex_voxels.add(idx)
            elif group == "ligand":
                self.ligand_voxels.add(idx)

            # Track atom IDs and groups per voxel
            self.atom_map.setdefault(idx, []).append((atom_id, group))

    def add_atoms(self, elements, coords, atom_ids=None, group="complex", auto_label=False):
        """
        Add multiple atoms to the voxel grid.

        Args:
            elements: list of str, element symbols (e.g. ['C', 'H', 'N'])
            coords: (N, 3) ndarray of coordinates
            atom_ids: optional list of atom IDs (e.g. integers or labels)
            group: 'complex' or 'ligand'
            auto_label: if True, generates labels like C1, C2, N1, ...
        """
        if len(elements) != len(coords):
            raise ValueError("elements and coords must have the same length")

        if atom_ids is not None and len(atom_ids) != len(coords):
            raise ValueError("If provided, atom_ids must match number of atoms")

        # Automatically generate atom labels like N1, C3 if requested
        if auto_label:
            element_counts = defaultdict(int)
            atom_ids = []
            for elem in elements:
                element_counts[elem] += 1
                atom_ids.append(f"{elem}{element_counts[elem]}")
        elif atom_ids is None:
            atom_ids = list(range(len(elements)))  # default to 0-based index

        for elem, coord, aid in zip(elements, coords, atom_ids):
            self.add_atom(elem, coord, atom_id=aid, group=group)

    def get_voxel_status(self, coord, radius):
        """
        Returns:
            dict with counts of 'complex', 'ligand', 'both'
        """
        counts = {"complex": 0, "ligand": 0, "both": 0}
        for idx in self._get_voxel_sphere_indices(np.array(coord), radius):
            owners = self.grid.get(idx, set())
            if "complex" in owners and "ligand" in owners:
                counts["both"] += 1
            elif "complex" in owners:
                counts["complex"] += 1
            elif "ligand" in owners:
                counts["ligand"] += 1
        return counts

    def has_cross_clash(self, coord, radius):
        """True if this atom would occupy a voxel already held by the opposite group"""
        for idx in self._get_voxel_sphere_indices(np.array(coord), radius):
            owners = self.grid.get(idx, set())
            if "complex" in owners and "ligand" in owners:
                return True
        return False

    def get_clashing_atoms(self, verbose=False):
        """Returns list of tuples: (ligand_atom_id, complex_atom_id)"""
        clashes = set()
        ligand_ids = set()
        complex_ids = set()
        overlapping_voxels = self.complex_voxels & self.ligand_voxels

        for voxel in overlapping_voxels:
            atoms_here = self.atom_map.get(voxel, [])
            ligands = [a for a in atoms_here if a[1] == "ligand"]
            complexes = [a for a in atoms_here if a[1] == "complex"]
            for l in ligands:
                for c in complexes:
                    clashes.add((l[0], c[0]))
                    ligand_ids.add(l[0])
                    complex_ids.add(c[0])

        if verbose:
            if clashes:
                print("⚠️ Atoms involved in clashes:")
                for lig_id, comp_id in clashes:
                    print(f"  Ligand atom {lig_id} clashes with complex atom {comp_id}")

        return list(clashes), ligand_ids, complex_ids

    def get_clashing_ligand_atoms(self):
        return list(set(lig for (lig, _) in self.get_clashing_atoms()))

    def has_voxel_clash(self):
        """Return True if any voxel is filled by both complex and ligand."""
        return len(self.complex_voxels & self.ligand_voxels) > 0

def plot_voxels(voxel_grid, group="both", mode="cube", size=50, silent=False):
    """
    Visualize occupied voxels from a VoxelGrid.

    Args:
        voxel_grid: VoxelGrid instance
        group: 'complex', 'ligand', or 'both'
        mode: 'scatter' (points) or 'cube' (cubic voxels)
        size: scatter size (ignored for cube)
        silent: if True, skip plotting when no voxels match
    """
    voxel_size = voxel_grid.voxel_size
    x, y, z = [], [], []
    colors = []

    for idx, owners in voxel_grid.grid.items():
        if group == "both" and {"complex", "ligand"} <= owners:
            pass
        elif group == "complex" and "complex" not in owners:
            continue
        elif group == "ligand" and "ligand" not in owners:
            continue
        elif group == "both" and {"complex", "ligand"} > owners:
            continue  # not a true clash
        x.append(idx[0] * voxel_size)
        y.append(idx[1] * voxel_size)
        z.append(idx[2] * voxel_size)

        if {"complex", "ligand"} <= owners:
            colors.append("red")  # clash
        elif "complex" in owners:
            colors.append("green")
        else:
            colors.append("blue")

    # Handle empty voxel set
    if len(x) == 0:
        if not silent:
            print(f"[plot_voxels] No voxels found for group='{group}' and mode='{mode}'. Showing empty plot.")
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Empty Voxel Grid")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            plt.tight_layout()
            plt.show()
        return

    # Actual plotting
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if mode == "scatter":
        ax.scatter(x, y, z, c=colors, s=size, alpha=0.6)
    elif mode == "cube":
        ax.bar3d(x, y, z,
                 dx=voxel_size, dy=voxel_size, dz=voxel_size,
                 color=colors, alpha=0.5, edgecolor='k', linewidth=0.1)
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Use 'scatter' or 'cube'.")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Voxel Visualization: {group} ({mode})")
    plt.tight_layout()
    plt.show()
