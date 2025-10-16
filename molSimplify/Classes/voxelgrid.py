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

    def copy(self):
        from copy import deepcopy
        new_grid = VoxelGrid(voxel_size=self.voxel_size, vdw_scale=self.vdw_scale)
        new_grid.grid = deepcopy(self.grid)
        new_grid.complex_voxels = set(self.complex_voxels)
        new_grid.ligand_voxels = set(self.ligand_voxels)
        new_grid.atom_map = deepcopy(self.atom_map)
        return new_grid

    def get_clash_severity(self, coords_lig, elements_lig, coords_complex, elements_complex, vdw_radii=None, scale=0.9):
        """
        Computes the total severity of all ligand-complex clashes.
        Returns:
            total_severity: float (sum over all clash pairs)
            severity_dict: dict mapping (lig_idx, comp_idx) -> severity
        """
        if vdw_radii is None:
            vdw_radii = self.vdw_radii  # your global dict
    
        # Get all clashing atom pairs
        clash_pairs, ligand_ids, complex_ids = self.get_clashing_atoms()
    
        total_severity = 0.0
        severity_dict = {}
        for lig_idx, comp_idx in clash_pairs:
            d = np.linalg.norm(coords_lig[lig_idx] - coords_complex[comp_idx])
            r_sum = vdw_radii.get(elements_lig[lig_idx], 1.5) + vdw_radii.get(elements_complex[comp_idx], 1.5)
            r_sum *= scale
            if d < r_sum:
                severity = r_sum - d
                severity_dict[(lig_idx, comp_idx)] = severity
                total_severity += severity
        return total_severity, severity_dict
    
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

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def plot_all_voxels(
    voxel_grid, center=None, span=12.0,  # span in Å (adjust to your system)
    mode="cube", size=50, silent=False, ax=None, filename=None, show_legend=True
):
    """
    Plots all voxels, centered and with equal axes/borders for movie-quality frames.
    - center: (x, y, z) tuple to center the plot (defaults to metal or mean complex position)
    - span: length of each axis (Å) (e.g., 12.0 means axes go from center-6 to center+6)
    """
    voxel_size = voxel_grid.voxel_size
    x, y, z, colors = [], [], [], []

    for idx, owners in voxel_grid.grid.items():
        owners = set(owners)
        x.append(idx[0] * voxel_size)
        y.append(idx[1] * voxel_size)
        z.append(idx[2] * voxel_size)
        if "complex" in owners and "ligand" in owners:
            colors.append("red")      # overlap/clash
        elif "complex" in owners:
            colors.append("green")    # complex only
        elif "ligand" in owners:
            colors.append("blue")     # ligand only
        else:
            colors.append("grey")     # just in case

    if len(x) == 0:
        if not silent:
            print("[plot_voxels] No voxels to plot. Showing empty plot.")
        return

    if ax is None:
        fig = plt.figure(figsize=(8, 6), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # Center and span: set axes
    if center is None:
        # Default: center on mean of complex voxels (green)
        cx = [a for a,c in zip(x,colors) if c == "green"]
        cy = [a for a,c in zip(y,colors) if c == "green"]
        cz = [a for a,c in zip(z,colors) if c == "green"]
        if len(cx) > 0:
            center = (np.mean(cx), np.mean(cy), np.mean(cz))
        else:
            center = (np.mean(x), np.mean(y), np.mean(z))
    xlim = (center[0] - span/2, center[0] + span/2)
    ylim = (center[1] - span/2, center[1] + span/2)
    zlim = (center[2] - span/2, center[2] + span/2)

    if mode == "scatter":
        ax.scatter(x, y, z, c=colors, s=size, alpha=0.6)
    elif mode == "cube":
        ax.bar3d(x, y, z,
                 dx=voxel_size, dy=voxel_size, dz=voxel_size,
                 color=colors, alpha=0.8, edgecolor='none')
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Use 'scatter' or 'cube'.")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_axis_off()
    ax.set_facecolor('white')
    if hasattr(fig, 'patch'):
        fig.patch.set_facecolor('white')

    # Optionally add legend
    if show_legend:
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green', label='Complex', markersize=10),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', label='Ligand', markersize=10),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', label='Clash', markersize=10),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close(fig)
    
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
    plt.close(fig)
