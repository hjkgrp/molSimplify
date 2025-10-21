import sys
import os
import numpy as np
import glob
from collections import deque
import pandas as pd

sys.path.append(os.path.expanduser("~/molSimplify/molSimplify/Classes/"))
sys.path.append(os.path.expanduser("~/molSimplify/molSimplify/Scripts/"))

from mol3D import mol3D
from atom3D import atom3D
from geometry import PointTranslateSph
from structgen import *

def getAtomsDistance(mol, atom_1, atom_2):
    """
    Calculate the distance between two atoms in a mol3D object.
    Parameters
    ----------
    mol : mol3D
        mol3D class instance to calculate distance.
    atom_1: int
        index of first atom.
    atom_2: int
        index of second atom.
    Returns
    -------
    distance : float
        Distance between the two atoms.
    """
    comp3D = mol3D()
    comp3D.copymol3D(mol)
    coords_1 = np.array(comp3D.getAtomCoords(atom_1))
    coords_2 = np.array(comp3D.getAtomCoords(atom_2))
    vector = coords_1 - coords_2
    distance = np.linalg.norm(vector)
    return [vector, distance]

def getAllAtomsIdx(mol):
    """
    Get all atom indices in a mol3D object.
    Parameters
    ----------
    mol : mol3D
        mol3D class instance to get atom indices from.
    Returns
    -------
    all_ids : list
        List of all atom indices in the mol3D object.
    """
    comp3D = mol3D()
    comp3D.copymol3D(mol)
    atom_types = comp3D.getAtomTypes()
    all_idx =[]
    # Get all atom indices for each atom type
    for i in atom_types:
        all_idx.extend(comp3D.findAtomsbySymbol(i))
    return all_idx

def getClosestAtoms(mol,ref_idx,cdist, symbol='None'):
    """
    Get closest atoms to a reference atom given a set distance.
    Parameters
    ----------
    mol: mol3D
        mol3D class instace where the function should operate.
    ref_idx: int
        index of reference atom
    cdist: float
        Cutoff distance to neighbor atoms
    symbol: str
        Symbol of the atom type to consider
    Returns
    -------
    neighbor_list : list
        List of indices for closest atoms.
    """
    comp3D=mol3D()
    comp3D.copymol3D(mol)
    metal_ids = comp3D.findMetal()
    metal_idx=metal_ids[0]
    index_list = getAllAtomsIdx(comp3D)
    neighbor_list = []
    if symbol != 'None':
        index_list = comp3D.findAtomsbySymbol(symbol)
    for idx in index_list:
        if idx != ref_idx:
            distance = getAtomsDistance(comp3D, ref_idx, idx)[1]
            if distance < cdist:
                neighbor_list.append((idx, distance))
    sorted_neighbor_list = sorted(neighbor_list, key=lambda x: x[1])
    return sorted_neighbor_list

def getAtomswithinDistance(mol, ref_idx, cdist):
    """
    Get atoms within a certain distance from a reference atom.
    Parameters
    ----------
    mol : mol3D
        mol3D class instance to search for atoms.
    ref_idx: int
        Index of the reference atom.
    cdist: float
        Cutoff distance to search for neighboring atoms.
    Returns
    -------
    atoms_within_distance : list
        List of indices of atoms within the specified distance from the reference atom.
    """
    comp3D = mol3D()
    comp3D.copymol3D(mol)
    all_idx = getAllAtomsIdx(comp3D)
    atoms_within_distance = []
    for atom_idx in all_idx:
        if atom_idx != ref_idx:
            distance = getAtomsDistance(comp3D, ref_idx, atom_idx)[1]
            if distance < cdist:
                atoms_within_distance.append(atom_idx)
    return atoms_within_distance

def getAllDistances(mol,ref, remove_atoms=None):
    """
    Get all distances from a reference atom to all other atoms in the mol3D object.
    Parameters
    ----------
    mol : mol3D
        mol3D class instance to calculate distances.
    ref: int or coords
        Index of the reference atom or coordinates.
    remove_atoms: list, optional
        List of atom indices to exclude from the distance calculation.
    Returns
    -------
    all_distances : list
        List of tuples containing atom index and distance from the reference atom.
    """
    comp3D = mol3D()
    comp3D.copymol3D(mol)
    all_idx = getAllAtomsIdx(comp3D)
    if isinstance(ref, int):
        ref_coords = comp3D.getAtomCoords(ref)
    else:
        ref_coords = ref
    if remove_atoms is None:
        remove_atoms = []
    all_distances = []
    all_distances_idx = []
    for atom_idx in all_idx:
        if atom_idx != ref if isinstance(ref,int) and atom_idx not in remove_atoms else atom_idx not in remove_atoms:
            atom_coords = comp3D.getAtomCoords(atom_idx)
            distance = np.linalg.norm(np.array(ref_coords) - np.array(atom_coords))
            all_distances.append(distance)
            all_distances_idx.append(atom_idx)
    return all_distances, all_distances_idx

def getDihedralAngle(mol, atom_list):
    """
    Calculate the dihedral angle (in degrees) between planes (a, b, c) and (a, b, d).
    Parameters
    ----------
    atom_list: list of int [a, b, c, d]
        Atom indices for the four points. A and B should form the reference vector.
    Returns
    -------
    dihedral : float
        Dihedral angle in degrees, in the range (-180, 180]
    """
    comp3D = mol3D()
    comp3D.copymol3D(mol)
    a, b, c, d = atom_list
    a = np.array(comp3D.getAtomCoords(a))
    b = np.array(comp3D.getAtomCoords(b))
    c = np.array(comp3D.getAtomCoords(c))
    d = np.array(comp3D.getAtomCoords(d))
    # Vectors in each plane
    ab = b - a
    ac = c - a
    ad = d - a
    # Normals to the planes
    n1 = np.cross(ab, ac)
    n2 = np.cross(ab, ad)
    # Normalize normals
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    # Angle between normals
    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, n2), ab / np.linalg.norm(ab))
    angle = np.degrees(np.arctan2(y, x))
    return angle

def get_possible_bond_angles(element):
    """
    Return a list of typical bond angles (in degrees) for a given element.
    Parameters
    ----------
    element : str
        The atomic symbol (e.g., 'O', 'C', 'N', 'H', etc.)
    Returns
    -------
    angles : list of float
        List of possible bond angles in degrees.
    """
    element = element.upper()
    bond_angles = {
        'H': [],
        'O': [104.5],                # e.g., H2O (bent)
        'N': [107, 120, 180],        # e.g., NH3 (pyramidal), sp2, sp
        'C': [109.5, 120, 180],      # sp3 (tetrahedral), sp2 (trigonal planar), sp (linear)
        'S': [92, 104.5, 107, 120],  # e.g., H2S, SO2, SO3
        'P': [93.5, 107, 120],       # e.g., PH3, PCl3, POCl3
        'F': [180],                  # e.g., F2 (linear)
        'CL': [180],                 # e.g., Cl2 (linear)
        'BR': [180],                 # e.g., Br2 (linear)
        'I': [180],                  # e.g., I2 (linear)
        'B': [120],                  # e.g., BF3 (trigonal planar)
        'SI': [109.5, 120, 180],     # similar to carbon
        'AL': [120],                 # e.g., AlCl3 (trigonal planar)
        'MG': [180],                 # e.g., MgCl2 (linear)
        'BE': [180],                 # e.g., BeCl2 (linear)
        # Add more elements as needed
    }
    return bond_angles.get(element, [])

def get_positions_around_point(center, radius=1.0, num_phi_angles=72, num_theta_angles=72):
    """
    Generate points on a sphere around a given center.
    Parameters
    ----------
    center : list or np.ndarray
        The [x, y, z] coordinates of the center point.
    radius : float, optional
        The radius of the sphere. Default is 1.0.
    num_phi_angles : int, optional
        Number of phi (azimuthal) angles. Default is 18.
    num_theta_angles : int, optional
        Number of theta (polar) angles. Default is 36.
    Returns
    -------
    points : list
        List of [x, y, z] coordinates on the sphere.
    """
    center = np.array(center)
    points = []
    phi_vals = np.linspace(0, 2 * np.pi, num_phi_angles, endpoint=False)
    theta_vals = np.linspace(0, np.pi, num_theta_angles, endpoint=True)
    for theta in theta_vals:
        for phi in phi_vals:
            x = center[0] + radius * np.sin(theta) * np.cos(phi)
            y = center[1] + radius * np.sin(theta) * np.sin(phi)
            z = center[2] + radius * np.cos(theta)
            points.append([x, y, z])
    return points

def reflect_bond(mol,atom_1,atom_2):
    """
    Find coordinates of reflection of bond through a given atom
    Parameters
    ----------
    mol : mol3D
        mol3D class instance of metal complex reflect bond.
    atom_1: int
        index of atom to work as a reflection point
    atom_2: int
        index of atom to be reflected
    Returns
    -------
    ref_site: lis
        Reflected site coordinates
    """
    # Find position of reflected site 
    c3D=mol3D()
    c3D.copymol3D(mol)
    a1_coords=c3D.getAtomCoords(atom_1)
    a1_array=np.array(a1_coords)
    a2_coords=c3D.getAtomCoords(atom_2)
    a2_array=np.array(a2_coords)
    ref_array = 2*a1_array-a2_array
    ref_site = ref_array.tolist()
    return ref_site

def find_vacant_site(mol):
    """
    Find vacant site in metal complex.
    Parameters
    ----------
    mol : mol3D
        mol3D class instance of metal complex to find vacant site.
    Returns
    -------
    vacant_site: lis
        Vacant site coordinates
    """
    # Create mol3D for transition metal complex
    comp3D = mol3D()
    comp3D.copymol3D(mol)
    metal_ids = comp3D.findMetal()
    metal_idx=metal_ids[0]
    # Identify ligand index
    ligands = comp3D.getBondedAtoms(metal_idx)
    lig_opposite_open_site=[]
    # Find ligand in a position with opposite to vacant site
    for lig in ligands:
        lig_angles=[]
        for i in ligands:
            if i!=lig:
                lig_angles.append(comp3D.getAngle(lig, metal_idx,i))
        if all(angles < 170 for angles in lig_angles):
            lig_opposite_open_site.append(lig)
    # Find position of opposing open site 
    ax_lig_idx = lig_opposite_open_site[0]
    vacant_site=reflect_bond(comp3D, metal_idx, ax_lig_idx)
    return vacant_site

def extract_min_distance(path_to_xyz, ref_idx, atom_symbol,cdist):
    """    
    Extract the minimum distance between a reference atom and atoms of a specific type.
    Parameters
    ----------
    path_to_xyz : str
        Path to the .xyz file containing the molecular structure.
    ref_idx : int
        Index of the reference atom.
    atom_symbol : str
        Symbol of the atoms to search for (e.g., 'H' for hydrogen).
    cdist : float
        Cutoff distance to consider for finding the closest atoms.
    Returns
    -------
    min_distance_atom : int
        Index of the atom with the minimum distance to the reference atom.
    min_distance : float
        Minimum distance between the reference atom and the closest atom of the specified type.
    """
    # Retrieve optimized geometry
    comp3d = mol3D()
    comp3d.readfromxyz(path_to_xyz,read_final_optim_step=True)
    # Find closest hydrogen in methane to oxo
    closest_atoms = [i[0] for i in getClosestAtoms(comp3d, ref_idx, cdist)]
    desired_atoms = comp3d.findAtomsbySymbol(atom_symbol)
    # Filter closest atoms to only include those of the desired type
    filtered_atoms = [atom for atom in closest_atoms if atom in desired_atoms]
    if not filtered_atoms:
        print(f"No atoms of type {atom_symbol} found within {cdist} Å of atom index {ref_idx}.")
        return None, None
    # Calculate distances to the filtered atoms
    distances = []
    for atom in filtered_atoms:
        distance = getAtomsDistance(comp3d, atom, ref_idx)[1]
        distances.append((atom, distance))
    # Find the atom with the minimum distance
    min_distance_atom, min_distance = min(distances, key=lambda x: x[1])
    return min_distance_atom, min_distance

def rotate_point_around_axis(rot_point, axis_point, axis_vector, bond_angle):
    """
    Rotates a point in 3D around a line defined by an axis point and direction vector.

    Parameters:
        rot_point : list
            The point to rotate (e.g., [x, y, z]).
        axis_point : list
            A point on the axis of rotation.
        axis_vector : list
            Direction vector of the axis.
        theta : float
            Angle of rotation in degrees.

    Returns:
        rotated_point : lis
            The rotated point.
    """
    P = np.array(rot_point)
    A = np.array(axis_point)
    v = axis_vector
    v = v / np.linalg.norm(v)  # Normalize the axis vector
    theta=bond_angle*np.pi/180
    p = P - A  # Translate point to origin
    v_cross_p = np.cross(v, p)
    v_dot_p = np.dot(v, p)
    rotated_vector=(p * np.cos(theta) + v_cross_p * np.sin(theta) + v * v_dot_p * (1 - np.cos(theta)))
    rotated_point=(A+rotated_vector).tolist()
    return rotated_point

def freeze_atoms(mol, list_idx_not_to_freeze=False):
    """
    Creates the list of atoms to freeze, excluding list
    Parameters:
    ----
        list_idx_not_to_freeze : list
            List of atoms indices not to freeze.
    Returns:
    ---_
        freeze : lis
            List of atoms to freeze.
    """
    # Create list to freeze complex for FF optimization
    fe_complex=mol3D()
    fe_complex.copymol3D(mol)
    atom_types = fe_complex.getAtomTypes()
    freeze =[]
    for i in atom_types:
        freeze.extend(fe_complex.findAtomsbySymbol(i))
    # Remove oxo to FF optimize it
    for i in list_idx_not_to_freeze:
        if  i in freeze:
            freeze.remove(i)
    return freeze

def define_best_orientation(mol, ref_atom, radius=1.6, num_phi_angles=36, num_theta_angles=36, remove_atoms=None, phi_list=None, theta_list=None):
    """
    Define the best orientation for a ligand based on minimizing distance to closest atoms.
    Parameters
    ----------
    mol : mol3D
        mol3D class instance of metal complex to orient ligand.
    ref_atom : int
        Index of the reference atom (metal or other atom).
    radius : float, optional
        Radius around the reference atom to search for ligand orientations. Default is 1.0.
    num_phi_angles : int, optional
        Number of phi angles to test for ligand orientation. Default is 36.
    num_theta_angles : int, optional
        Number of theta angles to test for ligand orientation. Default is 36.
    Returns
    -------
    comp3D : mol3D
        mol3D class instance of the metal complex with the ligand in the best orientation.
    """
    # Create mol3D for transition metal complex
    comp3D = mol3D()
    comp3D.copymol3D(mol)
    ref_coords = comp3D.getAtomCoords(ref_atom)
    possible_positions = get_positions_around_point(ref_coords, radius=radius, num_phi_angles=num_phi_angles, num_theta_angles=num_theta_angles, phi_list=phi_list, theta_list=theta_list)
    min_distances = []
    for pos in possible_positions:
        distances, ids = getAllDistances(comp3D, pos, remove_atoms=remove_atoms)
        sorted_distances = sorted(distances)
        if sorted_distances:
            min_distance = sorted_distances[0]
            min_distances.append(min_distance)
        else:
            min_distances.append(float('-inf'))
    max_distance = max(min_distances)
    best_rotated_site = possible_positions[min_distances.index(max_distance)]
    return best_rotated_site

def define_possible_sites(mol, phi_angles, ref_atom_1, ref_atom_2, num_theta_angles=36, distance = None, linear = True):
    """
    Define possible sites for a ligand based on given angles.
    Parameters
    ----------
    mol : mol3D
        mol3D class instance of metal complex to orient ligand.
    phi_angles : list
        List of phi angles in degrees to test for ligand orientation.
    num_theta_angles : int
        Number of theta angles to test for ligand orientation.
    ref_atom_1 : int
        Index of the first reference atom (metal or other atom).
    ref_atom_2 : int
        Index of the second reference atom (bonded atom).
    Returns
    -------
    possible_sites : list
        List of possible sites for the ligand, each site is a list containing coordinates, type, and angle.    
    """
    comp3D = mol3D()
    comp3D.copymol3D(mol)
    # Set of sites for first nitrogen
    vacant_site = reflect_bond(comp3D,ref_atom_1,ref_atom_2)
    lig=atom3D('N',vacant_site)
    rot_atom=ref_atom_1+1
    comp3D.addAtom(lig,rot_atom,auto_populate_bo_dict=False)
    if distance is not None:
        comp3D.BCM(rot_atom,ref_atom_1,distance)
    # Get reference atom coords
    ref1_coords = comp3D.getAtomCoords(ref_atom_1)
    rot_coords = comp3D.getAtomCoords(rot_atom)
    ref2_coords = comp3D.getAtomCoords(ref_atom_2)
    # Get metal reference bond vector and length
    ref_bond_vector, ref_bond_length = getAtomsDistance(comp3D, ref_atom_1, ref_atom_2)
    bond_vector, bond_length = getAtomsDistance(comp3D, rot_atom, ref_atom_1)
    theta_angles = np.linspace(0, 360, num_theta_angles, endpoint=False)
    possible_sites = []
    if linear:
        possible_sites.append([vacant_site, "linear", 0, 0])
    for phi in phi_angles:
        bond_angle_rad = (180 - phi) * np.pi / 180
        # Rotate the point around the axis defined by the reference bond vector
        first_rotation = PointTranslateSph(ref1_coords, rot_coords, [bond_length, 0, bond_angle_rad])
        # Add the rotated site to the list of possible sites
        for ang in theta_angles:
            rotated_site = rotate_point_around_axis(first_rotation, rot_coords, ref_bond_vector, ang)
            possible_sites.append([rotated_site, "bent", ang, phi])
    return possible_sites

def define_best_site(mol, candidate_sites, exclude_indices=None):
    """
    For each candidate site, find the minimum distance to all atoms in the molecule,
    excluding specified atom indices.
    Parameters
    ----------
    mol : mol3D
    mol3D class instance.
    candidate_sites : list
    List of candidate coordinates (each is a list or array of 3 floats).
    exclude_indices : list, optional
    List of atom indices to exclude from the distance calculation.
    Returns
    -------
    min_distances : list of tuples
    Each tuple is (site, closest_atom_idx, min_distance)
    """
    comp3D = mol3D()
    comp3D.copymol3D(mol)
    if exclude_indices is None:
        exclude_indices = []
    all_atom_idx = getAllAtomsIdx(comp3D)
    for idx in exclude_indices:
        if idx in all_atom_idx:
            all_atom_idx.remove(idx)
    all_atom_coords = [(idx, comp3D.getAtomCoords(idx)) for idx in all_atom_idx]
    min_distances = []
    for site in candidate_sites:
        distances = []
        for coords in all_atom_coords:
            site_coords = site[0] if isinstance(site, (list, tuple)) and len(site) > 0 and isinstance(site[0], (list, tuple, np.ndarray)) else site
            distance = np.linalg.norm(np.array(site_coords) - np.array(coords[1]))
            distances.append((coords[0], distance))
        if distances:
            sorted_distances = sorted(distances, key=lambda x: x[1])
            sorted_only_distances = [d[1] for d in sorted_distances]
            min_distances.append([site, sorted_only_distances[0], sorted_only_distances[1], sorted_only_distances[2], sorted_only_distances[3], sorted_distances[4], sorted_distances[5]])
        else:
            min_distances.append([site, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')])
    best_guess = max(min_distances, key=lambda x: (x[1],x[2],x[3], x[4], x[5]), default=None)
    # best_guess[0] is the site, best_guess[1:4] are the first, second, third min distances
    return best_guess

def define_best_n2o_orientation(mol, n1_bond_angles, n2_bond_angles, ref_atom_1, ref_atom_2, num_theta_angles=36):
    """
    Define the best orientation for a nitrogen dimer based on minimizing distance to closest atoms.
    Parameters
    ----------
    mol : mol3D
        mol3D class instance of metal complex to orient nitrogen dimer.
    n1_bond_angles : float
        Angle in degrees between the bond vector and the reference atom.
    n2_bond_angles : float
        Angle in degrees between the bond vector of the second nitrogen and the first nitrogen.
    ref_atom_1 : int
        Index of the first reference atom (oxygen).
    ref_atom_2 : int
        Index of the second reference atom (metal).
    num_phi_angles : int, optional
        Number of phi angles to test for ligand orientation. Default is 36.
    Returns
    -------
    comp3D : mol3D
        mol3D class instance of the metal complex with the nitrogen dimer in the best orientation.
    """
    comp3D = mol3D()
    comp3D.copymol3D(mol)
    # Set of sites for first nitrogen
    if isinstance(n1_bond_angles, int):
        n1_possible_angles = np.linspace(0, 60, n1_bond_angles, endpoint=False)
    elif isinstance(n1_bond_angles, list):
        n1_possible_angles = n1_bond_angles
    else:
        raise ValueError("n1_bond_angles should be either an integer or a list of angles.")
    if isinstance(n2_bond_angles, int):
        n2_possible_angles = np.linspace(0, 60, n2_bond_angles, endpoint=False)
    elif isinstance(n2_bond_angles, list):
        n2_possible_angles = n2_bond_angles
    else:
        raise ValueError("n2_bond_angles_no should be either an integer or a list of angles.")
    n1_possible_sites = define_possible_sites(comp3D, n1_possible_angles, ref_atom_1, ref_atom_2, num_theta_angles=num_theta_angles)
    distances = []
    n1_total = len(n1_possible_sites)
    for i, site in enumerate(n1_possible_sites, 1):
        print(f"Processing N1 site {i}/{n1_total} ({100*i/n1_total:.1f}%)")
        ang3D = mol3D()
        ang3D.copymol3D(mol)
        # Add nitrogen atom at the best rotated site
        n_atom = atom3D('N', site[0])
        n_idx = ref_atom_1 + 1
        ang3D.addAtom(n_atom, n_idx, auto_populate_bo_dict=False)
        n1_closest_atoms = [i[0] for i in getClosestAtoms(ang3D, n_idx, 4.0)]  # Using cutoff distance of 4.0 Å
        if ref_atom_1 in n1_closest_atoms:
            n1_closest_atoms.remove(ref_atom_1)
        if ref_atom_2 in n1_closest_atoms:
            n1_closest_atoms.remove(ref_atom_2)           
        n1_closest_atoms_distance = [(a, getAtomsDistance(ang3D, n_idx, a)[1]) for a in n1_closest_atoms]
        n1_min_distance = min(n1_closest_atoms_distance, key=lambda x: x[1], default=(None, float('inf')))
        # Check if the structure is intact
        # Get all atom coords
        all_atom_idx = getAllAtomsIdx(ang3D)
        if ref_atom_1 in all_atom_idx:
            all_atom_idx.remove(ref_atom_1)
        if n_idx in all_atom_idx:
            all_atom_idx.remove(n_idx)
        all_atom_coords = [ang3D.getAtomCoords(idx) for idx in all_atom_idx]
        # Add second nitrogen atom at the reflected site
        n2_distances =[]
        n2_possible_sites = define_possible_sites(ang3D, n2_possible_angles, n_idx, ref_atom_1, num_theta_angles=num_theta_angles)
        for n2_site in n2_possible_sites:
            for atom in all_atom_coords:
                distance = np.linalg.norm(np.array(n2_site[0]) - np.array(atom))
                n2_distances.append((atom, distance))
            n2_min_distance = min(n2_distances, key=lambda x: x[1], default=(None, float('inf')))
            distances.append((site, n2_site, n1_min_distance[1], n2_min_distance[1]))
    best_orientation = max(distances, key=lambda x: (x[3], x[2]), default=None)
    if best_orientation:
        print(f"Best orientation found with N1 at {best_orientation[0][0]} is {best_orientation[0][1]} at an angle of {best_orientation[0][2]} degrees")
        print(f"and N2 at {best_orientation[1][0]} is {best_orientation[1][1]} at an angle of {best_orientation[1][2]} degrees")
    return (best_orientation[0], best_orientation[1]) if best_orientation else None

def find_terminal_ligand(mol, element='O'):
    """
    Find the terminal ligand in the complex.
    Parameters
    ----------
    mol : mol3D
        mol3D class instance of the complex.
    element : str
        Element symbol of the ligand to find. Of the form 'O'. Default is 'O'.
    moiety : list
        List of element symbols that make up the ligand moiety, maximum of 3 atom moiety. Default is ['O', 'H'].
    Returns
    -------
    o_idx : int
        Index of the oxo ligand atom.
    """
    comp3D = mol3D()
    comp3D.copymol3D(mol)
    metal_ids = comp3D.findMetal()
    for mif in metal_ids:
        ligands = comp3D.getBondedAtoms(mif)
        x_ligand = []
        if element:
            all_x = comp3D.findAtomsbySymbol(element)
            for lig in ligands:
                if lig in all_x:
                    x_ligand.append(lig)
    terminals = []
    for x in x_ligand:
        x_bonded_atoms = comp3D.getBondedAtoms(x)
        num_bonded_atoms = len(x_bonded_atoms)
        if num_bonded_atoms == 1:
            terminals.append(x)
    if len(terminals) > 1:
        print("Error: More than 1 terminal ligand found")
        return False
    elif len(terminals) == 0:
        print("Error: No terminal ligand found")
        return False
    else:
        t_idx = terminals[0]
    return t_idx

def find_moieties(mol, moiety, metal_ids=None, distance_cutoff=4.0):
    """
    Find all occurrences of a moiety in the molecule.
    Parameters
    ----------
    mol : mol3D
        mol3D class instance of the molecule.
    moiety : list
        List of [parent_index, element] pairs defining the moiety. 
        Example: [[0, 'C'], [1, 'H'], [1, 'H'], [1, 'H']] for a methyl group.
    metal_ids : list, optional
        List of metal atom indices to consider as potential parents.
    Returns
    -------
    matches: list
        List of matching atom indices for the moiety. 
        Example: [metal_idx, c_idx, h1_idx, h2_idx, h3_idx],[metal_idx, c2_idx, h4_idx, h5_idx, h6_idx]...
    """
    comp3D = mol3D()
    comp3D.copymol3D(mol)
    if metal_ids is None:
        metal_ids = comp3D.findMetal()
    matches = []
    def search(path, depth):
        if depth == len(moiety):
            matches.append(path[:])
            return
        parent_idx = moiety[depth][0]
        element = moiety[depth][1]
        parent_atom = path[parent_idx]
        bonded = comp3D.getBondedAtoms(parent_atom)
        found = False
        for atom in bonded:
            if atom not in path and comp3D.getAtomwithinds([atom])[0].symbol() == element:
                found = True
                search(path + [atom], depth + 1)
        # Fallback: if not found, try closest atom of correct type within cutoff
        if not found:
            all_candidates = comp3D.findAtomsbySymbol(element)
            parent_coords = comp3D.getAtomCoords(parent_atom)
            closest = None
            min_dist = float('inf')
            for atom in all_candidates:
                if atom not in path:
                    coords = comp3D.getAtomCoords(atom)
                    dist = np.linalg.norm(np.array(parent_coords) - np.array(coords))
                    if dist < distance_cutoff and dist < min_dist:
                        min_dist = dist
                        closest = atom
            if closest is not None:
                search(path + [closest], depth + 1)
    # Start search from each possible first atom (e.g., O for oxo)
    first_element = moiety[0][1]
    for mid in metal_ids:
        candidates = [a for a in comp3D.getBondedAtoms(mid) if comp3D.getAtomwithinds([a])[0].symbol() == first_element]
        for atom in candidates:
            search([mid, atom], 1)
    # Error handling and fallback
    if len(matches) > 1:
        print("More than 1 moiety found")
        return matches
    elif len(matches) == 0:
        print("Error: No exact moiety found, trying closest match")
        # Try a looser search: start from all atoms of the first element
        all_first = comp3D.findAtomsbySymbol(first_element)
        for atom in all_first:
            search([atom], 1)
        if len(matches) == 0:
            print("Error: No moiety found even with loose search")
            return False
    return matches if matches else False

def find_oxo_n2_ligand(path_to_xyz): 
    """
    Find the oxo ligand and its bonded nitrogen atom in the given molecule.
    Parameters
    ----------
    mol : mol3D
        The 3D molecular structure to search within.
    Returns
    -------
    oxo_n2_ligand : list
        A list containing the indices of the oxo ligand and its two bonded nitrogen atoms.
    """
    # Create mol3D for transition metal complex
    comp3D = mol3D()
    comp3D.readfromxyz(path_to_xyz, read_final_optim_step=True)
    metal_ids = comp3D.findMetal()
    metal_idx = metal_ids[0]
    # Identify oxo ligand
    ligands=comp3D.getBondedAtoms(metal_idx)
    all_oxy=comp3D.findAtomsbySymbol('O')
    all_nit=comp3D.findAtomsbySymbol('N')
    # Find oxo ligand with N₂ bonded
    o_ligand = []
    oxo_n2_ligands = []
    oxo_n2_ligan = []
    # Get surrounding atoms of the metal
    surrounding_atoms = getAtomswithinDistance(comp3D, metal_idx, 3.0)
    surrounding_os = []
    for atom in surrounding_atoms:
        if atom in all_oxy and atom not in ligands:
            surrounding_os.append(atom)
    ligands.extend(surrounding_os)
    print (ligands)
    # Filter ligands to find oxo ligands
    for lig in ligands:
        if lig in all_oxy:
            o_ligand.append(lig)
    for o in o_ligand:
        o_bonded_atoms=comp3D.getBondedAtoms(o)
        for n in o_bonded_atoms:
            if n in all_nit:
                o_n_bonded_atoms=comp3D.getBondedAtoms(n)
                for n2 in o_n_bonded_atoms:
                    if n2 in all_nit:
                        o_n2_bonded_atoms=comp3D.getBondedAtoms(n2)
                        num_bonded_atoms=len(o_n2_bonded_atoms)
                        if num_bonded_atoms==1:
                            oxo_n2_ligands.append([o,n,n2])
            else:
                oxo_n2_ligands = []
    if len(oxo_n2_ligands)>1:
        print("Error: More than 1 oxo with N₂ bonded")
        return False
    elif len(oxo_n2_ligands)==0:
        print("Error: No oxo with N₂ bonded, finding oxo with closest N2")
        oxo_n_pairs = []
        for oxo in o_ligand:
            initial_cutoff = 4
            closest_n = extract_min_distance(path_to_xyz, oxo, 'N', initial_cutoff)
            oxo_n_pairs.append((oxo, closest_n[0], closest_n[1]))
        for oxo_n in oxo_n_pairs:
            n_bonded_atoms = comp3D.getBondedAtoms(oxo_n[1])
            for n in n_bonded_atoms:
                if n in all_nit:
                    n_bonded_atoms = comp3D.getBondedAtoms(n)
                    num_bonded_atoms = len(n_bonded_atoms)
                    if num_bonded_atoms == 1:
                        oxo_n2_ligan.append((oxo_n[0], oxo_n[1], n))
        if len(oxo_n2_ligan) == 3:
            o_idx = oxo_n2_ligan[0]
            n1_idx = oxo_n2_ligan[1]
            n2_idx = oxo_n2_ligan[2]
        else:
            print("Error: No oxo with N₂ close by")
            return False
    else:
        oxo_n2_ligand = oxo_n2_ligands[0]
        o_idx=oxo_n2_ligand[0]
        n1_idx=oxo_n2_ligand[1]
        n2_idx=oxo_n2_ligand[2]
    return [o_idx, n1_idx, n2_idx]

def read_substrate_xyz(path_to_xyz, bonding_atom):
    """
    Read a substrate .xyz file and extract internal coordinates bonding information for all atoms.
    Parameters:
    -----------
    path_to_xyz : str
        Path to the substrate .xyz file.
    bonding_atom : int
        Index of the atom to which the substrate will be bonded.
    Returns:
    -------
    atom_data: list
        A list of dictionaries containing bonding information for each atom containing:
        - grandparent: The atom bonded to the parent atom
        - parent: The parent atom
        - atom: The current atom
        - element: The element type of the current atom
        - distance: The distance from the parent atom
        - angle: The angle between the parent, current, and the grandparent atom
        - dihedral_angle: The dihedral angle involving the current atom
        - dihedral_neighbor: The neighbor atom involved in the dihedral angle
    """
    sub3D = mol3D()
    sub3D.readfromxyz(path_to_xyz, read_final_optim_step=True)
    bonding_element = sub3D.getAtomwithinds([bonding_atom])[0].symbol()
    queue = deque([(bonding_atom, None)])  # (current_atom, parent_atom)
    visited = set([bonding_atom])
    atom_data = [{
        "grandparent": None, 
        "parent": None, 
        "atom": bonding_atom, 
        "element": bonding_element, 
        "distance": 0, 
        "angle": None,
        "dihedral_angle": None,
        "dihedral_neighbor": None
    }]
    all_atom_idx = getAllAtomsIdx(sub3D)
    while queue:
        current_atom, parent_atom = queue.popleft()
        bonded_atoms = sub3D.getBondedAtoms(current_atom)
        for atom in bonded_atoms:
            if atom not in visited:
                # Calculate position, distance, angle, etc.
                distance = getAtomsDistance(sub3D, current_atom, atom)[1]
                element = sub3D.getAtomwithinds([atom])[0].symbol()
                # Check if there are other atoms at the same level (i.e., bonded to the same parent)
                same_level_atom = [entry for entry in atom_data if entry["parent"] == current_atom]
                if same_level_atom:
                    dihedral_angle = getDihedralAngle(sub3D, [parent_atom, current_atom, atom, same_level_atom[0]["atom"]])
                    dihedral_neighbor = same_level_atom[0]["atom"]
                else:
                    dihedral_angle = None
                    dihedral_neighbor = None
                if parent_atom is not None:
                    angle = sub3D.getAngle(parent_atom, current_atom, atom)
                    # Store the path and geometric info
                    atom_data.append({
                        "grandparent": parent_atom,
                        "parent": current_atom,
                        "atom": atom,
                        "element": element,
                        "distance": distance,
                        "angle": angle,
                        "dihedral_angle": dihedral_angle,
                        "dihedral_neighbor": dihedral_neighbor
                    })
                else:
                    # No angle for the first step
                    atom_data.append({
                        "grandparent": None,
                        "parent": current_atom,
                        "atom": atom,
                        "element": element,
                        "distance": distance,
                        "angle": None,
                        "dihedral_angle": dihedral_angle,
                        "dihedral_neighbor": dihedral_neighbor
                    })
                visited.add(atom)
                queue.append((atom, current_atom))
    not_visited = set(all_atom_idx) - visited
    if not_visited: 
        for a in not_visited:
            element = sub3D.getAtomwithinds([a])[0].symbol()
            closest_atoms = getClosestAtoms(sub3D, a, 4.0)
            if closest_atoms:
                closest_atom = closest_atoms[0][0]
                # Find the parent in atom_data
                parent_entry = next((entry for entry in atom_data if entry["neighbor"] == closest_atom), None)
                grandparent = parent_entry["parent"] if parent_entry else None
                distance = getAtomsDistance(sub3D, closest_atom, a)[1]
                angle = sub3D.getAngle(grandparent, closest_atom, a) if grandparent is not None else None
                same_level_atom = [e for e in atom_data if e["parent"] == current_atom]
                if same_level_atom:
                    # If there are, we can calculate the dihedral angle
                    dihedral_angle = getDihedralAngle(sub3D, [parent_atom, current_atom, a, same_level_atom[0]["atom"]])
                    dihedral_neighbor = same_level_atom[0]["atom"]
                else:
                    dihedral_angle = None
                    dihedral_neighbor = None
            else:
                closest_atom = None
                grandparent = None
                distance = None
                angle = None
            atom_data.append({
            "grandparent": grandparent,
            "parent": closest_atom,
            "atom": a,
            "element": element,
            "distance": distance,
            "angle": angle,
            "dihedral_angle": dihedral_angle,
            "dihedral_neighbor": dihedral_neighbor
        })
    return atom_data

def insert_ligand_2_vacant_site(path_to_complex_xyz, bl_length=1.6, element='O'):
    """
    Insert a ligand into a metal complex.
    Parameters
    ----------
    path_to_complex_xyz : str
        Path to the metal complex .xyz file.
    bl_length : float, optional
        Bond length for the metal-oxo bond. Default is 1.6 Å.
    element : str, optional
        Element symbol of the oxo ligand. Default is 'O'.
    Returns
    -------
    oxo_complex : mol3D
        mol3D class instance of the complex with the oxo ligand inserted.
    """
    # Create mol3D for transition metal complex
    comp3D = mol3D()
    comp3D.readfromxyz(path_to_complex_xyz,read_final_optim_step=True)
    metal_ids = comp3D.findMetal()
    metal_idx=metal_ids[0]
    # Find vacant site
    vacant_site = find_vacant_site(comp3D)
    # Add oxygen to complex in the open site
    if element:
        o_atom=atom3D(element,vacant_site)
        o_idx=metal_idx+1
        comp3D.addAtom(o_atom, o_idx, auto_populate_bo_dict = False)
    # Bond oxygen to metal and adjust length
    comp3D.add_bond(metal_idx,o_idx,1)
    comp3D.BCM(o_idx,metal_idx, bl_length)
    idc = [metal_idx, o_idx]
    return comp3D, idc

def insert_bonded_atom(path_to_complex_xyz, parent_idx, distance, element, angle=None,grandparent_idx=None, dihedral_angle=None, dihedral_neighbor=None):
    """
    Insert a bonded atom into a metal complex.
    Parameters
    ----------
    path_to_complex_xyz : str
        Path to the metal complex .xyz file.
    bl_length : float, optional
        Bond length for the metal-oxo bond. Default is 1.6 Å.
    element : str, optional
        Element symbol of the oxo ligand. Default is 'O'.
    Returns
    -------
    oxo_complex : mol3D
        mol3D class instance of the complex with the oxo ligand inserted.
    """
    # Create mol3D for transition metal complex
    comp3D = mol3D()
    comp3D.readfromxyz(path_to_complex_xyz,read_final_optim_step=True)
    # Find indices of parent and grandparent atoms
    parent_coords = comp3D.getAtomCoords(parent_idx)
    if grandparent_idx is None:
        grandparent_idx = comp3D.getBondedAtom(parent_idx)[0]
    grandparent_coords = comp3D.getAtomCoords(grandparent_idx) if grandparent_idx is not None else None
    position_found = False
    if dihedral_angle is not None and dihedral_neighbor is not None:
            dihedral_neighbor_position = comp3D.getAtomCoords(dihedral_neighbor) 
            bond_vector, bond_length = getAtomsDistance(comp3D, parent_idx, grandparent_idx)
            best_position = rotate_point_around_axis(dihedral_neighbor_position, parent_coords, bond_vector, dihedral_angle)
            position_found = True
    if angle is None:
        if not position_found:
            parent_element = comp3D.getAtomwithinds([parent_idx])[0].symbol()
            possible_angles = get_possible_bond_angles(parent_element)
            if possible_angles:
                possible_sites = define_possible_sites(comp3D, possible_angles, parent_idx, grandparent_idx, num_theta_angles=18, distance=distance, linear=False)
                best_position = define_best_site(comp3D, possible_sites, [parent_idx])[0]
            else:
                possible_sites = define_possible_sites(comp3D, [90, 108, 120, 180], parent_idx, grandparent_idx, num_theta_angles=18, distance=distance, linear=False)
                best_position = define_best_site(comp3D, possible_sites, [parent_idx])[0]
    if grandparent_coords is not None and angle is not None:
        if not position_found:
            possible_sites = define_possible_sites(comp3D, [angle], parent_idx, grandparent_idx, num_theta_angles=18, distance=distance, linear=False)
            if possible_sites:
                best_position = define_best_site(comp3D, possible_sites, [parent_idx])[0]
    # Add atom to complex bonded to parent atom
    if element:
        coords = best_position[0] if isinstance(best_position, (list, tuple)) and len(best_position) > 0 and isinstance(best_position[0], (list, tuple, np.ndarray)) else best_position
        atom = atom3D(element, coords)
        idx = len(getAllAtomsIdx(comp3D))  # New index for the atom
        comp3D.addAtom(atom, idx, auto_populate_bo_dict=False)
        # Bond atom to parent atom and adjust length
        comp3D.add_bond(parent_idx,idx,1)
        comp3D.BCM(idx,parent_idx, distance)
    idc = [parent_idx, idx]
    return comp3D, idc

def insert_substrate(path_to_complex_xyz, substrate_relative_positions, 
                     bonding_site = None, linear = False):
    """
    Insert a substrate into the given molecule at specified relative positions.
    Parameters:
    ----------
    path_to_complex_xyz : str
        Path to the metal complex .xyz file.
    substrate_relative_positions : list of dict
        List of dictionaries containing information about the substrate atoms to be inserted.
    bonding_site : int, optional
        Index of the atom to which the substrate will be bonded. If None, the function will find an appropriate bonding site.
    Returns:
    -------
    comp3D : mol3D
        The updated metal complex with the substrate inserted.
    """
    init3D = mol3D()
    init3D.readfromxyz(path_to_complex_xyz,read_final_optim_step=True)
    metal_idx = init3D.findMetal()[0]
    bonding_element = substrate_relative_positions[0]["element"]
    # If bonding site is not specified, use metal with an open site
    bonding_assigned = False
    if bonding_site is None:
        metal_idx = init3D.findMetal()
        ligand = find_terminal_ligand(init3D, bonding_element)
        if ligand:
            bonding_idx = ligand
            print(f"Found bonding ligand at index {bonding_idx}")
            comp3D = mol3D()
            comp3D.copymol3D(init3D)
            bonding_assigned = True
        elif ligand is False:
            if not bonding_assigned:
                comp3D, indices = insert_ligand_2_vacant_site(path_to_complex_xyz, bl_length=2.0, element=bonding_element)
                metal_idx, bonding_idx = indices
                bonding_assigned = True
    elif bonding_site is not None:
        if not bonding_assigned:
            bonded_atoms = init3D.getBondedAtoms(bonding_site)
            for ba in bonded_atoms:
                if init3D.getAtomwithinds([ba])[0].symbol() == bonding_element:
                    if len(init3D.getBondedAtoms(ba)) == 1:
                        bonding_idx = ba
                    comp3D = mol3D()
                    comp3D.copymol3D(init3D)
                    bonding_assigned = True
                    break
            if len(bonded_atoms) == 1:
                if not bonding_assigned:
                    comp3D, indices = insert_bonded_atom(path_to_complex_xyz, bonding_site, 2.0, bonding_element, grandparent_idx=bonded_atoms[0])
                    metal_idx, bonding_idx = indices
                    bonding_assigned = True
            elif len(bonded_atoms) > 1: 
                if not bonding_assigned:
                    for atom in bonded_atoms:
                        if init3D.getAtomwithinds([atom])[0].symbol() == bonding_element:
                            if len(init3D.getBondedAtoms(atom)) == 1:
                                bonding_idx = atom
                                bonding_assigned = True
                                break
                if not bonding_assigned:
                    comp3D, indices = insert_bonded_atom(path_to_complex_xyz, bonding_site, 2.0, bonding_element, grandparent_idx=bonded_atoms[0])
                    metal_idx, bonding_idx = indices
                    bonding_assigned = True
    inserted_atoms = []
    bonding_atom_info = substrate_relative_positions[0]
    bonding_atom_coords = comp3D.getAtomCoords(bonding_idx)
    inserted_atoms.append({
        "grandparent": bonding_atom_info["grandparent"],
        "parent": bonding_atom_info["parent"],
        "atom": bonding_atom_info["atom"],
        "element": bonding_atom_info["element"],
        "distance": bonding_atom_info["distance"],
        "angle": bonding_atom_info["angle"],
        "dihedral_angle": bonding_atom_info["dihedral_angle"],
        "dihedral_neighbor": bonding_atom_info["dihedral_neighbor"],
        "new_idx": bonding_idx,  # You can set this if you know the index
        "position": bonding_atom_coords  # You can set this if you know the position
    })
    bonding_atom = substrate_relative_positions.pop(0)
    queue = deque(substrate_relative_positions)  # (current_atom, parent_atom)
    visited = set([bonding_idx])
    while queue:
        atom_info = queue.popleft()
        # Extract grandparent info:
        grandparent_idx = atom_info["grandparent"]
        grandparent_entry = next((entry for entry in inserted_atoms if entry["atom"] == grandparent_idx), None)
        if grandparent_entry:
            grandparent_coords = grandparent_entry["position"]
            grandparent_idx = grandparent_entry["new_idx"]
        elif grandparent_idx is None:
            grandparent_idx = metal_idx
            grandparent_coords = comp3D.getAtomCoords(metal_idx)
        # Extract parent info:
        parent_idx = atom_info["parent"]
        parent_entry = next((entry for entry in inserted_atoms if entry["atom"] == parent_idx), None)
        if parent_entry:
            parent_coords = parent_entry["position"]
            parent_idx = parent_entry["new_idx"]
        elif parent_idx is None:
            parent_idx = bonding_idx
            parent_coords = comp3D.getAtomCoords(bonding_idx)
        # Extract atom info:
        atom_idx = atom_info["atom"]
        element = atom_info["element"]
        distance = atom_info["distance"]
        angle = atom_info["angle"]
        best_position = None
        # Extract dihedral neighbor info
        dihedral_angle = atom_info["dihedral_angle"]
        dihedral_neighbor = atom_info["dihedral_neighbor"]
        position_found = False
        if parent_idx == bonding_idx:
            if linear:
                possible_sites= get_positions_around_point(parent_coords, radius=distance)
                # best_position = reflect_bond(comp3D, parent_idx, grandparent_idx)
                best_position = define_best_site(comp3D, possible_sites, [parent_idx])[0]
                print(f"Best position for atom {atom_idx} ({element}): {best_position} found by reflecting bond")
                position_found = True
        if dihedral_angle is not None and dihedral_neighbor is not None:
            if not position_found:
                dihedral_entry = next((entry for entry in inserted_atoms if entry["atom"] == dihedral_neighbor), None)
                dihedral_neighbor_position = dihedral_entry["position"] if dihedral_entry else None
                bond_vector, bond_length = getAtomsDistance(comp3D, parent_idx, grandparent_idx)
                best_position = rotate_point_around_axis(dihedral_neighbor_position, parent_coords, bond_vector, dihedral_angle)
                print(f"Best position for atom {atom_idx} ({element}): {best_position} found by dihedral angle")
                position_found = True
        if angle is None:
            if not position_found:
                parent_element = comp3D.getAtomwithinds([parent_idx])[0].symbol()
                possible_angles = get_possible_bond_angles(parent_element)
                if possible_angles:
                    possible_sites = define_possible_sites(comp3D, possible_angles, parent_idx, grandparent_idx, num_theta_angles=18, distance=distance, linear=False)
                    best_position = define_best_site(comp3D, possible_sites, [parent_idx])[0][0]
                    print(f"Best position for atom {atom_idx} ({element}): {best_position} found by angle")
                else:
                    possible_sites = define_possible_sites(comp3D, [90, 108, 120, 180], parent_idx, grandparent_idx, num_theta_angles=18, distance=distance, linear=False)
                    best_position = define_best_site(comp3D, possible_sites, [parent_idx])[0][0]
                    print(f"Best position for atom {atom_idx} ({element}): {best_position} found by angle")
                position_found = True
        if grandparent_idx is not None and angle is not None:
            if not position_found:
                possible_sites = define_possible_sites(comp3D, [angle], parent_idx, grandparent_idx, num_theta_angles=18, distance=distance, linear=False)
                if possible_sites:
                    best_position = define_best_site(comp3D, possible_sites, [parent_idx])[0][0]
                    print(f"Best position for atom {atom_idx} ({element}): {best_position} found by angle and grandparent")
                    position_found = True
        if best_position is False:
            print(f"Could not find a suitable position for atom {atom_idx} ({element})")
            continue
        else:
            new_atom = atom3D(element, best_position)
            new_idx = len(getAllAtomsIdx(comp3D))
            comp3D.addAtom(new_atom, new_idx, auto_populate_bo_dict=False)
            comp3D.add_bond(parent_idx if parent_idx is not None else bonding_idx, new_idx, 1)
            comp3D.BCM(new_idx, parent_idx if parent_idx is not None else bonding_idx, distance)
            visited.add(new_idx)
            inserted_atoms.append({
            "grandparent": grandparent_idx,
            "parent": parent_idx,
            "atom": atom_idx,
            "element": element,
            "distance": distance,
            "angle": angle,
            "dihedral_angle": dihedral_angle,
            "dihedral_neighbor": dihedral_neighbor,
            "new_idx": new_idx,
            "position": best_position
            })
    return comp3D, inserted_atoms

def insert_methyl(mol,carbon_site, c_idx, bonding_atom_idx):
    """
    Add a methyl group to the metal complex at the specified carbon site.
    Parameters
    ----------
    mol : mol3D
        mol3D class instance of the metal complex.
    carbon_site : list
        List of 3 floats representing the x, y, z coordinates of the carbon site.
    c_idx : int
        Index of the carbon atom in the metal complex.
    bonding_atom_idx : int
        Index of the atom to which the methyl group will be bonded.
    Returns
    -------
    comp3D : mol3D
        mol3D class instance of the metal complex with the methyl group added.
    ind : list
        List of indices for the carbon and hydrogen atoms in the methyl group.
        """
    comp3D = mol3D()
    comp3D.copymol3D(mol)
    # Add methyl carbon bound to hydrogen
    c_atom=atom3D('C',carbon_site)
    comp3D.addAtom(c_atom,c_idx,auto_populate_bo_dict=False)
    comp3D.add_bond(bonding_atom_idx,c_idx,1)
    c_coords=comp3D.getAtomCoords(c_idx)
    ch_bond, ch_bond_length = getAtomsDistance(comp3D, bonding_atom_idx, c_idx)
    # Add second methyl hydrogen to carbon
    central_site=reflect_bond(comp3D,c_idx,bonding_atom_idx)
    c_central_site_bond_length=np.linalg.norm(np.array(central_site)-np.array(c_coords))
    bond_angle_rad=(180-109.5)*np.pi/180
    h2_site=PointTranslateSph(c_coords, central_site,[c_central_site_bond_length,0,bond_angle_rad])
    h2_atom=atom3D('H',h2_site)
    h2_idx=c_idx+1
    comp3D.addAtom(h2_atom,h2_idx,auto_populate_bo_dict=False)
    comp3D.add_bond(c_idx,h2_idx,1)
    h2_coords=comp3D.getAtomCoords(h2_idx)
    comp3D.BCM(h2_idx,c_idx,1.1)
    # Add third methane hydrogen to carbon
    h3_site=rotate_point_around_axis(h2_coords,c_coords,ch_bond,120)
    h3_atom=atom3D('H',h3_site)
    h3_idx=h2_idx+1
    comp3D.addAtom(h3_atom,h3_idx,auto_populate_bo_dict=False)
    comp3D.add_bond(c_idx,h3_idx,1)
    comp3D.BCM(h3_idx,c_idx,1.1)
    # Add fourth methane hydrogen to carbon
    h4_site=rotate_point_around_axis(h2_coords,c_coords,ch_bond,240)
    h4_atom=atom3D('H',h4_site)
    h4_idx=h3_idx+1
    comp3D.addAtom(h4_atom,h4_idx,auto_populate_bo_dict=False)
    comp3D.add_bond(c_idx,h4_idx,1)
    comp3D.BCM(h4_idx,c_idx,1.1)
    ind = [c_idx, h2_idx, h3_idx, h4_idx]
    return comp3D, ind

def oxo_n2o_intermediate(path_to_complex_xyz):
    """
    Build a N₂O-based oxo complex from a square pyramidal metal complex.
    Parameters
    ----------
    path_to_complex_xyz : str
        Path to metal complex .xyz file.
    Returns
    -------
    n2o_complex_xyz : str
        Path to the written .xyz file of the optimized complex.
    """
    # Create mol3D for transition metal complex
    fe_complex = mol3D()
    fe_complex.readfromxyz(path_to_complex_xyz,read_final_optim_step=True)
    metal_idx = fe_complex.findMetal()[0]
    oxo_ligand = find_terminal_ligand(fe_complex, 'O')
    if oxo_ligand:
        o_idx = oxo_ligand
        print(f"Found oxo ligand at index {o_idx}")
    elif oxo_ligand is False:
        fe_complex, indices = insert_ligand_2_vacant_site(path_to_complex_xyz, bl_length=2.0)
        metal_idx, o_idx = indices
    # Add nitrogen to oxygen
    n1_site, n2_site=define_best_n2o_orientation(fe_complex,[104,120],[104,120],o_idx,metal_idx,num_theta_angles=9)
    n_atom=atom3D('N',n1_site[0])
    n_idx=o_idx+1
    fe_complex.addAtom(n_atom,n_idx,auto_populate_bo_dict=False)
    fe_complex.add_bond(o_idx,n_idx,1)
    fe_complex.BCM(n_idx,o_idx,1.2)
    # Add second nitrogen
    n2_atom=atom3D('N',n2_site[0])
    n2_idx=n_idx+1
    fe_complex.addAtom(n2_atom,n2_idx,auto_populate_bo_dict=False)
    fe_complex.add_bond(n_idx,n2_idx,2)
    fe_complex.BCM(n2_idx,n_idx,1.1)
    # Create list to freeze complex for FF optimization
    ids = [o_idx, n_idx, n2_idx]
    ind = [metal_idx, o_idx, n_idx, n2_idx]
    freeze = freeze_atoms(fe_complex,ids)
    return fe_complex, ind, freeze

def hat_ch4_intermediate(path_to_complex_xyz, indices=False, oxo_ligand=False):
    """
    Build a hydroxo complex intermediate attaching a methane to the oxygen.
    Parameters
    ----
    path_to_complex_xyz : str
        Path to metal complex .xyz file, the complex needs an oxo ligand.
    indices : list, optional
        List of indices for the metal and oxo atoms. If not provided, the function will find them automatically.
    oxo_ligand : bool, optional
        If True, the function assumes the oxo ligand is already present in the complex. Default is False.
    Return
    ----
    complex : mol3D
        mol3D class instance of complex with added methane near the oxo.
    ind : list
        List of indices for the metal, oxo, carbon, and hydrogen atoms in the complex
    """
    # Create mol3D for transition metal complex
    comp3D = mol3D()
    comp3D.readfromxyz(path_to_complex_xyz,read_final_optim_step=True)
    metal_idx = comp3D.findMetal()[0]
    oxo_ligand = find_terminal_ligand(comp3D, 'O')
    if oxo_ligand:
        o_idx = oxo_ligand
        print(f"Found oxo ligand at index {o_idx}")
    elif oxo_ligand is False:
        comp3D, indices = insert_ligand_2_vacant_site(path_to_complex_xyz, bl_length=2.0)
        metal_idx, o_idx = indices
    # Add first methane hydrogen to oxygen
    rotated_site=define_best_orientation(comp3D,o_idx, radius=1.1, remove_atoms=[o_idx])
    h1_atom=atom3D('H',rotated_site)
    h1_idx=o_idx+1
    comp3D.addAtom(h1_atom,h1_idx,auto_populate_bo_dict=False)
    comp3D.add_bond(o_idx,h1_idx,1)
    comp3D.BCM(h1_idx,o_idx,1.6)
    h1_coords=comp3D.getAtomCoords(h1_idx)
    # Add methane carbon bound to hydrogen
    c_site=reflect_bond(comp3D,h1_idx,o_idx)
    c_atom=atom3D('C',c_site)
    c_idx=h1_idx+1
    comp3D.addAtom(c_atom,c_idx,auto_populate_bo_dict=False)
    comp3D.add_bond(h1_idx,c_idx,1)
    comp3D.BCM(c_idx,h1_idx,1.1)
    c_coords=comp3D.getAtomCoords(c_idx)
    ch_bond, ch_bond_length = getAtomsDistance(comp3D, h1_idx, c_idx)
    # Add second methane hydrogen to carbon
    central_site=reflect_bond(comp3D,c_idx,h1_idx)
    c_central_site_bond_length=np.linalg.norm(np.array(central_site)-np.array(c_coords))
    bond_angle_rad=(180-109.5)*np.pi/180
    h2_site=PointTranslateSph(c_coords, central_site,[c_central_site_bond_length,0,bond_angle_rad])
    h2_atom=atom3D('H',h2_site)
    h2_idx=c_idx+1
    comp3D.addAtom(h2_atom,h2_idx,auto_populate_bo_dict=False)
    comp3D.add_bond(c_idx,h2_idx,1)
    h2_coords=comp3D.getAtomCoords(h2_idx)
    # Add third methane hydrogen to carbon
    h3_site=rotate_point_around_axis(h2_coords,c_coords,ch_bond,120)
    h3_atom=atom3D('H',h3_site)
    h3_idx=h2_idx+1
    comp3D.addAtom(h3_atom,h3_idx,auto_populate_bo_dict=False)
    comp3D.add_bond(c_idx,h3_idx,1)
    # Add fourth methane hydrogen to carbon
    h4_site=rotate_point_around_axis(h2_coords,c_coords,ch_bond,240)
    h4_atom=atom3D('H',h4_site)
    h4_idx=h3_idx+1
    comp3D.addAtom(h4_atom,h4_idx,auto_populate_bo_dict=False)
    comp3D.add_bond(c_idx,h4_idx,1)
    ind = [metal_idx, o_idx, c_idx, h1_idx, h2_idx, h3_idx, h4_idx]
    freeze = freeze_atoms(comp3D, ind)
    return comp3D, ind, freeze
    
def oxo_n2o_bond_breaking_distance(path_to_xyz):
    """
    Extract the minimum distance between a reference atom and atoms of a specific type.
    Parameters
    ----------
    path_to_xyz : str
        Path to the .xyz file containing the optimized geometry.
    Returns
    -------
    oxo_idx : int
        Index of the oxo ligand atom.
    n_idx : int
        Index of the nitrogen atom bonded to the oxo ligand.
    distance : float
        Minimum distance found to the nitrogen atoms bonded to the oxo ligand.
    """
    # Create mol3D for transition metal complex
    comp3D = mol3D()
    comp3D.readfromxyz(path_to_xyz,read_final_optim_step=True)
    metal_ids = comp3D.findMetal()
    if metal_ids:
        metal_idx=metal_ids[0]
    else:
        print("Error: No metal found in the complex")
        sys.exit(1)
    # Find oxo ligand and its bonded nitrogen atoms
    oxo_n2_ligand = find_oxo_n2_ligand(path_to_xyz)
    if not oxo_n2_ligand:
        print("Error: No oxo ligand with N₂ bonded found")
        sys.exit(1)
    o_idx, n1_idx, n2_idx = oxo_n2_ligand
    # Extract distance to atoms of a specific type
    distance_metal = getAtomsDistance(comp3D, o_idx, metal_idx)[1]
    distance_n = getAtomsDistance(comp3D, o_idx, n1_idx)[1]
    distance_n2 = getAtomsDistance(comp3D, o_idx, n2_idx)[1]
    distance_no = min(distance_n, distance_n2)
    # Find the index of the closest atom
    if distance_n < distance_n2:
        n_idx = n1_idx
    else:
        n_idx = n2_idx
    return o_idx, n_idx, metal_idx, distance_no, distance_metal

def hydroxo_ch4_bond_breaking_distance(path_to_complex_xyz):
    """
    Extract the minimum distance between a reference atom and atoms of a specific type.
    
    Parameters
    ----------
    path_to_xyz : str
        Path to the .xyz file containing the optimized geometry.
    
    Returns
    -------
    min_distance_atom : int
        Index of the atom with the minimum distance.
    min_distance : float
        Minimum distance found.
    """
    # Create mol3D for transition metal complex
    comp3D = mol3D()
    comp3D.readfromxyz(path_to_complex_xyz,read_final_optim_step=True)
    metal_ids = comp3D.findMetal()
    metal_idx=metal_ids[0]
    metal_coords = comp3D.getAtomCoords(metal_idx)
    # Identify oxo ligand
    ligands=comp3D.getBondedAtoms(metal_idx)
    all_oxy=comp3D.findAtomsbySymbol('O')
    all_carb=comp3D.findAtomsbySymbol('C')
    all_hydr=comp3D.findAtomsbySymbol('H')
    # Find CH₄
    methane = []
    for c in all_carb:
        c_bonded_atoms=comp3D.getBondedAtoms(c)
        if len(c_bonded_atoms) in [3, 4]:  
            # Count how many bonded atoms are hydrogens
            num_hydr = sum(1 for h in c_bonded_atoms if h in all_hydr)
            if num_hydr == len(c_bonded_atoms):
                methane.append(c)
    if len(methane)==0:
        print("Error: No CH₄ or CH₃ found")
        raise SystemExit
    elif len(methane)>1:
        print("Error: More than 1 CH₄ or CH₃ found")
        raise SystemExit
    else:
        ch4_idx=methane[0]
    # Find methane hydrogens
    bonded_hydrogens = comp3D.getBondedAtoms(ch4_idx)
    if len(bonded_hydrogens) == 4:
        methane_hydrogens = bonded_hydrogens
    elif len(bonded_hydrogens) == 3:
        distances = []
        for h in all_hydr:
            if h not in bonded_hydrogens:
                dist = getAtomsDistance(comp3D, ch4_idx, h)[1]
                distances.append((h, dist))
            if not distances:
                print("Error: No suitable hydrogen found")
                raise SystemExit
        methane_hydrogens = bonded_hydrogens + [min(distances, key=lambda x: x[1])[0]]
    # Find oxo ligand
    o_ligand = []
    oxo = False
    for lig in ligands:
        if lig in all_oxy:
            o_ligand.append(lig)
    if len(o_ligand)==0:
        print("Error: No oxo ligand")
        sys.exit(1)
    elif len(o_ligand)>1:
        for o in o_ligand:
            o_bonded_atoms=comp3D.getBondedAtoms(o)
            for h in methane_hydrogens:
                if h in o_bonded_atoms:
                    oxo=o
                elif ch4_idx in o_bonded_atoms:
                    oxo=o
        if not oxo:
            for o in o_ligand:
                o_bonded_atoms = comp3D.getBondedAtoms(o)
                if len(o_bonded_atoms) == 1:
                    oxo = o
            if not oxo:
                print("Error: No oxo ligand found bonded to methane or hydrogen")
                sys.exit(1)
    else:
        oxo=o_ligand[0]
    # Extract distance to methane hydrogens to oxo
    distances = []
    for h in methane_hydrogens:
        distance = getAtomsDistance(comp3D, oxo, h)[1]
        distances.append((h, distance))
    # Find the atom with the minimum distance
    h_idx, oh_distance = min(distances, key=lambda x: x[1])
    # Find C-H distance
    ch_distance = getAtomsDistance(comp3D, ch4_idx, h_idx)[1]
    return oxo, h_idx, ch4_idx, oh_distance, ch_distance

def check_structure(mol, mech_step, intermediate_ligands, metal_connected_atoms=False):
    """
    Check if the Terachem optimization structure is correct.
    Parameters
    ----------
    mol : mol3D
        mol3D class instance of the optimized structure.
    mech_step : str
        Mechanism step, either 'oxo' or 'hat'.
    intermediate_ligands : list
        List of indices of intermediate ligands (e.g., oxo, nitrogen atoms).
    metal_connected_atoms : list
        List of indices of atoms connected to the metal atom.
    Returns
    -------
    bool
        True if structure check passed, False otherwise.
    """
    check3D = mol3D()
    check3D.copymol3D(mol)
    # Check if the metal is still present
    metal_idx = check3D.findMetal()
    if not metal_idx:
        print("Error: Metal atom is missing in the optimized structure.")
        return False
    metal_idx = metal_idx[0]
    # Check if the metal is still bonded to the connected atoms
    if metal_connected_atoms:
        for atom in metal_connected_atoms:
            if atom not in check3D.getBondedAtoms(metal_idx):
                print(f"Error: Metal atom is not bonded to {atom} in the optimized structure.")
                if atom in intermediate_ligands:
                    distance = getAtomsDistance(check3D, metal_idx, atom)[1]
                    if distance > 4.0:
                        print(f"Warning: {atom} is too far from the metal atom in the optimized structure.")
                    return False
                else:
                    print(f"Error: {atom} is not bonded to the metal atom in the optimized structure.")
                    return False
    # Check if the intermediate is still present
    if mech_step == 'oxo':
        oxo_idx = find_terminal_ligand(check3D, 'O')
        if not oxo_idx:
            print("Error: Oxo ligand is missing in the optimized structure.")
            return False
        n1 = intermediate_ligands[2]
        n2 = intermediate_ligands[3]
        if n2 not in check3D.getBondedAtoms(n1):
            print(f"Error: Nitrogen atoms {n2} {n1} are not bonded in the optimized structure.")
            return False
    elif mech_step == 'hat':
        oxo_idx = find_terminal_ligand(check3D, 'O')
        if not oxo_idx:
            print("Error: Hydroxo ligand is missing in the optimized structure.")
            return False
        c_idx = intermediate_ligands[2]
        h1_idx = intermediate_ligands[3]
        h2_idx = intermediate_ligands[4]
        h3_idx = intermediate_ligands[5]
        h4_idx = intermediate_ligands[6]
        ch4_hydrogens = []
        for h in [h1_idx, h2_idx, h3_idx, h4_idx]:
            if h in check3D.getBondedAtoms(c_idx):
                ch4_hydrogens.append(h)
        if len(ch4_hydrogens) <3:
            print("Error: Not all methane hydrogens are bonded to the carbon in the optimized structure.")
            return False
    else:
        print("Error: Mechanism step must be either 'oxo' or 'hat' for this script.")
        return False
    return True

def intermediate_gen(mech_step, spinmult=1):
    """
    Generate an intermediate structure for a given mechanism step.
    Parameters
    ----------
    mech_step : str
        Mechanism step, either 'oxo' or 'hat'.
    spinmult : int
        Spin multiplicity for the calculation.  Default is 1 (singlet state).
    Returns
    -------
    ind : list
        List of indices for the metal, oxo, and nitrogen atoms in the generated intermediate.
    """
    # Create intermediate
    base_dir = os.getcwd()
    xyz_files=glob.glob(f"{base_dir}/*.xyz")
    for file in xyz_files:
        filename = os.path.basename(file)
        if "fe" in filename.lower() or "mn" in filename.lower():
            xyz = file
    # Terachem optimization for intermediate
    if mech_step == 'oxo':
        terachem_opt_name = "n2o"
        guess, ind, freeze = oxo_n2o_intermediate(xyz)
    elif mech_step == 'hat':
        terachem_opt_name = "ch4"
        guess, ind, freeze = hat_ch4_intermediate(xyz)
    else:
        print("Error: Mechanism step must be either 'oxo' or 'hat' for this script.")
        sys.exit()
    metal_idx=guess.findMetal()
    metal_connected_atoms = guess.getBondedAtoms(metal_idx[0])
    initial_guess = mol3D()
    initial_guess.copymol3D(guess)
    ffoptimized, energy = ffopt('uff',guess,metal_connected_atoms,1, frozenats=freeze,frozenangles=False, mlbonds=[],nsteps=100,spin = spinmult,debug=False)
    structure = check_structure(ffoptimized, mech_step, ind, metal_connected_atoms)
    if not structure:
        print("Force Field optimization structure check failed after optimization, using original guess.")
        ffoptimized.writexyz(f"failed_opt.xyz")
    else:
        print("Force Field optimization structure check passed.")
        ffoptimized.writexyz(f"{terachem_opt_name}_guess.xyz")
    initial_guess.writexyz(f"{terachem_opt_name}_guess.xyz")
    return ind, terachem_opt_name

def intermediate_generation_substrate(path_to_substrate_xyz, path_to_tmc_xyz, 
                                        bonding_atom, bonding_site, outfile_name):
    """
    Generate an intermediate structure for the substrate-bound metal complex.
    Parameters
    ----------
    path_to_substrate_xyz : str
        Path to the substrate XYZ file.
    path_to_tmc_xyz : str
        Path to the transition metal complex XYZ file.
    bonding_atom : int
        Index of the atom in the substrate to which the metal complex will bond.
    bonding_site : int
        Index of the bonding site in the metal complex.
    outfile_name : str
        Name of the output file for the generated intermediate structure.
    """
    atom_data = read_substrate_xyz(f"{path_to_substrate_xyz}", bonding_atom)
    df = pd.DataFrame(atom_data)
    print(df)
    df.to_csv("atom_data.csv", index=False)
    run3D, inserted_atoms = insert_substrate(f"{path_to_tmc_xyz}", atom_data, bonding_site = bonding_site)
    inserted_idx = [atom["new_idx"] for atom in inserted_atoms]
    closest_atoms = []
    for idx in inserted_idx:
        closest = getClosestAtoms(run3D, idx, 5.0)
        if closest:
            closest_atoms.append(closest[0])
    min_distance = min(closest_atoms, key=lambda x: x[1])
    if min_distance[1] < 0.8:
        print(f"Warning: Atom {min_distance[0]} is too close to the substrate.")
        print("Repositioning substrate...")
        atom_data2 = read_substrate_xyz(f"{path_to_substrate_xyz}", bonding_atom)
        rerun3D, inserted_atoms2 = insert_substrate(f"{path_to_tmc_xyz}", atom_data2, bonding_site = bonding_site, linear = True)
        rerun3D.writexyz(outfile_name)
        return inserted_atoms2
    else:
        run3D.writexyz(outfile_name)
        return inserted_atoms
