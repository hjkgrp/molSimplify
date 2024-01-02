from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.atom3D import atom3D
from molSimplify.Informatics.MOF.PBC_functions import (
    compute_adj_matrix,
    compute_distance_matrix3,
    fractional2cart,
    get_closed_subgraph,
    include_extra_shells,
    ligand_detect,
    linker_length,
    mkcell,
    readcif,
    returnXYZandGraph,
    slice_mat,
    write2file,
    writeXYZandGraph,
    XYZ_connected,
    )
from molSimplify.Scripts.cellbuilder_tools import import_from_cif
import numpy as np
from scipy import sparse
import networkx as nx
import copy
import itertools
import os


def periodic_checker(graph, coords):
    """
    Checks if a graph is periodic or not.
    This does the same task as molSimplify.Informatics.MOF.MOF_descriptors.detect_1D_rod, but in a different way.

    Parameters
    ----------
    graph : numpy.matrix
        Adjacency matrix. Shape is (number of atoms, number of atoms).
    coords : list of list of float
        Cartesian coordinates of atoms. Length of the outer list is the number of atoms, while each inner list is length 3.

    Returns
    -------
    periodic : bool
        Whether or not a graph is periodic.

    """
    from scipy.sparse import csgraph
    csg = csgraph.csgraph_from_dense(graph)
    x, y = csg.nonzero()
    maxdist = 0
    periodic = False
    for row1, row2 in zip(x, y):
        a = np.array(coords[row1])
        b = np.array(coords[row2])
        dist = np.linalg.norm(a-b)
        if dist > maxdist:
            maxdist = dist
    # If any connected atoms are more than four angstroms apart, they are very likely to be offset by a cell vector. Periodic.
    if maxdist > 4:
        periodic = True
    return periodic


def branch(molcif, main_paths, atoms_in_sbu, new_atoms=None):
    """
    Climbs out from a given atom and adds the atoms that are in the branch.
    This is important for getting all atoms in a branched functional group of a linker.

    Parameters
    ----------
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    main_paths : list of int
        Indices of main path atoms (atoms that are part of a linker).
    atoms_in_sbu : list of numpy.int64
        Indices of atoms in the SBU.
    new_atoms : list of numpy.int64
        Indices of new atoms to be included.

    Returns
    -------
    new_atoms : list of numpy.int64
        Indices of new atoms to be included.
    atoms_in_sbu : list of numpy.int64
        Indices of atoms in the SBU.

    """
    if new_atoms is None:
        new_atoms = []
    original_atoms = atoms_in_sbu.copy()
    for atom in new_atoms:
        bonded_list = molcif.getBondedAtoms(atom)
        if (len(set(bonded_list)-set(main_paths)-set(atoms_in_sbu)) > 0):
            new_atoms += list(set(bonded_list)-set(main_paths))
            new_atoms = list(set(new_atoms))
            atoms_in_sbu += new_atoms
    if len(original_atoms) == len(atoms_in_sbu):
        return new_atoms, atoms_in_sbu
    else:
        branch_atoms, branch_atoms_in_sbu = branch(molcif, main_paths, atoms_in_sbu, new_atoms)
        new_atoms += branch_atoms
        atoms_in_sbu += branch_atoms_in_sbu
        return new_atoms, atoms_in_sbu

def identify_main_chain(temp_mol, link_list):
    """
    Identifies the atom that are directly present from one
    connecting point to another. Identifies cases that can be functional groups.

    Parameters
    ----------
    temp_mol : molSimplify.Classes.mol3D.mol3D
        mol3D of a linker.
    link_list : list of int
        The indices of the anchoring atoms of the linker.

    Returns
    -------
    main : list of int
        Any atoms that lie on the path between two connection points.
    shortest : int
        The shortest path length between two anchoring atoms.
    longest : int
        The longest path length between two anchoring atoms.

    """
    G = nx.from_numpy_matrix(temp_mol.graph)
    pairs = []
    shortest = 0
    longest = 0
    if len(link_list) == 1:
        main = list(G.nodes)
        shortest = 1
        longest = 1
        return main, shortest, longest
    else:
        for a, b in itertools.combinations(link_list, 2):
            pair = (a, b)
            pairs.append(pair)
        shorts = []
        for i in pairs:
            short = list(nx.shortest_path(G, source=i[0], target=i[1]))
            shorts.append(short)
        shortest, longest = min([len(short) for short in shorts]), max([len(short) for short in shorts])
        paths = list(itertools.chain(*shorts))
        min_cycles = (nx.minimum_cycle_basis(G)) # gets all closed rings in graph
        min_cycles_copy = min_cycles.copy()
        min_cycles_copy_2 = []
        paths_copy = paths.copy()
        while len(min_cycles_copy) != len(min_cycles_copy_2):
            min_cycles_copy_2 = min_cycles_copy.copy()
            for i in min_cycles:
                paths = paths_copy.copy()
                if set(paths) & set(i):
                    # I believe this identifies potential functional groups.
                    # Identifies and adds minimum cycles that have atoms in common with any shortest anchoring atom to anchoring atom path.
                    if not set(i).issubset(set((paths))):
                        paths_copy += set(i)
                        min_cycles_copy.remove(i)

        main = paths
        return main, shortest, longest


def get_molcif_cycles_no_metal(molcif):
    """
    Makes the graph and get all cycles in the graph.

    Parameters
    ----------
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.

    Returns
    -------
    subcycle_list : list of list of int
        The individual subcycles. Each inner list is a subcycle.
    flat_subcycle_list : list of int
        Flattened list of subcycle atoms (indices).

    """
    G=nx.from_numpy_matrix(molcif.graph)
    cycles = nx.minimum_cycle_basis(G) # gets all closed rings in graph
    subcycle_list = []
    for cycle in cycles:
        skip_row = False
        for element in cycle:
            # don't include any cycles with metal in it
            # This is necessary to not get malformed cycles.
            if molcif.getAtom(element).ismetal():
                skip_row = True
                break
        if not skip_row:
            subcycle_list.append(cycle)
    # Flatten list to contain all atoms in subcycles
    flat_subcycle_list = [item for sublist in subcycle_list for item in sublist]
    return subcycle_list, flat_subcycle_list

def breakdown_MOF(SBUlist, SBU_subgraph, molcif, name, cell, anchoring_atoms, sbupath=False, connections_list=False, connections_subgraphlist=False, linkerpath=False):
    """
    Writes SBU and linker XYZ files.

    Output codes are as follows:
    2: There exist short (i.e. 2 atom) and longer linkers. We could not split the MOF apart consistently.
    4: The MOF contains a 1D rod, which cannot be easily reassembled into a new MOF.
    None: The MOF was split correctly

    Parameters
    ----------
    SBUlist : list of list of numpy.int64
        Each inner list is its own separate SBU. The ints are the atom indices of that SBU. Length is # of SBUs.
    SBU_subgraph : list of scipy.sparse.csr.csr_matrix
        The atom connections in the SBU subgraph. Length is # of SBUs.
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    name : str
        The name of the cif being analyzed.
    cell : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    anchoring_atoms : set of numpy.int64
        The indices of the anchoring atoms of the linkers.
    sbupath : str
        The path to which the SBU XYZ files will be written.
    connections_list : list of list of int
        Each inner list is its own separate linker. The ints are the atom indices of that linker. Length is # of linkers.
    connections_subgraphlist : list of numpy.matrix
        The atom connections in the linker subgraph. Length is # of linkers.
    linkerpath : str
        The path to which the linker XYZ files will be written.

    Returns
    -------
    None

    """
    n_sbu = len(SBUlist)
    all_SBU_atoms = []
    all_SBU_X_atoms = []

    # make the graph and get all cycles in the graph
    # return the flattened list of the subcycle atoms
    # The subcycle list contains all of the individual subcycles (if they need to be compared)
    subcycle_list, flat_subcycle_list = get_molcif_cycles_no_metal(molcif)

    '''
    Loop over all SBUs as identified by subgraphs. Then create the mol3Ds for each SBU.
    '''
    for i, SBU in enumerate(SBUlist):
        # For a given SBU, make a list of main paths. This contains atoms that are part of the linker.
        main_paths = []
        linker_length_dict = {}
        current_longest = 0
        for j, linker in enumerate(connections_list):
            # For each SBU and linker combo, make a mol3D define the linklist for that linker
            linker_mol = mol3D()
            # keep track of added atoms
            linker_added = []
            link_list = []
            linker_dict = {}
            for jj, val2 in enumerate(linker):
                # add anchoring atom to link list. Val2 has molcif numbering
                linker_dict[jj] = val2
                if val2 in anchoring_atoms:
                    link_list.append(jj)
                # This builds a mol object for the linker --> even though it is in the SBU section.
                if not (val2 in linker_added):
                    linker_mol.addAtom(molcif.getAtom(val2))
                    linker_added.append(val2)
            linker_mol.graph = connections_subgraphlist[j]
            # This identifies anything on the simple path from end to end
            main, shortest, longest = identify_main_chain(linker_mol, link_list)
            if longest > current_longest:
                current_longest = longest
            # Currently, main is in linker indices. Get them back in molcif indices.
            # This is the main chain for a given linker.
            main = [linker_dict[val] for val in main]
            main_paths.extend(main)
            min_length, max_length = linker_length(connections_subgraphlist[j],link_list)
            linker_length_j = max(min_length, max_length)
            # Make a dictionary that will identify the linker length and atoms in the linker by the linker number
            linker_length_dict[j] = {'length':linker_length_j, 'atoms':linker, 'longest':longest}
        if current_longest <= 2:
            return 2
        # put all main path atoms into the main path list
        main_paths = list(set(main_paths))
        SBU_mol = mol3D()

        # This list keeps track of if an atom has been added to the SBU
        SBU_added = []
        # This dictionary keeps a mapping between molcif indices and SBU_mol indices
        SBU_dict = {}
        # Keeps track of the branches off of a linker for instance.
        branches = []
        # Keeps track of the atoms bonded to a cycle.
        bonded_atoms_to_cycle = []
        # Tuple list keeps track of the atoms that are coordinated to X atoms
        tuple_list_sbu = []
        # Keep track of the indices that should be the X atoms
        atoms_that_are_X = []
        X_checked_list = []
        # Make an atom3D list of the X atoms. These atoms should be added to the end of the XYZ.
        X_atom3D_list = []
        for val in SBU:
            # make SBU mol, add new atom if never added before.
            if val not in SBU_added:
                SBU_mol.addAtom(molcif.getAtom(val))
                # Create a mapping between the molcif indices (values) and the SBUmol indices (keys)
                SBU_dict[SBU_mol.natoms-1] = val
                SBU_added.append(val)
            # Check if any of the atoms added to the SBU are part of a cycle. Checks overlap between first
            # two coordination shells and any rings in the SBU.
            in_cycles = any([val in cycle for cycle in subcycle_list])
            if in_cycles:
                # Some atoms overlap with the cycles that are formally part of a linker.
                cycles_with_overlap = []
                for cycle in subcycle_list:
                    if val in cycle:
                        if cycle not in cycles_with_overlap:
                            cycles_with_overlap.append(cycle)
                        temp_bonded_list = []
                        for cycle_val in cycle:
                            temp_bonded = molcif.getBondedAtoms(cycle_val)
                            temp_bonded = list(set(temp_bonded)-set(cycle))
                            temp_bonded_list.extend(temp_bonded)
                            if cycle_val not in SBU_added:
                                SBU_mol.addAtom(molcif.getAtom(cycle_val))
                                SBU_dict[SBU_mol.natoms-1] = cycle_val
                                SBU_added.append(cycle_val)
                        bonded_atoms_to_cycle.append(temp_bonded_list)
            # Check how many atoms are branched
            additional_branched_atoms,_ = branch(molcif, main_paths, SBU_added.copy(), [val])
            for branched_atom in additional_branched_atoms:
                if branched_atom not in SBU_added:
                    SBU_mol.addAtom(molcif.getAtom(branched_atom))
                    SBU_dict[SBU_mol.natoms-1] = branched_atom
                    SBU_added.append(branched_atom)
        if len(bonded_atoms_to_cycle)>1:
            new_bonded_atoms_to_cycle = []
            # Don't let things that are part of another cycle be included here
            for bonded_atoms_to_indiv_cycle in bonded_atoms_to_cycle:
                new_bonded_atoms_to_cycle.append(list(set(bonded_atoms_to_indiv_cycle)-set(flat_subcycle_list)))
            combos = itertools.combinations(new_bonded_atoms_to_cycle, 2)
            for comboval in combos:
                if comboval[0] == comboval[1]:
                    continue
                intersection = list(set(comboval[0])&set(comboval[1]))
                if len(intersection)>0:
                    for comboval_intersection in intersection:
                        if comboval_intersection not in SBU_added:
                            SBU_mol.addAtom(molcif.getAtom(comboval_intersection))
                            SBU_dict[SBU_mol.natoms-1] = comboval_intersection
                            SBU_added.append(comboval_intersection)
            intersection_atoms = list(set.intersection(*map(set,bonded_atoms_to_cycle)))
            for intersection_atom in intersection_atoms:
                if intersection_atom not in SBU_added:
                    SBU_mol.addAtom(molcif.getAtom(intersection_atom))
                    SBU_dict[SBU_mol.natoms-1] = intersection_atom
                    SBU_added.append(intersection_atom)
        for SBU_added_atoms in SBU_added.copy():
            bonded_atoms = molcif.getBondedAtoms(SBU_added_atoms)
            for bonded_atom in bonded_atoms:
                if molcif.getAtom(bonded_atom).symbol() == 'H':
                    if bonded_atom not in SBU_added:
                        SBU_mol.addAtom(molcif.getAtom(bonded_atom))
                        SBU_dict[SBU_mol.natoms-1] = bonded_atom
                        SBU_added.append(bonded_atom)
                if (bonded_atom in main_paths) and (not ((bonded_atom in SBU_added) or (bonded_atom in X_checked_list))):
                    temp_atom = molcif.getAtom(bonded_atom)
                    temp_atom_coords = temp_atom.coords()
                    new_atom = atom3D(Sym='X', xyz=temp_atom_coords.copy())
                    X_atom3D_list.append((new_atom, bonded_atom, SBU_added_atoms))
                    X_checked_list.append(bonded_atom)

        final_X_indices = []
        for X_atom in X_atom3D_list:
            if X_atom[1] in SBU_added:
                continue
            else:
                SBU_added.append(X_atom[1])
            SBU_mol.addAtom(X_atom[0])
            SBU_dict[SBU_mol.natoms-1] = X_atom[1]
            tuple_list_sbu.append((SBU_mol.natoms-1, X_atom[2]))
            final_X_indices.append(SBU_mol.natoms-1)
            atoms_that_are_X.append(X_atom[1])
        SBU_added_no_X = list(set(SBU_added)-set(atoms_that_are_X))
        inv_SBU_dict = {v: k for k, v in SBU_dict.items()}
        tempgraph = molcif.graph[np.ix_(SBU_added, SBU_added)]
        no_X_graph = molcif.graph[np.ix_(SBU_added_no_X, SBU_added_no_X)]
        SBU_mol.graph = tempgraph
        SBU_mol_cart_coords = np.array([atom.coords() for atom in SBU_mol.atoms])
        SBU_mol_atom_labels =[atom.sym for atom in SBU_mol.atoms]
        SBU_mol_adj_mat = np.array(SBU_mol.graph)

        SBU_mol_fcoords_connected = XYZ_connected(cell, SBU_mol_cart_coords, SBU_mol_adj_mat)
        coord_list, molgraph = returnXYZandGraph(None, SBU_mol_atom_labels, cell, SBU_mol_fcoords_connected, SBU_mol_adj_mat)
        for r in range(SBU_mol.natoms):
            SBU_mol.getAtom(r).setcoords(coord_list[r])
        for val in tuple_list_sbu:
            SBU_mol.BCM(val[0],inv_SBU_dict[val[1]],0.75)
        new_coords = [[float(val2) for val2 in val.split()[1:]] for val in SBU_mol.coords().split('\n')[2:-1]]
        is_periodic = periodic_checker(tempgraph, new_coords)
        # if is_periodic is true, the SBU is periodic in nature --> 1D rod.

        ###### WRITE THE SBU MOL TO THE PLACE
        if sbupath and not os.path.exists(sbupath+"/"+str(name)+str(i)+'.xyz'):
            if is_periodic:
                xyzname = sbupath+"/"+str(name)+"_sbu1Drod_"+str(i)+".xyz"
            else:
                xyzname = sbupath+"/"+str(name)+"_sbu_"+str(i)+".xyz"

        if len(final_X_indices)>0:
            X_string = '   '.join([str(val) for val in final_X_indices])
        else:
            X_string = '   '
        coord_list, molgraph = returnXYZandGraph(xyzname, SBU_mol_atom_labels, cell, SBU_mol_fcoords_connected, SBU_mol_adj_mat)
        SBU_mol.writexyz(xyzname, withgraph=True, specialheader='   '+X_string)
        all_SBU_atoms.extend(SBU_added)
        if '1Drod' in xyzname:
            # if SBU is a 1D rod, end it here
            return 4
    atoms_to_be_deleted_from_linker = list(set(all_SBU_atoms))
    for i, linker in enumerate(connections_list):
        linker_mol = mol3D()
        # This list keeps track of if an atom has been added to the SBU
        linker_added = []
        # This dictionary keeps a mapping between molcif indices and SBU_mol indices
        linker_dict = {}
        # Tuple list keeps track of the atoms that are coordinated to X atoms
        tuple_list_linker = []
        # Keep track of the indices that should be the X atoms
        atoms_that_are_X_linker = []
        X_checked_list_linker = []
        # Make an atom3D list of the X atoms. These atoms should be added to the end of the XYZ.
        X_atom3D_list_linker = []
        for val in linker.copy():
            # loop over atoms in linker
            if (val not in atoms_to_be_deleted_from_linker):
                # if current atom should not be deleted (not X), add it.
                linker_mol.addAtom(molcif.getAtom(val))
                linker_added.append(val)
                # keep mapping between linker and molcif
                linker_dict[linker_mol.natoms-1] = val
                current_atom = linker_mol.natoms-1
                # get all of the atoms bonded to the original atom
                for bonded_atom in molcif.getBondedAtoms(val):
                    # add the atom if it's in the SBU set
                    if (bonded_atom in all_SBU_atoms) and (bonded_atom not in linker_added):
                        linker_mol.addAtom(molcif.getAtom(bonded_atom))
                        linker_added.append(bonded_atom)
                        linker_dict[linker_mol.natoms-1] = bonded_atom
                        subatoms = molcif.getBondedAtoms(bonded_atom)
                        for subatom in subatoms:
                            if (subatom in atoms_to_be_deleted_from_linker) and (not ((subatom in linker_added) or (subatom in X_checked_list_linker))):
                                temp_atom_linker = molcif.getAtom(subatom)
                                temp_atom_coords_linker = temp_atom_linker.coords()
                                new_atom_linker = atom3D(Sym='X', xyz=temp_atom_coords_linker.copy())
                                X_atom3D_list_linker.append((new_atom_linker,subatom,bonded_atom))
                                X_checked_list.append(bonded_atom)
        final_X_indices_linker = []
        for X_atom_linker in X_atom3D_list_linker:
            if X_atom_linker[1] in linker_added:
                continue
            else:
                linker_added.append(X_atom_linker[1])
            linker_mol.addAtom(X_atom_linker[0])
            linker_dict[linker_mol.natoms-1] = X_atom_linker[1]
            tuple_list_linker.append((linker_mol.natoms-1, X_atom_linker[2]))
            final_X_indices_linker.append(linker_mol.natoms-1)
            atoms_that_are_X_linker.append(X_atom_linker[1])

        tempgraph = molcif.graph[np.ix_(linker_added, linker_added)]
        linker_added_no_X = list(set(linker_added)-set(atoms_that_are_X_linker))
        no_X_graph_linker = molcif.graph[np.ix_(linker_added_no_X, linker_added_no_X)]
        linker_mol.graph = tempgraph

        # make sure that the single graph is not multiple
        n_components, labels_components = sparse.csgraph.connected_components(csgraph=no_X_graph_linker)
        linker_mol_cart_coords = np.array([atom.coords() for atom in linker_mol.atoms])
        linker_mol_atom_labels = [atom.sym for atom in linker_mol.atoms]
        linker_mol_adj_mat = np.array(linker_mol.graph)
        inv_linker_dict = {v: k for k, v in linker_dict.items()}
        heavy_atom_count = linker_mol.count_atoms()
        if (linker_mol.natoms == 0) or (n_components > 1) or (heavy_atom_count < 3):
            continue
        linker_mol_fcoords_connected = XYZ_connected(cell, linker_mol_cart_coords , linker_mol_adj_mat )
        coord_list, molgraph = returnXYZandGraph(None , linker_mol_atom_labels , cell , linker_mol_fcoords_connected, linker_mol_adj_mat)
        for r in range(linker_mol.natoms):
            linker_mol.getAtom(r).setcoords(coord_list[r])
        for val in tuple_list_linker:
            linker_mol.BCM(val[0],inv_linker_dict[val[1]],0.75)
        ###### WRITE THE LINKER MOL TO THE PLACE
        if linkerpath and not os.path.exists(linkerpath+"/"+str(name)+str(i)+".xyz"):
            xyzname = linkerpath+"/"+str(name)+"_linker_"+str(i)+".xyz"

            if len(final_X_indices_linker)>0:
                X_string = '   '.join([str(val) for val in final_X_indices_linker])
            else:
                X_string = '   '
            coord_list, molgraph = returnXYZandGraph(xyzname, linker_mol_atom_labels , cell , linker_mol_fcoords_connected, linker_mol_adj_mat)
            linker_mol.writexyz(xyzname, withgraph=True, specialheader='   '+X_string)
    return None

def prepare_initial_SBU(molcif, allatomtypes, metal_list, logpath, name):
    """
    Prepares removelist and SBUlist, which indicate which atoms to remove from linkers and which atoms belong to SBUs.

    Parameters
    ----------
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    allatomtypes : list of str
        The atom types of the cif file, indicated by periodic symbols like 'O' and 'Cu'. Length is the number of atoms.
    metal_list : set of int
        The indices of metal atoms in the mol3D.
    logpath : str
        The path to which log files are written.
    name : str
        The name of the cif being analyzed.

    Returns
    -------
    removelist : set of int
        The indices of atoms to remove.
    SBUlist : set of numpy.int64
        The indices of atoms in SBUs. removelist + 1st coordination shell of the metals

    """
    SBUlist = set()
    metal_list = set([at for at in molcif.findMetal(transition_metals_only=False)])
    [SBUlist.update(set([metal])) for metal in molcif.findMetal(transition_metals_only=False)] # Remove all metals as part of the SBU
    [SBUlist.update(set(molcif.getBondedAtomsSmart(metal))) for metal in molcif.findMetal(transition_metals_only=False)]
    removelist = set()
    [removelist.update(set([metal])) for metal in molcif.findMetal(transition_metals_only=False)] # Remove all metals as part of the SBU
    for metal in removelist:
        bonded_atoms = set(molcif.getBondedAtomsSmart(metal))
        bonded_atoms_types = set([str(allatomtypes[at]) for at in set(molcif.getBondedAtomsSmart(metal))])
        cn = len(bonded_atoms)
        cn_atom = ",".join([at for at in bonded_atoms_types])
        tmpstr = "atom %i with type of %s found to have %i coordinates with atom types of %s\n"%(metal, allatomtypes[metal], cn, cn_atom)
        write2file(logpath, "/%s.log"%name, tmpstr)
    [removelist.update(set([atom])) for atom in SBUlist if all((molcif.getAtom(val).ismetal() or molcif.getAtom(val).symbol().upper() == 'H') for val in molcif.getBondedAtomsSmart(atom))]
    '''
    adding hydrogens connected to atoms which are only connected to metals. In particular interstitial OH, like in UiO SBU.
    '''
    for atom in SBUlist:
        for val in molcif.getBondedAtomsSmart(atom):
            if molcif.getAtom(val).symbol().upper() == 'H':
               removelist.update(set([val]))
    return removelist, SBUlist

def identify_lc_atoms(molcif, removelist, metal_list):
    """
    Returns linker information including the indices of atoms that anchor onto SBUs.

    Parameters
    ----------
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    removelist : set of int
        The indices of atoms to remove, i.e. the SBU atoms.
    metal_list : set of int
        The indices of metal atoms in the mol3D.

    Returns
    -------
    anc_atoms : set of numpy.int64
        The indices of the anchoring atoms of the linkers.
    linkers : set of int
        The indices of linkers.
    linker_list : list of list of int
        Each inner list is its own separate linker. The ints are the atom indices of that linker. Length is # of linkers.
    linker_subgraphlist : list of numpy.matrix
        The atom connections in the linker subgraph. Length is # of linkers.
    allatoms : set of int
        The indices of all of the atoms in the MOF.
    connections_list : list of list of int
        Each inner list is its own separate linker. The ints are the atom indices of that linker. Length is # of linkers.
    connections_subgraphlist : list of numpy.matrix
        The atom connections in the linker subgraph. Length is # of linkers.

    """
    allatoms = set(range(0, molcif.graph.shape[0]))
    linkers = allatoms - removelist # Anything that is in the remove list (SBU) is removed, leaving linkers
    # Use the atoms for linkers and the remove list, along with the original full unit cell graph to make the linker subgraphs
    linker_list, linker_subgraphlist = get_closed_subgraph(linkers.copy(), removelist.copy(), molcif.graph)
    # Next, we have to determine which atoms on the linkers are the connecting points to the SBU.
    linker_length_list = [len(linker_val) for linker_val in linker_list]
    adjmat = molcif.graph.copy()
    connections_list = copy.deepcopy(linker_list)
    connections_subgraphlist = copy.deepcopy(linker_subgraphlist)
    '''
    find all anchoring atoms on linkers and ligands (lc identification)
    '''
    anc_atoms = set()
    for linker in linker_list:
        for atom_linker in linker:
            # We check from the graph if the anchor atom is bonded to a metal. If it is then it is an anchoring atom
            bonded2atom = np.nonzero(molcif.graph[atom_linker,:])[1]
            if set(bonded2atom) & metal_list:
                anc_atoms.add(atom_linker)
    # return the anchoring atoms, the atoms we leave as linkers
    return anc_atoms, linkers, linker_list, linker_subgraphlist, allatoms, connections_list, connections_subgraphlist

def identify_short_linkers(molcif, initial_SBU_list, initial_SBU_subgraphlist, removelist, linkers, linker_list, linker_subgraphlist, adj_matrix, SBUlist, logpath, linkerpath, name, cell_v):
    """
    Helps determine whether a MOF has long or short linkers.

    Parameters
    ----------
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    initial_SBU_list : list of list of numpy.int32
        Each inner list is its own separate SBU. The ints are the atom indices of that SBU. Length is # of SBUs.
    initial_SBU_subgraphlist : list of scipy.sparse.csr.csr_matrix
        The atom connections in the SBU subgraph. Length is # of SBUs.
    removelist : set of int
        The indices of atoms to remove.
    linkers : set of int
        The indices of linkers.
    linker_list : list of list of int
        Each inner list is its own separate linker. The ints are the atom indices of that linker. Length is # of linkers.
    linker_subgraphlist : list of numpy.matrix
        The atom connections in the linker subgraph. Length is # of linkers.
    adj_matrix : scipy.sparse.csr.csr_matrix
        Adjacency matrix. 1 represents a bond, 0 represents no bond. Shape is (number of atoms, number of atoms).
    SBUlist : set of numpy.int64
        The indices of atoms in SBUs. removelist + 1st coordination shell of the metals
    logpath : str
        The path to which log files are written.
    linkerpath : str
        Path of the folder to make TXT files in.
    name : str
        The name of the cif being analyzed.
    cell_v : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).

    Returns
    -------
    min_max_linker_length : int
        The longest path length between two anchors in a linker.
    long_ligands : bool
        Indicates whether the linkers are short.
    SBUlist : set of numpy.int64
        The indices of atoms in SBUs. removelist + 1st coordination shell of the metals
    removelist : set of int
        The indices of atoms to remove.
    linker_list : list of list of int
        Each inner list is its own separate linker. The ints are the atom indices of that linker. Length is # of linkers.
    linker_subgraphlist : list of numpy.matrix
        The atom connections in the linker subgraph. Length is # of linkers.

    """
    templist = linker_list[:]
    tempgraphlist = linker_subgraphlist[:]
    long_ligands = False
    max_min_linker_length, min_max_linker_length = (0,100) # The maximum value of the minimum linker length, and the minimum value of the maximum linker length. Updated later.
    for ii, atoms_list in reversed(list(enumerate(linker_list))): #Loop over all linker subgraphs
        linkeranchors_list = set()
        linkeranchors_atoms = set()
        sbuanchors_list = set()
        sbu_connect_list = set()
        """""""""
        Here, we are trying to identify what is actually a linker and what is a ligand.
        To do this, we check if something is connected to more than one SBU. Set to
        handle cases where primitive cell is small, ambiguous cases are recorded.
        """""""""
        for iii, atoms in enumerate(atoms_list): #loop over all atoms in a linker
            connected_atoms = np.nonzero(adj_matrix[atoms,:])[1]
            for kk, sbu_atoms_list in enumerate(initial_SBU_list): #loop over all SBU subgraphs
                for sbu_atoms in sbu_atoms_list: #Loop over SBU
                    if sbu_atoms in connected_atoms:
                        linkeranchors_list.add(iii)
                        linkeranchors_atoms.add(atoms)
                        sbuanchors_list.add(sbu_atoms)
                        sbu_connect_list.add(kk) #Add if unique SBUs
        min_length, max_length = linker_length(linker_subgraphlist[ii], linkeranchors_list)

        if len(linkeranchors_list) >=2 : # linker, and in one ambiguous case, could be a ligand.
            if len(sbu_connect_list) >= 2: #Something that connects two SBUs is certain to be a linker
                max_min_linker_length = max(min_length, max_min_linker_length)
                min_max_linker_length = min(max_length, min_max_linker_length)
                continue
            else:
                # check number of times we cross PBC :
                # TODO: we still can fail in multidentate ligands!
                linker_cart_coords = np.array([at.coords() \
                        for at in [molcif.getAtom(val) for val in atoms_list]])
                linker_adjmat = np.array(linker_subgraphlist[ii])
                pr_image_organic = ligand_detect(cell_v, linker_cart_coords, linker_adjmat, linkeranchors_list)
                sbu_temp = linkeranchors_atoms.copy()
                sbu_temp.update({val for val in initial_SBU_list[list(sbu_connect_list)[0]]})
                sbu_temp = list(sbu_temp)
                sbu_cart_coords = np.array([at.coords() \
                       for at in [molcif.getAtom(val) for val in sbu_temp]])
                sbu_adjmat = slice_mat(adj_matrix.todense(), sbu_temp)
                pr_image_sbu = ligand_detect(cell_v, sbu_cart_coords, sbu_adjmat,set(range(len(linkeranchors_list))))
                if not (len(np.unique(pr_image_sbu, axis=0))==1 and len(np.unique(pr_image_organic, axis=0))==1): # linker
                    max_min_linker_length = max(min_length, max_min_linker_length)
                    min_max_linker_length = min(max_length, min_max_linker_length)
                    tmpstr = str(name)+','+' Anchors list: '+str(sbuanchors_list) \
                            +','+' SBU connectlist: '+str(sbu_connect_list)+' set to be linker\n'
                    write2file(linkerpath, "/ambiguous.txt", tmpstr)
                    continue
                else: #  all anchoring atoms are in the same unitcell -> ligand
                    removelist.update(set(templist[ii])) # we also want to remove these ligands
                    SBUlist.update(set(templist[ii])) # we also want to remove these ligands
                    linker_list.pop(ii)
                    linker_subgraphlist.pop(ii)
                    tmpstr = str(name)+','+' Anchors list: '+str(sbuanchors_list) \
                            +','+' SBU connectlist: '+str(sbu_connect_list)+' set to be ligand\n'
                    write2file(linkerpath, "/ambiguous.txt", tmpstr)
                    tmpstr = str(name)+str(ii)+','+' Anchors list: '+ \
                            str(sbuanchors_list)+','+' SBU connectlist: '+str(sbu_connect_list)+'\n'
                    write2file(linkerpath, "/ligand.txt", tmpstr)
        else: #definite ligand
            write2file(logpath, "/%s.log"%name, "found ligand\n")
            removelist.update(set(templist[ii])) # we also want to remove these ligands
            SBUlist.update(set(templist[ii])) # we also want to remove these ligands
            linker_list.pop(ii)
            linker_subgraphlist.pop(ii)
            tmpstr = str(name)+','+' Anchors list: '+str(sbuanchors_list) \
         +','+' SBU connectlist: '+str(sbu_connect_list)+'\n'
            write2file(linkerpath, "/ligand.txt", tmpstr)

    tmpstr = str(name) + ", (min_max_linker_length,max_min_linker_length): " + \
                str(min_max_linker_length) + " , " +str(max_min_linker_length) + "\n"
    write2file(logpath, "/%s.log"%name, tmpstr)
    if min_max_linker_length < 3:
        write2file(linkerpath, "/short_ligands.txt", tmpstr)
    if min_max_linker_length > 2:
        # for N-C-C-N ligand ligand
        if max_min_linker_length == min_max_linker_length:
            long_ligands = True
        elif min_max_linker_length > 3:
            long_ligands = True
    return min_max_linker_length, long_ligands, SBUlist, removelist, linker_list, linker_subgraphlist

def make_MOF_fragments(data, path=False, xyzpath=False):
    """
    Breaks a MOF into fragments for use with pormake (in silico MOF construction).

    Output codes are as follows:
    2: There exist short (i.e. 2 atom) and longer linkers. We could not split the MOF apart consistently.
    3: The MOF consists only of very short 2 atom linkers.
    4: The MOF contains a 1D rod, which cannot be easily reassembled into a new MOF.
    None: The MOF was split correctly

    Parameters
    ----------
    data : str
        The path to the cif file for which SBUs and linkers will be identified.
    path : str
        The parent path to which output will be written. Will contain a folder for SBUs and another for linkers.
    xyzpath : str
        The path to which an xyz file and a net (connectivity) file of the MOF will be written.

    Returns
    -------
    return_code : int or None
        See function description for possible return codes and their meanings.

    """
    if not path:
        print('Need a directory to place all of the linker and SBU objects. Exiting now.')
        raise ValueError('Base path must be specified in order to write descriptors.')
    else:
        if path.endswith('/'):
            path = path[:-1]
        if not os.path.isdir(path+'/linkers'):
            os.mkdir(path+'/linkers')
        if not os.path.isdir(path+'/sbus'):
            os.mkdir(path+'/sbus')
        if not os.path.isdir(path+'/xyz'):
            os.mkdir(path+'/xyz')
        if not os.path.isdir(path+'/logs'):
            os.mkdir(path+'/logs')
    linkerpath = path+'/linkers'
    sbupath = path+'/sbus'
    logpath = path+"/logs"

    '''
    Input cif file and get the cell parameters and adjacency matrix. If overlap, do not featurize.
    Simultaneously prepare mol3D class for MOF for future RAC featurization (molcif)
    '''

    cpar, allatomtypes, fcoords = readcif(data)
    cell_v = mkcell(cpar)
    cart_coords = fractional2cart(fcoords, cell_v)
    name = os.path.basename(data).strip(".cif")
    if len(cart_coords) > 2000:
        print("cif file is too large, skipping it for now...")
        tmpstr = "Failed to featurize %s: large primitive cell\n"%(name)
        write2file(path,"/FailedStructures.log", tmpstr)
        return None, None
    distance_mat = compute_distance_matrix3(cell_v, cart_coords)
    try:
        adj_matrix, _ = compute_adj_matrix(distance_mat, allatomtypes)
    except NotImplementedError:
        tmpstr = "Failed to featurize %s: atomic overlap\n"%(name)
        write2file(path,"/FailedStructures.log", tmpstr)
        return None, None

    writeXYZandGraph(xyzpath, allatomtypes, cell_v, fcoords, adj_matrix.todense())
    molcif,_,_,_,_ = import_from_cif(data, True)
    molcif.graph = adj_matrix.todense()

    '''
    check number of connected components.
    if more than 1: it checks if the structure is interpenetrated. Fails if no metal in one of the connected components (identified by the graph).
    This includes floating solvent molecules.
    '''

    n_components, labels_components = sparse.csgraph.connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    metal_list = set([at for at in molcif.findMetal(transition_metals_only=False)])
    if not len(metal_list) > 0:
        tmpstr = "Failed to featurize %s: no metal found\n"%(name)
        write2file(path,"/FailedStructures.log", tmpstr)
        return None, None

    for comp in range(n_components):
        inds_in_comp = [i for i in range(len(labels_components)) if labels_components[i]==comp]
        if not set(inds_in_comp)&metal_list:
            tmpstr = "Failed to featurize %s: solvent molecules\n"%(name)
            write2file(path,"/FailedStructures.log", tmpstr)
            return None, None

    if n_components > 1 :
        print("structure is interpenetrated")
        tmpstr = "%s found to be an interpenetrated structure\n"%(name)
        write2file(logpath, "/%s.log"%name, tmpstr)

    '''
    step 1: metallic part
        removelist = metals (1) + atoms only connected to metals (2) + H connected to (1+2)
        SBUlist = removelist + 1st coordination shell of the metals
    removelist = set()
    Logs the atom types of the connecting atoms to the metal in logpath.
    '''
    removelist, SBUlist = prepare_initial_SBU(molcif, allatomtypes, metal_list, logpath, name)

    '''
    At this point:
    The remove list only removes metals and things ONLY connected to metals or hydrogens.
    Thus the coordinating atoms are double counted in the linker.

    step 2: organic part
        removelist = linkers are all atoms - the removelist (assuming no bond between
        organic linkers)
    '''
    anc_atoms, linkers, linker_list, linker_subgraphlist, allatoms, connections_list, connections_subgraphlist = identify_lc_atoms(molcif, removelist, metal_list)

    '''
    step 3: linker or ligand ?
    checking to find the anchors and #SBUs that are connected to an organic part
    anchor <= 1 -> ligand
    anchor > 1 and #SBU > 1 -> linker
    else: walk over the linker graph and count #crossing PBC
        if #crossing is odd -> linker
        else -> ligand
    '''
    initial_SBU_list, initial_SBU_subgraphlist = get_closed_subgraph(removelist.copy(), linkers.copy(), adj_matrix)
    min_max_linker_length, long_ligands, SBUlist, removelist, linker_list, linker_subgraphlist = identify_short_linkers(molcif, initial_SBU_list, initial_SBU_subgraphlist, removelist, linkers, linker_list, linker_subgraphlist, adj_matrix, SBUlist, logpath, linkerpath, name, cell_v)

    '''
    In the case of long linkers, add second coordination shell without further checks. In the case of short linkers, start from metal
    and grow outwards using the include_extra_shells function
    '''
    linker_length_list = [len(linker_val) for linker_val in linker_list]
    if len(set(linker_length_list)) != 1:
        write2file(linkerpath, "/uneven.txt", str(name)+'\n')
    if min_max_linker_length > 2: # treating the 2 atom ligands differently! Need caution
        if long_ligands:
            tmpstr = "\nStructure has LONG LINKER\n\n"
            write2file(logpath, "/%s.log"%name, tmpstr)
            [[SBUlist.add(val) for val in molcif.getBondedAtomsSmart(zero_first_shell)] for zero_first_shell in SBUlist.copy()] #First account for all of the carboxylic acid type linkers, add in the carbons.
        truncated_linkers = allatoms - SBUlist
        SBU_list, SBU_subgraphlist = get_closed_subgraph(SBUlist, truncated_linkers, adj_matrix)
        if not long_ligands:
            tmpstr = "\nStructure has SHORT LINKER\n\n"
            write2file(logpath, "/%s.log"%name, tmpstr)
            SBU_list , SBU_subgraphlist = include_extra_shells(SBU_list, SBU_subgraphlist, molcif, adj_matrix)
            print('=== SKIPPING DUE TO LINKER BEING TOO SHORT!')
            return 2
    else:
        tmpstr = "Structure %s has extremely short linkers, check the outputs\n"%name
        write2file(linkerpath, "/short.txt", tmpstr)
        tmpstr = "Structure has extremely short linkers\n"
        write2file(logpath, "/%s.log"%name, tmpstr)
        truncated_linkers = allatoms - removelist
        SBU_list, SBU_subgraphlist = get_closed_subgraph(removelist, truncated_linkers, adj_matrix)
        SBU_list, SBU_subgraphlist = include_extra_shells(SBU_list, SBU_subgraphlist, molcif, adj_matrix)
        SBU_list, SBU_subgraphlist = include_extra_shells(SBU_list, SBU_subgraphlist, molcif, adj_matrix)
        print('=== SKIPPING DUE TO LINKER BEING TOO SHORT!')
        return 3

    return_code = breakdown_MOF(SBU_list, SBU_subgraphlist, molcif, name, cell_v, anc_atoms, sbupath, connections_list, connections_subgraphlist, linkerpath)
    return return_code
