from molSimplify.Scripts.io import getlicores, lig_load
from copy import copy, deepcopy

# NOTE: -isomers does not currently support ligands with denticity > 2 or complexes with 3 bidentate molecules

# These adjacency matrices represent what an atom at each position can "see."
# For example, in an octahedral complex, an atom at the 0th position in an
# octahedral complex can see atoms at positions (1,3,4,5). Used to check
# whether a given isomer is unique. NOTE: uses python numbering (zero indexed)
oct_adjacency = {0: (1, 3, 4, 5), 1: (0, 2, 4, 5), 2: (1, 3, 4, 5), 3: (0, 2, 4, 5),
                 4: (0, 1, 2, 3), 5: (0, 1, 2, 3)}
thd_adjacency = {0: (1, 2, 3), 1: (0, 2, 3), 2: (0, 1, 3), 3: (0, 1, 2)}
sqp_adjacency = {0: (1, 3), 1: (0, 2), 2: (1, 3), 3: (0, 2)}
tbp_adjacency = {0: (2, 3, 4), 1: (2, 3, 4), 2: (0, 1), 3: (0, 1), 4: (0, 1)}
spy_adjacency = {0: (1, 3, 4), 1: (0, 2, 4), 2: (1, 3, 4), 3: (0, 2, 4), 4: (0, 1, 2, 3)}
pbp_adjacency = {0: (1, 4), 1: (0, 2), 2: (1, 3), 3: (2, 4), 4: (3, 0), 5: (0, 1, 2, 3, 4), 6: (0, 1, 2, 3, 4)}

# Core functionality of isomers.py
#  @param args A class containing the user specified input
#  @return collapsed A list of lists, each list contains the args.lig for a unique isomer


def generateisomers(args):
    # Check if this is a non-supported case
    dents = []
    for ligands in args.lig:
        denticity = checkdenticity(ligands)
        dents.append(denticity)
    if max(dents) > 2:
        print('WARNING, -isomers does not support ligand denticities greater than two. Quiting...')
        return []

    # If supported, run the program
    expanded = expandrepresentation(args)
    permutations = findpermutations(expanded)
    unique_permutations = checkunique(args, permutations)
    isomers = collapserepresentation(args, unique_permutations)
    if args.stereos:
        isomers = generatestereo(isomers)

    return isomers

# Takes the user specified -lig and rewrites it in an explicit form
#  @param args The list of user specified inputs
#  @return expanded The ligand list in expanded form


def expandrepresentation(args):
    expanded_temp = []
    for occ, ligand in enumerate(args.lig):
        occ = int(args.ligocc[occ])
        for duplicate in range(occ):
            expanded_temp.append(ligand)

    expanded = []
    for counter, ligand in enumerate(expanded_temp):
        if checkdenticity(ligand) == 2:
            expanded.append(ligand+'_alphabond')
            expanded.append(ligand+'_betabond')
        elif checkdenticity(ligand) > 2:
            print(
                'WARNING, -isomers does not fully support ligand denticities greater than two!')
            expanded.append(ligand)
        else:
            expanded.append(ligand)
    return expanded

# Given a list, generates all possible permutations of that list
#  @param lst The list to permute
#  @return master A list of lists, with each specifying a permutation


def findpermutations(lst, master=None):
    if master is None:
        master = []
    for _ in range(len(lst)-1):
        if not master:
            for element in lst:
                master.append([element])

        master_updated = []
        for permutation in master:
            master_tmp = []
            lst_tmp = copy(lst)
            for element in permutation:
                lst_tmp.remove(element)
            for element in lst_tmp:
                master_tmp.append(permutation+[element])
            master_updated = master_updated + master_tmp

        master = deepcopy(master_updated)

    return master

# Filters a list of lig lists for permutations which correspond to unique compounds
#
#  based on the geometry of the metal complex
#
#  @param args The user specified input
#  @param master Should be left empty and allowed to default to an empty list
#  @return A smaller list of lists, corresponding only to unique compounds


def checkunique(args, permutations):
    adjacency = getadjacency(args.geometry)
    unique_representations = []
    unique_simple_geometries = []

    for permutation in permutations:
        geometry = []
        for index, ligand in enumerate(permutation):
            visible_indices = adjacency[index]

            visible_ligands = []
            for i in visible_indices:
                visible_ligands.append(permutation[i])
            visible_ligands.sort()

            geometry.append([permutation[index]] + visible_ligands)

        if checkallowedbidentates(permutation, args.geometry):
            geometry.sort()
            if checkincluded(geometry, unique_representations):
                unique_representations.append(geometry)
                unique_simple_geometries.append(permutation)

    return unique_simple_geometries

# Changes the lig list from the expanded representation back to the traditional molsimplify representation
#
#  Also, filters out compounds which are not actually unique due to symmetry in bidentate ligands
#
#  @param args The user specified input
#  @param permutations A list of lists. The lig lists in expanded representation to be collapsed
#  @return recheck_unique A further filtered -lig list, back in the standard representation


def collapserepresentation(args, permutations):

    # First, relabel symmetric ligands
    permutations_tmp = deepcopy(permutations)
    for geo in permutations_tmp:
        for counter, ligand in enumerate(geo):
            if ligand.endswith('_alphabond'):
                name = ligand.split('_')[0]
                if checksymmetric(name):
                    geo[counter] = name+'_sym'
                    geo[counter+1] = name+'_sym'
            elif ligand.endswith('_betabond'):
                name = ligand.split('_')[0]
                if checksymmetric(name):
                    geo[counter] = name+'_sym'
                    geo[counter+1] = name+'_sym'

    recheck_unique = checkunique(args, permutations_tmp)

    for geo in recheck_unique:
        for counter, ligand in enumerate(geo):
            if ligand.endswith('_alphabond'):
                name = ligand.split('_')[0]
                geo[counter] = name
                del geo[counter+1]
            elif ligand.endswith('_betabond'):
                name = ligand.split('_')[0]
                geo[counter] = name+'_flipped'
                del geo[counter+1]
            elif ligand.endswith('_sym'):
                name = ligand.split('_')[0]
                geo[counter] = name
                del geo[counter+1]

    return recheck_unique

# If requested, generates the stereoisomer of all unique isomers found by isomers.py
#
#  Stereoisomers are generated by reflecting the ligands over the plane containing ligands 2,4,5 and 6
#  There is no check to determine if the generated "stereoisomer" is unique from the original compound
#
#  @param collapsed_representation A list of lists specifying all possible isomers. Output by collapserepresentation()
#  @return stereoisomers_Final A list of lists containing twice as many values as collapsed_representation, corresponding to the stereoisomers for each compound.


def generatestereo(collapsed_representation):
    stereoisomers_tmp = deepcopy(collapsed_representation)

    stereoisomers = []
    for isomer in stereoisomers_tmp:
        stereoisomer = []
        for ligand in isomer:
            if checkdenticity(ligand) == 2:
                if ligand.endswith('_flipped'):
                    stereoisomer.append(ligand.split('_')[0]+'_betabond')
                    stereoisomer.append(ligand.split('_')[0]+'_alphabond')
                else:
                    stereoisomer.append(ligand.split('_')[0]+'_alphabond')
                    stereoisomer.append(ligand.split('_')[0]+'_betabond')
            else:
                stereoisomer.append(ligand)
        if len(stereoisomer) == 6:
            stereoisomers.append(stereoisomer)
        else:
            print(
                'WARNING, isomers.py has detected a non-octahedral complex in stereoisomer generation.')
            print('Stereoisomer generation only supports octahedral complexes!')

    stereoisomers_final = []
    for isomer in stereoisomers:
        stereo = copy(isomer)
        # flip ligand 1 and ligand 3 to generate a stereoisomer
        stereo[0], stereo[2] = stereo[2], stereo[0]
        # skip rotation if sites 4 and 5 are bonded
        if not stereo[4].endswith('_alphabond') and not stereo[4].endswith('_betabond'):
            # molSimplify can't handle a bidentate ligand at sites 1 and 4, to address this, rotate the compound 90 degrees in the equatorial plane
            stereo[0], stereo[1], stereo[2], stereo[3] = stereo[3], stereo[0], stereo[1], stereo[2]
        stereoisomers_final.append(isomer)
        stereoisomers_final.append(stereo)

    for stereoisomer in stereoisomers_final:
        for counter, ligand in enumerate(stereoisomer):
            if ligand.endswith('_alphabond'):
                stereoisomer[counter] = ligand.split('_')[0]
                del stereoisomer[counter+1]
            elif ligand.endswith('_betabond'):
                stereoisomer[counter] = ligand.split('_')[0]+'_flipped'
                del stereoisomer[counter+1]
    return stereoisomers_final


# Filters to only allow bidentates in adjacent locations on the metal complex
#
#  This is enforced by ensuring that bidentates are adjacent in the exoanded -lig list
#
#  @param simple_geometry A lig list in expanded form.
#  @return allowed Returns a boolean. True if the bidentates are in allowed positions.
def checkallowedbidentates(simple_geometry, geometry):
    simple_geometry_tmp = copy(simple_geometry)

    # Check if bidentate is in a position where there's nothing to coordinate to

    if geometry in ['oct', 'pbp']:
        if simple_geometry[-1].endswith('_alphabond') or simple_geometry[-1].endswith('_betabond'):
            return False
    if geometry in ['tbp']:
        if simple_geometry[0].endswith('_alphabond') or simple_geometry[0].endswith('_betabond'):
            return False

    for counter, bonds in enumerate(simple_geometry_tmp):
        if counter > 0:
            bond_previous = simple_geometry_tmp[counter - 1]
        else:
            bond_previous = 'None'
        if counter < (len(simple_geometry) - 1):
            bond_next = simple_geometry_tmp[counter + 1]
        else:
            bond_next = 'None'

        ligand_name = bonds.split('_')[0]

        if bonds.endswith('_alphabond'):
            if bond_previous.endswith('_betabond') and bond_previous.startswith(ligand_name):
                simple_geometry_tmp[counter] = 'None'
                simple_geometry_tmp[counter - 1] = 'None'
            elif bond_next.endswith('_betabond') and bond_next.startswith(ligand_name):
                simple_geometry_tmp[counter] = 'None'
                simple_geometry_tmp[counter + 1] = 'None'
            else:
                return False

        elif bonds.endswith('_betabond'):
            if bond_previous.endswith('_alphabond') and bond_previous.startswith(ligand_name):
                simple_geometry_tmp[counter] = 'None'
                simple_geometry_tmp[counter - 1] = 'None'
            elif bond_next.endswith('_alphabond') and bond_next.startswith(ligand_name):
                simple_geometry_tmp[counter] = 'None'
                simple_geometry_tmp[counter + 1] = 'None'
            else:
                return False
    return True
# Fetches the ligand dictionary and returns the denticity of a ligand
#  @param ligand The name of the ligand, as a string
#  @return dent The denticity of the ligand.


def checkdenticity(ligand):
    ligands_dict = getlicores()
    connecting_atoms = ligands_dict[ligand][2]
    dent = len(connecting_atoms)
    return dent

# Fetches the adjacency dictionary (beginning of this script) for a given geometry
#  @param The metal complex geometry, in short form, as a string
#  @return adjacency The adjacency matrix, if it exists. Else, returns None.


def getadjacency(geo):
    if geo == 'oct':
        return oct_adjacency
    elif geo == 'thd':
        return thd_adjacency
    elif geo == 'sqp':
        return sqp_adjacency
    elif geo == 'tbp':
        return tbp_adjacency
    elif geo == 'spy':
        return spy_adjacency
    elif geo == 'pbp':
        return pbp_adjacency
    else:
        print('****************************************************')
        print(('****** WARNING, '+geo+' not supported by -isomers! *****'))
        print('****************************************************')
        return None

# Searches through a list and returns the index(es) as which a value occurs
#  @param lst The list to search through
#  @param key The value to search for
#  @retrun indices A list of indices at which key is found


def searchlist(lst, key):
    indices = []
    for counter, element in enumerate(lst):
        if element == key:
            indices.append(counter)
    return indices

# Checks if a list is in a list of lists
#  @param geometry The list which may or may not be included
#  @param unique_representations The list of lists to search through
#  @return unique Returns a boolean. True if the list is NOT in the list of lists


def checkincluded(geometry, unique_representations):
    unique_array = []
    unique = False
    if unique_representations:
        for saved_geometry in unique_representations:
            for counter, foo in enumerate(geometry):
                breaker = False
                try:
                    for counter2, spam in enumerate(geometry[counter]):
                        if saved_geometry[counter][counter2] != geometry[counter][counter2]:
                            unique_array.append(True)
                            breaker = True
                        if breaker:
                            break
                    if breaker:
                        break
                except IndexError:
                    pass

        if len(unique_array) == len(unique_representations):
            unique = True
        else:
            unique = False
    else:
        unique = True

    return unique

# Checks if a ligand is symmetric when it binds a metal
#  @param lig The name of the ligand, as a string
#  @return symmetric Returns a boolean. True if the ligand is symmetric


def checksymmetric(lig):
    symmetric = True
    ligand, emsg = lig_load(lig)
    ligand.convert2mol3D()
    ligands_dict = getlicores()
    connecting_atoms = ligands_dict[lig][2]

    # Each bonding atom will be represented as a bonding enviroment. This
    # is a list of lists, where each individual list corresponds to atoms
    # a certain number of bonds away from the bonding atom. The
    # bonding_atom_environments variable holds a list of bonding environemnts,
    # one for each bonding atom.
    bonding_atom_environments = []
    for atom in connecting_atoms:
        index = int(atom)
        coordination_spheres = [[index]]
        used_atoms = {index}

        finding_atoms = True
        while finding_atoms:
            current_sphere = coordination_spheres[-1]
            length = len(coordination_spheres)

            next_sphere = set([])
            for atoms in current_sphere:
                # get a set containing elements from both sets
                next_sphere = next_sphere | set(ligand.getBondedAtoms(atoms))

            next_sphere = next_sphere - used_atoms  # subtracting sets
            used_atoms = used_atoms | next_sphere
            if list(next_sphere):
                coordination_spheres.append(list(next_sphere))

            if length == len(coordination_spheres):
                finding_atoms = False
        bonding_atom_environments.append(coordination_spheres)

    bonding_atom_environments0 = deepcopy(bonding_atom_environments)
    # Change the list of atoms from atom indices to atom names, also sort them
    for counter1, bonding_atoms in enumerate(bonding_atom_environments0):
        for counter2, sphere in enumerate(bonding_atoms):
            for counter3, atom in enumerate(sphere):
                atom = ligand.getAtom(int(atom))
                bonding_atom_environments0[counter1][counter2][counter3] = atom.name
            sphere.sort()
    if bonding_atom_environments0[0] == bonding_atom_environments0[1]:
        symmetric = True
    else:
        symmetric = False
    return symmetric
