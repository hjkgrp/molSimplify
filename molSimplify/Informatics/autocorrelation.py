import numpy as np
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import ligand_breakdown, ligand_assign
from molSimplify.Scripts.geometry import distance
from molSimplify.Classes.globalvars import globalvars

# ########## UNIT CONVERSION
HF_to_Kcal_mol = 627.503


def autocorrelation(mol, prop_vec, orig, d, oct=True, catoms=None, use_dist=False):
    """Calculate and return the products autocorrelation for a single atom

    Parameters
    ----------
        mol : mol3D
            mol3D object to calculate autocorrelation over
        prop_vec : list
            property of atoms in mol in order of index
        orig : int
            zero-indexed starting atom
        d : int
            number of hops to travel
        oct : bool, optional
            Flag is octahedral complex, by default True
        catoms: list, optional
            List of connecting atoms, by default None (uses mol3D.getBondedAtomsSmart)
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False

    Returns
    -------
        result_vector : list
            assembled products autocorrelations

    """
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    if not use_dist:
        result_vector[hopped] = prop_vec[orig] * prop_vec[orig]
    else:
        result_vector[hopped] = 0.5 * abs(prop_vec[orig]) ** 2.4 / mol.natoms
    while hopped < (d):

        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # prepare all atoms attached to this connection
            # print('called in AC')
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        # print('new active set at hop = ' +str(hopped) + ' is ' +str(new_active_set))
        for inds in new_active_set:
            if not use_dist:
                result_vector[hopped] += prop_vec[orig] * prop_vec[inds]
            else:
                this_dist = distance(mol.getAtom(orig).coords(), mol.getAtom(inds).coords())
                result_vector[hopped] += prop_vec[orig] * prop_vec[inds] / (this_dist * mol.natoms)
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def autocorrelation_derivative(mol, prop_vec, orig, d, oct=True, catoms=None):
    """Returns derivative vector of products autocorrelations

    Parameters
    ----------
        mol : mol3D
            mol3D object to calculate derivatives over
        prop_vec : list
            property of atoms in mol in order of index
        orig : int
            zero-indexed starting atom
        d : int
            number of hops to travel
        oct : bool, optional
            Flag is octahedral complex, by default True
        catoms : list, optional
            List of connecting atom, by default None (use mol3D.getBondedAtomsSmart)
    Returns
    -------
        derivative_mat : list
            RAC derivatives matrix

    """
    derivative_mat = np.zeros((d + 1, len(prop_vec)))

    # loop for each atom

    hopped = 0
    active_set = set([orig])
    historical_set = set()
    for derivate_ind in range(0, len(prop_vec)):
        if derivate_ind == orig:
            derivative_mat[hopped, derivate_ind] = 2 * prop_vec[orig]
        else:
            derivative_mat[hopped, derivate_ind] = 0
    while hopped < (d):

        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # prepare all atoms attached to this connection
            # print('called in AC')
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        # print('new active set at hop = ' +str(hopped) + ' is ' +str(new_active_set))
        for inds in new_active_set:
            for derivate_ind in range(0, len(prop_vec)):
                if derivate_ind == orig:
                    derivative_mat[hopped, derivate_ind] += prop_vec[inds]
                elif derivate_ind == inds:
                    derivative_mat[hopped, derivate_ind] += prop_vec[orig]
            historical_set.update(active_set)
        active_set = new_active_set
    return (derivative_mat)


def ratiometric(mol, prop_vec_num, prop_vec_den, orig, d, oct=True, catoms=None):
    """This function returns the ratiometrics for one atom

    Parameters
    ----------
        mol : mol3D class
        prop_vec : vector, property of atoms in mol in order of index
        orig : int, zero-indexed starting atom
        d : int, number of hops to travel
        oct : bool, if complex is octahedral, will use better bond checks

    Returns
    -------
        result_vector : vector of prop_vec_num / prop_vec_den

    """
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    result_vector[hopped] = prop_vec_num[orig] / prop_vec_den[orig]
    """
        if oct:
            print('using OCT autocorrelation')
        else:
            print('NOT using OCT autocorrelation')
    """
    while hopped < (d):

        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # prepare all atoms attached to this connection
            # print('called in AC')
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        # print('new active set at hop = ' +str(hopped) + ' is ' +str(new_active_set))
        for inds in new_active_set:
            result_vector[hopped] += prop_vec_num[orig] / prop_vec_den[inds]
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def summetric(mol, prop_vec, orig, d, oct=True, catoms=None):
    """This function returns the summetrics for one atom

    Parameters
    ----------
        mol : mol3D class
        prop_vec : vector, property of atoms in mol in order of index
        orig : int, zero-indexed starting atom
        d : int, number of hops to travel
        oct : bool, if complex is octahedral, will use better bond checks

    Returns
    -------
        result_vector : vector of prop_vec_num / prop_vec_den

    """
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    result_vector[hopped] = prop_vec[orig] + prop_vec[orig]
    """
        if oct:
            print('using OCT autocorrelation')
        else:
            print('NOT using OCT autocorrelation')
    """
    while hopped < (d):

        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # prepare all atoms attached to this connection
            # print('called in AC')
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        # print('new active set at hop = ' +str(hopped) + ' is ' +str(new_active_set))
        for inds in new_active_set:
            result_vector[hopped] += prop_vec[orig] + prop_vec[inds]
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def deltametric(mol: mol3D, prop_vec, orig, d: int, oct=True, catoms=None):
    # # this function returns the deltametric
    # # over the whole molecule
    # Inputs:
    #   mol - mol3D class
    #   prop_vec - vector, property of atoms in mol in order of index
    #   orig -  int, zero-indexed starting atom
    #   d - int, number of hops to travel
    #   oct - bool, if complex is octahedral, will use better bond checks
    #   if oct:
    #       print('using OCT delta autocorrelation')
    #   else:
    #       print('NOT using OCT delta autocorrelation')
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    result_vector[hopped] = 0.00
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # prepare all atoms attached to this connection
            # print('called in DAC')
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        # print('new active set at hop = ' +str(hopped) + ' is ' +str(new_active_set))
        for inds in new_active_set:
            result_vector[hopped] += prop_vec[orig] - prop_vec[inds]
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def autocorrelation_catoms(mol, prop_vec, orig, d, oct=True, catoms=None):
    # Calculate the autocorrelation for the orig to certain connecting atoms.
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    result_vector[hopped] = prop_vec[orig] * prop_vec[orig]
    #    if oct:
    #        print('using OCT autocorrelation')
    #    else:
    #        print('NOT using OCT autocorrelation')
    while hopped < (d):

        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # prepare all atoms attached to this connection
            # print('called in AC')
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            # print('--1--:', this_atoms_neighbors)
            if this_atom == orig and (catoms is not None):
                this_atoms_neighbors = catoms
            # print('--2--:', this_atoms_neighbors)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        # print('new active set at hop = ' +str(hopped) + ' is ' +str(new_active_set))
        for inds in new_active_set:
            result_vector[hopped] += prop_vec[orig] * prop_vec[inds]
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def deltametric_derivative(mol, prop_vec, orig, d, oct=True, catoms=None):
    # # this function returns the derivative vector
    # # of the scalar autocorrelation
    # # starting at orig with depth d,
    # # with respect to the atomic properties
    # # in prop_vec, for all atoms.
    # # The return type is np.array for
    # # Be sure to read this carefully!
    # Inputs:
    #    mol - mol3D class
    #    prop_vec - vector, property of atoms in mol in order of index
    #    orig -  int, zero-indexed starting atom
    #    d - int, number of hops to travel
    #    oct - bool, if complex is octahedral, will use better bond checks
    #    if oct:
    #        print('using OCT delta autocorrelation')
    #    else:
    #        print('NOT using OCT delta autocorrelation')

    derivative_mat = np.zeros((d + 1, len(prop_vec)))

    hopped = 0
    active_set = set([orig])
    historical_set = set()

    # the zero-depth element is always zero
    for derivate_ind in range(0, len(prop_vec)):
        derivative_mat[hopped, derivate_ind] = 0.0

    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # prepare all atoms attached to this connection
            # print('called in DAC')
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        # print('new active set at hop = ' +str(hopped) + ' is ' +str(new_active_set))
        for inds in new_active_set:

            for derivate_ind in range(0, len(prop_vec)):
                if derivate_ind == orig:
                    derivative_mat[hopped, derivate_ind] += 1
                elif derivate_ind == inds:
                    derivative_mat[hopped, derivate_ind] += -1

        historical_set.update(active_set)
        active_set = new_active_set
    return (derivative_mat)


def deltametric_catoms(mol, prop_vec, orig, d, oct=True, catoms=None):
    # Calculate the deltametrics for the orig to certain connecting atoms.
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    result_vector[hopped] = 0.00
    # metal_idx = mol.findMetal()[0]
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # prepare all atoms attached to this connection
            # print('called in DAC')
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            # print('--1--:', this_atoms_neighbors)
            if this_atom == orig and (catoms is not None):
                this_atoms_neighbors = catoms
            # print('--2--:', this_atoms_neighbors)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        # print('new active set at hop = ' +str(hopped) + ' is ' +str(new_active_set))
        for inds in new_active_set:
            result_vector[hopped] += prop_vec[orig] - prop_vec[inds]
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def full_autocorrelation(mol, prop, d, oct=oct, modifier=False, use_dist=False, transition_metals_only=True):
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier, transition_metals_only=transition_metals_only)
    index_set = list(range(0, mol.natoms))
    autocorrelation_vector = np.zeros(d + 1)
    for centers in index_set:
        autocorrelation_vector += autocorrelation(mol, w, centers, d, oct=oct, use_dist=use_dist)
    return (autocorrelation_vector)


def full_autocorrelation_derivative(mol, prop, d, oct=oct, modifier=False):
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
    index_set = list(range(0, mol.natoms))
    autocorrelation_derivative_mat = np.zeros((d + 1, mol.natoms))
    for centers in index_set:
        autocorrelation_derivative_mat += autocorrelation_derivative(mol, w, centers, d, oct=oct)
    return (autocorrelation_derivative_mat)


def atom_only_autocorrelation(mol, prop, d, atomIdx, oct=True):
    # atomIdx must be either a list of indicies
    # or a single index
    w = construct_property_vector(mol, prop, oct)
    autocorrelation_vector = np.zeros(d + 1)
    if hasattr(atomIdx, "__len__"): # Indicative of a list of indices
        for elements in atomIdx:
            autocorrelation_vector += autocorrelation(mol, w, elements, d, oct=oct)
        autocorrelation_vector = np.divide(autocorrelation_vector, len(atomIdx)) # averaging
    else: # Single index
        autocorrelation_vector += autocorrelation(mol, w, atomIdx, d, oct=oct)
    return (autocorrelation_vector)


def atom_only_autocorrelation_derivative(mol, prop, d, atomIdx, oct=True):
    # atomIdx must b either a list of indicies
    # or a single index
    w = construct_property_vector(mol, prop, oct)
    autocorrelation_derivative_mat = np.zeros((d + 1, mol.natoms))
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            autocorrelation_derivative_mat += autocorrelation_derivative(mol, w, elements, d, oct=oct)
        autocorrelation_derivative_mat = np.divide(autocorrelation_derivative_mat, len(atomIdx))
    else:
        autocorrelation_derivative_mat += autocorrelation_derivative(mol, w, atomIdx, d, oct=oct)
    return (autocorrelation_derivative_mat)


def metal_only_autocorrelation(mol, prop, d, oct=True, catoms=None,
                               func=autocorrelation, modifier=False):
    try:
        metal_ind = mol.findMetal()[0]
        w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
        autocorrelation_vector = func(mol, w, metal_ind, d, oct=oct,
                                      catoms=catoms)
    except IndexError:
        print('Error, no metal found in mol object!')
        return False
    return (autocorrelation_vector)


def metal_only_autocorrelation_derivative(mol, prop, d, oct=True, catoms=None,
                                          func=autocorrelation_derivative, modifier=False):
    try:
        metal_ind = mol.findMetal()[0]
        w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
        autocorrelation_vector_derivative = func(mol, w, metal_ind, d, oct=oct,
                                                 catoms=catoms)
    except IndexError:
        print('Error, no metal found in mol object!')
        return False
    return (autocorrelation_vector_derivative)


def multimetal_only_autocorrelation(mol, prop, d, oct=True, catoms=None,
                                    func=autocorrelation, modifier=False):
    autocorrelation_vector = np.zeros(d + 1)
    n_met = len(mol.findMetal())
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
    for metal_ind in mol.findMetal():
        autocorrelation_vector += func(mol, w, metal_ind, d, oct=oct, catoms=catoms)
    autocorrelation_vector = np.divide(autocorrelation_vector, n_met)
    return (autocorrelation_vector)


def multiatom_only_autocorrelation(mol, prop, d, oct=True, catoms=None,
                                   func=autocorrelation, modifier=False,
                                   additional_elements=False):
    autocorrelation_vector = np.zeros(d + 1)
    metal_list = mol.findMetal()
    if additional_elements:
        for element in additional_elements:
            metal_list += mol.findAtomsbySymbol(element)
    n_met = len(metal_list)
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
    for metal_ind in metal_list:
        autocorrelation_vector += func(mol, w, metal_ind, d, oct=oct, catoms=catoms)
    autocorrelation_vector = np.divide(autocorrelation_vector, n_met)
    return (autocorrelation_vector)


def atom_only_ratiometric(mol, prop_num, prop_den, d, atomIdx, oct=True):
    # atomIdx must b either a list of indicies
    # or a single index
    w_num = construct_property_vector(mol, prop_num, oct)
    w_den = construct_property_vector(mol, prop_den, oct)
    autocorrelation_vector = np.zeros(d + 1)
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            autocorrelation_vector += ratiometric(mol, w_num, w_den, elements, d, oct=oct)
        autocorrelation_vector = np.divide(autocorrelation_vector, len(atomIdx))
    else:
        autocorrelation_vector += ratiometric(mol, w_num, w_den, atomIdx, d, oct=oct)
    return (autocorrelation_vector)


def atom_only_summetric(mol, prop, d, atomIdx, oct=True):
    # atomIdx must b either a list of indicies
    # or a single index
    w = construct_property_vector(mol, prop, oct)
    autocorrelation_vector = np.zeros(d + 1)
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            autocorrelation_vector += summetric(mol, w, elements, d, oct=oct)
        autocorrelation_vector = np.divide(autocorrelation_vector, len(atomIdx))
    else:
        autocorrelation_vector += summetric(mol, w, atomIdx, d, oct=oct)
    return (autocorrelation_vector)


def atom_only_deltametric(mol, prop, d, atomIdx, oct=True, modifier=False):
    # atomIdx must b either a list of indicies
    # or a single index
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)

    deltametric_vector = np.zeros(d + 1)
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            deltametric_vector += deltametric(mol, w, elements, d, oct=oct)
        deltametric_vector = np.divide(deltametric_vector, len(atomIdx))
    else:
        deltametric_vector += deltametric(mol, w, atomIdx, d, oct=oct)
    return (deltametric_vector)


def atom_only_deltametric_derivative(mol, prop, d, atomIdx, oct=True, modifier=False):
    # atomIdx must b either a list of indicies
    # or a single index
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)

    deltametric_derivative_mat = np.zeros((d + 1, mol.natoms))
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            deltametric_derivative_mat += deltametric_derivative(mol, w, elements, d, oct=oct)
        deltametric_derivative_mat = np.divide(deltametric_derivative_mat, len(atomIdx))
    else:

        deltametric_derivative_mat += deltametric_derivative(mol, w, atomIdx, d, oct=oct)
    return (deltametric_derivative_mat)


def metal_only_deltametric_derivative(mol, prop, d, oct=True, catoms=None,
                                      func=deltametric_derivative, modifier=False):
    try:
        metal_ind = mol.findMetal()[0]
        w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
        deltametric_vector_derivative = func(mol, w, metal_ind, d, oct=oct,
                                             catoms=catoms)
    except IndexError:
        print('Error, no metal found in mol object!')
        return False
    return (deltametric_vector_derivative)


def metal_only_deltametric(mol, prop, d, oct=True, catoms=None,
                           func=deltametric, modifier=False):
    try:
        metal_ind = mol.findMetal()[0]
        w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
        deltametric_vector = func(mol, w, metal_ind, d, oct=oct,
                                  catoms=catoms)
    except IndexError:
        print('Error, no metal found in mol object!')
        return False
    return (deltametric_vector)


def multimetal_only_deltametric(mol, prop, d, oct=True, catoms=None,
                                func=deltametric, modifier=False):
    deltametric_vector = np.zeros(d + 1)
    n_met = len(mol.findMetal())

    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
    for metal_ind in mol.findMetal():
        deltametric_vector += func(mol, w, metal_ind, d, oct=oct,
                                   catoms=catoms)
    deltametric_vector = np.divide(deltametric_vector, n_met)
    return (deltametric_vector)


def multiatom_only_deltametric(mol, prop, d, oct=True, catoms=None,
                               func=deltametric, modifier=False,
                               additional_elements=False):
    deltametric_vector = np.zeros(d + 1)
    metal_list = mol.findMetal()
    if additional_elements:
        for element in additional_elements:
            metal_list += mol.findAtomsbySymbol(element)
    n_met = len(metal_list)
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
    for metal_ind in mol.findMetal():
        deltametric_vector += func(mol, w, metal_ind, d, oct=oct,
                                   catoms=catoms)
    deltametric_vector = np.divide(deltametric_vector, n_met)
    return (deltametric_vector)


def metal_only_layer_density(mol, prop, d, oct=True):
    try:
        metal_ind = mol.findMetal()[0]
        print(('metal_index is: %d' % metal_ind))
        w = construct_property_vector(mol, prop, oct=oct)
        density_vector = layer_density_in_3D(mol, w, metal_ind, d, oct=oct)
    except IndexError:
        print('Error, no metal found in mol object!')
        return False
    return density_vector


def layer_density_in_3D(mol, prop_vec, orig, d, oct=True):
    # # this function returns the density (prop^3/(d+1)^3)
    # # for one atom
    # Inputs:
    #    mol - mol3D class
    #    prop_vec - vector, property of atoms in mol in order of index
    #    orig -  int, zero-indexed starting atom
    #    d - int, number of hops to travel
    #    oct - bool, if complex is octahedral, will use better bond checks
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    result_vector[hopped] = prop_vec[orig] ** 3 / (hopped + 1) ** 3
    #    if oct:
    #        print('using OCT autocorrelation')
    #    else:
    #        print('NOT using OCT autocorrelation')
    while hopped < (d):

        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # prepare all atoms attached to this connection
            # print('called in AC')
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        # print('new active set at hop = ' +str(hopped) + ' is ' +str(new_active_set))
        for inds in new_active_set:
            result_vector[hopped] += prop_vec[inds] ** 3 / (hopped + 1) ** 3
            historical_set.update(active_set)
        active_set = new_active_set
    return result_vector


def construct_property_vector(mol: mol3D, prop: str, oct=True, modifier=False, transition_metals_only=True):
    # # assigns the value of property
    # # for atom i (zero index) in mol
    # # to position i in returned vector
    # # can be used to create weighted
    # # graph representations
    # # oct - bool, if complex is octahedral, will use better bond checks
    # # modifier - dict, used to modify prop vector (e.g. for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology',
                       'ox_nuclear_charge', 'size', 'vdwrad', 'group_number', 'polarizability',
                       'bondvalence', 'num_bonds', 'bondvalence_devi', 'bodavrg', 'bodstd', 'charge']
    # # note that ident just codes every atom as one, this gives
    # # a purely toplogical index. coord gives the number of
    # # connecting atom to attom i (similar to Randic index)
    # if not oct:
    #     print('NOT using octahedral bonding pattern')
    globs = globalvars()
    prop_dict = dict()
    w = np.zeros(mol.natoms)
    done = False
    if prop not in allowed_strings:
        print(('error, property  ' + str(prop) + ' is not a vaild choice'))
        print((' options are  ' + str(allowed_strings)))
        return False
    if prop == 'electronegativity':
        prop_dict = globs.endict()
    elif prop == 'size':
        at_keys = list(globs.amass().keys())
        for keys in at_keys:
            values = globs.amass()[keys][2]
            prop_dict.update({keys: values})
    elif prop == 'nuclear_charge':
        at_keys = list(globs.amass().keys())
        for keys in at_keys:
            values = globs.amass()[keys][1]
            prop_dict.update({keys: values})
    elif prop == 'group_number':  # Uses number of valence electrons
        # if not modifier:
        at_keys = list(globs.amass().keys())
        for keys in at_keys:
            values = globs.amass()[keys][-1]
            prop_dict.update({keys: values})
        # ###### 11/06/2019 -- Adjusted Gval RACs to not adjust on oxidation state. Confounded with O RACs. #####
        # # else:
        #     at_keys = globs.amass().keys()
        #     for keys in at_keys:
        #         values = globs.amass()[keys][3]
        #         if keys in modifier.keys():
        #             values -= float(modifier[keys]) # assumes oxidation state provided (i.e. Fe(IV))
        #         prop_dict.update({keys: values})
    elif prop == 'ox_nuclear_charge':
        if not modifier:
            print('Error, must give modifier with ox_nuclear_charge')
            return False
        else:
            at_keys = list(globs.amass().keys())
            for keys in at_keys:
                values = globs.amass()[keys][1]
                if keys in list(modifier.keys()):
                    values -= float(modifier[keys])  # assumes oxidation state provided (i.e. Fe(IV))
                prop_dict.update({keys: values})
    elif prop == 'polarizability':
        prop_dict = globs.polarizability()
        for i, atoms in enumerate(mol.getAtoms()):
            atom_type = atoms.symbol()
            w[i] = prop_dict[atom_type]
    elif prop == 'ident':
        at_keys = list(globs.amass().keys())
        for keys in at_keys:
            prop_dict.update({keys: 1})
    elif prop == 'topology':
        for i, atoms in enumerate(mol.getAtoms()):
            # print('atom # ' + str(i) + " symbol =  " + str(atoms.symbol()))
            w[i] = len(mol.getBondedAtomsSmart(i, oct=oct))
        done = True
    elif prop == 'vdwrad':
        prop_dict = globs.vdwrad()
        for i, atoms in enumerate(mol.getAtoms()):
            atom_type = atoms.symbol()
            if atom_type in globs.metalslist():
                w[i] = globs.amass()[atoms.symbol()][2]
            else:
                w[i] = prop_dict[atoms.symbol()]
        done = True
        # for keys in at_keys:
        #     prop_dict.update({keys: 1})
    elif prop == 'bondvalence':
        assert len(mol.getAtoms()) == len(mol.bv_dict)
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = mol.bv_dict[i]
        done = True
    elif prop == 'num_bonds':
        for i, atom in enumerate(mol.getAtoms()):
            if not atom.ismetal(transition_metals_only):
                w[i] = globs.bondsdict()[atom.symbol()]
            else:
                w[i] = len(mol.getBondedAtomsSmart(i, oct=oct))
        done = True
    elif prop == 'bondvalence_devi':
        assert len(mol.getAtoms()) == len(mol.bvd_dict)
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = mol.bvd_dict[i]
        done = True
    elif prop == 'bodavrg':
        assert len(mol.getAtoms()) == len(mol.bodavrg_dict)
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = mol.bodavrg_dict[i]
        done = True
    elif prop == 'bodstd':
        assert len(mol.getAtoms()) == len(mol.bodstd_dict)
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = mol.bodstd_dict[i]
        done = True
    elif prop == 'charge':
        assert len(mol.getAtoms()) == len(mol.charge_dict)
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = mol.charge_dict[i]
        done = True
    if not done:
        for i, atoms in enumerate(mol.getAtoms()):
            # print('atom # ' + str(i) + " symbol =  " + str(atoms.symbol()))
            w[i] = prop_dict[atoms.symbol()]
    return (w)


def find_ligand_autocorrelations_oct(mol, prop, loud, depth, name=False,
                                     oct=True, custom_ligand_dict=False):
    # # this function takes a
    # # symmetric (axial == axial,
    # # equatorial == equatorial)
    # # octahedral complex
    # # and returns autocorrelations for
    # # the axial an equatorial ligands
    # # custom_ligand_dict allows the user to skip the breakdown
    # # in cases where 3D geo is not correct/formed
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=oct)
        (ax_ligand_list, eq_ligand_list, ax_natoms_list,
         eq_natoms_list, ax_con_int_list, eq_con_int_list,
         ax_con_list, eq_con_list, built_ligand_list) = ligand_assign(
            mol, liglist, ligdents, ligcons, loud, name=False)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    # count ligands
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)
    # get full ligand AC
    ax_ligand_ac_full = []
    eq_ligand_ac_full = []
    for i in range(0, n_ax):
        if not list(ax_ligand_ac_full):
            ax_ligand_ac_full = full_autocorrelation(ax_ligand_list[i].mol, prop, depth)
        else:
            ax_ligand_ac_full += full_autocorrelation(ax_ligand_list[i].mol, prop, depth)
    ax_ligand_ac_full = np.divide(ax_ligand_ac_full, n_ax)
    for i in range(0, n_eq):
        if not list(eq_ligand_ac_full):
            eq_ligand_ac_full = full_autocorrelation(eq_ligand_list[i].mol, prop, depth)
        else:
            eq_ligand_ac_full += full_autocorrelation(eq_ligand_list[i].mol, prop, depth)
    eq_ligand_ac_full = np.divide(eq_ligand_ac_full, n_eq)

    # get partial ligand AC
    ax_ligand_ac_con = []
    eq_ligand_ac_con = []

    for i in range(0, n_ax):
        if not list(ax_ligand_ac_con):
            ax_ligand_ac_con = atom_only_autocorrelation(ax_ligand_list[i].mol, prop, depth, ax_con_int_list[i])
        else:
            ax_ligand_ac_con += atom_only_autocorrelation(ax_ligand_list[i].mol, prop, depth, ax_con_int_list[i])
    ax_ligand_ac_con = np.divide(ax_ligand_ac_con, n_ax)
    for i in range(0, n_eq):
        if not list(eq_ligand_ac_con):
            eq_ligand_ac_con = atom_only_autocorrelation(eq_ligand_list[i].mol, prop, depth, eq_con_int_list[i])
        else:
            eq_ligand_ac_con += atom_only_autocorrelation(eq_ligand_list[i].mol, prop, depth, eq_con_int_list[i])
    eq_ligand_ac_con = np.divide(eq_ligand_ac_con, n_eq)

    # ax_ligand_ac_con = atom_only_autocorrelation(ax_ligand.mol,prop,depth,ax_con_int)
    # eq_ligand_ac_con = atom_only_autocorrelation(eq_ligand.mol,prop,depth,eq_con_int)
    return ax_ligand_ac_full, eq_ligand_ac_full, ax_ligand_ac_con, eq_ligand_ac_con


def find_ligand_autocorrelation_derivatives_oct(mol, prop, loud, depth, name=False,
                                                oct=True, custom_ligand_dict=False):
    # # this function takes a
    # # symmetric (axial == axial,
    # # equatorial == equatorial)
    # # octahedral complex
    # # and returns autocorrelations for
    # # the axial an equatorial ligands
    # # custom_ligand_dict allows the user to skip the breakdown
    # # in cases where 3D geo is not correct/formed
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=oct)
        (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list,
         eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list) = ligand_assign(
            mol, liglist, ligdents, ligcons, loud, name=False)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    # count ligands
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)
    # get full ligand AC
    ax_ligand_ac_full_derivative = None
    eq_ligand_eq_full_derivative = None

    # allocate the full jacobian matrix
    ax_full_j = np.zeros([depth + 1, mol.natoms])
    eq_full_j = np.zeros([depth + 1, mol.natoms])
    ax_con_j = np.zeros([depth + 1, mol.natoms])
    eq_con_j = np.zeros([depth + 1, mol.natoms])

    # full ligand ACs
    for i in range(0, n_ax):  # for each ax ligand
        ax_ligand_ac_full_derivative = full_autocorrelation_derivative(ax_ligand_list[i].mol, prop, depth)
        # now we need to map back to full positions
        for ii, row in enumerate(ax_ligand_ac_full_derivative):
            for original_ids in list(ax_ligand_list[i].ext_int_dict.keys()):
                ax_full_j[ii, original_ids] += np.divide(row[ax_ligand_list[i].ext_int_dict[original_ids]], n_ax)

    for i in range(0, n_eq):  # for each eq ligand
        # now we need to map back to full positions
        eq_ligand_eq_full_derivative = full_autocorrelation_derivative(eq_ligand_list[i].mol, prop, depth)
        for ii, row in enumerate(eq_ligand_eq_full_derivative):
            for original_ids in list(eq_ligand_list[i].ext_int_dict.keys()):
                eq_full_j[ii, original_ids] += np.divide(row[eq_ligand_list[i].ext_int_dict[original_ids]], n_eq)

    # ligand connection ACs
    for i in range(0, n_ax):
        ax_ligand_ac_con_derivative = atom_only_autocorrelation_derivative(ax_ligand_list[i].mol, prop, depth,
                                                                           ax_con_int_list[i])
        # now we need to map back to full positions
        for ii, row in enumerate(ax_ligand_ac_con_derivative):
            for original_ids in list(ax_ligand_list[i].ext_int_dict.keys()):
                ax_con_j[ii, original_ids] += np.divide(row[ax_ligand_list[i].ext_int_dict[original_ids]], n_ax)

    for i in range(0, n_eq):
        eq_ligand_ac_con_derivative = atom_only_autocorrelation_derivative(eq_ligand_list[i].mol, prop, depth,
                                                                           eq_con_int_list[i])
        # now we need to map back to full positions
        for ii, row in enumerate(eq_ligand_ac_con_derivative):
            for original_ids in list(eq_ligand_list[i].ext_int_dict.keys()):
                eq_con_j[ii, original_ids] += np.divide(row[eq_ligand_list[i].ext_int_dict[original_ids]], n_eq)

    return ax_full_j, eq_full_j, ax_con_j, eq_con_j


def find_ligand_autocorrs_and_deltametrics_oct_dimers(mol, prop, loud, depth, name=False,
                                                      oct=True, custom_ligand_dict=False):
    # # this function takes a
    # # symmetric (axial == axial,
    # # equatorial == equatorial)
    # # octahedral complex
    # # and returns autocorrelations for
    # # the axial an equatorial ligands
    # # custom_ligand_dict allows the user to skip the breakdown
    # # in cases where 3D geo is not correct/formed
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    if not custom_ligand_dict:
        raise ValueError('No custom ligand dict provided!')
        # liglist, ligdents, ligcons = ligand_breakdown(mol)
        # ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list, eq_con_int_list,
        # ax_con_list, eq_con_list, built_ligand_list = ligand_assign(
        #    mol, liglist, ligdents, ligcons, loud, name=False)
    else:
        ax1_ligand_list = custom_ligand_dict["ax1_ligand_list"]
        ax2_ligand_list = custom_ligand_dict["ax2_ligand_list"]
        ax3_ligand_list = custom_ligand_dict["ax3_ligand_list"]
        ax1_con_int_list = custom_ligand_dict["ax1_con_int_list"]
        ax2_con_int_list = custom_ligand_dict["ax2_con_int_list"]
        ax3_con_int_list = custom_ligand_dict["ax3_con_int_list"]
        axligs = [ax1_ligand_list, ax2_ligand_list, ax3_ligand_list]
        axcons = [ax1_con_int_list, ax2_con_int_list, ax3_con_int_list]
        n_axs = [len(i) for i in axligs]

    # get full ligand AC
    ax_ligand_ac_fulls = [False, False, False]

    for axnum in range(3):
        ax_ligand_ac_full = list()
        for i in range(0, n_axs[axnum]):
            if not list(ax_ligand_ac_full):
                ax_ligand_ac_full = full_autocorrelation(axligs[axnum][i].mol, prop, depth)
            else:
                ax_ligand_ac_full += full_autocorrelation(axligs[axnum][i].mol, prop, depth)
        ax_ligand_ac_full = np.divide(ax_ligand_ac_full, n_axs[axnum])
        ax_ligand_ac_fulls[axnum] = ax_ligand_ac_full

    # get partial ligand AC
    ax_ligand_ac_cons = [False, False, False]

    for axnum in range(3):
        ax_ligand_ac_con = list()
        for i in range(0, n_axs[axnum]):
            if not list(ax_ligand_ac_con):
                ax_ligand_ac_con = atom_only_autocorrelation(axligs[axnum][i].mol, prop, depth, axcons[axnum][i])
            else:
                ax_ligand_ac_con += atom_only_autocorrelation(axligs[axnum][i].mol, prop, depth, axcons[axnum][i])
        ax_ligand_ac_con = np.divide(ax_ligand_ac_con, n_axs[axnum])
        ax_ligand_ac_cons[axnum] = ax_ligand_ac_con

    # get deltametrics
    ax_delta_cons = [False, False, False]

    for axnum in range(3):
        ax_delta_con = list()
        for i in range(0, n_axs[axnum]):
            if not list(ax_delta_con):
                ax_delta_con = atom_only_deltametric(axligs[axnum][i].mol, prop, depth, axcons[axnum][i])
            else:
                ax_delta_con += atom_only_deltametric(axligs[axnum][i].mol, prop, depth, axcons[axnum][i])
        ax_delta_con = np.divide(ax_delta_con, n_axs[axnum])
        ax_delta_cons[axnum] = ax_delta_con

    return ax_ligand_ac_fulls + ax_ligand_ac_cons + ax_delta_cons


def find_ligand_deltametrics_oct(mol, prop, loud, depth, name=False, oct=True, custom_ligand_dict=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    # # this function takes a
    # # octahedral complex
    # # and returns deltametrics for
    # # the axial an equatorial ligands
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=oct)
        (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list,
         eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list) = ligand_assign(
            mol, liglist, ligdents, ligcons, loud, name=False)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    # count ligands
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)

    # get partial ligand AC
    ax_ligand_ac_con = []
    eq_ligand_ac_con = []

    for i in range(0, n_ax):
        if not list(ax_ligand_ac_con):
            ax_ligand_ac_con = atom_only_deltametric(ax_ligand_list[i].mol, prop, depth, ax_con_int_list[i])
        else:
            ax_ligand_ac_con += atom_only_deltametric(ax_ligand_list[i].mol, prop, depth, ax_con_int_list[i])
    ax_ligand_ac_con = np.divide(ax_ligand_ac_con, n_ax)
    for i in range(0, n_eq):
        if not list(eq_ligand_ac_con):
            eq_ligand_ac_con = atom_only_deltametric(eq_ligand_list[i].mol, prop, depth, eq_con_int_list[i])
        else:
            eq_ligand_ac_con += atom_only_deltametric(eq_ligand_list[i].mol, prop, depth, eq_con_int_list[i])
    eq_ligand_ac_con = np.divide(eq_ligand_ac_con, n_eq)

    return ax_ligand_ac_con, eq_ligand_ac_con


def find_ligand_deltametric_derivatives_oct(mol, prop, loud, depth, name=False, oct=True, custom_ligand_dict=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    # # this function takes a
    # # octahedral complex
    # # and returns deltametrics for
    # # the axial an equatorial ligands
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=oct)
        (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list,
         eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list) = ligand_assign(
            mol, liglist, ligdents, ligcons, loud, name=False)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]

    # count ligands
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)

    # allocate the full jacobian matrix
    ax_con_j = np.zeros([depth + 1, mol.natoms])
    eq_con_j = np.zeros([depth + 1, mol.natoms])

    for i in range(0, n_ax):
        ax_ligand_ac_con_derivative = atom_only_deltametric_derivative(ax_ligand_list[i].mol, prop, depth,
                                                                       ax_con_int_list[i])
        # now we need to map back to full positions
        for ii, row in enumerate(ax_ligand_ac_con_derivative):
            for original_ids in list(ax_ligand_list[i].ext_int_dict.keys()):
                ax_con_j[ii, original_ids] += np.divide(row[ax_ligand_list[i].ext_int_dict[original_ids]], n_ax)

    for i in range(0, n_eq):
        eq_ligand_ac_con_derivative = atom_only_deltametric_derivative(eq_ligand_list[i].mol, prop, depth,
                                                                       eq_con_int_list[i])
        for ii, row in enumerate(eq_ligand_ac_con_derivative):
            for original_ids in list(eq_ligand_list[i].ext_int_dict.keys()):
                eq_con_j[ii, original_ids] += np.divide(row[eq_ligand_list[i].ext_int_dict[original_ids]], n_eq)

    return ax_con_j, eq_con_j


def find_mc_eq_ax_deltametrics_oct(mol, prop, loud, depth, name=False, oct=True,
                                   func=deltametric_catoms):
    # For octahedral complexes only.
    # Calculate mc/ax, mc/eq deltametrics.
    liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=oct)
    (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list,
     eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list) = ligand_assign(
        mol, liglist, ligdents, ligcons, loud, name=False)
    # shape reduce
    ax_con_list = [x[0] for x in ax_con_list]
    eq_con_list = [x[0] for x in eq_con_list]
    ax_ligand_del_mc = metal_only_deltametric(mol, prop, depth, catoms=ax_con_list, func=func)
    eq_ligand_del_mc = metal_only_deltametric(mol, prop, depth, catoms=eq_con_list, func=func)
    ax_ligand_del_mc = np.divide(ax_ligand_del_mc, len(ax_con_list))
    eq_ligand_del_mc = np.divide(eq_ligand_del_mc, len(eq_con_list))
    return ax_ligand_del_mc, eq_ligand_del_mc


def find_mc_eq_ax_autocorrelation_oct(mol, prop, loud, depth, name=False, oct=True,
                                      func=autocorrelation_catoms, modifier=False):
    # For octahedral complexes only.
    # Calculate mc/ax, mc/eq deltametrics.
    liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=oct)
    (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list,
     eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list) = ligand_assign(
        mol, liglist, ligdents, ligcons, loud, name=False)
    # shape reduce
    ax_con_list = [x[0] for x in ax_con_list]
    eq_con_list = [x[0] for x in eq_con_list]
    ax_ligand_ac_mc = metal_only_autocorrelation(mol, prop, depth, catoms=ax_con_list, func=func, modifier=modifier)
    eq_ligand_ac_mc = metal_only_autocorrelation(mol, prop, depth, catoms=eq_con_list, func=func, modifier=modifier)
    ax_ligand_ac_mc = np.divide(ax_ligand_ac_mc, len(ax_con_list))
    eq_ligand_ac_mc = np.divide(eq_ligand_ac_mc, len(eq_con_list))
    return ax_ligand_ac_mc, eq_ligand_ac_mc


def generate_mc_eq_ax_deltametrics(mol, loud, depth=4, name=False,
                                   func=deltametric_catoms, NumB=False, Gval=False):
    result_ax_mc = list()
    result_eq_mc = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        ax_ligand_ac_con, eq_ligand_ac_con = find_mc_eq_ax_deltametrics_oct(mol, properties, loud, depth, name,
                                                                            func=func)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result_ax_mc.append(ax_ligand_ac_con)
        result_eq_mc.append(eq_ligand_ac_con)
    results_dictionary = {'colnames': colnames, 'result_mc_ax_del': result_ax_mc,
                          'result_mc_eq_del': result_eq_mc}
    return results_dictionary


def generate_mc_eq_ax_autocorrelation(mol, loud, depth=4, name=False,
                                      func=autocorrelation_catoms, NumB=False, Gval=False):
    result_ax_mc = list()
    result_eq_mc = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        ax_ligand_ac_con, eq_ligand_ac_con = find_mc_eq_ax_autocorrelation_oct(mol, properties, loud, depth, name,
                                                                               func=func)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result_ax_mc.append(ax_ligand_ac_con)
        result_eq_mc.append(eq_ligand_ac_con)
    results_dictionary = {'colnames': colnames, 'result_mc_ax_ac': result_ax_mc,
                          'result_mc_eq_ac': result_eq_mc}
    return results_dictionary


def generate_all_ligand_autocorrelations(mol, loud, depth=4, name=False, flag_name=False,
                                         custom_ligand_dict=False, NumB=False, Gval=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    result_ax_full = list()
    result_eq_full = list()
    result_ax_con = list()
    result_eq_con = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        (ax_ligand_ac_full,
         eq_ligand_ac_full,
         ax_ligand_ac_con,
         eq_ligand_ac_con) = find_ligand_autocorrelations_oct(
             mol,
             properties,
             loud=loud,
             depth=depth,
             name=name,
             oct=True,
             custom_ligand_dict=custom_ligand_dict)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result_ax_full.append(ax_ligand_ac_full)
        result_eq_full.append(eq_ligand_ac_full)
        result_ax_con.append(ax_ligand_ac_con)
        result_eq_con.append(eq_ligand_ac_con)
    if flag_name:
        results_dictionary = {'colnames': colnames,
                              'result_ax_full_ac': result_ax_full,
                              'result_eq_full_ac': result_eq_full,
                              'result_ax_con_ac': result_ax_con,
                              'result_eq_con_ac': result_eq_con}
    else:
        results_dictionary = {'colnames': colnames,
                              'result_ax_full': result_ax_full,
                              'result_eq_full': result_eq_full,
                              'result_ax_con': result_ax_con,
                              'result_eq_con': result_eq_con}
    return results_dictionary


def generate_all_ligand_autocorrelation_derivatives(mol, loud, depth=4, name=False, flag_name=False,
                                                    custom_ligand_dict=False, NumB=False, Gval=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    result_ax_full = None
    result_eq_full = None
    result_ax_con = None
    result_eq_con = None

    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        ax_ligand_ac_full, eq_ligand_ac_full, ax_ligand_ac_con, eq_ligand_ac_con = find_ligand_autocorrelation_derivatives_oct(
            mol,
            properties,
            loud=loud,
            depth=depth,
            name=name,
            oct=True,
            custom_ligand_dict=custom_ligand_dict)
        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])
        if result_ax_full is None:
            result_ax_full = ax_ligand_ac_full
        else:
            result_ax_full = np.row_stack([result_ax_full, ax_ligand_ac_full])

        if result_eq_full is None:
            result_eq_full = eq_ligand_ac_full
        else:
            result_eq_full = np.row_stack([result_eq_full, eq_ligand_ac_full])

        if result_ax_con is None:
            result_ax_con = ax_ligand_ac_con
        else:
            result_ax_con = np.row_stack([result_ax_con, ax_ligand_ac_con])

        if result_eq_con is None:
            result_eq_con = eq_ligand_ac_con
        else:
            result_eq_con = np.row_stack([result_eq_con, eq_ligand_ac_con])

    if flag_name:
        results_dictionary = {'colnames': colnames, 'result_ax_full_ac': result_ax_full,
                              'result_eq_full_ac': result_eq_full,
                              'result_ax_con_ac': result_ax_con, 'result_eq_con_ac': result_eq_con}
    else:
        results_dictionary = {'colnames': colnames, 'result_ax_full': result_ax_full, 'result_eq_full': result_eq_full,
                              'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}
    return results_dictionary


def generate_all_ligand_autocorrs_and_deltametrics_dimers(mol, loud, depth=4, name=False, flag_name=False,
                                                          custom_ligand_dict=False, NumB=False, Gval=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    result_ax1_full = list()
    result_ax2_full = list()
    result_ax3_full = list()
    result_ax1_con = list()
    result_ax2_con = list()
    result_ax3_con = list()
    result_delta_ax1_con = list()
    result_delta_ax2_con = list()
    result_delta_ax3_con = list()

    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        # lig_autocorrs is a list of length 6 (ax{i}_ligand_ac_fulls, ax{i}_ligand_ac_cons)
        lig_autocorrs = find_ligand_autocorrs_and_deltametrics_oct_dimers(mol,
                                                                          properties,
                                                                          loud=loud,
                                                                          depth=depth,
                                                                          name=name,
                                                                          oct=True,
                                                                          custom_ligand_dict=custom_ligand_dict)
        this_colnames = []
        assert all([len(i) > 0 for i in lig_autocorrs]), 'Some ligand autocorrelations are empty! %s' % lig_autocorrs
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result_ax1_full.append(lig_autocorrs[0])
        result_ax2_full.append(lig_autocorrs[1])
        result_ax3_full.append(lig_autocorrs[2])
        result_ax1_con.append(lig_autocorrs[3])
        result_ax2_con.append(lig_autocorrs[4])
        result_ax3_con.append(lig_autocorrs[5])
        result_delta_ax1_con.append(lig_autocorrs[6])
        result_delta_ax2_con.append(lig_autocorrs[7])
        result_delta_ax3_con.append(lig_autocorrs[8])

    results_dictionary = {'colnames': colnames,
                          'result_ax1_full': result_ax1_full,
                          'result_ax2_full': result_ax2_full,
                          'result_ax3_full': result_ax3_full,
                          'result_ax1_con': result_ax1_con,
                          'result_ax2_con': result_ax2_con,
                          'result_ax3_con': result_ax3_con,
                          'result_delta_ax1_con': result_delta_ax1_con,
                          'result_delta_ax2_con': result_delta_ax2_con,
                          'result_delta_ax3_con': result_delta_ax3_con}
    # if flag_name:
    #    results_dictionary = {'colnames': colnames, 'result_ax_full_ac': result_ax_full,
    #                          'result_eq_full_ac': result_eq_full,
    #                          'result_ax_con_ac': result_ax_con, 'result_eq_con_ac': result_eq_con}
    # else:
    #    results_dictionary = {'colnames': colnames, 'result_ax_full': result_ax_full, 'result_eq_full': result_eq_full,
    #                          'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}
    return results_dictionary


def generate_all_ligand_deltametrics(mol, loud, depth=4, name=False, flag_name=False,
                                     custom_ligand_dict=False, NumB=False, Gval=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]

    result_ax_con = list()
    result_eq_con = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        ax_ligand_ac_con, eq_ligand_ac_con = find_ligand_deltametrics_oct(mol, properties, loud, depth, name, oct=True,
                                                                          custom_ligand_dict=custom_ligand_dict)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result_ax_con.append(ax_ligand_ac_con)
        result_eq_con.append(eq_ligand_ac_con)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'result_ax_con_del': result_ax_con,
                              'result_eq_con_del': result_eq_con}
    else:
        results_dictionary = {'colnames': colnames, 'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}
    return results_dictionary


def generate_all_ligand_deltametric_derivatives(mol, loud, depth=4, name=False, flag_name=False,
                                                custom_ligand_dict=False, NumB=False, Gval=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]

    result_ax_con = None
    result_eq_con = None
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        ax_ligand_ac_con, eq_ligand_ac_con = find_ligand_deltametric_derivatives_oct(mol, properties, loud, depth, name,
                                                                                     oct=True,
                                                                                     custom_ligand_dict=custom_ligand_dict)

        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])
        if result_ax_con is None:
            result_ax_con = ax_ligand_ac_con
        else:
            result_ax_con = np.row_stack([result_ax_con, ax_ligand_ac_con])
        if result_eq_con is None:
            result_eq_con = eq_ligand_ac_con
        else:
            result_eq_con = np.row_stack([result_eq_con, eq_ligand_ac_con])
    if flag_name:
        results_dictionary = {'colnames': colnames, 'result_ax_con_del': result_ax_con,
                              'result_eq_con_del': result_eq_con}
    else:
        results_dictionary = {'colnames': colnames, 'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}
    return results_dictionary


def generate_metal_autocorrelations(mol, loud, depth=4, oct=True, flag_name=False,
                                    modifier=False, NumB=False, Gval=False):
    # oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        metal_ac = metal_only_autocorrelation(mol, properties, depth, oct=oct, modifier=modifier)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_ac': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_autocorrelation_derivatives(mol, loud, depth=4, oct=True, flag_name=False,
                                               modifier=False, NumB=False, Gval=False):
    # oct - bool, if complex is octahedral, will use better bond checks
    result = None
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        metal_ac_der = metal_only_autocorrelation_derivative(mol, properties, depth, oct=oct, modifier=modifier)
        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])

        if result is None:
            result = metal_ac_der
        else:
            result = np.row_stack([result, metal_ac_der])
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_ac': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_multimetal_autocorrelations(mol, loud, depth=4, oct=True, flag_name=False, polarizability=False, Gval=False):
    # oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if polarizability:
        allowed_strings += ['polarizability']
        labels_strings += ['alpha']
    for ii, properties in enumerate(allowed_strings):
        metal_ac = multimetal_only_autocorrelation(mol, properties, depth, oct=oct)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_ac': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_multiatom_autocorrelations(mol, loud, depth=4, oct=True, flag_name=False, additional_elements=False):
    # oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    for ii, properties in enumerate(allowed_strings):
        metal_ac = multiatom_only_autocorrelation(mol, properties, depth, oct=oct, additional_elements=additional_elements)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_ac': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_autocorrelations(oxmodifier, mol, loud, depth=4, oct=True, flag_name=False):
    # # oxmodifier - dict, used to modify prop vector (e.g. for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    metal_ox_ac = metal_only_autocorrelation(mol, 'ox_nuclear_charge', depth, oct=oct, modifier=oxmodifier)
    this_colnames = []
    for i in range(0, depth + 1):
        this_colnames.append('O' + '-' + str(i))
    colnames.append(this_colnames)
    result.append(metal_ox_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_autocorrelation_derivatives(oxmodifier, mol, loud, depth=4, oct=True, flag_name=False):
    # # oxmodifier - dict, used to modify prop vector (e.g. for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = None
    colnames = []
    metal_ox_ac = metal_only_autocorrelation_derivative(mol, 'ox_nuclear_charge', depth, oct=oct, modifier=oxmodifier)
    for i in range(0, depth + 1):
        colnames.append(['d' + 'O' + '-' + str(i) + '/d' + 'O' + str(j) for j in range(0, mol.natoms)])
    result = metal_ox_ac
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_deltametrics(oxmodifier, mol, loud, depth=4, oct=True, flag_name=False):
    # # oxmodifier - dict, used to modify prop vector (e.g. for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    metal_ox_ac = metal_only_deltametric(mol, 'ox_nuclear_charge', depth, oct=oct, modifier=oxmodifier)
    this_colnames = []
    for i in range(0, depth + 1):
        this_colnames.append('O' + '-' + str(i))
    colnames.append(this_colnames)
    result.append(metal_ox_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_deltametric_derivatives(oxmodifier, mol, loud, depth=4, oct=True, flag_name=False):
    # # oxmodifier - dict, used to modify prop vector (e.g. for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    metal_ox_ac = metal_only_deltametric_derivative(mol, 'ox_nuclear_charge', depth, oct=oct, modifier=oxmodifier)
    for i in range(0, depth + 1):
        colnames.append(['d' + 'O' + '-' + str(i) + '/d' + 'O' + str(j) for j in range(0, mol.natoms)])

    result = metal_ox_ac
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_eff_autocorrelations(oxmodifier, mol, loud, depth=4, oct=True, flag_name=False):
    # # oxmodifier - dict, used to modify prop vector (e.g. for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    metal_ox_ac = metal_only_autocorrelation(mol, 'group_number', depth, oct=oct, modifier=oxmodifier)
    this_colnames = []
    for i in range(0, depth + 1):
        this_colnames.append('Gval' + '-' + str(i))
    colnames.append(this_colnames)
    result.append(metal_ox_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_eff_deltametrics(oxmodifier, mol, loud, depth=4, oct=True, flag_name=False):
    # # oxmodifier - dict, used to modify prop vector (e.g. for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    metal_ox_ac = metal_only_deltametric(mol, 'group_number', depth, oct=oct, modifier=oxmodifier)
    this_colnames = []
    for i in range(0, depth + 1):
        this_colnames.append('Gval' + '-' + str(i))
    colnames.append(this_colnames)
    result.append(metal_ox_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_deltametrics(mol, loud, depth=4, oct=True, flag_name=False,
                                modifier=False, NumB=False, Gval=False):
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        metal_ac = metal_only_deltametric(mol, properties, depth, oct=oct, modifier=modifier)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_del': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_deltametric_derivatives(mol, loud, depth=4, oct=True, flag_name=False,
                                           modifier=False, NumB=False, Gval=False):
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = None
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        metal_ac_der = metal_only_deltametric_derivative(mol, properties, depth, oct=oct, modifier=modifier)
        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])
        if result is None:
            result = metal_ac_der
        else:
            result = np.row_stack([result, metal_ac_der])
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_del': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_multimetal_deltametrics(mol, loud, depth=4, oct=True, flag_name=False, polarizability=False, Gval=False):
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if polarizability:
        allowed_strings += ['polarizability']
        labels_strings += ['alpha']
    for ii, properties in enumerate(allowed_strings):
        metal_ac = multimetal_only_deltametric(mol, properties, depth, oct=oct)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_del': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_multiatom_deltametrics(mol, loud, depth=4, oct=True, flag_name=False, additional_elements=False):
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    for ii, properties in enumerate(allowed_strings):
        metal_ac = multiatom_only_deltametric(mol, properties, depth, oct=oct, additional_elements=additional_elements)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_del': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_full_complex_autocorrelations(mol, loud,
                                           depth=4, oct=True,
                                           flag_name=False, modifier=False,
                                           use_dist=False, NumB=False, Gval=False, polarizability=False):
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    if polarizability:
        allowed_strings += ["polarizability"]
        labels_strings += ["alpha"]
    for ii, properties in enumerate(allowed_strings):
        metal_ac = full_autocorrelation(mol, properties, depth,
                                        oct=oct, modifier=modifier,
                                        use_dist=use_dist)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_f_all': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_full_complex_coulomb_autocorrelations(mol, loud,
                                                   depth=3, oct=True,
                                                   flag_name=False, modifier=False,
                                                   use_dist=False, transition_metals_only=True):
    result = list()
    colnames = []
    # allowed_strings = ['ident', 'topology', 'bondvalence', 'valenceelectron', 'bondvalence_devi', 'bodavrg', 'bodstd',
    #                    'charge']
    # labels_strings = ['I', 'T', 'BV', 'VE', 'BVD', "BODavrg", "BODstd", "Ch"]
    allowed_strings = ['ident', 'topology', 'group_number', "num_bonds"]
    labels_strings = ['I', 'T', 'Gval', "NumB"]
    for ii, properties in enumerate(allowed_strings):
        metal_ac = full_autocorrelation(mol, properties, depth,
                                        oct=oct, modifier=modifier,
                                        use_dist=use_dist,
                                        transition_metals_only=transition_metals_only)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_f_all': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_full_complex_autocorrelation_derivatives(mol, loud, depth=4, oct=True, flag_name=False,
                                                      modifier=False, NumB=False, Gval=False):
    result = None
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        f_ac_der = full_autocorrelation_derivative(mol, properties, depth, oct=oct, modifier=modifier)
        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])
        if result is None:
            result = f_ac_der
        else:
            result = np.row_stack([result, f_ac_der])
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_f_all': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_atomonly_autocorrelations(mol, atomIdx, loud, depth=4, oct=True, NumB=False, Gval=False, polarizability=False):
    # # this function gets autocorrelations for a molecule starting
    # # in one single atom only
    # Inputs:
    #       mol - mol3D class
    #       atomIdx - int, index of atom3D class; or list of indices
    #       loud - bool, print output
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    if polarizability:
        allowed_strings += ['polarizability']
        labels_strings += ['alpha']
    # print('The selected connection type is ' + str(mol.getAtom(atomIdx).symbol()))
    for ii, properties in enumerate(allowed_strings):
        atom_only_ac = atom_only_autocorrelation(mol, properties, depth, atomIdx, oct=oct)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(atom_only_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_atomonly_autocorrelation_derivatives(mol, atomIdx, loud, depth=4, oct=True, NumB=False, Gval=False):
    # # this function gets the d/dx for autocorrelations for a molecule starting
    # # in one single atom only
    # Inputs:
    #       mol - mol3D class
    #       atomIdx - int, index of atom3D class
    #       loud - bool, print output
    result = None
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    # print('The selected connection type is ' + str(mol.getAtom(atomIdx).symbol()))
    for ii, properties in enumerate(allowed_strings):
        atom_only_ac = atom_only_autocorrelation_derivative(mol, properties, depth, atomIdx, oct=oct)
        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])
        if result is None:
            result = atom_only_ac
        else:
            result = np.row_stack([result, atom_only_ac])
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_atomonly_deltametrics(mol, atomIdx, loud, depth=4, oct=True, NumB=False, Gval=False, polarizability=False):
    # # this function gets deltametrics for a molecule starting
    # # in one single atom only
    # Inputs:
    #       mol - mol3D class
    #       atomIdx - int, index of atom3D class
    #       loud - bool, print output
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    if polarizability:
        allowed_strings += ["polarizability"]
        labels_strings += ["alpha"]
    # print('The selected connection type is ' + str(mol.getAtom(atomIdx).symbol()))
    for ii, properties in enumerate(allowed_strings):
        atom_only_ac = atom_only_deltametric(mol, properties, depth, atomIdx, oct=oct)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(atom_only_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_atomonly_deltametric_derivatives(mol, atomIdx, loud, depth=4, oct=True, NumB=False, Gval=False):
    # # this function gets deltametrics for a molecule starting
    # # in one single atom only
    # Inputs:
    #       mol - mol3D class
    #       atomIdx - int, index of atom3D class
    #       loud - bool, print output
    result = None
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    # print('The selected connection type is ' + str(mol.getAtom(atomIdx).symbol()))
    for ii, properties in enumerate(allowed_strings):
        atom_only_ac_der = atom_only_deltametric_derivative(mol, properties, depth, atomIdx, oct=oct)
        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])
        if result is None:
            result = atom_only_ac_der
        else:
            result = np.row_stack([result, atom_only_ac_der])
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary
