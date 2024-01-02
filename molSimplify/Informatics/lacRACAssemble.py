# Written JP Janet
# for HJK Group
# Dpt of Chemical Engineering, MIT

# #########################################################
# ####### Defines methods for assembling    ###############
# #######     RACs from lists of ligands    ###############
# #########################################################
from __future__ import print_function
import numpy as np
from molSimplify.Classes.ligand import ligand_breakdown
from molSimplify.Classes.ligand import ligand_assign_consistent as ligand_assign
from molSimplify.Classes.globalvars import globalvars

globs = globalvars()


def get_descriptor_vector(this_complex, custom_ligand_dict=False,
                          ox_modifier=False, NumB=False, Gval=False,
                          lacRACs=True, loud=False, metal_ind=None,
                          smiles_charge=False, eq_sym=False,
                          use_dist=False, size_normalize=False,
                          alleq=False, MRdiag_dict={}, depth=3):
    """ Calculate and return all geo-based RACs for a given octahedral complex (featurize).

    Parameters
    ----------
        this_complex : mol3D
            Transition metal complex to be featurized.
        custom_ligand_dict : bool, optional
            Custom ligand dictionary to evaluate for complex if passed, by default False
            Skip the ligand breakdown steps -
            in cases where 3D geo is not correct/formed
            custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
            ax_con_int_list ,eq_con_int_list
            with types: eq/ax_ligand_list list of mol3D
            eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
        ox_modifier : bool, optional
            dict, used to modify prop vector (e.g. for adding
            ONLY used with  ox_nuclear_charge ox or charge)
            {"Fe":2, "Co": 3} etc, by default False
        NumB : bool, optional
            Use Number of Bonds as additional RAC, by default False
        Gval : bool, optional
            Use group number as RAC, by default False
        lacRACs : bool, optional
            Use ligand_assign_consistent (lac) to represent mol3D given
            if False, use ligand_assign (older), default True
        loud : bool, optional
            Print debugging information, by default False
        metal_ind : bool, optional
            index of the metal atom to generate property, by default False
        smiles_charge : bool, optional
            use obmol conversion through smiles to assign ligand_misc_charges, by default False
        use_dist : bool, optional
            Whether or not CD-RACs used.
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        alleq : bool, optional
            Whether or not all ligands are equatorial.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.
        depth : int, optional
            The depth of the RACs (how many bonds out the RACs go).

    Returns
    -------
        descriptor_names : list
            Compiled list of descriptor names
        descriptors : list
            Compiled list of descriptor values

    """
    # modifier -
    descriptor_names = []
    descriptors = []
    # Generate custom_ligand_dict if one not passed!
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(this_complex, BondedOct=True) # Complex is assumed to be octahedral
        # print(liglist, ligdents, ligcons)
        if not alleq:
            if lacRACs:
                from molSimplify.Classes.ligand import ligand_assign_consistent as ligand_assign
            else:
                from molSimplify.Classes.ligand import ligand_assign as ligand_assign
            ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, \
                ax_con_int_list, eq_con_int_list, ax_con_list, eq_con_list, \
                built_ligand_list = ligand_assign(this_complex, liglist, ligdents, ligcons, loud, eq_sym_match=eq_sym)
        else:
            from molSimplify.Classes.ligand import ligand_assign_alleq
            ax_ligand_list, eq_ligand_list, ax_con_int_list, eq_con_int_list = ligand_assign_alleq(
                this_complex, liglist, ligdents, ligcons)
        custom_ligand_dict = {'ax_ligand_list': ax_ligand_list,
                              'eq_ligand_list': eq_ligand_list,
                              'ax_con_int_list': ax_con_int_list,
                              'eq_con_int_list': eq_con_int_list}
    # misc descriptors
    results_dictionary = generate_all_ligand_misc(this_complex, loud=False,
                                                  custom_ligand_dict=custom_ligand_dict,
                                                  smiles_charge=smiles_charge)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['result_ax'],
                                                       'misc', 'ax')
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['result_eq'],
                                                       'misc', 'eq')

    # full ACs
    results_dictionary = generate_full_complex_autocorrelations(this_complex, depth=depth,
                                                                loud=False, flag_name=False,
                                                                modifier=ox_modifier, NumB=NumB,
                                                                Gval=Gval, use_dist=use_dist,
                                                                size_normalize=size_normalize,
                                                                MRdiag_dict=MRdiag_dict)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['results'],
                                                       'f', 'all')
    # # ligand ACs
    results_dictionary = generate_all_ligand_autocorrelations(this_complex, depth=depth,
                                                              loud=False, flag_name=False,
                                                              custom_ligand_dict=custom_ligand_dict,
                                                              NumB=NumB, Gval=Gval, use_dist=use_dist,
                                                              size_normalize=size_normalize,
                                                              MRdiag_dict=MRdiag_dict)
    if not alleq:
        descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                           results_dictionary['colnames'],
                                                           results_dictionary['result_ax_full'],
                                                           'f', 'ax')
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['result_eq_full'],
                                                       'f', 'eq')
    if not alleq:
        descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                           results_dictionary['colnames'],
                                                           results_dictionary['result_ax_con'],
                                                           'lc', 'ax')
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['result_eq_con'],
                                                       'lc', 'eq')

    results_dictionary = generate_all_ligand_deltametrics(this_complex, depth=depth, loud=False,
                                                          custom_ligand_dict=custom_ligand_dict,
                                                          NumB=NumB, Gval=Gval, use_dist=use_dist,
                                                          size_normalize=size_normalize,
                                                          MRdiag_dict=MRdiag_dict)
    if not alleq:
        descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                           results_dictionary['colnames'],
                                                           results_dictionary['result_ax_con'],
                                                           'D_lc', 'ax')
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['result_eq_con'],
                                                       'D_lc', 'eq')

    # metal ACs
    # print('getting metal ACs')
    results_dictionary = generate_metal_autocorrelations(this_complex, depth=depth,
                                                         loud=False, modifier=ox_modifier,
                                                         NumB=NumB, Gval=Gval, metal_ind=metal_ind,
                                                         use_dist=use_dist,
                                                         size_normalize=size_normalize,
                                                         MRdiag_dict=MRdiag_dict)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['results'],
                                                       'mc', 'all')

    results_dictionary = generate_metal_deltametrics(this_complex, depth=depth,
                                                     loud=False, modifier=ox_modifier,
                                                     NumB=NumB, Gval=Gval, metal_ind=metal_ind,
                                                     use_dist=use_dist,
                                                     size_normalize=size_normalize,
                                                     MRdiag_dict=MRdiag_dict)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['results'],
                                                       'D_mc', 'all')

    # ## ox-metal ACs, if ox available
    if ox_modifier:
        results_dictionary = generate_metal_ox_autocorrelations(ox_modifier, this_complex,
                                                                depth=depth, loud=False,
                                                                metal_ind=metal_ind, use_dist=use_dist,
                                                                size_normalize=size_normalize)
        descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                           results_dictionary['colnames'],
                                                           results_dictionary['results'],
                                                           'mc', 'all')
        results_dictionary = generate_metal_ox_deltametrics(ox_modifier, this_complex,
                                                            depth=depth, loud=False,
                                                            metal_ind=metal_ind,
                                                            use_dist=use_dist,
                                                            size_normalize=size_normalize)
        descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                           results_dictionary['colnames'],
                                                           results_dictionary['results'],
                                                           'D_mc', 'all')
    return descriptor_names, descriptors


def get_descriptor_derivatives(this_complex, custom_ligand_dict=False, ox_modifier=False,
                               lacRACs=True, depth=4, loud=False, metal_ind=None):
    """ Calculate and return all derivatives of RACs for a given octahedral complex.

    Parameters
    ----------
        this_complex : mol3D
            Transition metal complex to be featurized.
        custom_ligand_dict : bool, optional
            Custom ligand dictionary to evaluate for complex if passed, by default False
            Skip the ligand breakdown steps -
            in cases where 3D geo is not correct/formed
            custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
            ax_con_int_list ,eq_con_int_list
            with types: eq/ax_ligand_list list of mol3D
            eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
        ox_modifier : bool, optional
            dict, used to modify prop vector (e.g. for adding
            ONLY used with  ox_nuclear_charge ox or charge)
            {"Fe":2, "Co": 3} etc, by default False
        lacRACs : bool, optional
            Use ligand_assign_consistent (lac) to represent mol3D given
            if False, use ligand_assign (older), default True
        depth : int, optional
            depth of RACs to calculate, by default 4
        loud : bool, optional
            Print debugging information, by default False
        metal_ind : bool, optional
            index of the metal atom to generate RACs from, by default False

    Returns
    -------
        descriptor_derivative_names : list
            Compiled list (matrix) of descriptor derivative names
        descriptor_derivatives : list
            Derivatives of RACs w.r.t atomic props (matrix)

    """
    if not custom_ligand_dict:
        if lacRACs:
            from molSimplify.Classes.ligand import ligand_assign_consistent as ligand_assign
        else:
            from molSimplify.Classes.ligand import ligand_assign as ligand_assign
        liglist, ligdents, ligcons = ligand_breakdown(this_complex, BondedOct=True) # Complex is assumed to be octahedral
        (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list,
         ax_con_int_list, eq_con_int_list, ax_con_list, eq_con_list,
         built_ligand_list) = ligand_assign(this_complex, liglist, ligdents, ligcons, loud)
        custom_ligand_dict = {'ax_ligand_list': ax_ligand_list,
                              'eq_ligand_list': eq_ligand_list,
                              'ax_con_int_list': ax_con_int_list,
                              'eq_con_int_list': eq_con_int_list}
    #  cannot do misc descriptors !
    descriptor_derivative_names = []
    descriptor_derivatives = None
    # full ACs
    results_dictionary = generate_full_complex_autocorrelation_derivatives(this_complex, depth=depth,
                                                                           loud=False, flag_name=False,
                                                                           modifier=ox_modifier)
    descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                        descriptor_derivatives,
                                                                                        results_dictionary['colnames'],
                                                                                        results_dictionary['results'],
                                                                                        'f', 'all')
    # ligand ACs
    # print('getting ligand AC derivatives')
    results_dictionary = generate_all_ligand_autocorrelation_derivatives(this_complex, depth=depth, loud=False,
                                                                         custom_ligand_dict=custom_ligand_dict)
    descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                        descriptor_derivatives,
                                                                                        results_dictionary['colnames'],
                                                                                        results_dictionary['result_ax_full'],
                                                                                        'f', 'ax')
    descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                        descriptor_derivatives,
                                                                                        results_dictionary['colnames'],
                                                                                        results_dictionary['result_eq_full'],
                                                                                        'f', 'eq')
    descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                        descriptor_derivatives,
                                                                                        results_dictionary['colnames'],
                                                                                        results_dictionary['result_ax_con'],
                                                                                        'lc', 'ax')
    descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                        descriptor_derivatives,
                                                                                        results_dictionary['colnames'],
                                                                                        results_dictionary['result_eq_con'],
                                                                                        'lc', 'eq')
    results_dictionary = generate_all_ligand_deltametric_derivatives(this_complex, depth=depth, loud=False,
                                                                     custom_ligand_dict=custom_ligand_dict)
    descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                        descriptor_derivatives,
                                                                                        results_dictionary['colnames'],
                                                                                        results_dictionary['result_ax_con'],
                                                                                        'D_lc', 'ax')
    descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                        descriptor_derivatives,
                                                                                        results_dictionary['colnames'],
                                                                                        results_dictionary['result_eq_con'],
                                                                                        'D_lc', 'eq')
    # metal ACs
    # print('getting metal AC derivatives')
    results_dictionary = generate_metal_autocorrelation_derivatives(this_complex, depth=depth, loud=False,
                                                                    modifier=ox_modifier, metal_ind=metal_ind)
    descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                        descriptor_derivatives,
                                                                                        results_dictionary['colnames'],
                                                                                        results_dictionary['results'],
                                                                                        'mc', 'all')
    results_dictionary = generate_metal_deltametric_derivatives(this_complex, depth=depth, loud=False,
                                                                modifier=ox_modifier, metal_ind=metal_ind)
    descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                        descriptor_derivatives,
                                                                                        results_dictionary['colnames'],
                                                                                        results_dictionary['results'],
                                                                                        'D_mc', 'all')
    # ## ox-metal ACs
    if ox_modifier:
        results_dictionary = generate_metal_ox_autocorrelation_derivatives(ox_modifier, this_complex, depth=depth,
                                                                           loud=False, metal_ind=metal_ind)
        descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                            descriptor_derivatives,
                                                                                            results_dictionary['colnames'],
                                                                                            results_dictionary['results'],
                                                                                            'mc', 'all')

        results_dictionary = generate_metal_ox_deltametric_derivatives(ox_modifier, this_complex, depth=depth,
                                                                       loud=False, metal_ind=metal_ind)
        descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                            descriptor_derivatives,
                                                                                            results_dictionary['colnames'],
                                                                                            results_dictionary['results'],
                                                                                            'D_mc', 'all')

    return descriptor_derivative_names, descriptor_derivatives


def append_descriptors(descriptor_names, descriptors, list_of_names, list_of_props, prefix, suffix):
    """Utility to build standardly formated RACS

    Parameters
    ----------
        descriptor_names : list
            Descriptors names list to be appended to
        descriptors : list
            Descriptors list to be appended to
        list_of_names : list
            nmaes to be added
        list_of_props : list
            Types of RACs
        prefix : str
            Prefix to be added to names
        suffix : str
            Suffix to be added to names

    Returns
    -------
        descriptor_names : list
            Compiled list of descriptor names
        descriptors : list
            Compiled list of descriptor values

    """
    try:
        basestring
    except NameError:
        basestring = str

    for names in list_of_names:
        if not isinstance(names, basestring):
            names = ["-".join([prefix, str(i), suffix]) for i in names]
            descriptor_names += names
        else:
            names = "-".join([prefix, str(names), suffix])
            descriptor_names.append(names)
    for values in list_of_props:
        if not isinstance(names, basestring):
            descriptors.extend(values)
        else:
            descriptors.append(values)
    return descriptor_names, descriptors


def append_descriptor_derivatives(descriptor_derivative_names, descriptor_derivatives,
                                  mat_of_names, dmat, prefix, suffix):
    """Utility to build standardly formated RACS derivatives

    Parameters
    ----------
        descriptor_derivative_names : list
            RAC names, will be a matrix, will be appended to
        descriptor_derivatives : list
            RAC, will be appended to
        mat_of_names : list
            names, will be added
        dmat : list
            mat of RAC derivatives
        prefix : str
            RAC prefix
        suffix : str
            RAC suffix

    Returns
    -------
        descriptor_derivative_names : list
            Compiled list (matrix) of descriptor derivative names
        descriptor_derivatives : list
            Derivatives of RACs w.r.t atomic props (matrix)

    """
    # try:
    #     basestring
    # except NameError:
    #     basestring = str

    for names in mat_of_names:
        jnames = ["-".join([prefix, str(i), suffix]) for i in names]
        descriptor_derivative_names.append(jnames)
    if descriptor_derivatives is None:
        descriptor_derivatives = dmat
    else:
        descriptor_derivatives = np.row_stack([descriptor_derivatives, dmat])
    return descriptor_derivative_names, descriptor_derivatives


def autocorrelation(mol, prop_vec, orig, d, oct=True, use_dist=False, size_normalize=False):
    """Calculate and return the products autocorrelation

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
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.

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
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        for inds in new_active_set:
            if not use_dist:
                result_vector[hopped] += prop_vec[orig] * prop_vec[inds]
            else:
                this_dist = mol.getDistToMetal(orig, inds)
                if size_normalize:
                    result_vector[hopped] += prop_vec[orig] * prop_vec[inds] / (this_dist * mol.natoms)
                else:
                    result_vector[hopped] += prop_vec[orig] * prop_vec[inds] / (this_dist)
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def autocorrelation_derivative(mol, prop_vec, orig, d, oct=True):
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


def deltametric(mol, prop_vec, orig, d, oct=True, use_dist=False, size_normalize=False):
    """Returns the deltametric autocorrelation

    Parameters
    ----------
        mol : mol3D
            mol3D object to calculate deltametric autocorrelation over
        prop_vec : list
            property of atoms in mol in order of index
        orig : int
            zero-indexed starting atom
        d : int
            number of hops to travel
        oct : bool, optional
            Flag is octahedral complex, by default True
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.

    Returns
    -------
        results_vector : list
            deltametric autocorrelations

    """
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
            if not use_dist:
                result_vector[hopped] += prop_vec[orig] - prop_vec[inds]
            else:
                this_dist = mol.getDistToMetal(orig, inds)
                if size_normalize:
                    result_vector[hopped] += (prop_vec[orig] - prop_vec[inds]) / (this_dist * mol.natoms + 1e-6)
                else:
                    result_vector[hopped] += (prop_vec[orig] - prop_vec[inds]) / (this_dist + 1e-6)
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def deltametric_derivative(mol, prop_vec, orig, d, oct=True):
    """Returns the deltametric autocorrelation derivative vector

    Parameters
    ----------
        mol : mol3D
            mol3D object to calculate deltametric autocorrelation derivative over
        prop_vec : list
            property of atoms in mol in order of index
        orig : int
            zero-indexed starting atom
        d : int
            number of hops to travel
        oct : bool, optional
            Flag is octahedral complex, by default True

    Returns
    -------
        derivative_mat : list
            Deltametric autocorrelation derivatives matrix

    """
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


def construct_property_vector(mol, prop, oct=True, modifier=False, MRdiag_dict={}):
    """Assigns the value of property for atom i (zero index) in mol.

    Parameters
    ----------
        mol : mol3D
            molecule to generate property vector for
        prop : str
            Property to generate vector for - Acceptable prop values: ['electronegativity',
            'nuclear_charge', 'ident', 'topology', 'ox_nuclear_charge', 'size', 'vdwrad',
            'group_number', 'polarizability', 'bondvalence', 'num_bonds',
            'bondvalence_devi', 'bodavrg', 'bodstd', 'charge']
        oct : bool, optional
            Flag is octahedral complex, by default True
        modifier : bool, optional
            if passed - dict, used to modify prop vector (e.g. for adding
            ONLY used with  ox_nuclear_charge    ox or charge)
            {"Fe":2, "Co": 3} etc, by default False
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.

    Returns
    -------
        w : list
            property vector for mol by atom

    """
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology',
                       'ox_nuclear_charge', 'size', 'vdwrad', 'group_number', 'polarizability',
                       'bondvalence', 'num_bonds', 'bondvalence_devi', 'bodavrg', 'bodstd', 'charge',
                       ]
    if len(MRdiag_dict):
        for k in list(MRdiag_dict):
            allowed_strings += [k]
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
        at_keys = list(globs.amass().keys())
        for keys in at_keys:
            values = globs.amass()[keys][3]
            prop_dict.update({keys: values})
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
    elif prop == 'bondvalence':
        assert len(mol.getAtoms()) == len(mol.bv_dict)
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = mol.bv_dict[i]
        done = True
    elif prop == 'num_bonds':
        for i, atom in enumerate(mol.getAtoms()):
            if not atom.ismetal():
                w[i] = globs.bondsdict()[atom.symbol()]
            else:
                w[i] = len(mol.getBondedAtomsSmart(i, oct=True))
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
    elif prop in ['A25PBE', 'B1', 'rND_PBE', 'IND_PBE', 'nLUMO_MP2', 'T1',
                  'largest_amp', 'TAE', 'C0^2', 'nLUMO_CAS', '%CorrE_orca']:
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = MRdiag_dict[prop][mol.getAtom(i).symbol()]
        done = True
    if not done:
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = prop_dict[atoms.symbol()]
    return (w)


def full_autocorrelation(mol, prop, d, oct=True, modifier=False, use_dist=False, size_normalize=False, MRdiag_dict={}):
    """Calculate full scope product autocorrelations (i.e. start at every atom up to depth d)

    Parameters
    ----------
        mol : mol3D
            molecule to calculate full scope RAC over
        prop : str
            Property to evaluete
        d : int
            depth of full scope autocorrelation
        oct : bool, optional
            Is octahedral flag, by default True
        modifier : bool, optional
            Use ox modifier, by default False
        use_dist : bool, optional
            Weigh autocorrelation by distance of atoms from each other, by default False
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.

    Returns
    -------
        autocorrelation_vector : list
            full scope product autocorrelation values

    """
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier, MRdiag_dict=MRdiag_dict)
    index_set = list(range(0, mol.natoms))
    autocorrelation_vector = np.zeros(d + 1)
    for centers in index_set:
        autocorrelation_vector += autocorrelation(mol, w, centers, d, oct=oct, use_dist=use_dist,
                                                  size_normalize=size_normalize)
    return (autocorrelation_vector)


def full_autocorrelation_derivative(mol, prop, d, oct=True, modifier=False):
    """Calculate full scope product autocorrelations derivatives
    (i.e. start at every atom up to depth d)

    Parameters
    ----------
        mol : mol3D
            molecule to calculate full scope RAC over
        prop : str
            Property to evaluate
        d : int
            depth of scope to evalue
        oct : bool, optional
            Is octahedral flag, by default True
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False
        modifier : bool, optional
            Use ox modifier, by default False

    Returns
    -------
        autocorrelation_derivative_mat : list
            full scope autocorrelation derivative matrix

    """
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
    index_set = list(range(0, mol.natoms))
    autocorrelation_derivative_mat = np.zeros((d + 1, mol.natoms))
    for centers in index_set:
        autocorrelation_derivative_mat += autocorrelation_derivative(mol, w, centers, d, oct=oct)
    return (autocorrelation_derivative_mat)


def generate_full_complex_autocorrelations(mol, loud,
                                           depth=4, oct=True,
                                           flag_name=False, modifier=False,
                                           use_dist=False, size_normalize=False,
                                           NumB=False, Gval=False, polarizability=False,
                                           MRdiag_dict={}):
    """Utility to manage full complex autocorrelation generation and labeling.

    Parameters
    ----------
        mol : mol3D
            molecule used for full scope
        loud : bool
            print debugging information
        depth : int, optional
            depth of autocorrelations to evaluate, by default 4
        oct : bool, optional
            is an octahedral complex, by default True
        flag_name : bool, optional
            Prepend "f_all" to results to track full complex, by default False
        modifier : bool, optional
            Use ox_modifier on metal charge, by default False
        use_dist : bool, optional
            Weigh autocorrelations by interatomic distances, by default False
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        NumB : bool, optional
            use number of bonds as RAC, by default False
        Gval : bool, optional
            use G value as RAC, by default False
        polarizability : bool, optional
            Use polarizability (alpha) as RAC, by default False
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.

    Returns
    -------
        results_dictionary : dict
            formatted dictionary with {'colnames': colnames, 'results': result}

    """
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
    if len(MRdiag_dict):
        allowed_strings, labels_strings = [], []
        for k in list(MRdiag_dict):
            allowed_strings += [k]
            labels_strings += [k]
    for ii, properties in enumerate(allowed_strings):
        metal_ac = full_autocorrelation(mol, properties, depth,
                                        oct=oct, modifier=modifier,
                                        use_dist=use_dist, size_normalize=size_normalize, MRdiag_dict=MRdiag_dict)
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
    """Utility to manage full complex autocorrelation derivative generation and labeling.

    Parameters
    ----------
        mol : mol3D
            molecule used for full scope
        loud : bool
            print debugging information
        depth : int, optional
            depth of autocorrelations to evaluate, by default 4
        oct : bool, optional
            is an octahedral complex, by default True
        flag_name : bool, optional
            Prepend "f_all" to results to track full complex, by default False
        modifier : bool, optional
            Use ox_modifier on metal charge, by default False
        NumB : bool, optional
            use number of bonds as RAC, by default False
        Gval : bool, optional
            use G value as RAC, by default False

    Returns
    -------
        results_dictionary : dict
            formatted dictionary with {'colnames': colnames, 'results': result}

    """
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
        # colnames.append(this_colnames)
        if result is None:
            result = f_ac_der
        else:
            result = np.row_stack([result, f_ac_der])
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_f_all': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def atom_only_autocorrelation(mol, prop, d, atomIdx, oct=True, use_dist=False, size_normalize=False, MRdiag_dict={}):
    """Calculate product autocorrelation vectors from a given atom or list of atoms
    (e.g. up to depth 4 from the connecting atoms)

    Parameters
    ----------
        mol : mol3D
            molecule to calculate atom-only autocorrelations from
        prop : str
            property to calculate
        d : int
            depth to calculate derivatives over
        atomIdx : int or list
            atoms from which the autocorrelation vector should be centered
        oct : bool, optional
            use octahedral flag, by default True
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.

    Returns
    -------
        autocorrelation_vector : list
            list of atom-only autocorrelations

    """
    w = construct_property_vector(mol, prop, oct, MRdiag_dict=MRdiag_dict)
    autocorrelation_vector = np.zeros(d + 1)
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            autocorrelation_vector += autocorrelation(mol, w, elements, d, oct=oct, use_dist=use_dist,
                                                      size_normalize=size_normalize)
        autocorrelation_vector = np.divide(autocorrelation_vector, len(atomIdx))
    else:
        autocorrelation_vector += autocorrelation(mol, w, atomIdx, d, oct=oct, use_dist=use_dist,
                                                  size_normalize=size_normalize)
    return (autocorrelation_vector)


def atom_only_autocorrelation_derivative(mol, prop, d, atomIdx, oct=True):
    """Calculate product autocorrelation derivative vectors from a given atom or list of atoms
    (e.g. up to depth 4 from the connecting atoms)

    Parameters
    ----------
        mol : mol3D
            molecule to calculate atom-only autocorrelation derivatives from
        prop : str
            property to calculate
        d : int
            depth to calculate derivatives over
        atomIdx : int or list
            atoms from which the autocorrelation vector should be centered
        oct : bool, optional
            use octahedral flag, by default True

    Returns
    -------
        autocorrelation_vector : list
            list of atom-only autocorrelation derivatives

    """
    w = construct_property_vector(mol, prop, oct)
    autocorrelation_derivative_mat = np.zeros((d + 1, mol.natoms))
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            autocorrelation_derivative_mat += autocorrelation_derivative(mol, w, elements, d, oct=oct)
        autocorrelation_derivative_mat = np.divide(autocorrelation_derivative_mat, len(atomIdx))
    else:
        autocorrelation_derivative_mat += autocorrelation_derivative(mol, w, atomIdx, d, oct=oct)
    return (autocorrelation_derivative_mat)


def metal_only_autocorrelation(mol, prop, d, oct=True, metal_ind=None,
                               func=autocorrelation, modifier=False, use_dist=False, size_normalize=False, MRdiag_dict={}):
    """Calculate the metal_only product autocorrelations
    (e.g. metal-centered atom-only RACs)

    Parameters
    ----------
        mol : mol3D
            molecule with metal to calculate MC product RACs for
        prop : str
            Property to evaluate
        d : int
            depth of autocorrelation
        oct : bool, optional
            use octahedral geometry evaluations, by default True
        metal_ind : bool, optional
            index of the metal atom to generate property, by default False
        func : function, optional
            which function to evaluate mc-racs by, by default autocorrelation
        modifier : bool, optional
            use ox_modifier, by default False
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.

    Returns
    -------
        autocorrelation_vector: list
            MC atom-only RACs vector

    """
    try:
        if not isinstance(metal_ind, int):
            metal_ind = mol.findMetal()[0]
        w = construct_property_vector(mol, prop, oct=oct, modifier=modifier, MRdiag_dict=MRdiag_dict)
        autocorrelation_vector = func(mol, w, metal_ind, d, oct=oct, use_dist=use_dist, size_normalize=size_normalize)
    except IndexError:
        print('Error, no metal found in mol object!')
        return False
    return (autocorrelation_vector)


def metal_only_autocorrelation_derivative(mol, prop, d, oct=True, metal_ind=None,
                                          func=autocorrelation_derivative, modifier=False):
    """Calculate the metal_only product autocorrelation derivatives
    (e.g. metal-centered atom-only RAC derivatives)

    Parameters
    ----------
        mol : mol3D
            molecule with metal to calculate MC product RAC derivatives for
        prop : str
            Property to evaluate
        d : int
            depth of autocorrelation
        oct : bool, optional
            use octahedral geometry evaluations, by default True
        metal_ind : bool, optional
            index (int) of metal atom to consider, default False
        func : function, optional
            which function to evaluate mc-racs by, by default autocorrelation_derivative
        modifier : bool, optional
            use ox_modifier, by default False

    Returns
    -------
        autocorrelation_vector: list
            MC atom-only RAC derivatives vector (matrix)

    """
    try:
        if not isinstance(metal_ind, int):
            metal_ind = mol.findMetal()[0]
        w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
        autocorrelation_vector_derivative = func(mol, w, metal_ind, d, oct=oct)
    except IndexError:
        print('Error, no metal found in mol object!')
        return False
    return (autocorrelation_vector_derivative)


def atom_only_deltametric(mol, prop, d, atomIdx, oct=True, modifier=False,
                          use_dist=False, size_normalize=False, MRdiag_dict={}):
    """Calculate deltametric autocorrelation vectors from a given atom or list of atoms
    (e.g. up to depth 4 from the connecting atoms)

    Parameters
    ----------
        mol : mol3D
            molecule to calculate atom-only autocorrelations from
        prop : str
            property to calculate
        d : int
            depth to calculate derivatives over
        atomIdx : int or list
            atoms from which the autocorrelation vector should be centered
        oct : bool, optional
            use octahedral flag, by default True
        modifier : TODO
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.

    Returns
    -------
        autocorrelation_vector : list
            list of atom-only deltametric autocorrelations

    """
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier, MRdiag_dict=MRdiag_dict)
    deltametric_vector = np.zeros(d + 1)
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            deltametric_vector += deltametric(mol, w, elements, d, oct=oct, use_dist=use_dist, size_normalize=size_normalize)
        deltametric_vector = np.divide(deltametric_vector, len(atomIdx))
    else:
        deltametric_vector += deltametric(mol, w, atomIdx, d, oct=oct, use_dist=use_dist, size_normalize=size_normalize)
    return (deltametric_vector)


def atom_only_deltametric_derivative(mol, prop, d, atomIdx, oct=True, modifier=False):
    """Calculate deltametric autocorrelation derivative vectors
    from a given atom or list of atoms
    (e.g. up to depth 4 from the connecting atoms)

    Parameters
    ----------
        mol : mol3D
            molecule to calculate atom-only deltametric autocorrelation derivatives from
        prop : str
            property to calculate
        d : int
            depth to calculate derivatives over
        atomIdx : int or list
            atoms from which the autocorrelation vector should be centered
        oct : bool, optional
            use octahedral flag, by default True
        modifier : bool, optional
            use ox_modifier, by default False

    Returns
    -------
        deltametric_derivative_mat : list
            matrix of atom-only deltametric autocorrelation derivatives

    """
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
    deltametric_derivative_mat = np.zeros((d + 1, mol.natoms))
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            deltametric_derivative_mat += deltametric_derivative(mol, w, elements, d, oct=oct)
        deltametric_derivative_mat = np.divide(deltametric_derivative_mat, len(atomIdx))
    else:

        deltametric_derivative_mat += deltametric_derivative(mol, w, atomIdx, d, oct=oct)
    return (deltametric_derivative_mat)


def metal_only_deltametric_derivative(mol, prop, d, oct=True, metal_ind=None,
                                      func=deltametric_derivative, modifier=False):
    """Gets the metal atom-only deltametric derivatives

    Parameters
    ----------
        mol : mol3D
            molecule with metal to calculate MC deltametric RAC derivatives for
        prop : str
            Property to evaluate
        d : int
            depth of autocorrelation
        oct : bool, optional
            use octahedral geometry evaluations, by default True
        metal_ind : bool, optional
            index of metal atom to consider, by default False
        func : function, optional
            which function to evaluate mc-racs by, by default deltametric_derivative
        modifier : bool, optional
            use ox_modifier, by default False

    Returns
    -------
        deltametric_vector_derivative : list
            metal-centerted deltametric derivatives vector (matrix)

    """
    try:
        if not isinstance(metal_ind, int):
            metal_ind = mol.findMetal()[0]
        w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
        deltametric_vector_derivative = func(mol, w, metal_ind, d, oct=oct)
    except IndexError:
        print('Error, no metal found in mol object!')
        return False
    return (deltametric_vector_derivative)


def metal_only_deltametric(mol, prop, d, oct=True, metal_ind=None,
                           func=deltametric, modifier=False, use_dist=False, size_normalize=False, MRdiag_dict={}):
    """Gets the metal atom-only deltametric RAC

    Parameters
    ----------
        mol : mol3D
            molecule with metal to calculate MC deltametric RACs
        prop : str
            Property to evaluate
        d : int
            depth of autocorrelation
        oct : bool, optional
            use octahedral geometry evaluations, by default True
        metal_ind : bool, optional
            index of metal atom to consider, by default False
        func : function, optional
            which function to evaluate mc-racs by, by default deltametric
        modifier : bool, optional
            use ox_modifier, by default False
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.

    Returns
    -------
        deltametric_vector : list
            metal-centerted deltametric RAC vector

    """
    try:
        if not isinstance(metal_ind, int):
            metal_ind = mol.findMetal()[0]
        w = construct_property_vector(mol, prop, oct=oct, modifier=modifier, MRdiag_dict=MRdiag_dict)
        deltametric_vector = func(mol, w, metal_ind, d, oct=oct, use_dist=use_dist, size_normalize=size_normalize)
    except IndexError:
        print('Error, no metal found in mol object!')
        return False
    return (deltametric_vector)


# Get the ligand_misc_descriptors
# custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
# #                                    ax_con_int_list ,eq_con_int_list
# @param mol, mol3D class
# @param loud, bool, print out statements for debugging
# @return results_dictionary, vector, ax vs eq. charge (from OBMol) and denticity
def generate_all_ligand_misc(mol, loud, custom_ligand_dict=False, smiles_charge=False):
    """Get the ligand_misc_descriptors (axial vs. equatorial
     charge (from OBMol) and denticity)

    Parameters
    ----------
        mol : mol3D
            molecule to get the ligand_misc descriptors from
        loud : bool
            print debugging information
        custom_ligand_dict : bool, optional
            custom_ligand_dictionary if passed, by default False
        smiles_charge : bool, optional
            Whether or not to use the smiles charge assignent, default is False

    Returns
    -------
        results_dictionary : dict
            Labels and results of ligand_misc RACs - {'colnames': colnames,
                            'result_ax': result_ax, 'result_eq': result_eq}

    """
    result_ax = list()
    result_eq = list()
    colnames = ['dent', 'charge']
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True) # Complex is assumed to be octahedral
        ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list, eq_con_int_list, \
            ax_con_list, eq_con_list, built_ligand_list = ligand_assign(
                mol, liglist, ligdents, ligcons, loud)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        # ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        # eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    # count ligands
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)
    # allocate
    result_ax_dent = False
    result_eq_dent = False
    result_ax_charge = False
    result_eq_charge = False
    # loop over axial ligands
    if n_ax > 0:
        for i in range(0, n_ax):
            if mol.bo_dict:
                ax_ligand_list[i].mol.convert2OBMol2()
            else:
                ax_ligand_list[i].mol.convert2OBMol()
            if not (i == 0):
                result_ax_dent += ax_ligand_list[i].dent
                if smiles_charge:
                    result_ax_charge += ax_ligand_list[i].mol.get_smilesOBmol_charge()
                else:
                    result_ax_charge += ax_ligand_list[i].mol.OBMol.GetTotalCharge()
            else:
                result_ax_dent = ax_ligand_list[i].dent
                if smiles_charge:
                    result_ax_charge = ax_ligand_list[i].mol.get_smilesOBmol_charge()
                else:
                    result_ax_charge = ax_ligand_list[i].mol.OBMol.GetTotalCharge()
        # average axial results
        result_ax_dent = np.divide(result_ax_dent, n_ax)
        result_ax_charge = np.divide(result_ax_charge, n_ax)
    # loop over eq ligands
    if n_eq > 0:
        for i in range(0, n_eq):
            if mol.bo_dict:
                eq_ligand_list[i].mol.convert2OBMol2()
            else:
                eq_ligand_list[i].mol.convert2OBMol()
            if not (i == 0):
                result_eq_dent += eq_ligand_list[i].dent
                if smiles_charge:
                    result_eq_charge += eq_ligand_list[i].mol.get_smilesOBmol_charge()
                else:
                    result_eq_charge += eq_ligand_list[i].mol.OBMol.GetTotalCharge()
            else:
                result_eq_dent = eq_ligand_list[i].dent
                if smiles_charge:
                    result_eq_charge = eq_ligand_list[i].mol.get_smilesOBmol_charge()
                else:
                    result_eq_charge = eq_ligand_list[i].mol.OBMol.GetTotalCharge()
        # average eq results
        result_eq_dent = np.divide(result_eq_dent, n_eq)
        result_eq_charge = np.divide(result_eq_charge, n_eq)
        # save the results
    result_ax.append(result_ax_dent)
    result_ax.append(result_ax_charge)
    result_eq.append(result_eq_dent)
    result_eq.append(result_eq_charge)
    results_dictionary = {'colnames': colnames,
                          'result_ax': result_ax, 'result_eq': result_eq}
    return results_dictionary


def generate_all_ligand_autocorrelations(mol, loud, depth=4, flag_name=False,
                                         custom_ligand_dict=False, NumB=False, Gval=False,
                                         use_dist=False, size_normalize=False, MRdiag_dict={}):
    """Utility for generating all ligand-based product autocorrelations for a complex

    Parameters
    ----------
        mol : mol3D
            molecule to get lc-RACs for
        loud : bool
            print debugging information
        depth : int, optional
            depth of RACs to calculate, by default 4
        flag_name : bool, optional
            Shift RAC names slightly, by default False
        custom_ligand_dict : bool, optional
            Dict of ligands if passed - see generate_descriptor_vector, by default False
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False
        Gval : bool, optional
            Use G value as descriptor property, by default False
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.

    Returns
    -------
        results_dictionary: dict
            Dictionary of all geo-based ligand product descriptors (both full and connecting atom scopes) -
            {'colnames': colnames, 'result_ax_full': result_ax_full, 'result_eq_full': result_eq_full,
            'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}

    """
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    if len(MRdiag_dict):
        allowed_strings, labels_strings = [], []
        for k in list(MRdiag_dict):
            allowed_strings += [k]
            labels_strings += [k]
    result_ax_full = list()
    result_eq_full = list()
    result_ax_con = list()
    result_eq_con = list()
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True) # Complex is assumed to be octahedral
        (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list, eq_con_int_list,
         ax_con_list, eq_con_list, built_ligand_list) = ligand_assign(
            mol, liglist, ligdents, ligcons, loud)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    # count ligands
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)
    colnames = []
    for ii, properties in enumerate(allowed_strings):
        # ############## replaced find_ligand_autocorrelations_oct function here
        # get full ligand AC
        ax_ligand_ac_full = []
        eq_ligand_ac_full = []
        for i in range(0, n_ax):
            if not list(ax_ligand_ac_full):
                ax_ligand_ac_full = full_autocorrelation(ax_ligand_list[i].mol, properties, depth, use_dist=use_dist,
                                                         size_normalize=size_normalize, MRdiag_dict=MRdiag_dict)
            else:
                ax_ligand_ac_full += full_autocorrelation(ax_ligand_list[i].mol, properties, depth, use_dist=use_dist,
                                                          size_normalize=size_normalize, MRdiag_dict=MRdiag_dict)
        ax_ligand_ac_full = np.divide(ax_ligand_ac_full, n_ax)
        for i in range(0, n_eq):
            if not list(eq_ligand_ac_full):
                eq_ligand_ac_full = full_autocorrelation(eq_ligand_list[i].mol, properties, depth, use_dist=use_dist,
                                                         size_normalize=size_normalize, MRdiag_dict=MRdiag_dict)
            else:
                eq_ligand_ac_full += full_autocorrelation(eq_ligand_list[i].mol, properties, depth, use_dist=use_dist,
                                                          size_normalize=size_normalize, MRdiag_dict=MRdiag_dict)
        eq_ligand_ac_full = np.divide(eq_ligand_ac_full, n_eq)
        ax_ligand_ac_con = []
        eq_ligand_ac_con = []
        for i in range(0, n_ax):
            if not list(ax_ligand_ac_con):
                ax_ligand_ac_con = atom_only_autocorrelation(ax_ligand_list[i].mol, properties, depth, ax_con_int_list[i],
                                                             use_dist=use_dist, size_normalize=size_normalize,
                                                             MRdiag_dict=MRdiag_dict)
            else:
                ax_ligand_ac_con += atom_only_autocorrelation(ax_ligand_list[i].mol, properties, depth, ax_con_int_list[i],
                                                              use_dist=use_dist, size_normalize=size_normalize,
                                                              MRdiag_dict=MRdiag_dict)
        ax_ligand_ac_con = np.divide(ax_ligand_ac_con, n_ax)
        for i in range(0, n_eq):
            if not list(eq_ligand_ac_con):
                eq_ligand_ac_con = atom_only_autocorrelation(eq_ligand_list[i].mol, properties, depth, eq_con_int_list[i],
                                                             use_dist=use_dist, size_normalize=size_normalize,
                                                             MRdiag_dict=MRdiag_dict)
            else:
                eq_ligand_ac_con += atom_only_autocorrelation(eq_ligand_list[i].mol, properties, depth, eq_con_int_list[i],
                                                              use_dist=use_dist, size_normalize=size_normalize,
                                                              MRdiag_dict=MRdiag_dict)
        eq_ligand_ac_con = np.divide(eq_ligand_ac_con, n_eq)
        ################
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result_ax_full.append(ax_ligand_ac_full)
        result_eq_full.append(eq_ligand_ac_full)
        result_ax_con.append(ax_ligand_ac_con)
        result_eq_con.append(eq_ligand_ac_con)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'result_ax_full_ac': result_ax_full,
                              'result_eq_full_ac': result_eq_full,
                              'result_ax_con_ac': result_ax_con, 'result_eq_con_ac': result_eq_con}
    else:
        results_dictionary = {'colnames': colnames, 'result_ax_full': result_ax_full, 'result_eq_full': result_eq_full,
                              'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}
    return results_dictionary


def generate_all_ligand_autocorrelation_derivatives(mol, loud, depth=4, flag_name=False,
                                                    custom_ligand_dict=False, NumB=False, Gval=False):
    """Utility for generating all ligand-based autocorrelation derivatives for a complex

    Parameters
    ----------
        mol : mol3D
            molecule to get lc-RAC derivatives for
        loud : bool
            print debugging information
        depth : int, optional
            depth of RACs to calculate, by default 4
        flag_name : bool, optional
            Shift RAC names slightly, by default False
        custom_ligand_dict : bool, optional
            Dict of ligands if passed - see generate_descriptor_vector, by default False
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False
        Gval : bool, optional
            Use G value as descriptor property, by default False

    Returns
    -------
        results_dictionary: dict
            Dictionary of all geo-based ligand product descriptor derivatives
            (both full and connecting atom scopes)
            {'colnames': colnames, 'result_ax_full': result_ax_full, 'result_eq_full': result_eq_full,
            'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}

    """
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
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True) # Complex is assumed to be octahedral
        ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list, eq_con_int_list, \
            ax_con_list, eq_con_list, built_ligand_list = ligand_assign(
                mol, liglist, ligdents, ligcons, loud)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    # count ligands
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)
    for ii, properties in enumerate(allowed_strings):
        # allocate the full jacobian matrix
        ax_full_j = np.zeros([depth + 1, mol.natoms])
        eq_full_j = np.zeros([depth + 1, mol.natoms])
        ax_con_j = np.zeros([depth + 1, mol.natoms])
        eq_con_j = np.zeros([depth + 1, mol.natoms])
        #################
        # full ligand ACs
        for i in range(0, n_ax):  # for each ax ligand
            ax_ligand_ac_full_derivative = full_autocorrelation_derivative(ax_ligand_list[i].mol, properties, depth)
            # now we need to map back to full positions
            for jj, row in enumerate(ax_ligand_ac_full_derivative):
                for original_ids in list(ax_ligand_list[i].ext_int_dict.keys()):
                    ax_full_j[jj, original_ids] += np.divide(row[ax_ligand_list[i].ext_int_dict[original_ids]], n_ax)
        for i in range(0, n_eq):  # for each eq ligand
            # now we need to map back to full positions
            eq_ligand_eq_full_derivative = full_autocorrelation_derivative(eq_ligand_list[i].mol, properties, depth)
            for jj, row in enumerate(eq_ligand_eq_full_derivative):
                for original_ids in list(eq_ligand_list[i].ext_int_dict.keys()):
                    eq_full_j[jj, original_ids] += np.divide(row[eq_ligand_list[i].ext_int_dict[original_ids]], n_eq)
        # ligand connection ACs
        for i in range(0, n_ax):
            ax_ligand_ac_con_derivative = atom_only_autocorrelation_derivative(ax_ligand_list[i].mol, properties, depth,
                                                                               ax_con_int_list[i])
            # now we need to map back to full positions
            for jj, row in enumerate(ax_ligand_ac_con_derivative):
                for original_ids in list(ax_ligand_list[i].ext_int_dict.keys()):
                    ax_con_j[jj, original_ids] += np.divide(row[ax_ligand_list[i].ext_int_dict[original_ids]], n_ax)
        for i in range(0, n_eq):
            eq_ligand_ac_con_derivative = atom_only_autocorrelation_derivative(eq_ligand_list[i].mol, properties, depth,
                                                                               eq_con_int_list[i])
            # now we need to map back to full positions
            for jj, row in enumerate(eq_ligand_ac_con_derivative):
                for original_ids in list(eq_ligand_list[i].ext_int_dict.keys()):
                    eq_con_j[jj, original_ids] += np.divide(row[eq_ligand_list[i].ext_int_dict[original_ids]], n_eq)
        ax_ligand_ac_full, eq_ligand_ac_full, ax_ligand_ac_con, eq_ligand_ac_con = ax_full_j, eq_full_j, ax_con_j, eq_con_j
        #################
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


def generate_all_ligand_deltametrics(mol, loud, depth=4, flag_name=False,
                                     custom_ligand_dict=False, NumB=False, Gval=False,
                                     use_dist=False, size_normalize=False, MRdiag_dict={}):
    """Utility for generating all ligand-based deltametric autocorrelations for a complex

    Parameters
    ----------
        mol : mol3D
            molecule to get D_lc-RACs for
        loud : bool
            print debugging information
        depth : int, optional
            depth of RACs to calculate, by default 4
        flag_name : bool, optional
            Shift RAC names slightly, by default False
        custom_ligand_dict : bool, optional
            Dict of ligands if passed - see generate_descriptor_vector, by default False
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False
        Gval : bool, optional
            Use G value as descriptor property, by default False
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.

    Returns
    -------
        results_dictionary: dict
            Dictionary of all geo-based ligand deltametric descriptors (both full and connecting atom scopes) -
            {'colnames': colnames, 'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}

    """
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
    if len(MRdiag_dict):
        allowed_strings, labels_strings = [], []
        for k in list(MRdiag_dict):
            allowed_strings += [k]
            labels_strings += [k]
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True) # Complex is assumed to be octahedral
        (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list, eq_con_int_list,
         ax_con_list, eq_con_list, built_ligand_list) = ligand_assign(mol, liglist, ligdents, ligcons, loud)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    # count ligands
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)
    for ii, properties in enumerate(allowed_strings):
        ####################
        # get partial ligand AC
        ax_ligand_ac_con = []
        eq_ligand_ac_con = []
        for i in range(0, n_ax):
            if not list(ax_ligand_ac_con):
                ax_ligand_ac_con = atom_only_deltametric(ax_ligand_list[i].mol, properties, depth, ax_con_int_list[i],
                                                         use_dist=use_dist, size_normalize=size_normalize,
                                                         MRdiag_dict=MRdiag_dict)
            else:
                ax_ligand_ac_con += atom_only_deltametric(ax_ligand_list[i].mol, properties, depth, ax_con_int_list[i],
                                                          use_dist=use_dist, size_normalize=size_normalize,
                                                          MRdiag_dict=MRdiag_dict)
        ax_ligand_ac_con = np.divide(ax_ligand_ac_con, n_ax)
        for i in range(0, n_eq):
            if not list(eq_ligand_ac_con):
                eq_ligand_ac_con = atom_only_deltametric(eq_ligand_list[i].mol, properties, depth, eq_con_int_list[i],
                                                         use_dist=use_dist, size_normalize=size_normalize,
                                                         MRdiag_dict=MRdiag_dict)
            else:
                eq_ligand_ac_con += atom_only_deltametric(eq_ligand_list[i].mol, properties, depth, eq_con_int_list[i],
                                                          use_dist=use_dist, size_normalize=size_normalize,
                                                          MRdiag_dict=MRdiag_dict)
        eq_ligand_ac_con = np.divide(eq_ligand_ac_con, n_eq)
        ####################
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


def generate_all_ligand_deltametric_derivatives(mol, loud, depth=4, flag_name=False,
                                                custom_ligand_dict=False, NumB=False, Gval=False):
    """Utility for generating all ligand-based deltametric derivatives for a complex

    Parameters
    ----------
        mol : mol3D
            molecule to get lc-RAC deltametric derivatives for
        loud : bool
            print debugging information
        depth : int, optional
            depth of RACs to calculate, by default 4
        flag_name : bool, optional
            Shift RAC names slightly, by default False
        custom_ligand_dict : bool, optional
            Dict of ligands if passed - see generate_descriptor_vector, by default False
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False
        Gval : bool, optional
            Use G value as descriptor property, by default False

    Returns
    -------
        results_dictionary: dict
            Dictionary of all geo-based ligand deltametric descriptor derivatives
            (both full and connecting atom scopes) -
            {'colnames': colnames, 'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}

    """
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
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True) # Complex is assumed to be octahedral
        ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list, eq_con_int_list, \
            ax_con_list, eq_con_list, built_ligand_list = ligand_assign(
                mol, liglist, ligdents, ligcons, loud)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    # count ligands
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)
    for ii, properties in enumerate(allowed_strings):
        # allocate the full jacobian matrix
        ax_con_j = np.zeros([depth + 1, mol.natoms])
        eq_con_j = np.zeros([depth + 1, mol.natoms])
        #################
        for i in range(0, n_ax):
            ax_ligand_ac_con_derivative = atom_only_deltametric_derivative(ax_ligand_list[i].mol, properties, depth,
                                                                           ax_con_int_list[i])
            for jj, row in enumerate(ax_ligand_ac_con_derivative):
                for original_ids in list(ax_ligand_list[i].ext_int_dict.keys()):
                    ax_con_j[jj, original_ids] += np.divide(row[ax_ligand_list[i].ext_int_dict[original_ids]], n_ax)
        for i in range(0, n_eq):
            eq_ligand_ac_con_derivative = atom_only_deltametric_derivative(eq_ligand_list[i].mol, properties, depth,
                                                                           eq_con_int_list[i])
            for jj, row in enumerate(eq_ligand_ac_con_derivative):
                for original_ids in list(eq_ligand_list[i].ext_int_dict.keys()):
                    eq_con_j[jj, original_ids] += np.divide(row[eq_ligand_list[i].ext_int_dict[original_ids]], n_eq)
        #################
        ax_ligand_ac_con, eq_ligand_ac_con = ax_con_j, eq_con_j
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
                                    modifier=False, NumB=False, Gval=False, metal_ind=None,
                                    use_dist=False, size_normalize=False, MRdiag_dict={}):
    """Utility for generating all metal-centered product autocorrelations for a complex

    Parameters
    ----------
        mol : mol3D
            molecule to get mc-RACs for
        loud : bool
            print debugging information
        depth : int, optional
            depth of RACs to calculate, by default 4
        oct : bool, optional
            Use octahedral criteria for structure evaluation, by default True
        flag_name : bool, optional
            Shift RAC names slightly, by default False
        modifier : bool, optional
            Use ox_modifier for metal, by default False
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False
        Gval : bool, optional
            Use G value as descriptor property, by default False
        metal_ind : bool, optional
            index of the metal atom to generate property, by default False
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.

    Returns
    -------
        results_dictionary: dict
            Dictionary of all geo-based MC-RAC product descriptors  -
            {'colnames': colnames, 'results': result}

    """
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
    if len(MRdiag_dict):
        allowed_strings, labels_strings = [], []
        for k in list(MRdiag_dict):
            allowed_strings += [k]
            labels_strings += [k]
    for ii, properties in enumerate(allowed_strings):
        metal_ac = metal_only_autocorrelation(mol, properties, depth, oct=oct,
                                              modifier=modifier, metal_ind=metal_ind, use_dist=use_dist,
                                              size_normalize=size_normalize, MRdiag_dict=MRdiag_dict)
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
                                               modifier=False, NumB=False, Gval=False, metal_ind=None):
    """Utility for generating all metal-centered product autocorrelation derivatives for a complex

    Parameters
    ----------
        mol : mol3D
            molecule to get mc-RAC derivatives for
        loud : bool
            print debugging information
        depth : int, optional
            depth of RACs to calculate, by default 4
        oct : bool, optional
            Use octahedral criteria for structure evaluation, by default True
        flag_name : bool, optional
            Shift RAC names slightly, by default False
        modifier : bool, optional
            Use ox_modifier for metal, by default False
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False
        Gval : bool, optional
            Use G value as descriptor property, by default False
        metal_ind : bool, optional
            index of the metal atom to generate property, by default False

    Returns
    -------
        results_dictionary : dict
            Dictionary of all geo-based MC-RAC product descriptor derivatives
            {'colnames': colnames, 'results': result}

    """
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
        metal_ac_der = metal_only_autocorrelation_derivative(mol, properties, depth,
                                                             oct=oct, modifier=modifier, metal_ind=metal_ind)
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


def generate_metal_deltametrics(mol, loud, depth=4, oct=True, flag_name=False,
                                modifier=False, NumB=False, Gval=False, metal_ind=None,
                                use_dist=False, size_normalize=False, MRdiag_dict={}):
    """Utility for generating all metal-centered deltametric autocorrelations for a complex

    Parameters
    ----------
        mol : mol3D
            molecule to get D_mc-RACs for
        loud : bool
            print debugging information
        depth : int, optional
            depth of RACs to calculate, by default 4
        oct : bool, optional
            Use octahedral criteria for structure evaluation, by default True
        flag_name : bool, optional
            Shift RAC names slightly, by default False
        modifier : bool, optional
            Use ox_modifier for metal, by default False
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False
        Gval : bool, optional
            Use G value as descriptor property, by default False
        metal_ind : bool, optional
            index of the metal atom to generate property, by default False
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.

    Returns
    -------
        results_dictionary: dict
            Dictionary of all geo-based MC-RAC deltametric descriptors  -
            {'colnames': colnames, 'results': result}

    """
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
    if len(MRdiag_dict):
        allowed_strings, labels_strings = [], []
        for k in list(MRdiag_dict):
            allowed_strings += [k]
            labels_strings += [k]
    for ii, properties in enumerate(allowed_strings):
        metal_ac = metal_only_deltametric(mol, properties, depth, oct=oct,
                                          modifier=modifier, metal_ind=metal_ind,
                                          use_dist=use_dist, size_normalize=size_normalize, MRdiag_dict=MRdiag_dict)
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
                                           modifier=False, NumB=False, Gval=False, metal_ind=None):
    """Utility for generating all metal-centered deltametric autocorrelation derivatives
    for a complex

    Parameters
    ----------
        mol : mol3D
            molecule to get D_mc-RAC derivatives for
        loud : bool
            print debugging information
        depth : int, optional
            depth of RACs to calculate, by default 4
        oct : bool, optional
            Use octahedral criteria for structure evaluation, by default True
        flag_name : bool, optional
            Shift RAC names slightly, by default False
        modifier : bool, optional
            Use ox_modifier for metal, by default False
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False
        Gval : bool, optional
            Use G value as descriptor property, by default False
        metal_ind : bool, optional
            index of the metal atom to generate property, by default False

    Returns
    -------
        results_dictionary: dict
            Dictionary of all geo-based MC-RAC deltametric descriptor derivatives  -
            {'colnames': colnames, 'results': result}

    """
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
        metal_ac_der = metal_only_deltametric_derivative(mol, properties, depth, oct=oct,
                                                         metal_ind=metal_ind, modifier=modifier)
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


# ################# Possibly Needed - ox_ utilities
def generate_metal_ox_autocorrelations(oxmodifier, mol, loud, depth=4,
                                       oct=True, flag_name=False, metal_ind=None,
                                       use_dist=False, size_normalize=False):
    # # oxmodifier - dict, used to modify prop vector (e.g. for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #    oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    metal_ox_ac = metal_only_autocorrelation(mol, 'ox_nuclear_charge', depth, oct=oct,
                                             modifier=oxmodifier, metal_ind=metal_ind,
                                             use_dist=use_dist, size_normalize=size_normalize)
    this_colnames = []
    for i in range(0, depth + 1):
        this_colnames.append('O' + '-' + str(i))
    colnames.append(this_colnames)
    result.append(metal_ox_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_autocorrelation_derivatives(oxmodifier, mol, loud, depth=4, oct=True, flag_name=False,
                                                  metal_ind=None):
    # # oxmodifier - dict, used to modify prop vector (e.g. for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #    oct - bool, if complex is octahedral, will use better bond checks
    result = None
    colnames = []
    metal_ox_ac = metal_only_autocorrelation_derivative(mol, 'ox_nuclear_charge', depth, oct=oct,
                                                        modifier=oxmodifier, metal_ind=metal_ind)
    for i in range(0, depth + 1):
        colnames.append(['d' + 'O' + '-' + str(i) + '/d' + 'O' + str(j) for j in range(0, mol.natoms)])
    result = metal_ox_ac
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_deltametrics(oxmodifier, mol, loud, depth=4, oct=True,
                                   flag_name=False, metal_ind=None, use_dist=False, size_normalize=False):
    # # oxmodifier - dict, used to modify prop vector (e.g. for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #    oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    metal_ox_ac = metal_only_deltametric(mol, 'ox_nuclear_charge', depth, oct=oct,
                                         metal_ind=metal_ind, modifier=oxmodifier, use_dist=use_dist,
                                         size_normalize=size_normalize)
    this_colnames = []
    for i in range(0, depth + 1):
        this_colnames.append('O' + '-' + str(i))
    colnames.append(this_colnames)
    result.append(metal_ox_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_deltametric_derivatives(oxmodifier, mol, loud, depth=4, oct=True,
                                              flag_name=False, metal_ind=False):
    # # oxmodifier - dict, used to modify prop vector (e.g. for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #    oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    metal_ox_ac = metal_only_deltametric_derivative(mol, 'ox_nuclear_charge',
                                                    depth, oct=oct, modifier=oxmodifier, metal_ind=metal_ind)
    for i in range(0, depth + 1):
        colnames.append(['d' + 'O' + '-' + str(i) + '/d' + 'O' + str(j) for j in range(0, mol.natoms)])

    result = metal_ox_ac
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary
