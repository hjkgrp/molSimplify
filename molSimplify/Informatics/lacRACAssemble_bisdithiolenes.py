from molSimplify.Classes.ligand import ligand_breakdown, ligand
from molSimplify.Classes.globalvars import globalvars
from molSimplify.Informatics.lacRACAssemble import (generate_all_ligand_autocorrelations,
                                                    generate_all_ligand_misc,
                                                    append_descriptors,
                                                    generate_metal_deltametrics,
                                                    generate_full_complex_autocorrelations,
                                                    generate_metal_autocorrelations,
                                                    generate_metal_ox_autocorrelations,
                                                    generate_metal_ox_deltametrics,
                                                    generate_all_ligand_deltametrics
                                                    )

globs = globalvars()


def lig_assign_bisdithiolene(inmol, liglist, ligdents, ligcons):
    """[summary]

    Parameters
    ----------
    inmol : mol3D object
        5-coordinate mol3D object
    liglist : list
        see ligand class
    ligdents : list
        see ligand class
    ligcons : list
        [see ligand class

    Returns
    -------
    ax_ligand_list, eq_ligand_list, ax_con_int_list, eq_con_int_list
        see ligands class

    Raises
    ------
    ValueError
        If unrecognized ligdent.
    """
    ax_ligand_list = list()
    eq_ligand_list = list()
    ax_con_int_list = list()
    eq_con_int_list = list()
    for i, ligand_indices in enumerate(liglist):
        this_ligand = ligand(inmol, ligand_indices, ligdents[i])
        this_ligand.obtain_mol3d()
        if ligdents[i] == 1:
            ax_con = ligcons[i]
            ax_ligand_list.append(this_ligand)
            current_ligand_index_list = this_ligand.index_list
            ax_con_int_list.append([current_ligand_index_list.index(x) for x in ax_con])
        elif ligdents[i] == 2:
            eq_con = ligcons[i]
            eq_ligand_list.append(this_ligand)
            current_ligand_index_list = this_ligand.index_list
            eq_con_int_list.append([current_ligand_index_list.index(x) for x in eq_con])
        else:
            raise ValueError('Ligdent unknown: ' + str(ligdents[i]))
    return ax_ligand_list, eq_ligand_list, ax_con_int_list, eq_con_int_list


def get_descriptor_vector(this_complex, custom_ligand_dict=False,
                          ox_modifier=False, NumB=False, Gval=False,
                          lacRACs=True, loud=False, metal_ind=None,
                          smiles_charge=False, eq_sym=False,
                          use_dist=False, size_normalize=False):
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
            Print degubbging information, by default False
        metal_ind : bool, optional
            index of the metal atom to generate property, by default False
        smiles_charge : bool, optional
            use obmol conversion through smiles to assign ligand_misc_charges, by default False

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
        if lacRACs:
            from molSimplify.Classes.ligand import ligand_assign_consistent as ligand_assign
        else:
            from molSimplify.Classes.ligand import ligand_assign as ligand_assign
        liglist, ligdents, ligcons = ligand_breakdown(this_complex, BondedOct=True) # Complex is assumed to be octahedral
        if sum(ligdents) == 6:
            (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list,
             ax_con_int_list, eq_con_int_list, ax_con_list, eq_con_list,
             built_ligand_list) = ligand_assign(this_complex, liglist,
                                                ligdents, ligcons, loud,
                                                eq_sym_match=eq_sym)
            custom_ligand_dict = {'ax_ligand_list': ax_ligand_list,
                                  'eq_ligand_list': eq_ligand_list,
                                  'ax_con_int_list': ax_con_int_list,
                                  'eq_con_int_list': eq_con_int_list}
        elif sum(ligdents) == 5:
            (ax_ligand_list, eq_ligand_list,
             ax_con_int_list, eq_con_int_list) = lig_assign_bisdithiolene(
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
    results_dictionary = generate_full_complex_autocorrelations(this_complex, depth=3, loud=False,
                                                                flag_name=False, modifier=ox_modifier,
                                                                NumB=NumB, Gval=Gval, use_dist=use_dist,
                                                                size_normalize=size_normalize)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['results'],
                                                       'f', 'all')

    # ligand ACs
    results_dictionary = generate_all_ligand_autocorrelations(this_complex, depth=3, loud=False,
                                                              flag_name=False,
                                                              custom_ligand_dict=custom_ligand_dict,
                                                              NumB=NumB, Gval=Gval, use_dist=use_dist,
                                                              size_normalize=size_normalize)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['result_ax_full'],
                                                       'f', 'ax')
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['result_eq_full'],
                                                       'f', 'eq')
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['result_ax_con'],
                                                       'lc', 'ax')
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['result_eq_con'],
                                                       'lc', 'eq')

    results_dictionary = generate_all_ligand_deltametrics(this_complex, depth=3, loud=False,
                                                          custom_ligand_dict=custom_ligand_dict,
                                                          NumB=NumB, Gval=Gval,
                                                          use_dist=use_dist,
                                                          size_normalize=size_normalize)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['result_ax_con'],
                                                       'D_lc', 'ax')
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['result_eq_con'],
                                                       'D_lc', 'eq')

    # metal ACs
    results_dictionary = generate_metal_autocorrelations(this_complex, depth=3, loud=False,
                                                         modifier=ox_modifier, NumB=NumB,
                                                         Gval=Gval, metal_ind=metal_ind,
                                                         use_dist=use_dist,
                                                         size_normalize=size_normalize)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['results'],
                                                       'mc', 'all')

    results_dictionary = generate_metal_deltametrics(this_complex, depth=3, loud=False,
                                                     modifier=ox_modifier, NumB=NumB,
                                                     Gval=Gval, metal_ind=metal_ind,
                                                     use_dist=use_dist,
                                                     size_normalize=size_normalize)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['results'],
                                                       'D_mc', 'all')

    # ## ox-metal ACs, if ox available
    if ox_modifier:
        results_dictionary = generate_metal_ox_autocorrelations(ox_modifier, this_complex, depth=3,
                                                                loud=False, metal_ind=metal_ind,
                                                                use_dist=use_dist,
                                                                size_normalize=size_normalize)
        descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                           results_dictionary['colnames'],
                                                           results_dictionary['results'],
                                                           'mc', 'all')
        results_dictionary = generate_metal_ox_deltametrics(ox_modifier, this_complex, depth=3,
                                                            loud=False, metal_ind=metal_ind,
                                                            use_dist=use_dist,
                                                            size_normalize=size_normalize)
        descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                           results_dictionary['colnames'],
                                                           results_dictionary['results'],
                                                           'D_mc', 'all')
    return descriptor_names, descriptors
