# Written JP Janet
# for HJK Group
# Dpt of Chemical Engineering, MIT

# #########################################################
# ####### Defines methods for assembling    ###############
# #######     RACs from lists of ligands    ###############
# #########################################################

# lac: ligand assign consistent

from __future__ import print_function
from molSimplify.Classes.ligand import (
    ligand_assign_consistent,
    ligand_assign_original,
    ligand_breakdown,
    )
from molSimplify.Informatics.autocorrelation import (
    append_descriptor_derivatives,
    append_descriptors,
    atom_only_autocorrelation_derivative,
    atom_only_deltametric_derivative,
    full_autocorrelation_derivative,
    generate_atomonly_autocorrelations,
    generate_atomonly_deltametrics,
    generate_full_complex_autocorrelation_derivatives,
    generate_full_complex_autocorrelations,
    generate_metal_autocorrelation_derivatives,
    generate_metal_autocorrelations,
    generate_metal_deltametric_derivatives,
    generate_metal_deltametrics,
    generate_metal_ox_autocorrelation_derivatives,
    generate_metal_ox_autocorrelations,
    generate_metal_ox_deltametric_derivatives,
    generate_metal_ox_deltametrics,
    )
import numpy as np


def get_descriptor_vector(this_complex,
                          custom_ligand_dict=False,
                          ox_modifier=False,
                          NumB=False, Gval=False,
                          lacRACs=True, loud=False,
                          smiles_charge=False, eq_sym=False,
                          use_dist=False, size_normalize=False,
                          alleq=False, custom_property_dict={},
                          depth=3, non_trivial=False,
                          ):
    """
    Calculate and return all geo-based RACs for a given octahedral TM complex (featurize).

    Parameters
    ----------
        this_complex : mol3D
            Transition metal complex to be featurized.
        custom_ligand_dict : bool, optional
            Custom ligand dictionary to evaluate for complex if passed, by default False.
            Skip the ligand breakdown steps -
            in cases where 3D geo is not correct/formed
            custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list,
            ax_con_int_list, and eq_con_int_list,
            with types: eq/ax_ligand_list list of mol3D
            eq/ax_con_int_list list of list/tuple of int e.g., [[1,2] [1,2]]
        ox_modifier : bool, optional
            dict, used to modify prop vector (e.g., for adding
            ONLY used with ox_nuclear_charge ox or charge)
            {"Fe": 2, "Co": 3} etc., by default False.
        NumB : bool, optional
            Use Number of Bonds as an atomic property, by default False.
        Gval : bool, optional
            Use group number as an atomic property, by default False.
        lacRACs : bool, optional
            Use ligand_assign_consistent (lac) to assign ligands as
            equatorial and axial.
            If False, use ligand_assign_original (older), default True.
        loud : bool, optional
            Print debugging information, by default False.
        smiles_charge : bool, optional
            Use obmol conversion through smiles to assign ligand_misc_charges, by default False.
        eq_sym : bool, optional
            Enforce equatorial plane to have connecting atoms with same symbol. Default is False.
        use_dist : bool, optional
            Whether or not CD-RACs used.
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        alleq : bool, optional
            Whether or not all ligands are equatorial.
        custom_property_dict : dict, optional
            Keys are custom property names (str),
            values are dictionaries mapping atom symbols (str, e.g., "H", "He") to
            the numerical property (float) for that atom.
            If provided, other property RACs (e.g., Z, S, T)
            will not be made.
        depth : int, optional
            The maximum depth of the RACs (how many bonds out the RACs go).
            For example, if set to 3, depths considered will be 0, 1, 2, and 3.
        non_trivial : bool, optional
            Flag to exclude difference RACs of I, and depth zero difference
            RACs. These RACs are always zero. By default False.

    Returns
    -------
        descriptor_names : list of str
            Compiled list of descriptor names.
        descriptors : list of float
            Compiled list of descriptor values.

    """
    if depth < 0:
        raise Exception('depth must be a non-negative integer.')

    metals = this_complex.findMetal()
    if len(metals) != 1:
        raise Exception('Molecule does not have the expected number of transition metals (1).')

    descriptor_names = []
    descriptors = []
    # Generate custom_ligand_dict if one not passed!
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(this_complex, BondedOct=True) # Complex is assumed to be octahedral
        if alleq:
            from molSimplify.Classes.ligand import ligand_assign_alleq
            ax_ligand_list, eq_ligand_list, ax_con_int_list, eq_con_int_list = ligand_assign_alleq(
                this_complex, liglist, ligdents, ligcons)
        else:
            if lacRACs:
                assignment_func = ligand_assign_consistent
            else:
                assignment_func = ligand_assign_original
            ax_ligand_list, eq_ligand_list, _, _, \
                ax_con_int_list, eq_con_int_list, _, _, \
                _ = assignment_func(this_complex, liglist, ligdents, ligcons, loud=loud, eq_sym_match=eq_sym)

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
                                                                flag_name=False,
                                                                modifier=ox_modifier, NumB=NumB,
                                                                Gval=Gval, use_dist=use_dist,
                                                                size_normalize=size_normalize,
                                                                custom_property_dict=custom_property_dict)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['results'],
                                                       'f', 'all')
    # # ligand ACs
    results_dictionary = generate_all_ligand_autocorrelations_lac(this_complex, depth=depth,
                                                              loud=False, flag_name=False,
                                                              custom_ligand_dict=custom_ligand_dict,
                                                              NumB=NumB, Gval=Gval, use_dist=use_dist,
                                                              size_normalize=size_normalize,
                                                              custom_property_dict=custom_property_dict)
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

    results_dictionary = generate_all_ligand_deltametrics_lac(this_complex, depth=depth, loud=False,
                                                          custom_ligand_dict=custom_ligand_dict,
                                                          NumB=NumB, Gval=Gval, use_dist=use_dist,
                                                          size_normalize=size_normalize,
                                                          custom_property_dict=custom_property_dict,
                                                          non_trivial=non_trivial)
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
    results_dictionary = generate_metal_autocorrelations(this_complex, depth=depth,
                                                         modifier=ox_modifier,
                                                         NumB=NumB, Gval=Gval,
                                                         use_dist=use_dist,
                                                         size_normalize=size_normalize,
                                                         custom_property_dict=custom_property_dict)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['results'],
                                                       'mc', 'all')

    results_dictionary = generate_metal_deltametrics(this_complex, depth=depth,
                                                     modifier=ox_modifier,
                                                     NumB=NumB, Gval=Gval,
                                                     use_dist=use_dist,
                                                     size_normalize=size_normalize,
                                                     custom_property_dict=custom_property_dict,
                                                     non_trivial=non_trivial)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['results'],
                                                       'D_mc', 'all')

    # ## ox-metal ACs, if ox available
    if ox_modifier:
        results_dictionary = generate_metal_ox_autocorrelations(ox_modifier, this_complex,
                                                                depth=depth,
                                                                use_dist=use_dist,
                                                                size_normalize=size_normalize)
        descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                           results_dictionary['colnames'],
                                                           results_dictionary['results'],
                                                           'mc', 'all')
        results_dictionary = generate_metal_ox_deltametrics(ox_modifier, this_complex,
                                                            depth=depth,
                                                            use_dist=use_dist,
                                                            size_normalize=size_normalize,
                                                            non_trivial=non_trivial)
        descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                           results_dictionary['colnames'],
                                                           results_dictionary['results'],
                                                           'D_mc', 'all')
    return descriptor_names, descriptors


def get_descriptor_derivatives(this_complex, custom_ligand_dict=False, ox_modifier=False,
                               lacRACs=True, depth=4, loud=False):
    """
    Calculate and return all derivatives of RACs for a given octahedral complex.

    Parameters
    ----------
        this_complex : mol3D
            Transition metal complex to be featurized.
        custom_ligand_dict : bool, optional
            Custom ligand dictionary to evaluate for complex if passed, by default False.
            Skip the ligand breakdown steps -
            in cases where 3D geo is not correct/formed
            custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list,
            ax_con_int_list, and eq_con_int_list,
            with types: eq/ax_ligand_list list of mol3D
            eq/ax_con_int_list list of list/tuple of int e.g., [[1,2] [1,2]]
        ox_modifier : bool, optional
            dict, used to modify prop vector (e.g., for adding
            ONLY used with ox_nuclear_charge ox or charge)
            {"Fe":2, "Co": 3} etc., by default False.
        lacRACs : bool, optional
            Use ligand_assign_consistent (lac) to assign ligands as
            equatorial and axial.
            if False, use ligand_assign_original (older), default True.
        depth : int, optional
            Depth of RACs to calculate, by default 4.
        loud : bool, optional
            Print debugging information, by default False.

    Returns
    -------
        descriptor_derivative_names : list
            Compiled list (matrix) of descriptor derivative names
        descriptor_derivatives : list
            Derivatives of RACs w.r.t atomic props (matrix)

    """
    if not custom_ligand_dict:
        if lacRACs:
            assignment_func = ligand_assign_consistent
        else:
            assignment_func = ligand_assign_original
        liglist, ligdents, ligcons = ligand_breakdown(this_complex, BondedOct=True) # Complex is assumed to be octahedral
        (ax_ligand_list, eq_ligand_list, _, _,
         ax_con_int_list, eq_con_int_list, _, _,
         _) = assignment_func(this_complex, liglist, ligdents, ligcons, loud=loud)
        custom_ligand_dict = {'ax_ligand_list': ax_ligand_list,
                              'eq_ligand_list': eq_ligand_list,
                              'ax_con_int_list': ax_con_int_list,
                              'eq_con_int_list': eq_con_int_list}
    #  cannot do misc descriptors !
    descriptor_derivative_names = []
    descriptor_derivatives = None
    # full ACs
    results_dictionary = generate_full_complex_autocorrelation_derivatives(this_complex, depth=depth,
                                                                           flag_name=False,
                                                                           modifier=ox_modifier)
    descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                        descriptor_derivatives,
                                                                                        results_dictionary['colnames'],
                                                                                        results_dictionary['results'],
                                                                                        'f', 'all')
    # ligand ACs
    results_dictionary = generate_all_ligand_autocorrelation_derivatives_lac(this_complex, depth=depth, loud=False,
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
    results_dictionary = generate_all_ligand_deltametric_derivatives_lac(this_complex, depth=depth, loud=False,
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
    results_dictionary = generate_metal_autocorrelation_derivatives(this_complex, depth=depth, modifier=ox_modifier)
    descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                        descriptor_derivatives,
                                                                                        results_dictionary['colnames'],
                                                                                        results_dictionary['results'],
                                                                                        'mc', 'all')
    results_dictionary = generate_metal_deltametric_derivatives(this_complex, depth=depth, modifier=ox_modifier)
    descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                        descriptor_derivatives,
                                                                                        results_dictionary['colnames'],
                                                                                        results_dictionary['results'],
                                                                                        'D_mc', 'all')
    # ## ox-metal ACs
    if ox_modifier:
        results_dictionary = generate_metal_ox_autocorrelation_derivatives(ox_modifier, this_complex, depth=depth)
        descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                            descriptor_derivatives,
                                                                                            results_dictionary['colnames'],
                                                                                            results_dictionary['results'],
                                                                                            'mc', 'all')

        results_dictionary = generate_metal_ox_deltametric_derivatives(ox_modifier, this_complex, depth=depth)
        descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                            descriptor_derivatives,
                                                                                            results_dictionary['colnames'],
                                                                                            results_dictionary['results'],
                                                                                            'D_mc', 'all')

    return descriptor_derivative_names, descriptor_derivatives


def generate_all_ligand_misc(mol, loud=False, custom_ligand_dict=False, smiles_charge=False):
    """
    Get the ligand_misc_descriptors (axial vs. equatorial
    charge (from OBMol) and denticity).

    custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    ax_con_int_list, eq_con_int_list

    Parameters
    ----------
        mol : mol3D
            Molecule to get the ligand_misc descriptors from.
        loud : bool, optional
            Print debugging information, by default False.
        custom_ligand_dict : bool, optional
            custom_ligand_dictionary if passed, by default False.
        smiles_charge : bool, optional
            Whether or not to use the smiles charge assignment, default is False.

    Returns
    -------
        results_dictionary : dict
            Labels and results of ligand_misc RACs - {'colnames': colnames,
            'result_ax': result_ax, 'result_eq': result_eq}.
            Ax. vs eq. charge (from OBMol) and denticity.

    """
    result_ax = list()
    result_eq = list()
    colnames = ['dent', 'charge']
    if custom_ligand_dict:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        # ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        # eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    else:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True) # Complex is assumed to be octahedral
        ax_ligand_list, eq_ligand_list, _, _, _, _, \
            _, _, _ = ligand_assign_consistent(
                mol, liglist, ligdents, ligcons, loud=loud)

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
            if i == 0:
                result_ax_dent = ax_ligand_list[i].dent
                if smiles_charge:
                    result_ax_charge = ax_ligand_list[i].mol.get_smilesOBmol_charge()
                else:
                    result_ax_charge = ax_ligand_list[i].mol.OBMol.GetTotalCharge()
            else:
                result_ax_dent += ax_ligand_list[i].dent
                if smiles_charge:
                    result_ax_charge += ax_ligand_list[i].mol.get_smilesOBmol_charge()
                else:
                    result_ax_charge += ax_ligand_list[i].mol.OBMol.GetTotalCharge()

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

            if i == 0:
                result_eq_dent = eq_ligand_list[i].dent
                if smiles_charge:
                    result_eq_charge = eq_ligand_list[i].mol.get_smilesOBmol_charge()
                else:
                    result_eq_charge = eq_ligand_list[i].mol.OBMol.GetTotalCharge()
            else:
                result_eq_dent += eq_ligand_list[i].dent
                if smiles_charge:
                    result_eq_charge += eq_ligand_list[i].mol.get_smilesOBmol_charge()
                else:
                    result_eq_charge += eq_ligand_list[i].mol.OBMol.GetTotalCharge()

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


def generate_all_ligand_autocorrelations_lac(mol, loud=False, depth=4, flag_name=False,
                                         custom_ligand_dict=False, NumB=False, Gval=False,
                                         use_dist=False, size_normalize=False, custom_property_dict={}):
    """
    Utility for generating all ligand-based product autocorrelations for a complex.

    Parameters
    ----------
        mol : mol3D
            Molecule to get lc-RACs for.
        loud : bool, optional
            Print debugging information, by default False.
        depth : int, optional
            Depth of RACs to calculate, by default 4.
        flag_name : bool, optional
            Shift RAC names slightly, by default False.
        custom_ligand_dict : bool, optional
            Dict of ligands if passed - see generate_descriptor_vector, by default False.
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False.
        Gval : bool, optional
            Use G value as descriptor property, by default False.
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of scope atom from start atom,
            by default False.
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        custom_property_dict : dict, optional
            Keys are custom property names (str),
            values are dictionaries mapping atom symbols (str, e.g., "H", "He") to
            the numerical property (float) for that atom.
            If provided, other property RACs (e.g., Z, S, T)
            will not be made.

    Returns
    -------
        results_dictionary: dict
            Dictionary of all geo-based ligand product descriptors (both full and connecting atom scopes) -
            {'colnames': colnames, 'result_ax_full': result_ax_full, 'result_eq_full': result_eq_full,
            'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}.

    """
    if custom_ligand_dict:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    else:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True) # Complex is assumed to be octahedral
        (ax_ligand_list, eq_ligand_list, _, _, ax_con_int_list, eq_con_int_list,
         _, _, _) = ligand_assign_consistent(
            mol, liglist, ligdents, ligcons, loud=loud)

    # Count ligands.
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)

    # ############## replaced find_ligand_autocorrelations_oct function here
    # get full ligand AC
    result_ax_full, result_eq_full = [], []
    for i in range(0, n_ax):
        if list(result_ax_full):
            result_ax_full += generate_full_complex_autocorrelations(
                ax_ligand_list[i].mol,
                depth=depth,
                custom_property_dict=custom_property_dict,
                NumB=NumB,
                Gval=Gval,
                use_dist=use_dist,
                size_normalize=size_normalize,
                )['results']
        else:
            results_dict = generate_full_complex_autocorrelations(
                ax_ligand_list[i].mol,
                depth=depth,
                custom_property_dict=custom_property_dict,
                NumB=NumB,
                Gval=Gval,
                use_dist=use_dist,
                size_normalize=size_normalize,
                )
            result_ax_full = results_dict['results']
            result_ax_full = np.array(result_ax_full)
            colnames = results_dict['colnames']

    # Average over the number of axial ligands.
    result_ax_full = np.divide(result_ax_full, n_ax)

    for i in range(0, n_eq):
        if list(result_eq_full):
            result_eq_full += generate_full_complex_autocorrelations(
                eq_ligand_list[i].mol,
                depth=depth,
                custom_property_dict=custom_property_dict,
                NumB=NumB,
                Gval=Gval,
                use_dist=use_dist,
                size_normalize=size_normalize,
                )['results']
        else:
            results_dict = generate_full_complex_autocorrelations(
                eq_ligand_list[i].mol,
                depth=depth,
                custom_property_dict=custom_property_dict,
                NumB=NumB,
                Gval=Gval,
                use_dist=use_dist,
                size_normalize=size_normalize,
                )
            result_eq_full = results_dict['results']
            result_eq_full = np.array(result_eq_full)
            colnames = results_dict['colnames']

    # Average over the number of equatorial ligands.
    result_eq_full = np.divide(result_eq_full, n_eq)

    result_ax_con, result_eq_con = [], []
    for i in range(0, n_ax):
        if list(result_ax_con):
            result_ax_con += generate_atomonly_autocorrelations(
                ax_ligand_list[i].mol,
                ax_con_int_list[i],
                depth=depth,
                custom_property_dict=custom_property_dict,
                NumB=NumB,
                Gval=Gval,
                use_dist=use_dist,
                size_normalize=size_normalize,
                )['results']
        else:
            result_ax_con = generate_atomonly_autocorrelations(
                ax_ligand_list[i].mol,
                ax_con_int_list[i],
                depth=depth,
                custom_property_dict=custom_property_dict,
                NumB=NumB,
                Gval=Gval,
                use_dist=use_dist,
                size_normalize=size_normalize,
                )['results']
            result_ax_con = np.array(result_ax_con)

    # Average over the number of axial ligands.
    result_ax_con = np.divide(result_ax_con, n_ax)

    for i in range(0, n_eq):
        if list(result_eq_con):
            result_eq_con += generate_atomonly_autocorrelations(
                eq_ligand_list[i].mol,
                eq_con_int_list[i],
                depth=depth,
                custom_property_dict=custom_property_dict,
                NumB=NumB,
                Gval=Gval,
                use_dist=use_dist,
                size_normalize=size_normalize,
                )['results']
        else:
            result_eq_con = generate_atomonly_autocorrelations(
                eq_ligand_list[i].mol,
                eq_con_int_list[i],
                depth=depth,
                custom_property_dict=custom_property_dict,
                NumB=NumB,
                Gval=Gval,
                use_dist=use_dist,
                size_normalize=size_normalize,
                )['results']
            result_eq_con = np.array(result_eq_con)

    # Average over the number of equatorial ligands.
    result_eq_con = np.divide(result_eq_con, n_eq)

    result_ax_full = list(result_ax_full)
    result_eq_full = list(result_eq_full)
    result_ax_con = list(result_ax_con)
    result_eq_con = list(result_eq_con)

    if flag_name:
        results_dictionary = {'colnames': colnames, 'result_ax_full_ac': result_ax_full,
                              'result_eq_full_ac': result_eq_full,
                              'result_ax_con_ac': result_ax_con, 'result_eq_con_ac': result_eq_con}
    else:
        results_dictionary = {'colnames': colnames, 'result_ax_full': result_ax_full, 'result_eq_full': result_eq_full,
                              'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}
    return results_dictionary


def generate_all_ligand_autocorrelation_derivatives_lac(mol, loud=False, depth=4, flag_name=False,
                                                    custom_ligand_dict=False, NumB=False, Gval=False):
    """
    Utility for generating all ligand-based autocorrelation derivatives for a complex.

    Parameters
    ----------
        mol : mol3D
            Molecule to get lc-RAC derivatives for.
        loud : bool, optional
            Print debugging information, by default False.
        depth : int, optional
            Depth of RACs to calculate, by default 4.
        flag_name : bool, optional
            Shift RAC names slightly, by default False.
        custom_ligand_dict : bool, optional
            Dict of ligands if passed - see generate_descriptor_vector, by default False.
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False.
        Gval : bool, optional
            Use G value as descriptor property, by default False.

    Returns
    -------
        results_dictionary: dict
            Dictionary of all geo-based ligand product descriptor derivatives
            (both full and connecting atom scopes)
            {'colnames': colnames, 'result_ax_full': result_ax_full, 'result_eq_full': result_eq_full,
            'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}.

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
    if custom_ligand_dict:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    else:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True) # Complex is assumed to be octahedral
        ax_ligand_list, eq_ligand_list, _, _, ax_con_int_list, eq_con_int_list, \
            _, _, _ = ligand_assign_consistent(
                mol, liglist, ligdents, ligcons, loud=loud)

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


def generate_all_ligand_deltametrics_lac(mol, loud=False, depth=4, flag_name=False,
                                     custom_ligand_dict=False, NumB=False, Gval=False,
                                     use_dist=False, size_normalize=False, custom_property_dict={},
                                     non_trivial=False):
    """
    Utility for generating all ligand-based deltametric autocorrelations for a complex.

    Parameters
    ----------
        mol : mol3D
            Molecule to get D_lc-RACs for.
        loud : bool, optional
            Print debugging information, by default False.
        depth : int, optional
            Depth of RACs to calculate, by default 4.
        flag_name : bool, optional
            Shift RAC names slightly, by default False.
        custom_ligand_dict : bool, optional
            Dict of ligands if passed - see generate_descriptor_vector, by default False.
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False.
        Gval : bool, optional
            Use G value as descriptor property, by default False.
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False.
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        custom_property_dict : dict, optional
            Keys are custom property names (str),
            values are dictionaries mapping atom symbols (str, e.g., "H", "He") to
            the numerical property (float) for that atom.
            If provided, other property RACs (e.g., Z, S, T)
            will not be made.
        non_trivial : bool, optional
            Flag to exclude difference RACs of I, and depth zero difference
            RACs. These RACs are always zero. By default False.

    Returns
    -------
        results_dictionary: dict
            Dictionary of all geo-based ligand deltametric descriptors (both full and connecting atom scopes) -
            {'colnames': colnames, 'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}.

    """
    result_ax_con = list()
    result_eq_con = list()

    if custom_ligand_dict:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    else:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True) # Complex is assumed to be octahedral
        (ax_ligand_list, eq_ligand_list, _, _, ax_con_int_list, eq_con_int_list,
         _, _, _) = ligand_assign_consistent(mol, liglist, ligdents, ligcons, loud=loud)

    # count ligands
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)

    ####################
    # get partial ligand AC
    for i in range(0, n_ax):
        if list(result_ax_con):
            result_ax_con += generate_atomonly_deltametrics(
                ax_ligand_list[i].mol,
                ax_con_int_list[i],
                depth=depth,
                custom_property_dict=custom_property_dict,
                NumB=NumB,
                Gval=Gval,
                use_dist=use_dist,
                size_normalize=size_normalize,
                non_trivial=non_trivial,
                )['results']
        else:
            results_dict = generate_atomonly_deltametrics(
                ax_ligand_list[i].mol,
                ax_con_int_list[i],
                depth=depth,
                custom_property_dict=custom_property_dict,
                NumB=NumB,
                Gval=Gval,
                use_dist=use_dist,
                size_normalize=size_normalize,
                non_trivial=non_trivial,
                )
            result_ax_con = results_dict['results']
            result_ax_con = np.array(result_ax_con)
            colnames = results_dict['colnames']

    # Average over the number of axial ligands.
    result_ax_con = np.divide(result_ax_con, n_ax)

    for i in range(0, n_eq):
        if list(result_eq_con):
            result_eq_con += generate_atomonly_deltametrics(
                eq_ligand_list[i].mol,
                eq_con_int_list[i],
                depth=depth,
                custom_property_dict=custom_property_dict,
                NumB=NumB,
                Gval=Gval,
                use_dist=use_dist,
                size_normalize=size_normalize,
                non_trivial=non_trivial,
                )['results']
        else:
            results_dict = generate_atomonly_deltametrics(
                eq_ligand_list[i].mol,
                eq_con_int_list[i],
                depth=depth,
                custom_property_dict=custom_property_dict,
                NumB=NumB,
                Gval=Gval,
                use_dist=use_dist,
                size_normalize=size_normalize,
                non_trivial=non_trivial,
                )
            result_eq_con = results_dict['results']
            result_eq_con = np.array(result_eq_con)
            colnames = results_dict['colnames']

    # Average over the number of equatorial ligands.
    result_eq_con = np.divide(result_eq_con, n_eq)

    result_ax_con = list(result_ax_con)
    result_eq_con = list(result_eq_con)
    ####################

    if flag_name:
        results_dictionary = {'colnames': colnames, 'result_ax_con_del': result_ax_con,
                              'result_eq_con_del': result_eq_con}
    else:
        results_dictionary = {'colnames': colnames, 'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}
    return results_dictionary


def generate_all_ligand_deltametric_derivatives_lac(mol, loud=False, depth=4, flag_name=False,
                                                custom_ligand_dict=False, NumB=False, Gval=False):
    """
    Utility for generating all ligand-based deltametric derivatives for a complex.

    Parameters
    ----------
        mol : mol3D
            Molecule to get lc-RAC deltametric derivatives for.
        loud : bool, optional
            Print debugging information, by default False.
        depth : int, optional
            Depth of RACs to calculate, by default 4.
        flag_name : bool, optional
            Shift RAC names slightly, by default False.
        custom_ligand_dict : bool, optional
            Dict of ligands if passed - see generate_descriptor_vector, by default False.
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False.
        Gval : bool, optional
            Use G value as descriptor property, by default False.

    Returns
    -------
        results_dictionary: dict
            Dictionary of all geo-based ligand deltametric descriptor derivatives
            (both full and connecting atom scopes) -
            {'colnames': colnames, 'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}.

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
    if custom_ligand_dict:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    else:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True) # Complex is assumed to be octahedral
        ax_ligand_list, eq_ligand_list, _, _, ax_con_int_list, eq_con_int_list, \
            _, _, _ = ligand_assign_consistent(
                mol, liglist, ligdents, ligcons, loud=loud)

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
