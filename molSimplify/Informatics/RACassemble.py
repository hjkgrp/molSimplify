# Written JP Janet
# for HJK Group
# Dpt of Chemical Engineering, MIT

# #########################################################
# ####### Defines methods for assembling    ###############
# #######     RACs from lists of ligands    ###############
# #########################################################
from __future__ import print_function

import sys

from typing import Tuple, Union, List
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Informatics.autocorrelation import (generate_all_ligand_autocorrelation_derivatives,
                                                     generate_all_ligand_autocorrelations,
                                                     generate_all_ligand_deltametric_derivatives,
                                                     generate_all_ligand_deltametrics,
                                                     generate_full_complex_autocorrelation_derivatives,
                                                     generate_full_complex_autocorrelations,
                                                     generate_metal_autocorrelation_derivatives,
                                                     generate_metal_autocorrelations,
                                                     generate_metal_deltametric_derivatives,
                                                     generate_metal_deltametrics,
                                                     generate_metal_ox_autocorrelation_derivatives,
                                                     generate_metal_ox_autocorrelations,
                                                     generate_metal_ox_deltametric_derivatives,
                                                     generate_metal_ox_deltametrics)
from molSimplify.Informatics.misc_descriptors import (generate_all_ligand_misc)
from molSimplify.Informatics.rac155_geo import (rac155_list)
import numpy as np


# # Gets the connectivity matrix of an octahedral complex without geo
#  @param metal_mol mol3D() for the metal
#  @param custom_ligand_dict dict defining ligands (see below)
#  @return this_complex  mol3D with correct graph
def assemble_connectivity_from_parts(metal_mol: mol3D, custom_ligand_dict):
    # # custom_ligand_dict.keys() must be eq_ligand_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of ligand classes
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    # start with the connectivity matrix of the whole complex
    n_total = (1 + sum(m.mol.natoms for m in custom_ligand_dict["eq_ligand_list"])
               + sum(m.mol.natoms for m in custom_ligand_dict["ax_ligand_list"]))
    con_mat = np.zeros((n_total, n_total))

    this_complex = mol3D()
    this_complex.copymol3D(metal_mol)
    live_row = 1  # metal goes in row 0

    for i, class_lig in enumerate(custom_ligand_dict["eq_ligand_list"]):
        this_lig = class_lig.mol
        this_dim = this_lig.natoms
        this_con = custom_ligand_dict["eq_con_int_list"][i]
        this_lig.convert2OBMol()
        this_lig.createMolecularGraph()
        con_mat[live_row:live_row + this_dim, live_row:live_row + this_dim] = this_lig.graph
        for con_atoms in this_con:
            con_mat[0, live_row + con_atoms] = 1
            con_mat[live_row + con_atoms, 0] = 1
        live_row = live_row + this_dim
        this_complex.combine(this_lig, [], dirty=True)
    for i, class_lig in enumerate(custom_ligand_dict["ax_ligand_list"]):
        this_lig = class_lig.mol
        this_dim = this_lig.natoms
        this_con = custom_ligand_dict["ax_con_int_list"][i]
        this_lig.convert2OBMol()
        this_lig.createMolecularGraph()
        con_mat[live_row:live_row + this_dim, live_row:live_row + this_dim] = this_lig.graph
        for con_atoms in this_con:
            con_mat[0, live_row + con_atoms] = 1
            con_mat[live_row + con_atoms, 0] = 1
        live_row = live_row + this_dim
        this_complex.combine(this_lig, [], dirty=True)
    if not int(sum(con_mat[:, 0])) == 6:
        print('coordination number is ' + str(sum(con_mat[:, 0])))
        print('error, not oct ')
        sys.exit()
    # overwrite the connectivity matrix
    this_complex.graph = con_mat
    return this_complex


# # Gets the RACs of an octahedral complex without/without geo
#  @param this_complex mol3D() we want RACs for
#  @param custom_ligand_dict optional dict defining ligands (see below)
#  @return descriptor_names updated names
#  @return descriptors updated RACs
def get_descriptor_vector(this_complex, custom_ligand_dict=False,
                          ox_modifier=False, NumB=False, Gval=False):
    descriptor_names = []
    descriptors = []
    # misc descriptors
    results_dictionary = generate_all_ligand_misc(this_complex, loud=False,
                                                  custom_ligand_dict=custom_ligand_dict)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['result_ax'],
                                                       'misc', 'ax')
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['result_eq'],
                                                       'misc', 'eq')

    # full ACs
    results_dictionary = generate_full_complex_autocorrelations(this_complex, depth=3,
                                                                loud=False, flag_name=False,
                                                                modifier=ox_modifier,
                                                                NumB=NumB, Gval=Gval)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['results'],
                                                       'f', 'all')

    # ligand ACs
    results_dictionary = generate_all_ligand_autocorrelations(this_complex, depth=3,
                                                              loud=False, name=False,
                                                              flag_name=False,
                                                              custom_ligand_dict=custom_ligand_dict,
                                                              NumB=NumB, Gval=Gval)
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

    results_dictionary = generate_all_ligand_deltametrics(this_complex, depth=3,
                                                          loud=False, name=False,
                                                          custom_ligand_dict=custom_ligand_dict,
                                                          NumB=NumB, Gval=Gval)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['result_ax_con'],
                                                       'D_lc', 'ax')
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['result_eq_con'],
                                                       'D_lc', 'eq')

    # metal ACs
    results_dictionary = generate_metal_autocorrelations(this_complex, depth=3,
                                                         loud=False, modifier=ox_modifier,
                                                         NumB=NumB, Gval=Gval)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['results'],
                                                       'mc', 'all')

    results_dictionary = generate_metal_deltametrics(this_complex, depth=3,
                                                     loud=False, modifier=ox_modifier,
                                                     NumB=NumB, Gval=Gval)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                       results_dictionary['colnames'],
                                                       results_dictionary['results'],
                                                       'D_mc', 'all')

    # ## ox-metal ACs, if ox available
    if ox_modifier:
        results_dictionary = generate_metal_ox_autocorrelations(ox_modifier, this_complex,
                                                                depth=3, loud=False)
        descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                           results_dictionary['colnames'],
                                                           results_dictionary['results'],
                                                           'mc', 'all')
        results_dictionary = generate_metal_ox_deltametrics(ox_modifier, this_complex,
                                                            depth=3, loud=False)
        descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,
                                                           results_dictionary['colnames'],
                                                           results_dictionary['results'],
                                                           'D_mc', 'all')
    return descriptor_names, descriptors


# # Gets only the graph-based RAC155-RACs of an octahedral complex without/without geo
#  @param this_complex mol3D() we want RACs for
#  @param custom_ligand_dict optional dict defining ligands (see below)
#  @return descriptor_names_rac155 updated names
#  @return descriptors_rac155 updated RACs
def get_rac155_graph_based(this_complex, custom_ligand_dict=False, ox_modifier=False):
    descriptor_names, descriptors = get_descriptor_vector(this_complex,
                                                          custom_ligand_dict=custom_ligand_dict,
                                                          ox_modifier=False)
    descriptor_names_rac155 = []
    descriptors_rac155 = []
    try:
        for i, item in enumerate(descriptor_names):
            if item in rac155_list:
                descriptor_names_rac155.append(item)
                descriptors_rac155.append(descriptors[i])
    except IndexError:
        descriptor_names_rac155, descriptors_rac155 = None, None
        print('Complex could not featurize to all rac155 descriptors')

    return descriptor_names_rac155, descriptors_rac155


# # Gets the derivatives of RACs of an octahedral complex with geo only!
#  @param this_complex mol3D() we want derivatives for
#  @param custom_ligand_dict optional dict defining ligands (see below)
#  @return descriptor_derivative_names matrix of names
#  @return descriptor_derivatives derivatives of racs w.r.t atomic props
def get_descriptor_derivatives(this_complex, custom_ligand_dict=False, ox_modifier=False):
    descriptor_derivative_names = []
    descriptor_derivatives = None
    # cannot do misc descriptors !

    # full ACs
    results_dictionary = generate_full_complex_autocorrelation_derivatives(this_complex, depth=3,
                                                                           loud=False, flag_name=False,
                                                                           modifier=ox_modifier)
    descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                        descriptor_derivatives,
                                                                                        results_dictionary['colnames'],
                                                                                        results_dictionary['results'],
                                                                                        'f', 'all')

    # ligand ACs
    print('getting ligand AC derivatives')
    results_dictionary = generate_all_ligand_autocorrelation_derivatives(this_complex, depth=3,
                                                                         loud=True, name=False,
                                                                         flag_name=False)
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

    results_dictionary = generate_all_ligand_deltametric_derivatives(this_complex, depth=3,
                                                                     loud=False, name=False)
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
    print('getting metal AC derivatives')
    results_dictionary = generate_metal_autocorrelation_derivatives(this_complex, depth=3,
                                                                    loud=False, modifier=ox_modifier)
    descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                        descriptor_derivatives,
                                                                                        results_dictionary['colnames'],
                                                                                        results_dictionary['results'],
                                                                                        'mc', 'all')

    results_dictionary = generate_metal_deltametric_derivatives(this_complex, depth=3,
                                                                loud=False, modifier=ox_modifier)
    descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                        descriptor_derivatives,
                                                                                        results_dictionary['colnames'],
                                                                                        results_dictionary['results'],
                                                                                        'D_mc', 'all')

    # ## ox-metal ACs
    if ox_modifier:
        results_dictionary = generate_metal_ox_autocorrelation_derivatives(ox_modifier, this_complex,
                                                                           depth=3, loud=False)
        descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                            descriptor_derivatives,
                                                                                            results_dictionary['colnames'],
                                                                                            results_dictionary['results'],
                                                                                            'mc', 'all')

        results_dictionary = generate_metal_ox_deltametric_derivatives(ox_modifier, this_complex,
                                                                       depth=3, loud=False)
        descriptor_derivative_names, descriptor_derivatives = append_descriptor_derivatives(descriptor_derivative_names,
                                                                                            descriptor_derivatives,
                                                                                            results_dictionary['colnames'],
                                                                                            results_dictionary['results'],
                                                                                            'D_mc', 'all')

    return descriptor_derivative_names, descriptor_derivatives


# # utility to add one-hot encoded oxidation state
# # and d-electron count to some descriptors
#  @param descriptor_names RAC names, will be appended to
#  @param descriptors RAC, will be appended to
#  @param list_of_names names, will be added
#  @param list_of_props types of RACs
#  @param prefix RAC prefix
#  @param suffix RAC suffix
#  @return descriptor_names updated names
#  @return descriptors updated RACs
def create_OHE(metal: str,
               oxidation_state: Union[int, str]
               ) -> Tuple[List[str], List[int]]:
    # function to append OHE encoding of oxidation state
    # and d-electron countst
    OHE_names = ['ox2', 'ox3', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
    OHE_values = [0] * len(OHE_names)
    if int(oxidation_state) == 2:
        OHE_values[0] += 1
    elif int(oxidation_state) == 3:
        OHE_values[1] += 1
    if metal == "Cr" and int(oxidation_state) == 2:
        OHE_values[OHE_names.index("d4")] += 1
    elif metal == "Cr" and int(oxidation_state) == 3:
        OHE_values[OHE_names.index("d3")] += 1
    elif metal == "Mn" and int(oxidation_state) == 2:
        OHE_values[OHE_names.index("d5")] += 1
    elif metal == "Mn" and int(oxidation_state) == 3:
        OHE_values[OHE_names.index("d4")] += 1
    elif metal == "Fe" and int(oxidation_state) == 2:
        OHE_values[OHE_names.index("d6")] += 1
    elif metal == "Fe" and int(oxidation_state) == 3:
        OHE_values[OHE_names.index("d5")] += 1
    elif metal == "Co" and int(oxidation_state) == 2:
        OHE_values[OHE_names.index("d7")] += 1
    elif metal == "Co" and int(oxidation_state) == 3:
        OHE_values[OHE_names.index("d6")] += 1
    elif metal == "Ni" and int(oxidation_state) == 2:
        OHE_values[OHE_names.index("d8")] += 1
    else:
        raise NotImplementedError('Error: unknown metal and oxidation state '
                                  + str(metal) + '/' + str(oxidation_state))

    return OHE_names, OHE_values


# # utility to build standardly formated RACS
#  @param descriptor_names RAC names, will be appended to
#  @param descriptors RAC, will be appended to
#  @param list_of_names names, will be added
#  @param list_of_props types of RACs
#  @param prefix RAC prefix
#  @param suffix RAC suffix
#  @return descriptor_names updated names
#  @return descriptors updated RACs
def append_descriptors(descriptor_names, descriptors, list_of_names,
                       list_of_props, prefix, suffix):
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


# # utility to build standardly formated RACS derivatives
#  @param descriptor_derivative_names RAC names, will be a matrix!
#  @param descriptor_derivatives RAC, will be appended to
#  @param mat_of_names names, will be added
#  @param dmat mat of RACs
#  @param prefix RAC prefix
#  @param suffix RAC suffix
#  @return descriptor_derivative_names updated names
#  @return descriptor_derivatives updated RACs
def append_descriptor_derivatives(descriptor_derivative_names, descriptor_derivatives,
                                  mat_of_names, dmat, prefix, suffix):

    try:
        basestring
    except NameError:
        basestring = str

    for names in mat_of_names:
        jnames = ["-".join([prefix, str(i), suffix]) for i in names]
        descriptor_derivative_names.append(jnames)
    if descriptor_derivatives is None:
        descriptor_derivatives = dmat
    else:
        descriptor_derivatives = np.row_stack([descriptor_derivatives, dmat])
    return descriptor_derivative_names, descriptor_derivatives
