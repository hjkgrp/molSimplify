from molSimplify.Classes.ligand import ligand_breakdown, ligand_assign
from molSimplify.Informatics.graph_analyze import (get_lig_EN,
                                                   get_truncated_kier,
                                                   kier)
import numpy as np


def generate_all_ligand_misc(mol, loud, custom_ligand_dict=False, force_legacy=False):
    # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    ##                                    ax_con_int_list ,eq_con_int_list
    # with types: eq/ax_ligand_list list of mol3D
    # eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    # use force_legacy to get kier indices/MCDL
    result_ax = list()
    result_eq = list()
    if force_legacy:
        colnames = ['dent', 'maxDEN', 'ki', 'tki', 'charge']
    else:
        colnames = ['dent', 'charge']
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True) # Complex is assumed to be octahedral
        ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list, eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list = ligand_assign(
            mol, liglist, ligdents, ligcons, loud, name=False)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    # count ligands
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)
    # allocate
    result_ax_dent = False
    result_eq_dent = False
    result_ax_maxdelen = False
    result_eq_maxdelen = False
    result_ax_ki = False
    result_eq_ki = False
    result_ax_tki = False
    result_eq_tki = False
    result_ax_charge = False
    result_eq_charge = False
    # loop over axial ligands
    if n_ax > 0:
        for i in range(0, n_ax):
            ax_ligand_list[i].mol.convert2OBMol()
            if not (i == 0):
                result_ax_dent += ax_ligand_list[i].dent
                if force_legacy:
                    result_ax_maxdelen += get_lig_EN(
                        ax_ligand_list[i].mol, ax_con_int_list[i])
                    result_ax_ki += kier(ax_ligand_list[i].mol)
                    result_ax_tki += get_truncated_kier(
                        ax_ligand_list[i].mol, ax_con_int_list[i])
                result_ax_charge += ax_ligand_list[i].mol.OBMol.GetTotalCharge()
            else:
                result_ax_dent = ax_ligand_list[i].dent
                if force_legacy:
                    result_ax_maxdelen = get_lig_EN(
                        ax_ligand_list[i].mol, ax_con_int_list[i])
                    result_ax_ki = kier(ax_ligand_list[i].mol)
                    result_ax_tki = get_truncated_kier(
                        ax_ligand_list[i].mol, ax_con_int_list[i])
                result_ax_charge = ax_ligand_list[i].mol.OBMol.GetTotalCharge()
        # average axial results
        result_ax_dent = np.divide(result_ax_dent, n_ax)
        if force_legacy:
            result_ax_maxdelen = np.divide(result_ax_maxdelen, n_ax)
            result_ax_ki = np.divide(result_ax_ki, n_ax)
            result_ax_tki = np.divide(result_ax_tki, n_ax)
        result_ax_charge = np.divide(result_ax_charge, n_ax)

    # loop over eq ligands
    if n_eq > 0:
        for i in range(0, n_eq):
            eq_ligand_list[i].mol.convert2OBMol()
            if not (i == 0):
                result_eq_dent += eq_ligand_list[i].dent
                if force_legacy:
                    result_eq_maxdelen += get_lig_EN(
                        eq_ligand_list[i].mol, eq_con_int_list[i])
                    result_eq_ki += kier(eq_ligand_list[i].mol)
                    result_eq_tki += get_truncated_kier(
                        eq_ligand_list[i].mol, eq_con_int_list[i])
                result_eq_charge += eq_ligand_list[i].mol.OBMol.GetTotalCharge()
            else:
                result_eq_dent = eq_ligand_list[i].dent
                if force_legacy:
                    result_eq_maxdelen = get_lig_EN(
                        eq_ligand_list[i].mol, eq_con_int_list[i])
                    result_eq_ki = kier(eq_ligand_list[i].mol)
                    result_eq_tki = get_truncated_kier(
                        eq_ligand_list[i].mol, eq_con_int_list[i])
                result_eq_charge = eq_ligand_list[i].mol.OBMol.GetTotalCharge()
        # average eq results
        result_eq_dent = np.divide(result_eq_dent, n_eq)
        if force_legacy:
            result_eq_maxdelen = np.divide(result_eq_maxdelen, n_eq)
            result_eq_ki = np.divide(result_eq_ki, n_eq)
            result_eq_tki = np.divide(result_eq_tki, n_eq)
        result_eq_charge = np.divide(result_eq_charge, n_eq)
        # save the results
    result_ax.append(result_ax_dent)
    if force_legacy:
        result_ax.append(result_ax_maxdelen)
        result_ax.append(result_ax_ki)
        result_ax.append(result_ax_tki)
    result_ax.append(result_ax_charge)

    result_eq.append(result_eq_dent)
    if force_legacy:
        result_eq.append(result_eq_maxdelen)
        result_eq.append(result_eq_ki)
        result_eq.append(result_eq_tki)
    result_eq.append(result_eq_charge)

    results_dictionary = {'colnames': colnames,
                          'result_ax': result_ax, 'result_eq': result_eq}
    return results_dictionary


def generate_all_ligand_misc_dimers(mol, loud, custom_ligand_dict=False):
    # custom_ligand_dict.keys() must be ax1_ligands_list, ax2_ligand_list, ax3_ligand_list
    ##                                    ax1_con_int_list , ax2_con_int_list, ax3_con_int_list
    # with types: ax_{i}_ligand_list list of mol3D
    # ax_{i}_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    result_ax1 = list()
    result_ax2 = list()
    result_ax3 = list()
    result_axs = [result_ax1, result_ax2, result_ax3]
    colnames = ['dent', 'maxDEN', 'ki', 'tki', 'charge']
    if not custom_ligand_dict:
        raise ValueError('No custom_ligand_dict provided!')
        #liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True)
        # ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list, eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list = ligand_assign(
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
    # allocate
    '''
        result_ax_dent = False
        result_eq_dent = False
        result_ax_maxdelen = False
        result_eq_maxdelen = False
        result_ax_ki = False
        result_eq_ki = False
        result_ax_tki = False
        result_eq_tki = False
        result_ax_charge = False
        result_eq_charge = False
        '''
    # loop over axial ligands
    assert all([i > 0 for i in n_axs]
               ), 'At least one axis has no ligands. # Ligands: %s' % n_axs
    for ax_ligand_list, ax_con_int_list, n_ax, result_ax in zip(axligs, axcons, n_axs, result_axs):
        for i in range(0, n_ax):
            ax_ligand_list[i].mol.convert2OBMol()
            if i == 0:
                result_ax_dent = ax_ligand_list[i].dent
                result_ax_maxdelen = get_lig_EN(
                    ax_ligand_list[i].mol, ax_con_int_list[i])
                result_ax_ki = kier(ax_ligand_list[i].mol)
                result_ax_tki = get_truncated_kier(
                    ax_ligand_list[i].mol, ax_con_int_list[i])
                result_ax_charge = ax_ligand_list[i].mol.OBMol.GetTotalCharge()
            else:
                result_ax_dent += ax_ligand_list[i].dent
                result_ax_maxdelen += get_lig_EN(
                    ax_ligand_list[i].mol, ax_con_int_list[i])
                result_ax_ki += kier(ax_ligand_list[i].mol)
                result_ax_tki += get_truncated_kier(
                    ax_ligand_list[i].mol, ax_con_int_list[i])
                result_ax_charge += ax_ligand_list[i].mol.OBMol.GetTotalCharge()
        # average axial results
        result_ax_dent = np.divide(result_ax_dent, n_ax)
        result_ax_maxdelen = np.divide(result_ax_maxdelen, n_ax)
        result_ax_ki = np.divide(result_ax_ki, n_ax)
        result_ax_tki = np.divide(result_ax_tki, n_ax)
        result_ax_charge = np.divide(result_ax_charge, n_ax)

        result_ax.append(result_ax_dent)
        result_ax.append(result_ax_maxdelen)
        result_ax.append(result_ax_ki)
        result_ax.append(result_ax_tki)
        result_ax.append(result_ax_charge)
    assert all([len(i) > 0 for i in [result_ax1, result_ax2,
                                     result_ax3]]), 'Some results are empty.'
    results_dictionary = {'colnames': colnames, 'result_ax1': result_ax1,
                          'result_ax2': result_ax2, 'result_ax3': result_ax3}
    return results_dictionary
