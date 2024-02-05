# @file nn_prep.py
#  Helper routines for ANN integration
#
#   Written by JP Janet for HJK Group
#
#  Dpt of Chemical Engineering, MIT

import time
import os
from typing import Tuple, List, Dict, Any, Union
from molSimplify.Classes.ligand import ligand
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.atom3D import atom3D
from molSimplify.Scripts.io import lig_load
from molSimplify.Informatics.RACassemble import (assemble_connectivity_from_parts,
                                                 create_OHE)
from molSimplify.Informatics.lacRACAssemble import get_descriptor_vector
from molSimplify.Informatics.decoration_manager import (decorate_ligand)
from molSimplify.utils.tensorflow import tensorflow_silence
from molSimplify.utils.timer import DebugTimer
from molSimplify.python_nn.clf_analysis_tool import lse_trust
from molSimplify.python_nn.tf_ANN import (ANN_supervisor,
                                          find_ANN_10_NN_normalized_latent_dist,
                                          find_ANN_latent_dist,
                                          find_true_min_eu_dist)


def spin_classify(metal: str, spin: Union[int, str], ox: int) -> Tuple[bool, List[int]]:
    metal_spin_dictionary = {'co': {2: 4, 3: 5},
                             'cr': {2: 5, 3: 4},
                             'fe': {2: 5, 3: 6},
                             'mn': {2: 6, 3: 5},
                             'ni': {2: 3}}

    suggest_spin_dictionary = {'co': {2: [2, 4], 3: [1, 5]},
                               'cr': {2: [1, 5], 3: [2, 4]},
                               'fe': {2: [1, 5], 3: [2, 6]},
                               'mn': {2: [2, 6], 3: [1, 5]},
                               'ni': {2: [1, 3]}}

    high_spin = False
    if (int(spin) >= int(metal_spin_dictionary[metal][ox])):
        high_spin = True
    spin_ops = suggest_spin_dictionary[metal][ox]
    return high_spin, spin_ops


# wrapper to get ANN predictions from a known mol3D()
# generally unsafe
def invoke_ANNs_from_mol3d(mol: mol3D, oxidation_state: int,
                           alpha: float = 0.2, debug: bool = False) -> dict:

    tensorflow_silence()

    # check input
    if not oxidation_state == 2 and not oxidation_state == 3:
        print('Error, oxidation state must be 2 or 3')
        return {}

    # find the metal from RACs
    metal = mol.getAtom(mol.findMetal()[0]).symbol()
    # get RACs
    descriptor_names, descriptors = get_descriptor_vector(
        mol, ox_modifier={metal: oxidation_state})
    # get one-hot-encoding (OHE)
    ohe_names, ohe_values = create_OHE(metal.capitalize(), oxidation_state)
    descriptor_names += ohe_names
    descriptors += ohe_values
    # set exchange fraction
    descriptor_names += ['alpha']
    descriptors += [alpha]

    # call ANN for splitting
    split, latent_split = ANN_supervisor(
        'split', descriptors, descriptor_names, debug)

    # call ANN for bond lengths
    to_roman = {2: 'ii', 3: 'iii'}
    r_ls, latent_r_ls = ANN_supervisor(
        f'ls_{to_roman[oxidation_state]}', descriptors, descriptor_names, debug)
    r_hs, latent_r_hs = ANN_supervisor(
        f'hs_{to_roman[oxidation_state]}', descriptors, descriptor_names, debug)

    # ANN distance for splitting
    split_dist = find_true_min_eu_dist("split", descriptors, descriptor_names)

    # compile results and return
    results_dictionary = {"ls_bl": r_ls,
                          "hs_bl": r_hs,
                          "split": split,
                          "distance": split_dist}
    return (results_dictionary)


def tf_check_ligands(ligs: List[str], batslist: List[List[int]],
                     dents: List[int], tcats: List[str],
                     occs: List[int], debug: bool):
    # tests if ligand combination
    # is compatible with the ANN
    # INPUT:
    #   - ligs:  list of mol3D class, ligands
    #   - batslist: list of int, occupations
    #   - dents: list of int, denticity
    #   - tcats: list of int/bool
    # OUTPUT:
    #   - valid: bool
    # tcats controls
    # manual override
    # of connection atoms

    n_ligs = len(ligs)
    if debug:
        print(('nligs ' + str(n_ligs)))
        print(('ligs ' + str(ligs)))
        print(('occs in function  ' + str(occs)))
        print(('tcats in function  ' + str(tcats)))

    axial_ind_list = []
    equatorial_ind_list = []
    axial_ligs = []
    equatorial_ligs = []
    ax_dent = 0
    eq_dent = 0
    eq_tcat = ''
    ax_tcat = ''
    pentadentate = False
    ax_occs = []
    eq_occs = []
    valid = True
    if (set(dents) == set([2])):
        print('triple bidentate case')
        unique_ligs = []
        ucats = []
        unique_dict = {}
        if not (n_ligs) == 3:
            # something unexpected happened!
            valid = False
        for i in range(0, n_ligs):
            this_lig = ligs[i]
            # mulitple points
            if not (this_lig in unique_ligs):
                unique_ligs.append(this_lig)
                ucats.append(tcats[i])
                unique_dict.update({this_lig: 1})
            elif (this_lig in unique_ligs):
                unique_dict.update({this_lig: unique_dict[this_lig] + 1})

        if len(unique_ligs) == 1:
            axial_ligs.append(ligs[0])
            ax_dent = 2
            ax_tcat = tcats[0]
            ax_occs.append(1)
            equatorial_ligs.append(ligs[0])
            eq_dent = 2
            eq_tcat = tcats[0]
            eq_occs.append(2)
        elif len(unique_ligs) == 2:
            for key in list(unique_dict.keys()):
                if unique_dict[key] == 1:
                    axial_ligs.append(key)
                    ax_dent = 2
                    ax_occs.append(1)
                    ax_tcat = tcats[ligs.index(key)]
                elif unique_dict[key] == 2:
                    equatorial_ligs.append(key)
                    eq_dent = 2
                    eq_occs.append(2)
                    eq_tcat = tcats[ligs.index(key)]
        else:
            valid = False
    elif (set(dents) == set([1, 5])):
        print('THIS IS A PENTADENTATE!')
        pentadentate = True
        for i in range(0, n_ligs):
            this_bat = batslist[i]
            this_lig = ligs[i]
            this_dent = dents[i]
            this_occ = occs[i]
            if debug:
                print('\n')
                print(('iteration  ' + str(i)))
                print(('this_lig  ' + str(this_lig)))
                print(('this_dent  ' + str(this_dent)))
                print(('this_occ  ' + str(this_occ)))
                print(('this backbone atom  ' +
                      str(this_bat) + ' from ' + str(batslist)))
            # mulitple points
            if len(this_bat) > 1:
                if debug:
                    print(('adding ' + str(this_lig) + ' to equatorial'))
                equatorial_ligs.append(this_lig)
                eq_dent = 4
                eq_tcat = tcats[i]
                eq_occs.append(1)
                equatorial_ind_list.append(i)
            if debug:
                print(('adding ' + str(this_lig) + ' to axial'))
            axial_ligs.append(this_lig)
            ax_dent = 1
            ax_tcat = tcats[i]
            ax_occs.append(1)
            axial_ind_list.append(i)

    else:
        for i in range(0, n_ligs):
            this_bat = batslist[i]
            this_lig = ligs[i]
            this_dent = dents[i]
            this_occ = occs[i]
            if debug:
                print('\n')
                print(('iteration  ' + str(i)))
                print(('this_lig  ' + str(this_lig)))
                print(('this_dent  ' + str(this_dent)))
                print(('this_occ  ' + str(this_occ)))
                print(('this backbone atom  ' +
                       str(this_bat) + ' from ' + str(batslist)))
            # mulitple points
            if len(this_bat) == 1:
                if (5 in this_bat) or (6 in this_bat):
                    if debug:
                        print(('adding ' + str(this_lig) + ' to axial'))
                    axial_ligs.append(this_lig)
                    ax_dent = this_dent
                    if this_lig not in ['x', 'oxo', 'hydroxyl']:
                        ax_tcat = tcats[i]
                    ax_occs.append(occs[i])
                    axial_ind_list.append(i)
                else:
                    if debug:
                        print(('adding ' + str(this_lig) + ' to equatorial'))
                    equatorial_ligs.append(this_lig)
                    eq_dent = this_dent
                    eq_tcat = tcats[i]
                    eq_occs.append(occs[i])
                    equatorial_ind_list.append(i)
            else:
                equatorial_ligs.append(this_lig)
                eq_dent = this_dent
                eq_tcat = tcats[i]
                eq_occs.append(occs[i])
                equatorial_ind_list.append(i)
    if (len(axial_ligs) > 2):
        print(('axial lig error : ', axial_ligs, ax_dent, ax_tcat, ax_occs))
        valid = False
    if debug:
        print(('eq occupations  ' + str(eq_occs)))
        print(('eq dent   ' + str(eq_dent)))
    if not (4.0 / (float(eq_dent) * sum(eq_occs)) == 1):
        print(('equatorial ligs error: ', equatorial_ligs, eq_dent, eq_tcat))
        valid = False
    if valid and len(axial_ind_list) == 0:  # get the index position in ligs
        axial_ind_list = [ligs.index(ax_lig) for ax_lig in axial_ligs]
    if valid and len(equatorial_ind_list) == 0:  # get the index position in ligs
        equatorial_ind_list = [ligs.index(eq_lig)
                               for eq_lig in equatorial_ligs]

    return (valid, axial_ligs, equatorial_ligs, ax_dent, eq_dent, ax_tcat, eq_tcat,
            axial_ind_list, equatorial_ind_list, ax_occs, eq_occs, pentadentate)


def check_metal(metal: str, oxidation_state: str) -> Tuple[bool, str]:
    supported_metal_dict = {"fe": [2, 3], "mn": [2, 3], "cr": [2, 3],
                            "co": [2, 3], "ni": [2]}
    romans = {'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5', 'VI': '6'}
    if oxidation_state in list(romans.keys()):
        oxidation_state = romans[oxidation_state]
    outcome = False
    if metal in list(supported_metal_dict):
        #        print('metal in',supported_metal_dict[metal])
        if int(oxidation_state) in supported_metal_dict[metal]:
            outcome = True
    return outcome, oxidation_state


def tf_ANN_preproc(metal: str, oxstate, spin, ligs: List[str], occs: List[int], dents: List[int],
                   batslist: List[List[int]], tcats: List[str],
                   licores: dict, decoration, decoration_index, exchange, geometry: str = "oct",
                   debug: bool = False) -> Tuple[bool, str, dict, bool]:
    # prepares and runs ANN calculation

    start_time = time.perf_counter()

    ######################
    ANN_reason = 'None'
    ANN_attributes: Dict[str, Any] = {}
    ######################

    emsg = list()
    valid = True
    catalysis = False
    # Set default oxidation state variables
    ox = 0
    this_metal = metal.lower()
    if len(this_metal) > 2:
        this_metal = this_metal[0:2]
    newligs = []
    newcats = []
    newdents = []
    newoccs = []
    newdecs = [False] * 6
    newdec_inds = [[0]] * 6
    count = -1
    for i, lig in enumerate(ligs):
        this_occ = occs[i]
        if debug:
            print(('working on lig: ' + str(lig)))
            print(('occ is  ' + str(this_occ)))
        for j in range(0, int(this_occ)):
            count += 1
            newligs.append(lig)
            newdents.append(dents[i])
            newcats.append(tcats[i])
            newoccs.append(1)
            if decoration:
                newdecs[count] = (decoration[i])
                newdec_inds[count] = (decoration_index[i])

    ligs = newligs
    dents = newdents
    tcats = newcats
    occs = newoccs
    if debug:
        print('tf_nn has finished prepping ligands')

    if not geometry == "oct":
        emsg.append(
            "[ANN] Geometry is not supported at this time, MUST give -geometry = oct if you want an ANN prediction.")
        valid = False
        ANN_reason = 'geometry not octahedral'
        return valid, ANN_reason, ANN_attributes, catalysis
    if not oxstate:
        emsg.append("\n oxidation state must be given for an ANN prediction.")
        valid = False
        ANN_reason = 'oxstate not given'
        return valid, ANN_reason, ANN_attributes, catalysis
    if valid:
        oxidation_state = oxstate
        valid, oxidation_state = check_metal(this_metal, oxidation_state)
        if debug:
            print(f'valid after running check_metal? {valid}')
        if int(oxidation_state) in [3, 4, 5]:
            catalytic_moieties = ['oxo', 'x', 'hydroxyl', '[O--]', '[OH-]']
            if debug:
                print(('the ligands are', ligs))
                print((set(ligs).intersection(set(catalytic_moieties))))
            if len(set(ligs).intersection(set(catalytic_moieties))) > 0:
                catalysis = True
        # generate key in descriptor space
        ox = int(oxidation_state)
        if debug:
            print(('metal is ' + str(this_metal)))
            print(('metal validity', valid))
    if not valid and not catalysis:
        emsg.append("\n The only metals that are supported are Fe, Mn, Cr, Co, and Ni")
        emsg.append("\n Oxidation state not available for this metal")
        ANN_reason = 'metal / oxidation state combination not available'
        return valid, ANN_reason, ANN_attributes, catalysis
    if valid:
        try:
            spin_classify(this_metal, spin, ox)
        except KeyError:
            valid = False

    if emsg:
        print((str(" ".join(["ANN messages:"] + [str(i) for i in emsg]))))

    if not valid and not catalysis:
        emsg.append("\n this spin state not available for this metal")
        ANN_reason = 'spin state not available for metal'
        return valid, ANN_reason, ANN_attributes, catalysis
    # Else, i.e. valid or catalysis
    (valid, axial_ligs, equatorial_ligs, ax_dent, eq_dent, ax_tcat, eq_tcat, axial_ind_list,
        equatorial_ind_list, ax_occs, eq_occs, pentadentate) = tf_check_ligands(
        ligs, batslist, dents, tcats, occs, debug)

    if debug:
        print(("ligand validity is  " + str(valid)))
        print(('Occs', occs))
        print(('Ligands', ligs))
        print(('Dents', dents))
        print(('Bats (backbone atoms)', batslist))
        print(('lig validity', valid))
        print(('ax ligs', axial_ligs))
        print(('eq ligs', equatorial_ligs))
        print(('spin is', spin))

    if catalysis:
        valid = False
        if debug:
            print('tf_nn detects catalytic')
        ANN_reason = 'catalytic structure presented'

    net_lig_charge = 0
    if not valid and not catalysis:
        ANN_reason = 'found incorrect ligand symmetry'
        # or, an invalid metal, oxidation state, spin state combination was used
        return valid, ANN_reason, ANN_attributes, catalysis
    # Else, i.e. valid or catalysis
    ax_ligands_list = list()
    eq_ligands_list = list()
    if debug:
        print('loading axial ligands')
    for ii, axl in enumerate(axial_ligs):
        ax_lig3D, r_emsg = lig_load(axl, licores)  # load ligand
        net_lig_charge += ax_lig3D.charge
        if r_emsg:
            emsg += r_emsg
        if ax_tcat:
            ax_lig3D.cat = ax_tcat
            if debug:
                print(('custom ax connect atom given (0-ind) ' + str(ax_tcat)))
        if pentadentate and len(ax_lig3D.cat) > 1:
            ax_lig3D.cat = [ax_lig3D.cat[-1]]
        this_lig = ligand(mol3D(), [], ax_dent)
        this_lig.mol = ax_lig3D

        # check decoration index
        if newdecs:
            if newdecs[axial_ind_list[ii]]:
                print(('decorating ' + str(axl) + ' with ' + str(newdecs[axial_ind_list[ii]]) + ' at sites ' + str(
                    newdec_inds[axial_ind_list[ii]])))
                ax_lig3D = decorate_ligand(
                    axl, newdecs[axial_ind_list[ii]], newdec_inds[axial_ind_list[ii]], debug)
        ax_lig3D.convert2mol3D()  # mol3D representation of ligand
        for jj in range(0, ax_occs[ii]):
            ax_ligands_list.append(this_lig)
    if debug:
        print(f'Obtained the net ligand charge, which is... {net_lig_charge}')
        print('ax_ligands_list:')
        print(ax_ligands_list)
        print([h.mol.cat for h in ax_ligands_list])

    if debug:
        print(('loading equatorial ligands ' + str(equatorial_ligs)))
    for ii, eql in enumerate(equatorial_ligs):
        eq_lig3D, r_emsg = lig_load(eql, licores)  # load ligand
        net_lig_charge += eq_lig3D.charge
        if r_emsg:
            emsg += r_emsg
        if eq_tcat:
            eq_lig3D.cat = eq_tcat
            if debug:
                print(('custom eq connect atom given (0-ind) ' + str(eq_tcat)))
        if pentadentate and len(eq_lig3D.cat) > 1:
            eq_lig3D.cat = eq_lig3D.cat[0:4]

        if newdecs:
            if debug:
                print(('newdecs' + str(newdecs)))
                print(('equatorial_ind_list is ' + str(equatorial_ind_list)))
            c = 0
            if newdecs[equatorial_ind_list[ii]]:
                if debug:
                    print(('decorating ' + str(eql) + ' with ' + str(
                        newdecs[equatorial_ind_list[ii]]) + ' at sites ' + str(newdec_inds[equatorial_ind_list[ii]])))
                eq_lig3D = decorate_ligand(eql, newdecs[equatorial_ind_list[ii]],
                                           newdec_inds[equatorial_ind_list[ii]], debug)
                c += 1

        eq_lig3D.convert2mol3D()  # mol3D representation of ligand
        this_lig = ligand(mol3D(), [], eq_dent)
        this_lig.mol = eq_lig3D

        for jj in range(0, eq_occs[ii]):
            eq_ligands_list.append(this_lig)
    if debug:
        print('eq_ligands_list:')
        print(eq_ligands_list)
        print(
            ('writing copies of ligands as used  in ANN to currrent dir : ' + os.getcwd()))
        for kk, l in enumerate(ax_ligands_list):
            l.mol.writexyz('axlig-' + str(kk) + '.xyz')
        for kk, l in enumerate(eq_ligands_list):
            l.mol.writexyz('eqlig-' + str(kk) + '.xyz')
    # make description of complex
    custom_ligand_dict = {"eq_ligand_list": eq_ligands_list,
                          "ax_ligand_list": ax_ligands_list,
                          "eq_con_int_list": [h.mol.cat for h in eq_ligands_list],
                          "ax_con_int_list": [h.mol.cat for h in ax_ligands_list]}

    # placeholder for metal
    metal_mol = mol3D()
    metal_mol.addAtom(atom3D(metal))
    this_complex = assemble_connectivity_from_parts(
        metal_mol, custom_ligand_dict)

    if debug:
        print('custom_ligand_dict is : ')
        print(custom_ligand_dict)

    if debug:
        print(('finished checking ligands, valid is ' + str(valid)))
        print('assembling RAC custom ligand configuration dictionary')

    if valid:
        ANN_attributes = evaluate_tmc_anns(this_complex, this_metal, ox, spin,
                                           custom_ligand_dict, net_lig_charge,
                                           exchange, ligs, equatorial_ind_list,
                                           eq_occs, axial_ind_list, ax_occs,
                                           debug=debug)
        total_ANN_time = time.perf_counter() - start_time
        print(f'Total ML functions took {total_ANN_time:.2f} seconds')

    if catalysis:
        print('-----In Catalysis Mode-----')
        ANN_attributes = evaluate_catalytic_anns(this_complex, this_metal, ox, spin,
                                                 custom_ligand_dict, net_lig_charge,
                                                 exchange, debug=debug)
        total_ANN_time = time.perf_counter() - start_time
        print(f'Total Catalysis ML functions took {total_ANN_time:.2f} seconds')

    if not valid and not ANN_reason and not catalysis:
        ANN_reason = ' uncaught rejection (see sdout/stderr)'

    return valid, ANN_reason, ANN_attributes, catalysis


def evaluate_tmc_anns(this_complex: mol3D, metal: str, ox: int, spin: int,
                      custom_ligand_dict: Dict[str, list],
                      net_lig_charge: int, exchange: Union[str, float, int],
                      ligs, equatorial_ind_list, eq_occs, axial_ind_list, ax_occs,
                      debug: bool = False) -> Dict[str, Any]:
    ANN_attributes: Dict[str, Any] = {}
    high_spin, spin_ops = spin_classify(metal, spin, ox)
    # =====Classifiers:=====
    _descriptor_names = ["oxstate", "spinmult", "charge_lig"]
    _descriptors = [ox, spin, net_lig_charge]
    descriptor_names, descriptors = get_descriptor_vector(
        this_complex, custom_ligand_dict, ox_modifier={metal: ox})
    descriptor_names = _descriptor_names + descriptor_names
    descriptors = _descriptors + descriptors
    flag_oct, geo_lse = ANN_supervisor(
        "geo_static_clf", descriptors, descriptor_names, debug=debug)
    # Test for scikit-learn models
    # flag_oct, geo_lse = sklearn_supervisor("geo_static_clf", descriptors, descriptor_names, debug=False)
    sc_pred, sc_lse = ANN_supervisor(
        "sc_static_clf", descriptors, descriptor_names, debug=debug)
    ANN_attributes.update({"geo_label": 0 if flag_oct[0, 0] <= 0.5 else 1,
                           "geo_prob": flag_oct[0, 0],
                           "geo_LSE": geo_lse[0],
                           "geo_label_trust": lse_trust(geo_lse),
                           "sc_label": 0 if sc_pred[0, 0] <= 0.5 else 1,
                           "sc_prob": sc_pred[0, 0],
                           "sc_LSE": sc_lse[0],
                           "sc_label_trust": lse_trust(sc_lse)})

    # build RACs without geo
    with DebugTimer('getting RACs', debug):
        descriptor_names, descriptors = get_descriptor_vector(
            this_complex, custom_ligand_dict, ox_modifier={metal: ox})

        # get one-hot-encoding (OHE)
        ohe_names, ohe_values = create_OHE(metal.capitalize(), ox)
        descriptor_names += ohe_names
        descriptors += ohe_values

        # get alpha
        alpha = 0.2  # default for B3LYP
        if exchange:
            try:
                if float(exchange) > 1:
                    alpha = float(exchange) / 100  # if given as %
                elif float(exchange) <= 1:
                    alpha = float(exchange)
            except ValueError:
                print('cannot cast exchange argument as a float, using 20%')
        descriptor_names += ['alpha']
        descriptors += [alpha]
        descriptor_names += ['ox']
        descriptors += [ox]
        descriptor_names += ['spin']
        descriptors += [spin]

    # get spin splitting:
    with DebugTimer('split ANN', debug):
        split, latent_split = ANN_supervisor(
            'split', descriptors, descriptor_names, debug)

    # get bond lengths:
    with DebugTimer('GEO ANN', debug):
        if ox == 2:
            r_ls, latent_r_ls = ANN_supervisor(
                'ls_ii', descriptors, descriptor_names, debug)
            r_hs, latent_r_hs = ANN_supervisor(
                'hs_ii', descriptors, descriptor_names, debug)
        elif ox == 3:
            r_ls, latent_r_ls = ANN_supervisor(
                'ls_iii', descriptors, descriptor_names, debug)
            r_hs, latent_r_hs = ANN_supervisor(
                'hs_iii', descriptors, descriptor_names, debug)
        else:
            raise ValueError(f'Oxidation state {ox} not available')
    if high_spin:
        r = r_hs[0]
    else:
        r = r_ls[0]

    with DebugTimer('homo ANN', debug):
        homo, latent_homo = ANN_supervisor(
            'homo', descriptors, descriptor_names, debug)

    with DebugTimer('gap ANN', debug):
        gap, latent_gap = ANN_supervisor(
            'gap', descriptors, descriptor_names, debug)

    # get minimum distance to train (for splitting)
    with DebugTimer('min dist', debug):
        split_dist = find_true_min_eu_dist(
            "split", descriptors, descriptor_names)

    with DebugTimer('min HOMO dist', debug):
        homo_dist = find_true_min_eu_dist(
            "homo", descriptors, descriptor_names)
        homo_dist = find_ANN_latent_dist("homo", latent_homo, debug)

    with DebugTimer('min GAP dist', debug):
        gap_dist = find_true_min_eu_dist("gap", descriptors, descriptor_names)
        gap_dist = find_ANN_latent_dist("gap", latent_gap, debug)

    # save attributes for return
    ANN_attributes.update({'split': split[0][0]})
    ANN_attributes.update({'split_dist': split_dist})
    ANN_attributes.update({'This spin': spin})
    if split[0][0] < 0 and (abs(split[0]) > 5):
        ANN_attributes.update({'ANN_ground_state': spin_ops[1]})
    elif split[0][0] > 0 and (abs(split[0]) > 5):
        ANN_attributes.update({'ANN_ground_state': spin_ops[0]})
    else:
        ANN_attributes.update(
            {'ANN_ground_state': 'dgen ' + str(spin_ops)})

    ANN_attributes.update({'homo': homo[0][0]})
    ANN_attributes.update({'gap': gap[0][0]})
    ANN_attributes.update({'homo_dist': homo_dist})
    ANN_attributes.update({'gap_dist': gap_dist})

    # now that we have bond predictions, we need to map these
    # back to a length of equal size as the original ligand request
    # in order for molSimplify to understand it
    ANN_bondl = len(ligs) * [0.]
    added = 0
    for ii, eql in enumerate(equatorial_ind_list):
        for jj in range(0, eq_occs[ii]):
            ANN_bondl[added] = r[2]
            added += 1

    for ii, axl in enumerate(axial_ind_list):
        if debug:
            print((ii, axl, added, ax_occs))
        for jj in range(0, ax_occs[ii]):
            if debug:
                print((jj, axl, added, r[ii]))
            ANN_bondl[added] = r[ii]
            added += 1

    ANN_attributes.update({'ANN_bondl': 4 * [r[2]] + [r[0], r[1]]})

    HOMO_ANN_trust = 'not set'
    HOMO_ANN_trust_message = ""
    # Not quite sure if this should be divided by 3 or not, since RAC-155 descriptors
    if float(homo_dist) < 3:
        HOMO_ANN_trust_message = 'ANN results should be trustworthy for this complex '
        HOMO_ANN_trust = 'high'
    elif float(homo_dist) < 5:
        HOMO_ANN_trust_message = 'ANN results are probably useful for this complex '
        HOMO_ANN_trust = 'medium'
    elif float(homo_dist) <= 10:
        HOMO_ANN_trust_message = 'ANN results are fairly far from training data, be cautious '
        HOMO_ANN_trust = 'low'
    elif float(homo_dist) > 10:
        HOMO_ANN_trust_message = 'ANN results are too far from training data, be cautious '
        HOMO_ANN_trust = 'very low'
    ANN_attributes.update({'homo_trust': HOMO_ANN_trust})
    ANN_attributes.update({'gap_trust': HOMO_ANN_trust})

    ANN_trust = 'not set'
    ANN_trust_message = ""
    if float(split_dist / 3) < 0.25:
        ANN_trust_message = 'ANN results should be trustworthy for this complex '
        ANN_trust = 'high'
    elif float(split_dist / 3) < 0.75:
        ANN_trust_message = 'ANN results are probably useful for this complex '
        ANN_trust = 'medium'
    elif float(split_dist / 3) < 1.0:
        ANN_trust_message = 'ANN results are fairly far from training data, be cautious '
        ANN_trust = 'low'
    elif float(split_dist / 3) > 1.0:
        ANN_trust_message = 'ANN results are too far from training data, be cautious '
        ANN_trust = 'very low'
    ANN_attributes.update({'split_trust': ANN_trust})

    # print text to std out
    print("******************************************************************")
    print("************** ANN is engaged and advising on spin ***************")
    print("************** and metal-ligand bond distances    ****************")
    print("******************************************************************")
    if high_spin:
        print(f'You have selected a high-spin state, multiplicity = {spin}')
    else:
        print(f'You have selected a low-spin state, multiplicity = {spin}')
    # report to stdout
    if split[0] < 0 and not high_spin:
        if abs(split[0]) > 5:
            print('warning, ANN predicts a high spin ground state for this complex')
        else:
            print(
                'warning, ANN predicts a near degenerate ground state for this complex')
    elif split[0] >= 0 and high_spin:
        if abs(split[0]) > 5:
            print('warning, ANN predicts a low spin ground state for this complex')
        else:
            print(
                'warning, ANN predicts a near degenerate ground state for this complex')
    # print(f'delta is {float(split[0]):.2f} spin is {high_spin}')
    print(f"ANN predicts a spin splitting (HS - LS) of {float(split[0]):.2f} kcal/mol at {100 * alpha:.0f}% HFX")
    print(('ANN low spin bond length (ax1/ax2/eq) is predicted to be: ' + " /".join(
        ["{0:.2f}".format(float(i)) for i in r_ls[0]]) + ' angstrom'))
    print(('ANN high spin bond length (ax1/ax2/eq) is predicted to be: ' + " /".join(
        ["{0:.2f}".format(float(i)) for i in r_hs[0]]) + ' angstrom'))
    print(f'distance to splitting energy training data is {split_dist:.2f}')
    print(ANN_trust_message)
    print(f"ANN predicts a HOMO value of {float(homo[0]):.2f} eV at {100 * alpha:.0f}% HFX")
    print(f"ANN predicts a LUMO-HOMO energetic gap value of {float(gap[0]):.2f} eV at {100 * alpha:.0f}% HFX")
    print(HOMO_ANN_trust_message)
    print(f'distance to HOMO training data is {homo_dist:.2f}')
    print(f'distance to GAP training data is {gap_dist:.2f}')
    print("*******************************************************************")
    print("************** ANN complete, saved in record file *****************")
    print("*******************************************************************")
    from tensorflow.keras import backend as K
    # This is done to get rid of the attribute error that is a bug in tensorflow.
    K.clear_session()
    return ANN_attributes


def evaluate_catalytic_anns(this_complex: mol3D, metal: str, ox: int, spin: int,
                            custom_ligand_dict: Dict[str, list],
                            net_lig_charge: int, exchange: Union[str, float, int],
                            debug: bool = False) -> Dict[str, Any]:
    ANN_attributes: Dict[str, Any] = {}
    # build RACs without geo
    with DebugTimer('getting RACs', debug):
        descriptor_names, descriptors = get_descriptor_vector(
            this_complex, custom_ligand_dict, ox_modifier={metal: ox})
        # get alpha
        alpha = 20.0  # default for B3LYP
        if exchange:
            try:
                if float(exchange) < 1:
                    alpha = float(exchange) * 100  # if given as %
                elif float(exchange) >= 1:
                    alpha = float(exchange)
            except ValueError:
                print('cannot cast exchange argument to float, using 20%')
        descriptor_names += ['alpha', 'ox', 'spin', 'charge_lig']
        descriptors += [alpha, ox, spin, net_lig_charge]

    with DebugTimer('split ANN', debug):
        oxo, latent_oxo = ANN_supervisor('oxo', descriptors, descriptor_names, debug)

    with DebugTimer('min oxo dist', debug):
        oxo_dist, avg_10_NN_dist, avg_traintrain = find_ANN_10_NN_normalized_latent_dist("oxo", latent_oxo, debug)

    ANN_attributes.update({'oxo': oxo[0][0]})
    ANN_attributes.update({'oxo_dist': oxo_dist})

    with DebugTimer('HAT ANN', debug):
        hat, latent_hat = ANN_supervisor('hat', descriptors, descriptor_names, debug)

    with DebugTimer('min hat dist', debug):
        hat_dist, avg_10_NN_dist, avg_traintrain = find_ANN_10_NN_normalized_latent_dist("hat", latent_hat, debug)

    ANN_attributes.update({'hat': hat[0][0]})
    ANN_attributes.update({'hat_dist': hat_dist})

    # ######### for Oxo and HOMO optimization ##########
    with DebugTimer('oxo20 ANN', debug):
        oxo20, latent_oxo20 = ANN_supervisor('oxo20', descriptors, descriptor_names, debug)

    with DebugTimer('min oxo20 dist', debug):
        oxo20_dist, avg_10_NN_dist, avg_traintrain = find_ANN_10_NN_normalized_latent_dist("oxo20", latent_oxo20, debug)

    ANN_attributes.update({'oxo20': oxo20[0][0]})
    ANN_attributes.update({'oxo20_dist': oxo20_dist})
    # _ = find_ANN_latent_dist("oxo20", latent_oxo20, debug)
    # _ = find_true_min_eu_dist("oxo20", descriptors, descriptor_names, latent_space_vector=latent_oxo20)

    with DebugTimer('home_empty ANN', debug):
        homo_empty, latent_homo_empty = ANN_supervisor('homo_empty', descriptors, descriptor_names, debug)

    with DebugTimer('min homo_empty_dist', debug):
        homo_empty_dist, _, _ = find_ANN_10_NN_normalized_latent_dist("homo_empty", latent_homo_empty, debug)

    ANN_attributes.update({'homo_empty': homo_empty[0][0]})
    ANN_attributes.update({'homo_empty_dist': homo_empty_dist})
    # _ = find_ANN_latent_dist("homo_empty", latent_homo_empty, debug)
    # _ = find_true_min_eu_dist("homo_empty", descriptors, descriptor_names, latent_space_vector=latent_homo_empty)

    Oxo20_ANN_trust = 'not set'
    Oxo20_ANN_trust_message = ""
    # Not quite sure if this should be divided by 3 or not, since RAC-155 descriptors
    if float(oxo20_dist) < 0.75:
        Oxo20_ANN_trust_message = 'Oxo20 ANN results should be trustworthy for this complex '
        Oxo20_ANN_trust = 'high'
    elif float(oxo20_dist) < 1:
        Oxo20_ANN_trust_message = 'Oxo20 ANN results are probably useful for this complex '
        Oxo20_ANN_trust = 'medium'
    elif float(oxo20_dist) <= 1.25:
        Oxo20_ANN_trust_message = 'Oxo20 ANN results are fairly far from training data, be cautious '
        Oxo20_ANN_trust = 'low'
    elif float(oxo20_dist) > 1.25:
        Oxo20_ANN_trust_message = 'Oxo20 ANN results are too far from training data, be cautious '
        Oxo20_ANN_trust = 'very low'
    ANN_attributes.update({'oxo20_trust': Oxo20_ANN_trust})

    homo_empty_ANN_trust = 'not set'
    homo_empty_ANN_trust_message = ""
    # Not quite sure if this should be divided by 3 or not, since RAC-155 descriptors
    if float(homo_empty_dist) < 0.75:
        homo_empty_ANN_trust_message = 'homo_empty ANN results should be trustworthy for this complex '
        homo_empty_ANN_trust = 'high'
    elif float(homo_empty_dist) < 1:
        homo_empty_ANN_trust_message = 'homo_empty ANN results are probably useful for this complex '
        homo_empty_ANN_trust = 'medium'
    elif float(homo_empty_dist) <= 1.25:
        homo_empty_ANN_trust_message = 'homo_empty ANN results are fairly far from training data, be cautious '
        homo_empty_ANN_trust = 'low'
    elif float(homo_empty_dist) > 1.25:
        homo_empty_ANN_trust_message = 'homo_empty ANN results are too far from training data, be cautious '
        homo_empty_ANN_trust = 'very low'
    ANN_attributes.update({'homo_empty_trust': homo_empty_ANN_trust})

    ####################################################

    Oxo_ANN_trust = 'not set'
    Oxo_ANN_trust_message = ""
    # Not quite sure if this should be divided by 3 or not, since RAC-155 descriptors
    if float(oxo_dist) < 3:
        Oxo_ANN_trust_message = 'Oxo ANN results should be trustworthy for this complex '
        Oxo_ANN_trust = 'high'
    elif float(oxo_dist) < 5:
        Oxo_ANN_trust_message = 'Oxo ANN results are probably useful for this complex '
        Oxo_ANN_trust = 'medium'
    elif float(oxo_dist) <= 10:
        Oxo_ANN_trust_message = 'Oxo ANN results are fairly far from training data, be cautious '
        Oxo_ANN_trust = 'low'
    elif float(oxo_dist) > 10:
        Oxo_ANN_trust_message = 'Oxo ANN results are too far from training data, be cautious '
        Oxo_ANN_trust = 'very low'
    ANN_attributes.update({'oxo_trust': Oxo_ANN_trust})

    HAT_ANN_trust = 'not set'
    HAT_ANN_trust_message = ""
    # Not quite sure if this should be divided by 3 or not, since RAC-155 descriptors
    if float(hat_dist) < 3:
        HAT_ANN_trust_message = 'HAT ANN results should be trustworthy for this complex '
        HAT_ANN_trust = 'high'
    elif float(hat_dist) < 5:
        HAT_ANN_trust_message = 'HAT ANN results are probably useful for this complex '
        HAT_ANN_trust = 'medium'
    elif float(hat_dist) <= 10:
        HAT_ANN_trust_message = 'HAT ANN results are fairly far from training data, be cautious '
        HAT_ANN_trust = 'low'
    elif float(hat_dist) > 10:
        HAT_ANN_trust_message = 'HAT ANN results are too far from training data, be cautious '
        HAT_ANN_trust = 'very low'
    ANN_attributes.update({'hat_trust': HAT_ANN_trust})
    print("*******************************************************************")
    print("**************       CATALYTIC ANN ACTIVATED!      ****************")
    print("*********** Currently advising on Oxo and HAT energies ************")
    print("*******************************************************************")
    print(f"ANN predicts a Oxo20 energy of {float(oxo20[0]):.2f} kcal/mol at {alpha:.2f}% HFX")
    print(Oxo20_ANN_trust_message)
    print(f'Distance to Oxo20 training data in the latent space is {oxo20_dist:.2f}')
    print(f"ANN predicts a empty site beta HOMO level of {float(homo_empty[0]):.2f} eV at {alpha:.2f} % HFX")
    print(homo_empty_ANN_trust_message)
    print(f'Distance to empty site beta HOMO level training data in the latent space is {homo_empty_dist:.2f}')
    print('-------------------------------------------------------------------')
    print(f"ANN predicts a oxo formation energy of {float(oxo[0]):.2f} kcal/mol at {alpha:.2f}% HFX")
    print(Oxo_ANN_trust_message)
    print(f'Distance to oxo training data in the latent space is {oxo_dist:.2f}')
    print(f"ANN predicts a HAT energy of {float(hat[0]):.2f} kcal/mol at {alpha:.2f}% HFX")
    print(HAT_ANN_trust_message)
    print(f'Distance to HAT training data in the latent space is {hat_dist:.2f}')
    print("*******************************************************************")
    print("************** ANN complete, saved in record file *****************")
    print("*******************************************************************")
    from tensorflow.keras import backend as K
    # This is done to get rid of the attribute error that is a bug in tensorflow.
    K.clear_session()
    return ANN_attributes
