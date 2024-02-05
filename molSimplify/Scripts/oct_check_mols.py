import copy
import os
import re
import sys
# import time
import numpy as np

from molSimplify.Classes.atom3D import (atom3D,
                                        globalvars)
from molSimplify.Classes.globalvars import dict_oct_check_loose, dict_oct_check_st, dict_oneempty_check_st, \
    oct_angle_ref, oneempty_angle_ref
from molSimplify.Classes.ligand import ligand_breakdown
from molSimplify.Classes.ligand import (mol3D)
from molSimplify.Informatics.graph_analyze import obtain_truncation_metal
from molSimplify.Scripts.geometry import vecangle, distance, kabsch


# from openpyxl import load_workbook
# from openpyxl import Workbook
# import json

# Check whether the complex form an octahedral.
# Written by Chenru Duan
### upload: 3/7/2018
### update: 4/5/2018


# input: a xyz file
# output: a mol3D object.
def create_mol_with_xyz(_file_in):
    my_mol = mol3D()
    my_mol.readfromxyz(_file_in)
    return my_mol


def readfromtxt(mol, txt):
    # print('!!!!', filename)
    globs = globalvars()
    en_dict = globs.endict()
    mol.graph = []
    for line in txt:
        line_split = line.split()
        if len(line_split) == 4 and line_split[0]:
            # this looks for unique atom IDs in files
            lm = re.search(r'\d+$', line_split[0])
            # if the string ends in digits m will be a Match object, or None otherwise.
            if lm is not None:
                symb = re.sub(r'\d+', '', line_split[0])
                # number = lm.group()
                # # print('sym and number ' +str(symb) + ' ' + str(number))
                # globs = globalvars()
                atom = atom3D(symb, [float(line_split[1]), float(line_split[2]), float(line_split[3])],
                              name=line_split[0])
            elif line_split[0] in list(en_dict.keys()):
                atom = atom3D(line_split[0], [float(line_split[1]), float(
                    line_split[2]), float(line_split[3])])
            else:
                print('cannot find atom type')
                sys.exit()
            mol.addAtom(atom)
    return mol


def find_nearest_ind(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# deprecated
def comp_two_angle_array_(input_angle, target_angle, catoms_map, picked):
    _angs = input_angle[1][:]
    angs = copy.copy(_angs)
    # print("_angs: ", _angs)
    # print('target_angle', input_angle, target_angle)
    del_act = []
    output_angle, output_ind = [], []
    for idx, ele in enumerate(target_angle):
        del_arr = []
        for _idx, _ele in enumerate(_angs):
            del_arr.append([abs(ele - _ele), _idx, _ele])
        del_arr.sort()
        posi = del_arr[0][1]
        _angs.pop(posi)
        # print('!!!input_a:', input_angle[1])
        del_act.append(del_arr[0][0])
        output_angle.append(del_arr[0][2])
    # output_ind = [find_nearest_ind(angs, x) for x in output_angle]
    output_ind = []
    for x in output_angle:
        ind = find_nearest_ind(angs, x)
        output_ind.append(ind)
        angs[ind] = -1
    max_del_angle = max(del_act)
    sum_del = sum(del_act) / len(target_angle)
    return output_angle, output_ind, sum_del, max_del_angle


def comp_two_angle_array(input_angle, target_angle, catoms_map, picked_inds):
    '''
    input_angle: a list of angle (n_catom_candidates)
    target_angle: a list of angle (n_catom, < n_catom_candidates)
    catoms_map: a map of {atom_ind_in_mol, ind_in_angle_list}
    picked_inds: atom indexes (in the angle list) that have been picked already.
    '''
    _angs = input_angle[1][:]
    angs = copy.copy(_angs)
    picked_angles = [angs[x] for x in picked_inds]
    picked_inds_rev = [x for _, x in sorted(zip(picked_angles, picked_inds), reverse=True)]
    _target_angle = copy.copy(target_angle)
    # print("_angs: ", _angs)
    # print('target_angle', target_angle)
    del_act = []
    output_angle, output_ind = [], []
    for ind in picked_inds_rev:
        target_ind = find_nearest_ind(_target_angle, input_angle[1][:][ind])
        # print(_target_angle[target_ind], angs[ind])
        del_act.append(abs(_target_angle[target_ind]-angs[ind]))
        output_angle.append(angs[ind])
        _target_angle.pop(target_ind)
    for ind in sorted(picked_inds, reverse=True):
        _angs.pop(ind)
    for idx, ele in enumerate(_target_angle):
        del_arr = []
        for _idx, _ele in enumerate(_angs):
            del_arr.append([abs(ele - _ele), _idx, _ele])
        del_arr.sort()
        posi = del_arr[0][1]
        _angs.pop(posi)
        # print('!!!input_a:', input_angle[1])
        del_act.append(del_arr[0][0])
        output_angle.append(del_arr[0][2])
    # print("del_act: ", del_act)
    # output_ind = [find_nearest_ind(angs, x) for x in output_angle]
    output_ind = []
    for x in output_angle:
        ind = find_nearest_ind(angs, x)
        output_ind.append(ind)
        angs[ind] = -1
    max_del_angle = max(del_act)
    sum_del = sum(del_act) / len(target_angle)
    return output_angle, output_ind, sum_del, max_del_angle


# description: Given the target_angle, choose the input_angle that has
# the smallest angle deviation in input_array. In this process,
# the input_angle has already been filtered.
# input:       input_arr: array of input_angle, target_angle: array of angle you want.
# output:      output_angle: the input_angle that has the smallest deviation compared
# to target_arr. del_angle: the deviation. catoms: the connecting antom for
# the output_angle.
def comp_angle_pick_one_best(input_arr, target_angle, catoms_map, picked):
    '''
    Given the target_angle, choose the input_angle that has the smallest angle deviation
    in input_array.
    input_arr: array of input angles.
    target_angle: array of target angles.
    catoms_map: a map of {atom_ind_in_mol, ind_in_angle_list}.
    picked: atom indexes (in mol3D) that have been picked already.
    '''
    del_arr = []
    picked_inds = [catoms_map[x] for x in picked]
    # print("==============")
    # print("picked: ", picked, picked_inds)
    # print("input_arr", input_arr, len(input_arr))
    for ii, input_angle in enumerate(input_arr):
        out_angle, output_ind, sum_del, max_del_angle = comp_two_angle_array(
            input_angle, target_angle, catoms_map, picked_inds)
        del_arr.append([sum_del, ii, max_del_angle, out_angle, output_ind])
    del_arr.sort()
    # print("del_arr", del_arr)
    idx = 0
    posi = del_arr[idx][1]
    del_angle = del_arr[idx][0]
    output_angle = input_arr[posi][1]
    catoms = input_arr[posi][0]
    max_del_sig_angle = del_arr[idx][2]
    # print('!!!!posi', input_arr[posi])
    # print('!!!!out:', del_angle, del_arr[0][3])
    input_arr.pop(posi)
    return output_angle, del_angle, catoms, max_del_sig_angle


# description: Loop the target_angle in target_arr.
# output: three lists corresponding to the outputs of comp_angle_pick_one_best.
def loop_target_angle_arr(input_arr, target_arr, catoms_map):
    output_arr = []
    sum_del = []
    catoms_arr = []
    max_del_sig_angle_arr = []
    for idx, ele in enumerate(target_arr):
        output_angle, del_angle, catoms, max_del_sig_angle = comp_angle_pick_one_best(
            input_arr, ele, catoms_map, picked=catoms_arr)
        # print("catoms: ", catoms)
        # print("max_del_sig_angle: ", max_del_sig_angle)
        output_arr.append(output_angle)
        sum_del.append(del_angle)
        catoms_arr.append(catoms)
        max_del_sig_angle_arr.append(max_del_sig_angle)
    # print("!!!!", catoms_arr, target_arr)
    return output_arr, sum_del, catoms_arr, max(max_del_sig_angle_arr)


def sort_sec_ele(ele):
    return ele[1]


# standard 1: number of coordination for metal.
# For octehegral, if num_coord_metal < 6, it will be considered as
# problemetic. For num_coord_metal > 6, it might be caused by the
# ambiguity of C-N ring in the getBondedAtom function. We will use
# other metric (oct_comp) to see whether some of these catoms can
# construct an Oct.
# input: a xyz file
# output: coordination number for metal, and their indexes.
def get_num_coord_metal(file_in, debug=False):
    my_mol = create_mol_with_xyz(_file_in=file_in)
    metal_ind = my_mol.findMetal()[0]
    metal_coord = my_mol.getAtomCoords(metal_ind)
    catoms = my_mol.getBondedAtomsOct(ind=metal_ind)
    if debug:
        print(('metal coordinate:', metal_coord))
        print(('coordinations: ', catoms, len(catoms)))
    # standard 1: number of coordination for metal.
    num_coord_metal = len(catoms)
    return num_coord_metal, catoms


# standard 2: ligand changes (rmsd, max atom distance)
# for each ligand in the complex. Need input for
# original gemoetry.
# input: optimized and original xyz file.
# output: two scalar of maximum rmsd for ligands, and the
# maximum distance change in ligands.
def ligand_comp_org(file_in, file_init_geo, catoms_arr, flag_deleteH=True, flag_loose=False,
                    flag_lbd=True, debug=False, depth=3, BondedOct=False):
    liglist, liglist_init, flag_match = match_lig_list(file_in, file_init_geo,
                                                       catoms_arr,
                                                       flag_loose, flag_lbd,
                                                       debug=debug, depth=depth,
                                                       BondedOct=BondedOct)
    if debug:
        print(('lig_list:', liglist, len(liglist)))
        print(('lig_list_init:', liglist_init, len(liglist_init)))
    if flag_lbd:
        mymol_xyz = 'mymol_trunc_tmp.xyz'
        initmol_xyz = 'init_trunc_tmp.xyz'
    else:
        mymol_xyz = file_in
        initmol_xyz = file_init_geo
    if flag_match:
        rmsd_arr, max_atom_dist_arr = [], []
        for idx, lig in enumerate(liglist):
            lig_init = liglist_init[idx]
            if debug:
                print(('----This is %d th piece of ligand.' % (idx + 1)))
                print(('ligand is:', lig, lig_init))
            posi_shift = 2
            # Create mol3D without a tmp file.
            # _start = time.clock()
            with open(mymol_xyz, 'r') as fo:
                foo = []
                for ii, line in enumerate(fo):
                    if (ii - posi_shift) in lig:
                        if debug:
                            print(('line is', line))
                        foo.append(line)
            tmp_mol = mol3D()
            tmp_mol = readfromtxt(tmp_mol, foo)
            with open(initmol_xyz, 'r') as fo:
                foo = []
                for ii, line in enumerate(fo):
                    if (ii - posi_shift) in lig_init:
                        if debug:
                            print(('line is', line))
                        foo.append(line)
            tmp_org_mol = mol3D()
            tmp_org_mol = readfromtxt(tmp_org_mol, foo)
            # _elapsed = (time.clock() - _start)
            # print('-reading txt:', _elapsed)
            if debug:
                print(('# atoms: %d, init: %d' %
                       (tmp_mol.natoms, tmp_org_mol.natoms)))
                print(('!!!!atoms:', [x.symbol() for x in tmp_mol.getAtoms()],
                       [x.symbol() for x in tmp_org_mol.getAtoms()]))
            if flag_deleteH:
                tmp_mol.deleteHs()
                tmp_org_mol.deleteHs()
            _ = kabsch(tmp_org_mol, tmp_mol)
            rmsd = tmp_mol.rmsd(tmp_org_mol)
            rmsd_arr.append(rmsd)
            atom_dist_max = tmp_mol.maxatomdist(tmp_org_mol)
            max_atom_dist_arr.append(atom_dist_max)
            if debug:
                print(('rmsd:', rmsd))
                print(('atom_dist_max', atom_dist_max))
        rmsd_max = max(rmsd_arr)
        atom_dist_max = max(max_atom_dist_arr)
    else:
        rmsd_max, atom_dist_max = 'lig_mismatch', 'lig_mismatch'
    return rmsd_max, atom_dist_max


# Match the ligend list generated by ligand_breakdown.
# useful for cases where the init geo and opt geo have the
# different ligands arrangement (when you only have the opt geo
# but still want init geo)
def match_lig_list(file_in, file_init_geo, catoms_arr,
                   flag_loose, flag_lbd=True, debug=False,
                   depth=3, BondedOct=False):
    flag_match = True
    my_mol = create_mol_with_xyz(_file_in=file_in)
    init_mol = create_mol_with_xyz(_file_in=file_init_geo)
    if flag_lbd:  # Also do ligand breakdown for opt geo
        my_mol_trunc = obtain_truncation_metal(my_mol, depth)
        init_mol_trunc = obtain_truncation_metal(init_mol, depth)
        my_mol_trunc.createMolecularGraph()
        init_mol_trunc.createMolecularGraph()
        init_mol_trunc.writexyz('init_trunc_tmp.xyz')
        my_mol_trunc.writexyz('mymol_trunc_tmp.xyz')
        liglist_init, ligdents_init, ligcons_init = ligand_breakdown(
            init_mol_trunc, BondedOct=BondedOct)
        liglist, ligdents, ligcons = ligand_breakdown(my_mol_trunc, BondedOct=BondedOct)
        liglist_atom = [[my_mol_trunc.getAtom(x).symbol() for x in ele]
                        for ele in liglist]
        liglist_init_atom = [[init_mol_trunc.getAtom(x).symbol() for x in ele]
                             for ele in liglist_init]
        if debug:
            print(('!!!!:', [x.symbol() for x in init_mol_trunc.getAtoms()]))
            print(('liglist_init, ligdents_init, ligcons_init',
                   liglist_init, ligdents_init, ligcons_init))
            # print('liglist, ligdents, ligcons', liglist, ligdents, ligcons)
    else:  # ceate/use the liglist, ligdents, ligcons of initial geo as we just wanna track them down
        # _start = time.clock()
        if debug:
            print('Just inherit the ligand list from init structure.')
        liglist_init, ligdents_init, ligcons_init = ligand_breakdown(init_mol,
                                                                     BondedOct=BondedOct)
        # _elapsed = (time.clock() - _start)
        # print('time on lig_breakdoen:', _elapsed)
        liglist = liglist_init[:]
        liglist_atom = [[my_mol.getAtom(x).symbol() for x in ele]
                        for ele in liglist]
        liglist_init_atom = [[init_mol.getAtom(x).symbol() for x in ele]
                             for ele in liglist_init]

    if debug:
        print(('ligand_list opt in symbols:', liglist_atom))
        print(('ligand_list init in symbols: ', liglist_init_atom))
    liglist_shifted = []
    for ele in liglist_init_atom:
        # posi = liglist_atom.index(ele)
        # liglist_shifted.append(liglist[posi])
        # liglist_atom.pop(posi)
        try:
            _flag = False
            for idx, _ele in enumerate(liglist_atom):
                if set(ele) == set(_ele) and len(ele) == len(_ele):
                    if debug:
                        print(('fragment in liglist_init', ele))
                        print(('fragment in liglist', _ele))
                    posi = idx
                    _flag = True
            liglist_shifted.append(liglist[posi])
            liglist_atom.pop(posi)
            liglist.pop(posi)
            if not _flag:
                if debug:
                    print('Ligands cannot match!')
                flag_match = False
        except AssertionError:
            # To whoever encounters this: Please replace AssertionError
            # with whatever we are actually trying to except. RM 2022/02/17
            print('Ligands cannot match!')
            flag_match = False
    if debug:
        print(('!!!!!returns', liglist_shifted, liglist_init))
    return liglist_shifted, liglist_init, flag_match


def find_the_other_ind(arr, ind):
    arr.pop(arr.index(ind))
    return arr[0]


# The linear ligand here ius defined as:
# 1) ligands that contains only two atoms. For example CO
# 2) ligands with more than two atoms and looks linear itself.
# For example: A-B-C-.... with the angle of A-B and B-C larger than 170.
def is_linear_ligand(mol, ind):
    catoms = mol.getBondedAtomsSmart(ind)
    metal_ind = mol.findMetal()[0]
    flag = False
    if metal_ind in catoms and len(catoms) == 2:
        ind_next = find_the_other_ind(catoms[:], metal_ind)
        _catoms = mol.getBondedAtomsSmart(ind_next)
        if len(_catoms) == 1:
            flag = True
        elif len(_catoms) == 2:
            ind_next2 = find_the_other_ind(_catoms[:], ind)
            vec1 = (np.array(mol.getAtomCoords(ind))
                    - np.array(mol.getAtomCoords(ind_next)))
            vec2 = (np.array(mol.getAtomCoords(ind_next2))
                    - np.array(mol.getAtomCoords(ind_next)))
            ang = vecangle(vec1, vec2)
            if ang > 170:
                flag = True
    # print(flag, catoms)
    return flag, catoms


def get_linear_angle(mol, ind):
    flag, catoms = is_linear_ligand(mol, ind)
    if flag:
        vec1 = np.array(mol.getAtomCoords(
            catoms[0])) - np.array(mol.getAtomCoords(ind))
        vec2 = np.array(mol.getAtomCoords(
            catoms[1])) - np.array(mol.getAtomCoords(ind))
        ang = vecangle(vec1, vec2)
    else:
        ang = 0
    return flag, ang


def check_angle_linear(file_in, catoms_arr):
    mol = create_mol_with_xyz(file_in)
    dict_angle_linear = {}
    for ind in catoms_arr:
        flag, ang = get_linear_angle(mol, ind)
        dict_angle_linear[str(ind)] = [flag, ang]
    dict_orientation = {}
    devi_linear_avrg, devi_linear_max = 0, 0
    count = 0
    for key in dict_angle_linear:
        [flag, ang] = dict_angle_linear[key]
        if flag:
            count += 1
            devi_linear_avrg += 180 - ang
            if (180 - ang) > devi_linear_max:
                devi_linear_max = 180 - ang
    if count:
        devi_linear_avrg /= count
    else:
        devi_linear_avrg = 0
    dict_orientation['devi_linear_avrg'] = devi_linear_avrg
    dict_orientation['devi_linear_max'] = devi_linear_max
    return dict_angle_linear, dict_orientation


# standard 3: catoms structure compared to a perfect octahedral.
# Note: The same logic can be used to judge whether a complex is
# Tetrahedron or panor square.
# input: a xyz file.
# output: the summation for the angle deviation for the closest
# Oct structure. Candidates are from GetBondedAtom function
# for the metal. A metric for the distance deviation from the
# pefect Oct, including the diff in eq ligands, ax ligands and
# eq-ax ligands.
def oct_comp(file_in, angle_ref=oct_angle_ref, catoms_arr=None,
             debug=False):
    my_mol = create_mol_with_xyz(_file_in=file_in)
    num_coord_metal, catoms = get_num_coord_metal(file_in=file_in)
    # metal_ind = my_mol.findMetal()[0]
    metal_coord = my_mol.getAtomCoords(my_mol.findMetal()[0])
    catom_coord = []
    if catoms_arr is not None:
        catoms = catoms_arr
    theta_arr, oct_dist = [], []
    for atom in catoms:
        coord = my_mol.getAtomCoords(atom)
        catom_coord.append(coord)
    th_input_arr = []
    catoms_map = {}
    for idx1, coord1 in enumerate(catom_coord):
        delr1 = (np.array(coord1) - np.array(metal_coord)).tolist()
        catoms_map.update({catoms[idx1]: idx1})
        theta_tmp = []
        for idx2, coord2 in enumerate(catom_coord):
            if idx2 != idx1:
                delr2 = (np.array(coord2) - np.array(metal_coord)).tolist()
                theta = vecangle(delr1, delr2)
                theta_tmp.append(theta)
        th_input_arr.append([catoms[idx1], theta_tmp])
    th_output_arr, sum_del_angle, catoms_arr, max_del_sig_angle = loop_target_angle_arr(
        th_input_arr, angle_ref, catoms_map)
    if debug:
        print(('th:', th_output_arr))
        print(('sum_del:', sum_del_angle))
        print(('catoms_arr:', catoms_arr))
        print(('catoms_type:', [my_mol.getAtom(x).symbol()
                                for x in catoms_arr]))
    for idx, ele in enumerate(th_output_arr):
        theta_arr.append([catoms_arr[idx], sum_del_angle[idx], ele])
    # theta_arr.sort(key=sort_sec_ele)
    # theta_trunc_arr = theta_arr[0:6]
    theta_trunc_arr = theta_arr
    # print('truncated theta array:', theta_trunc_arr)
    theta_trunc_arr_T = list(map(list, list(zip(*theta_trunc_arr))))
    oct_catoms = theta_trunc_arr_T[0]
    oct_angle_devi = theta_trunc_arr_T[1]
    oct_angle_all = theta_trunc_arr_T[2]
    if debug:
        print(('Summation of deviation angle for catoms:', oct_angle_devi))
        print(('Angle for catoms:', oct_angle_all))
    for atom in oct_catoms:
        coord = catom_coord[catoms.index(atom)]
        dist = distance(coord, metal_coord)
        oct_dist.append(dist)
    oct_dist.sort()
    # print('!!!oct_dist:', oct_dist)
    try:  # For Oct
        dist_del_arr = np.array(
            [oct_dist[3] - oct_dist[0], oct_dist[4] - oct_dist[1], oct_dist[5] - oct_dist[2]])
        min_posi = np.argmin(dist_del_arr)
        if min_posi == 0:
            dist_eq, dist_ax = oct_dist[:4], oct_dist[4:]
        elif min_posi == 1:
            dist_eq, dist_ax = oct_dist[1:5], [oct_dist[0], oct_dist[5]]
        else:
            dist_eq, dist_ax = oct_dist[2:], oct_dist[:2]
    except IndexError:  # For one empty site
        if (oct_dist[3] - oct_dist[0]) > (oct_dist[4] - oct_dist[1]):
            dist_ax, dist_eq = oct_dist[:1], oct_dist[1:]  # ax dist is smaller
        else:
            dist_ax, dist_eq = oct_dist[4:], oct_dist[:4]  # eq dist is smaller
    dist_del_all = oct_dist[-1] - oct_dist[0]
    if debug:
        print(('dist:', dist_eq, dist_ax))
    dist_del_eq = max(dist_eq) - min(dist_eq)
    dist_del_ax = max(dist_ax) - min(dist_ax)
    dist_del_eq_ax = max(abs(max(dist_eq) - min(dist_ax)),
                         abs(max(dist_ax) - min(dist_eq)))
    oct_dist_del = [dist_del_eq, dist_del_ax, dist_del_eq_ax, dist_del_all]
    if debug:
        print(('distance difference for catoms to metal (eq, ax, eq_ax):', oct_dist_del))
    return oct_angle_devi, oct_dist_del, max_del_sig_angle, catoms_arr


def dict_check_processing(dict_info, dict_check, std_not_use,
                          num_coord=6, debug=False):
    if debug:
        print(('dict_oct_info', dict_info))
    for ele in std_not_use:
        dict_info[ele] = 'banned_by_user'
    flag_list = []
    for key, values in list(dict_check.items()):
        if not dict_info[key] == 'banned_by_user':
            print((dict_info[key]))
            print(values)
            if dict_info[key] > values:
                flag_list.append(key)
    if dict_info['num_coord_metal'] < num_coord:
        flag_list.append('num_coord_metal')
    if flag_list == ['num_coord_metal'] and \
            (dict_info['num_coord_metal'] == -1 or dict_info['num_coord_metal'] > num_coord):
        dict_info['num_coord_metal'] = num_coord
        flag_list.remove('num_coord_metal')
    if not len(flag_list):
        flag_oct = 1  # good structure
        flag_list = 'None'
    else:
        flag_oct = 0
        flag_list = ', '.join(flag_list)
        print('------bad structure!-----')
        print(('flag_list:', flag_list))
    return flag_oct, flag_list, dict_info


def Oct_inspection(file_in, file_init_geo=None, catoms_arr=None, dict_check=dict_oct_check_st,
                   std_not_use=[], angle_ref=oct_angle_ref, flag_loose=True, flag_lbd=False,
                   dict_check_loose=dict_oct_check_loose, BondedOct=True, debug=False):
    if catoms_arr is None:
        print('Error, must have ctoms! If not, please use IsOct.')
        quit()
    elif len(catoms_arr) != 6:
        print('Error, must have 6 connecting atoms for octahedral.')
        quit()
    num_coord_metal = 6
    oct_angle_devi, oct_dist_del, max_del_sig_angle = [-1, -1], [-1, -1, -1, -1], -1
    rmsd_max, atom_dist_max = -1, -1
    if file_init_geo is not None:
        # print('!!!Inspection,flag_loose:', flag_loose)
        # _start = time.clock()
        rmsd_max, atom_dist_max = ligand_comp_org(file_in, file_init_geo,
                                                  flag_loose=flag_loose,
                                                  flag_lbd=flag_lbd,
                                                  catoms_arr=catoms_arr,
                                                  debug=debug,
                                                  BondedOct=BondedOct)
        # _elapsed = time.clock() - _start
        # print('Time on ligand_comp_org:', _elapsed)
    if not rmsd_max == 'lig_mismatch':
        # _start = time.clock()
        oct_angle_devi, oct_dist_del, max_del_sig_angle, catoms_arr = oct_comp(file_in, angle_ref, catoms_arr,
                                                                               debug=debug)
        # _elapsed = time.clock() - _start
        # print('Time on oct_comp:', _elapsed)
    else:
        num_coord_metal = -1
        rmsd_max, atom_dist_max = -1, -1
        print('!!!!!Should always match. WRONG!!!!!')
        quit()
    dict_angle_linear, dict_orientation = check_angle_linear(
        file_in, catoms_arr)
    if debug:
        print('-------This is for the linear ligand orientation test-----')
        print(('!!!catoms_arr:', catoms_arr))
        print(('!!!dict_angle_linear', dict_angle_linear))
        print(('!!!dict_orientation', dict_orientation))
        print('---------orientation end.------------')
    dict_oct_info = {}
    dict_oct_info['num_coord_metal'] = num_coord_metal
    dict_oct_info['rmsd_max'] = rmsd_max
    dict_oct_info['atom_dist_max'] = atom_dist_max
    dict_oct_info['oct_angle_devi_max'] = max(oct_angle_devi)
    dict_oct_info['max_del_sig_angle'] = max_del_sig_angle
    dict_oct_info['dist_del_eq'] = oct_dist_del[0]
    dict_oct_info['dist_del_ax'] = oct_dist_del[1]
    dict_oct_info['dist_del_eq_ax'] = oct_dist_del[2]
    dict_oct_info['dist_del_all'] = oct_dist_del[3]
    dict_oct_info.update(dict_orientation)
    flag_oct, flag_list, dict_oct_info = dict_check_processing(dict_oct_info,
                                                               dict_check=dict_check,
                                                               std_not_use=std_not_use,
                                                               num_coord=6, debug=debug)
    flag_oct_loose, flag_list_loose, __ = dict_check_processing(dict_oct_info,
                                                                dict_check=dict_check_loose,
                                                                std_not_use=std_not_use,
                                                                num_coord=6, debug=debug)
    return flag_oct, flag_list, dict_oct_info, flag_oct_loose, flag_list_loose


# See whether a complex is Oct or not.
# output: flag: 1 is good and 0 is bad
# flag_list: if structure is bad, which test it fails
# dict_oct_info: values for each metric we check.
def IsOct(file_in, file_init_geo=None, dict_check=dict_oct_check_st,
          std_not_use=[], angle_ref=oct_angle_ref, flag_catoms=False,
          catoms_arr=None, debug=False):
    num_coord_metal, catoms = get_num_coord_metal(file_in, debug=debug)
    if catoms_arr is not None:
        catoms = catoms_arr
        num_coord_metal = len(catoms_arr)

    oct_angle_devi, oct_dist_del, max_del_sig_angle = [
                                                          -1, -1], [-1, -1, -1, -1], -1
    rmsd_max, atom_dist_max = -1, -1
    catoms_arr = catoms
    dict_orientation = {'devi_linear_max': -1, 'devi_linear_avrg': -1}
    if num_coord_metal >= 6:
        if not rmsd_max == 'lig_mismatch':
            num_coord_metal = 6
            oct_angle_devi, oct_dist_del, max_del_sig_angle, catoms_arr = oct_comp(file_in, angle_ref,
                                                                                   catoms_arr, debug=debug)
        if file_init_geo is not None:
            rmsd_max, atom_dist_max = ligand_comp_org(
                file_in, file_init_geo, catoms_arr, debug=debug)
        else:
            # num_coord_metal = -1
            rmsd_max, atom_dist_max = -1, -1
        dict_angle_linear, dict_orientation = check_angle_linear(
            file_in, catoms_arr)
        if debug:
            print('-------This is for the linear ligand orientation test-----')
            print(('!!!catoms_arr:', catoms_arr))
            print(('!!!dict_angle_linear', dict_angle_linear))
            print(('!!!dict_orientation', dict_orientation))
            print('---------orientation end.------------')
    dict_oct_info = {}
    dict_oct_info['num_coord_metal'] = num_coord_metal
    dict_oct_info['rmsd_max'] = rmsd_max
    dict_oct_info['atom_dist_max'] = atom_dist_max
    dict_oct_info['oct_angle_devi_max'] = max(oct_angle_devi)
    dict_oct_info['max_del_sig_angle'] = max_del_sig_angle
    dict_oct_info['dist_del_eq'] = oct_dist_del[0]
    dict_oct_info['dist_del_all'] = oct_dist_del[3]
    dict_oct_info.update(dict_orientation)
    flag_oct, flag_list, dict_oct_info = dict_check_processing(dict_oct_info, dict_check,
                                                               std_not_use, num_coord=6,
                                                               debug=debug)
    if not flag_catoms:
        return flag_oct, flag_list, dict_oct_info
    else:
        return flag_oct, flag_list, dict_oct_info, catoms_arr


def IsStructure(file_in, file_init_geo=None, dict_check=dict_oneempty_check_st,
                std_not_use=[], angle_ref=oneempty_angle_ref, num_coord=5,
                flag_catoms=False, debug=False):
    num_coord_metal, catoms = get_num_coord_metal(file_in, debug=debug)

    struct_angle_devi, struct_dist_del, max_del_sig_angle = [
                                                                -1, -1], [-1, -1, -1, -1], -1
    rmsd_max, atom_dist_max = -1, -1
    dict_orientation = {'devi_linear_max': -1, 'devi_linear_avrg': -1}
    if num_coord_metal >= num_coord:
        struct_angle_devi, struct_dist_del, max_del_sig_angle, catoms_arr = oct_comp(file_in, angle_ref,
                                                                                     debug=debug)
        if file_init_geo is not None:
            rmsd_max, atom_dist_max = ligand_comp_org(
                file_in, file_init_geo, catoms_arr, debug=debug)
        else:
            rmsd_max, atom_dist_max = -1, -1
        dict_angle_linear, dict_orientation = check_angle_linear(
            file_in, catoms_arr)
        if debug:
            print('-------This is for the linear ligand orientation test-----')
            print(('!!!catoms_arr:', catoms_arr))
            print(('!!!dict_angle_linear', dict_angle_linear))
            print(('!!!dict_orientation', dict_orientation))
            print('---------orientation end.------------')
    dict_struct_info = {}
    dict_struct_info['num_coord_metal'] = num_coord_metal
    dict_struct_info['rmsd_max'] = rmsd_max
    dict_struct_info['atom_dist_max'] = atom_dist_max
    dict_struct_info['oct_angle_devi_max'] = max(struct_angle_devi)
    dict_struct_info['max_del_sig_angle'] = max_del_sig_angle
    dict_struct_info['dist_del_eq'] = struct_dist_del[0]
    dict_struct_info['dist_del_all'] = struct_dist_del[3]
    dict_struct_info.update(dict_orientation)
    flag_struct, flag_list, dict_struct_info = dict_check_processing(dict_struct_info, dict_check,
                                                                     std_not_use, num_coord=num_coord,
                                                                     debug=debug)
    if not flag_catoms:
        return flag_struct, flag_list, dict_struct_info
    else:
        return flag_struct, flag_list, dict_struct_info, catoms_arr


# input: _path: path for opt geo
# path_init_geo: path for init geo
# output: list of info: ['unique_num', 'oxstate', 'spinmult', 'flag_oct', 'flag_list', 'dict_oct_info']
def loop_structure(_path, path_init_geo):
    charac = ['unique_num', 'oxstate', 'spinmult', 'flag_oct', 'flag_list']
    info_tot = []
    for dirpath, dirs, files in sorted(os.walk(_path)):
        for name in sorted(files):
            if name.split('.')[1] == 'xyz':
                unique_num, oxstate, spinmult = name.split('_')[0], name.split('_')[
                                                                        4][1:], name.split('_')[5][2:]
                print((unique_num, oxstate, spinmult))
                file_in = '%s/%s' % (dirpath, name)
                # ---You may add the function to find initial geo here.
                file_init_geo = find_file_with_unique_num(
                    path_init_geo, unique_num)
                # file_init_geo = gen_file_with_name(path_init_geo, name)
                print(('!!!!file info:!!!!!', file_in, file_init_geo))
                if os.path.exists(file_init_geo):
                    flag_oct, flag_list, dict_oct_info = IsOct(
                        file_in, file_init_geo)
                else:
                    print(('No init_geo!', name))
                    flag_oct, flag_list, dict_oct_info = IsOct(file_in)
                # dict_oct_info = json.dumps(dict_oct_info) # dictionary to string
                _c, _dict = [], []
                for key, value in list(dict_oct_info.items()):
                    _c.append(key)
                    _dict.append(value)
                info = [unique_num, oxstate, spinmult,
                        flag_oct, flag_list] + _dict
                info_tot.append(info)
    charac += _c  # append dict_oct_info
    # write_list_2_xlsx(charac, info_tot)
    return charac, info_tot


# find the file name by matching unique_ID and spin mulplicity
def find_file_with_unique_num(_path, unique_num):
    filename = None
    for dirpath, dirs, files in sorted(os.walk(_path)):
        for name in sorted(files):
            if name.split('_')[0] == str(unique_num):
                filename = '%s/%s' % (dirpath, name)
                break
    return filename


# Only use this function when they have roughly have the same naming.
def gen_file_with_name(path_init_geo, name_opt):
    name_opt = name_opt.split('_')
    name_opt = '_'.join(name_opt[:len(name_opt) - 1])
    name_init = '%s/%s_mols.xyz' % (path_init_geo, name_opt)  # noqa F841 (WIP)
