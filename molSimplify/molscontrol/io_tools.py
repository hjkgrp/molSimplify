import os
import json
import yaml
import numpy as np
import subprocess
from collections import OrderedDict
from sklearn.utils.extmath import randomized_svd
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import ligand_breakdown, ligand_assign_consistent, ligand_assign
from molSimplify.Informatics.RACassemble import assemble_connectivity_from_parts
from molSimplify.Classes.globalvars import dict_oct_check_st

"""
tools for io.
"""


def get_num_frame(geofile):
    with open(geofile, 'r') as fo:
        num_atoms = int(fo.readline().split()[0])
        num_lines = num_atoms + 2  # +2 for the xyz format.
    with open(geofile, 'r') as fo:
        txt = fo.readlines()
    return int(len(txt) / num_lines)


def get_configure(configure_file):
    with open(configure_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_ss_del(outfile, frame):
    _frame_now = -1
    ss_actual, ss_target = 0, 0
    with open(outfile, 'r') as fo:
        for line in fo:
            if "Spin multiplicity:" in line:
                spin = int(line.split()[-1])
                if spin == 1:
                    return {"del_ss": 0}
            elif "SPIN S-SQUARED:" in line:
                _frame_now += 1
                ss_actual = float(line.split()[2])
                ss_target = float(line.split()[-1].strip(")"))
            if _frame_now == frame:
                if not np.isnan(abs(ss_actual-ss_target)):
                    break
                else:
                    frame += 1
    return {"del_ss": abs(ss_actual-ss_target)}


def get_metal_spin_del(mullpopfile, spin, frame, idx=-1):
    if spin == 1:
        return {"del_metal_spin": 0}
    _frame_now = -2
    metal_spin, idx_spin = 0, 0
    with open(mullpopfile, 'r') as fo:
        for line in fo:
            ll = line.split()
            if "Spin-Averaged" in line:
                _frame_now += 1
                if _frame_now == frame:
                    del_metal_spin = abs(metal_spin+idx_spin-spin+1)
                    if not np.isnan(del_metal_spin):
                        return {"del_metal_spin": del_metal_spin}
                    else:
                        frame += 1
            elif ll[0] == "1":
                metal_spin = float(ll[-1])
            elif ll[0] == str(idx):
                idx_spin = float(ll[-1])
    del_metal_spin = abs(metal_spin+idx_spin-spin+1)
    return {"del_metal_spin": del_metal_spin}


def read_geometry_to_mol(geofile, frame=-1, txt=False):
    mol = mol3D()
    if not txt:
        with open(geofile, 'r') as fo:
            num_atoms = int(fo.readline().split()[0])
        num_lines = num_atoms + 2  # +2 for the xyz format.
        with open(geofile, 'r') as fo:
            if (frame + 1) * num_lines != 0:
                geotext = fo.readlines()[frame * num_lines:(frame + 1) * num_lines]
            else:
                geotext = fo.readlines()[frame * num_lines:]
        mol.readfromtxt(geotext)
    else:
        mol.readfromtxt(geofile.split('\n'))
    return mol


def obtain_jobinfo(xyzfile, frame=-1, txt=False):
    init_mol = read_geometry_to_mol(xyzfile, frame=frame, txt=txt)
    natoms = init_mol.natoms
    metal_ind = init_mol.findMetal()[0]
    liglist, ligdents, ligcons = ligand_breakdown(init_mol, BondedOct=False)
    # print(liglist)
    # print(ligdents)
    # print(ligcons)
    try:
        _, _, _, _, _, _, _ax_con, _eq_con, _ = ligand_assign_consistent(init_mol, liglist,
                                                                         ligdents, ligcons)
    except:
        print("bad initial geometry")
        return False
    print("metal_ind: ", metal_ind)
    print("ax_con: ", _ax_con)
    print("eq_con: ", _eq_con)
    job_info = {}
    info_list = ['ax_con', 'eq_con', 'ax_con_sym', 'eq_con_sym', 'catoms', 'natoms', 'metal_ind', "symbols"]
    eq_con, ax_con = [], []
    for x in _eq_con:
        eq_con += x
    for x in _ax_con:
        ax_con += x
    ax_con_sym = [init_mol.atoms[x].sym for x in ax_con]
    eq_con_sym = [init_mol.atoms[x].sym for x in eq_con]
    catoms = [x for x in eq_con] + [x for x in ax_con]
    symbols = [init_mol.atoms[ii].sym for ii in range(natoms)]
    for info in info_list:
        job_info.update({info: locals()[info]})
    return job_info


def get_geo_metrics(init_mol, job_info, geofile, frame=-1):
    mol_now = read_geometry_to_mol(geofile, frame=frame)
    dict_geo_metrics = {}
    info_list = ['flag_oct']
    dict_info = ['inspect_oct_angle_devi_max', 'inspect_max_del_sig_angle',
                 'inspect_dist_del_all', 'inspect_dist_del_eq', 'inspect_devi_linear_avrg',
                 'inspect_devi_linear_max']
    actural_dict_info = ['actural_rmsd_max']
    mol = mol3D()
    mol.copymol3D(mol_now)
    inspect_flag, _, dict_oct, inspect_flag_loose, _ = mol.Oct_inspection(init_mol=init_mol,
                                                                          catoms_arr=job_info['catoms'])
    _mol = mol3D()
    _mol.copymol3D(mol_now)
    flag_oct, _, dict_oct_info = _mol.IsOct(init_mol=init_mol)
    eqsym, maxdent, ligdents, homoleptic, ligsym = init_mol.get_symmetry_denticity()
    if not maxdent > 1:
        choice = 'mono'
    else:
        choice = 'multi'
    actural_dict_geo = {}
    inspect_dict_geo = {}
    for key in dict_oct_info:
        if "relative" in key:
            continue
        val = (dict_oct_info[key] if (dict_oct_info[key] != -1) and (dict_oct_info[key] != "lig_mismatch")
               else 1.20 * dict_oct_check_st[choice][key])
        actural_dict_geo['actural_%s' % key] = val
    for key in dict_oct:
        if "relative" in key:
            continue
        inspect_dict_geo['inspect_%s' % key] = dict_oct[key]
    for info in info_list:
        dict_geo_metrics.update({info: locals()[info]})
    for info in dict_info:
        dict_geo_metrics.update({info: inspect_dict_geo[info]})
    for info in actural_dict_info:
        dict_geo_metrics.update({info: actural_dict_geo[info]})
    return dict_geo_metrics


def get_bond_order(bofile, job_info, num_sv=4, frame=-1):
    metal_ind = job_info['metal_ind']
    natoms = job_info['natoms']
    dict_bondorder = OrderedDict()
    catoms = [metal_ind] + job_info['catoms']
    dict_patterns = {}
    for catom in catoms:
        dict_patterns[catom] = [metal_ind, catom]
    botext = list()
    while len(botext) < 2:
        with open(bofile, 'r') as fo:
            if frame == -1:
                for line in fo:
                    if "bond order list" in line:
                        botext = list()
                    else:
                        botext.append(line)
            else:
                c = -2
                for line in fo:
                    if "bond order list" in line:
                        c += 1
                        if c == frame:
                            break
                        botext = list()
                    else:
                        botext.append(line)
        frame += 1
        # print("botext: ", botext)
    bo_mat = np.zeros(shape=(natoms, natoms))
    for line in botext:
        ll = line.split()
        if not (len(ll) == 1 and ll[0].isdigit()):
            row_idx, col_idx = int(ll[0]), int(ll[1])
            bo_mat[row_idx, col_idx] = float(ll[2])
            bo_mat[col_idx, row_idx] = float(ll[2])
    U, Sigma, VT = randomized_svd(bo_mat, n_components=num_sv, n_iter=20)
    sigma = Sigma.tolist()
    for sv in range(num_sv):
        dict_bondorder.update({'bo_sv%d' % sv: sigma[sv]})
    bo_mat_off_diag = bo_mat.copy()
    np.fill_diagonal(bo_mat_off_diag, 0)
    _U, _Sigma, _VT = randomized_svd(bo_mat_off_diag, n_components=num_sv, n_iter=20)
    _sigma = _Sigma.tolist()
    for sv in range(num_sv):
        dict_bondorder.update({'bo_offsv%d' % sv: _sigma[sv]})
    for catom, vals in list(dict_patterns.items()):
        if catom != metal_ind:
            dict_bondorder.update({'bo_%d' % catom: bo_mat[vals[0], vals[1]]})
    dict_bondorder = symmetricalize_dict(job_info, feature_dict=dict_bondorder)
    for catom, vals in list(dict_patterns.items()):
        if catom == metal_ind:
            dict_bondorder.update({'bo_0': bo_mat[vals[0], vals[1]]})
    return dict_bondorder


def get_gradient(gradfile, job_info, num_sv=3, frame=-1):
    metal_ind = job_info['metal_ind']
    natoms = job_info['natoms']
    num_lines = natoms + 2
    dict_gradient = OrderedDict()
    catoms = [metal_ind] + job_info['catoms']
    gradtext = ["energy -nan"]
    while "energy -nan" in "".join(gradtext):
        with open(gradfile, 'r') as fo:
            if (frame + 1) * num_lines != 0:
                gradtext = fo.readlines()[frame * num_lines:(frame + 1) * num_lines]
            else:
                gradtext = fo.readlines()[frame * num_lines:]
        with open(gradfile, 'r') as fo:
            if not len(gradtext):
                gradtext = fo.readlines()[-1 * num_lines:]
        frame += 1
        # print("gradtext: ", gradtext)
    grad_mat = np.zeros(shape=(natoms, 3))
    for idx, line in enumerate(gradtext):
        ll = line.split()
        if ll[0] == 'terachem':
            dict_gradient.update({'grad_rms': float(ll[7][:-1])})
        if idx > 1:
            grad_mat[idx - 2, :] = [float(x) for x in ll[1:]]
    # print("grad_mat: ", grad_mat)
    U, Sigma, VT = randomized_svd(grad_mat, n_components=num_sv, n_iter=20)
    sigma = Sigma.tolist()
    for sv in range(num_sv):
        dict_gradient.update({'grad_sv%d' % sv: sigma[sv]})
    for catom in catoms:
        if catom != metal_ind:
            dict_gradient.update({'grad_%d' % catom: np.linalg.norm(grad_mat[catom, :])})
    max_norm = 0
    for ii in range(natoms):
        _norm = np.linalg.norm(grad_mat[ii, :])
        if _norm > max_norm:
            max_norm = _norm
    dict_gradient.update({'grad_maxnorm': max_norm})
    grad_mat_internal = grad_mat.copy()
    grad_mat_internal = grad_mat_internal - grad_mat_internal[metal_ind, :]
    _U, _Sigma, _VT = randomized_svd(grad_mat_internal, n_components=num_sv, n_iter=20)
    _sigma = _Sigma.tolist()
    for sv in range(num_sv):
        dict_gradient.update({'grad_intsv%d' % sv: _sigma[sv]})
    _max_norm = 0
    for ii in range(natoms):
        _norm = np.linalg.norm(grad_mat_internal[ii, :])
        if _norm > _max_norm:
            _max_norm = _norm
    dict_gradient.update({'grad_intmaxnorm': _max_norm})
    dict_gradient = symmetricalize_dict(job_info, feature_dict=dict_gradient)
    for catom in catoms:
        if catom == metal_ind:
            dict_gradient.update({'grad_0': np.linalg.norm(grad_mat[catom, :])})
    return dict_gradient


def get_mullcharge(chargefile, job_info, frame=-1):
    metal_ind = job_info['metal_ind']
    natoms = job_info['natoms']
    dict_mullcharge = OrderedDict()
    catoms = [metal_ind] + job_info['catoms']
    chargetext = ["energy -nan"]
    while "nan" in "".join(chargetext):
        with open(chargefile, 'r') as fo:
            if (frame + 1) * natoms != 0:
                chargetext = fo.readlines()[frame * natoms:(frame + 1) * natoms]
            else:
                chargetext = fo.readlines()[frame * natoms:]
        with open(chargefile, 'r') as fo:
            if not len(chargetext):
                chargetext = fo.readlines()[-1 * natoms:]
        frame += 1
        # print("chargetext: ", chargetext)
    for line in chargetext:
        ll = line.split()
        atom_ind = int(ll[0]) - 1
        if atom_ind in catoms:
            if atom_ind != metal_ind:
                if "nan" not in ll[-1]:
                    dict_mullcharge.update({'charge_%d' % atom_ind: float(ll[-1])})
                else:
                    dict_mullcharge.update({'charge_%d' % atom_ind: 0})
    dict_mullcharge = symmetricalize_dict(job_info, feature_dict=dict_mullcharge)
    for line in chargetext:
        ll = line.split()
        atom_ind = int(ll[0]) - 1
        if atom_ind in catoms:
            if atom_ind == metal_ind:
                if "nan" not in ll[-1]:
                    dict_mullcharge.update({'charge_0': float(ll[-1])})
                else:
                    dict_mullcharge.update({'charge_0': 0})
    return dict_mullcharge


def symmetricalize_dict(job_info, feature_dict):
    sym_list = ['eq_mean', 'ax_mean']
    catoms = job_info['catoms']
    # print("feature_dict: ", feature_dict)
    feature_type = get_feature_type(feature_dict)
    for sym in sym_list:
        eq_arr = list()
        ax_arr = list()
        for catom in catoms[:4]:
            eq_arr.append(feature_dict['%s_%d' % (feature_type, catom)])
        for catom in catoms[4:]:
            ax_arr.append(feature_dict['%s_%d' % (feature_type, catom)])
        eq_mean = np.mean(eq_arr)
        ax_mean = np.mean(ax_arr)
        for sym in sym_list:
            feature_dict.update({'%s_%s' % (feature_type, sym): locals()[sym]})
    for catom in catoms:
        del feature_dict['%s_%d' % (feature_type, catom)]
    return feature_dict


def get_feature_type(feature_dict):
    return list(feature_dict.keys())[0].split('_')[0]


def check_pid(pid):
    # print("PID: ", pid)
    if not pid:
        pid = 000000
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    else:
        return True


def kill_job(pid):
    cmd = 'kill -9 %s' % str(pid)
    q = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # Unused:
    _ = q.communicate()[0].decode("utf-8")
