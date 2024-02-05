import numpy as np

from molSimplify.Scripts.geometry import distance


def fpriority(mol):
    # setting up variables
    fidx_list = []
    sidx_list = []
    satno_list = []
    ref_list = []
    fd_list = []
    # fa_list = []
    # idx_list = [0] * 6
    exit_signal = True
    # getting bond-order matrix
    mol.convert2OBMol()
    BOMatrix = mol.populateBOMatrix()

    # preping for the loop
    fidx_list.append(mol.findMetal())
    for i in range(len(fidx_list)):
        for fidx in fidx_list[i]:
            for sidx in mol.getBondedAtoms(fidx):
                sidx_list.append([sidx])

    for i in range(len(fidx_list)):
        for fidx in fidx_list[i]:
            for j in range(len(sidx_list)):
                for sidx in sidx_list[j]:
                    BO = int(BOMatrix[fidx][sidx])
                    if BO == 0:
                        BO = 1
                    satno_str = str(mol.getAtom(sidx).atno)
                    satno_list.append(int(BO * satno_str))

    for satno in set(satno_list):
        satnocount = satno_list.count(satno)
        if satnocount > 1:
            s_sel_list = [i for i, atno in enumerate(satno_list) if atno is satno]
            exit_signal = False

    for i in range(len(fidx_list)):
        for fidx in fidx_list[i]:
            ref_list.append(fidx)

    # starting the loop
    tidx_list = []
    tatno_list = []
    for i in range(len(sidx_list)):
        tidx_list.append([])
        tatno_list.append([])

    while not exit_signal:
        fpriority_list = []
        for i in s_sel_list:
            t_list = []
            for sidx in sidx_list[i]:
                for tidx in mol.getBondedAtoms(sidx):
                    if tidx not in ref_list:
                        t_list.append(tidx)
            tidx_list[i] = t_list
        # print(sidx_list)
        # print(tidx_list)
        for i in s_sel_list:
            for sidx in sidx_list[i]:
                atno_list = tatno_list[i]
                ls = []
                for j in s_sel_list:
                    for tidx in tidx_list[j]:
                        BO = int(BOMatrix[sidx][tidx])
                        tatno_str = str(mol.getAtom(tidx).atno)
                        ls.append(BO * tatno_str)
                sorted(ls, reverse=True)
                for j in ls:
                    atno_list.append(j)
                a = ''.join(atno_list)
            tatno_list[i] = [a]
        sidx_list = []
        for i in range(len(tidx_list)):
            sidx_list.append(tidx_list[i])
        for i in s_sel_list:
            for sidx in sidx_list[i]:
                ref_list.append(sidx)
        test_list = []
        for i in range(len(sidx_list)):
            test_list.append([])
        # get priorities
        for i in range(len(satno_list)):
            atno_list = []
            atno_list.append(str(satno_list[i]))
            if tatno_list[i] == []:
                atno_list.append('')
            else:
                atno_list.append(tatno_list[i][0])
            a = '.'.join(atno_list)
            fpriority_list.append(float(a))
        if tidx_list == test_list or len(set(fpriority_list)) == 6:
            exit_signal = True
    # get distance
    fd_list = []
    mcoord = mol.atoms[mol.findMetal()[0]].coords()
    idx_list = np.argsort(np.array(fpriority_list))
    fpriority_list = np.array(fpriority_list)[idx_list].tolist()
    sidx_list = mol.getBondedAtoms(fidx_list[0][0])
    for idx in idx_list:
        scoord = mol.getAtom(sidx_list[idx]).coords()
        r = distance(mcoord, scoord)
        fd_list.append(r)

    # idx = np.argsort(np.array(fpriority_list))[-1]
    # sidx_list = mol.getBondedAtomsByCoordNo(fidx_list[0][0],6)
    # refcoord = mol.getAtom(sidx_list[idx]).coords()
    # mcoord = mol.getAtom(fidx_list[0][0]).coords()
    # idx0 = 0
    # dist5 = 0
    # idx5 = 0
    # idx1_4 = []
    # fprio1_4 = []
    # sxyzs = []
    # ssd_list = []
    # for i, sidx in enumerate(sidx_list):
    #     sxyz = mol.getAtom(sidx).coords()
    #     dist = distance(refcoord,sxyz)
    #     if dist == 0:
    #         idx0 = i
    #     elif dist > dist5:
    #         dist5 = dist
    #         idx5 = i
    #     idx1_4.append(i)
    #     fprio1_4.append(fpriority_list[i])
    #     sxyzs.append(sxyz)
    #     ssd_list.append(dist)
    #     fd_list.append(distance(mcoord,sxyz))
    # idx1_4.pop(idx0)
    # idx1_4.pop(idx5)
    # fprio1_4.pop(idx0)
    # fprio1_4.pop(idx5)
    # idx_list[0] = idx0
    # idx_list[5] = idx5
    # idx1 = idx1_4[np.argsort(np.array(fprio1_4))[3]]
    # sxyz1 = sxyzs[idx1]
    # idx2_ = idx1_4[np.argsort(np.array(fprio1_4))[2]]
    # sxyz2_ = sxyzs[idx2_]
    # idx3_ = idx1_4[np.argsort(np.array(fprio1_4))[1]]
    # sxyz3_ = sxyzs[idx3_]
    # idx4_ = idx1_4[np.argsort(np.array(fprio1_4))[0]]
    # sxyz4_ = sxyzs[idx4_]
    # fd1_4 = []
    # fd1_4.append(distance(sxyz1, sxyz1))
    # fd1_4.append(distance(sxyz1, sxyz2_))
    # fd1_4.append(distance(sxyz1, sxyz3_))
    # fd1_4.append(distance(sxyz1, sxyz4_))
    # idx3 = idx1_4[np.argsort(np.array(fd1_4))[-1]] + idx1 - 4
    # if idx3 == idx2_:
    #     if fpriority_list[idx3_] > fpriority_list[idx4_]:
    #         idx2 = idx3_
    #         idx4 = idx4_
    #     else:
    #         idx2 = idx4_
    #         idx4 = idx2_
    # elif idx3 == idx4_:
    #     if fpriority_list[idx2_] > fpriority_list[idx3_]:
    #         idx2 = idx2_
    #         idx4 = idx3_
    #     else:
    #         idx2 = idx3_
    #         idx4 = idx2_
    # else:
    #     if fpriority_list[idx2_] > fpriority_list[idx4_]:
    #         idx2 = idx2_
    #         idx4 = idx4_
    #     else:
    #         idx2 = idx4_
    #         idx4 = idx2_
    # # get ax, eq, ax idxes
    # idx_list[1] = idx1
    # idx_list[2] = idx2
    # idx_list[3] = idx3
    # idx_list[4] = idx4
    # fpriority_list = np.array(fpriority_list)[idx_list].tolist()
    # fd_list = np.array(fd_list)[idx_list].tolist()

    return fpriority_list, fd_list, idx_list


def fsym(mol):
    # getting idxs of interest
    midx = mol.findMetal()[0]  # monometallic complexes
    fidx_list = mol.getBondedAtoms(midx)  # list of idx of the first-coord sphere
    fsym_list = []
    for idx in fidx_list:
        sym = mol.getAtom(idx).sym
        fsym_list.append(sym)
    return fsym_list


def fvalency(mol):
    # getting idxs of interest
    midx = mol.findMetal()[0]  # monometallic complexes
    fidx_list = mol.getBondedAtoms(midx)  # list of idx of the first-coord sphere
    fvalency_list = []
    for idx in fidx_list:
        valency = len(mol.getBondedAtoms(idx)) - 1
        fvalency_list.append(valency)
    return fvalency_list


def fcharge(mol, charge, bond=False):
    # getting idxs of interest
    midx = mol.findMetal()[0]  # monometallic complexes
    fidx_list = mol.getBondedAtoms(midx)  # list of idx of the first-coord sphere
    mol.calcCharges(charge, bond)
    fcharge_list = []
    for idx in fidx_list:
        partialq = mol.partialcharges[idx]
        fcharge_list.append(float(partialq))
    return fcharge_list


def scharge_ave(mol, charge, bond=False):
    mol.calcCharges(charge)
    # getting idxs of interest
    midx = mol.findMetal()[0]  # monometallic complexes
    fidx_list = mol.getBondedAtoms(midx)  # list of idx of the first-coord sphere
    mol.calcCharges(charge, bond)
    sidx_list = [mol.getBondedAtoms(fidx) for fidx in fidx_list]
    scharge_ave_list = []
    for i in range(len(sidx_list)):
        partialq = 0
        for j in range(len(sidx_list[i])):
            idx = sidx_list[i][j]
            if idx is not midx:
                partialq += mol.partialcharges[idx]
        charge_ave = partialq/len(sidx_list[i])
        scharge_ave_list.append(float(charge_ave))

    return scharge_ave_list

# def fdistance(mol):
#     # getting idxs of interest
#     midx = mol.findMetal()[0] # monometallic complexes
#     mcoord = mol.getAtom(midx).coords()
#     fidx_list = mol.getBondedAtoms(midx) # list of idx of the first-coord sphere
#     fdistance_list = []
#     for idx in fidx_list:
#         fcoord = mol.getAtom(idx).coords()
#         d = distance(mcoord,fcoord)
#         fdistance_list.append(float(d))

#     return fdistance_list


def all_prop(mol, charge, bond=False):
    fprio_list, fd_list, idx_list = fpriority(mol)
    # fsym_list = fsym(mol)
    fva_list = fvalency(mol)
    fq_list = fcharge(mol, charge, bond)
    sq_ave_list = scharge_ave(mol, charge, bond)
    # fd_list = fdistance(mol)
    fd_list = np.array(fd_list)
    idx_list = idx_list(fprio_list, fd_list)
    # rearranging
    fprio_list = [fprio_list[idx] for idx in idx_list]
    fq_list = [fq_list[idx] for idx in idx_list]
    sq_ave_list = [sq_ave_list[idx] for idx in idx_list]
    fva_list = [fva_list[idx] for idx in idx_list]
    fd_list = [fd_list[idx] for idx in idx_list]

    prop_list = fprio_list + fq_list + sq_ave_list + fva_list + fd_list

    return prop_list


def f_prop(mol, charge, bond=False):
    fprio_list, fd_list, idx_list = fpriority(mol)
    # fsym_list = fsym(mol)
    fva_list = fvalency(mol)
    fq_list = fcharge(mol, charge, bond)
    # sq_ave_list = scharge_ave(mol, charge, bond)
    # fd_list = fdistance(mol)
    idx_list = idx_list(fprio_list, fd_list)
    # rearranging
    fprio_list = [fprio_list[idx] for idx in idx_list]
    fq_list = [fq_list[idx] for idx in idx_list]
    fva_list = [fva_list[idx] for idx in idx_list]

    prop_list = fprio_list + fq_list + fva_list

    return prop_list


def features(mol, charge, bond=False):
    feature = []
    feature_names = []
    prop_list = all_prop(mol, charge, bond)
    midx = mol.findMetal()[0]
    manto = mol.getAtom(midx).atno
    # a = np.array(prop_list)
    # b = a.T[a[0].argsort()].T
    # feature_list = b.tolist()
    fatno_list = [int(str(i).split('.')[0]) for i in prop_list[:6]]
    feature.append(manto)
    for i in range(len(fatno_list)):
        prop_list[i] = fatno_list[i]
    feature += prop_list
    # for i in range(len(feature_list)):
    #     for j in range(len(feature_list[i])):
    #         feature.append(feature_list[i][j])
    feature_names = [
        'mato', 'fatno_1', 'fatno_2', 'fatno_3', 'fatno_4', 'fatno_5', 'fatno_6',
        'fq_1', 'fq_2', 'fq_3', 'fq_4', 'fq_5', 'fq_6',
        'sqave_1', 'sqave_2', 'sqave_3', 'sqave_4', 'sqave_5', 'sqave_6',
        'fval_1', 'fval_2', 'fval_3', 'fval_4', 'fval_5', 'fval_6',
        'bl_1', 'bl_2', 'bl_3', 'bl_4', 'bl_5', 'bl_6']

    return feature_names, feature
    # return feature


def ffeatures(mol, charge, bond=False):
    feature = []
    feature_names = []
    prop_list = f_prop(mol, charge, bond)
    midx = mol.findMetal()[0]
    manto = mol.getAtom(midx).atno
    fatno_list = [int(str(i).split('.')[0]) for i in prop_list[:6]]
    feature.append(manto)
    for i in range(len(fatno_list)):
        prop_list[i] = fatno_list[i]
    feature += prop_list
    # a = np.array(prop_list)
    # b = a.T[a[0].argsort()].T
    # feature_list = b.tolist()
    # feature_list[0] = [int(str(i).split('.')[0]) for i in feature_list[0]]
    # feature.append(manto)
    # for i in range(len(feature_list)):
    #     for j in range(len(feature_list[i])):
    #         feature.append(feature_list[i][j])
    feature_names = [
        'mato', 'fatno_1', 'fatno_2', 'fatno_3', 'fatno_4', 'fatno_5', 'fatno_6',
        'fq_1', 'fq_2', 'fq_3', 'fq_4', 'fq_5', 'fq_6',
        'fval_1', 'fval_2', 'fval_3', 'fval_4', 'fval_5', 'fval_6']

    return feature_names, feature
