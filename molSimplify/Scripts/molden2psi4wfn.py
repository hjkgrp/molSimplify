#########################
# Currently, this script only work off the shell for lacvps_ecp
# (6-31g* for period 1-3 atoms and lanl2dz for others)
# It can easily been generalized to other basis sets by modifying
# the shell_sequence_mapping function.
# Written by Chenru Duan at 09/17/2020.
import numpy as np


def internal_ao_mapping(shell_type):
    '''
    Helper function to adjust the ordering of the internal AO in each shell if needed.
    Parameters
    ----------
        shell_type: int, 0 for s, 1 for p, 2 for d.

    Returns
    ----------
        a dictionary of how each AO should be shifted compared to their original location.
    '''
    if abs(shell_type) == 0:
        return {0: 0}
    elif abs(shell_type) == 1:
        return {0: 0, 1: 0, 2: 0}
    elif abs(shell_type) == 2:
        return {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    else:
        raise KeyError("only s, p, d shell is allowed.")


def shell_sequence_mapping(atom_shell_types):
    '''
    Helper function to adjust the ordering of each shell if needed.
    Parameters
    ----------
        atom_shell_types: list of int, 0 for s, 1 for p, 2 for d.

    Returns
    ----------
        a dictionary of how each AO shell should be shifted compared to their original location.
    '''
    # 6-31g* on C, O, N...
    if "-".join([str(x) for x in atom_shell_types]) == "-".join([str(x) for x in [0, 0, 0, 1, 1, 2]]):
        return {0: 0, 1: 0, 2: 3, 3: -1, 4: 0, 5: 0}  # for TC
    # 6-31g* on P, S, Cl...
    elif "-".join([str(x) for x in atom_shell_types]) == "-".join([str(x) for x in [0, 0, 0, 0, 1, 1, 1, 2]]):
        return {0: 0, 1: 0, 2: 3, 3: 6, 4: -2, 5: -1, 6: 0, 7: 0}
    # 6-31g* on metals
    elif "-".join([str(x) for x in atom_shell_types]) == "-".join([str(x) for x in [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2]]):
        return {0: 0, 1: 0, 2: 3, 3: 6, 4: 9, 5: -3, 6: -2, 7: -1, 8: 0, 9: 0, 10: 0}
    else:
        d = {}
        for ii, _ in enumerate(atom_shell_types):
            d.update({ii: 0})
        return d


def mocoeff_c2s(mocoeffs_c, shell_types):
    '''
    For Cartesian functions we need to add a basis function normalization constant of
    //      _______________________________
    //     / (2lx-1)!! (2ly-1)!! (2lz-1)!!
    //    /  -----------------------------
    //  \/             (2l-1)!!
    //
    d_conv, d_conv_inv: conversion metrix.
    source: https://github.com/psi4/psi4/blob/master/psi4/src/psi4/libmints/writer.cc, FCHKWriter
    Parameters
    ----------
        mocoeffs_c: numpy array, MO coeffs in the molden file.
        shell_types: list of int, 0 for s, 1 for p, 2 for d.

    Returns
    ----------
        mocoeffs_s: numpy array, scaled MO coeffs considering the basis function normalization difference.
    '''  # noqa W605 for the sqrt ACSII art in the description string
    pf1 = 1.0
    pf2 = np.sqrt(1.0 / 3.0)
    d_conv = np.array([
        [pf1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, pf1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, pf1],
        [0.0, pf2, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, pf2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, pf2, 0.0],
    ])
    d_conv_inv = np.linalg.inv(d_conv)
    mocoeffs_s = []
    start = 0
    end = 0
    for t in shell_types:
        if t == 0:
            end += 1
            mocoeffs_s += mocoeffs_c[start:end].tolist()
        elif t == 1:
            end += 3
            mocoeffs_s += mocoeffs_c[start:end].tolist()
        elif t == 2:
            end += 6
            tmp_c = mocoeffs_c[start:end]
            tmp_s = np.matmul(d_conv_inv, tmp_c)
            mocoeffs_s += tmp_s.tolist()
        else:
            raise KeyError("only support for s, p, and d for now.")
        start = end
    mocoeffs_s = np.array(mocoeffs_s)
    return mocoeffs_s


def mocoeff_mapping(d_molden, mapping, key='alpha'):
    '''
    Obtain the final MO coeffs.
    Parameters
    ----------
        d_molden: dict, obtained by using iodata to read a molden file
        mapping: dict, ow each AO shell should be shifted compared to their original location.

    Returns
    ----------
        mocoeff_psi4: numpy array, final MO coeffs.
    '''
    coeff = np.copy(d_molden['orb_%s_coeffs' % key])
    coeff = mocoeff_c2s(coeff, d_molden['obasis']['shell_types'])
    mocoeff_psi4 = np.zeros(shape=coeff.shape)
    for ii in range(len(mapping.keys())):
        mocoeff_psi4[mapping[ii], :] = coeff[ii, :]
    return mocoeff_psi4


def tcmolden2psi4wfn_ao_mapping(d_molden, restricted=False):
    '''
    The top level function to call to obtain the final MO coeffs.
    Parameters
    ----------
        d_molden: dict, obtained by using iodata to read a molden file
        restricted: whether the calculation is under restricted formalism.

    Returns
    ----------
        Ca: numpy array, final MO coeffs for alpha orb.
        Cb: numpy array, final MO coeffs for beta orb.
        mapping: dict, ow each AO shell should be shifted compared to their original location.
    '''
    mapping = {}
    start = 0
    ao_start = 0
    for atom_i in np.unique(d_molden['obasis']['shell_map']):
        num_shell = np.count_nonzero(d_molden['obasis']['shell_map'] == atom_i)
        atom_shell_types = d_molden['obasis']['shell_types'][start: start+num_shell]
        start += num_shell
        shell_mapping = shell_sequence_mapping(atom_shell_types)
#         print(shell_mapping)
        for ii in range(len(shell_mapping.keys())):
            shell_type = atom_shell_types[ii]
            ao_mapping_local = internal_ao_mapping(shell_type)
#             print(ii, shell_type, ao_mapping_local)
            for jj in range(len(ao_mapping_local.keys())):
                mapping.update(
                    {ao_start: ao_start+ao_mapping_local[jj]+shell_mapping[ii]})
                ao_start += 1
    Ca = mocoeff_mapping(d_molden, mapping, key='alpha')
    if not restricted:
        Cb = mocoeff_mapping(d_molden, mapping, key='beta')
    else:
        Cb = False
    return Ca, Cb, mapping
