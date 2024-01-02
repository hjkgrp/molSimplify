#########################
# Currently, this script only work off the shell for lacvps_ecp
# (6-31g* for period 1-3 atoms and lanl2dz for others)
# It can easily been generalized to other basis sets by modifying
# the shell_sequence_mapping function in molden2psi4wfn and
# change the corresponding Psi4 setting in this script.
# Written by Chenru Duan at 09/17/2020.
import psi4
import argparse
import os
import numpy as np
import shutil
import iodata
from molden2psi4wfn import tcmolden2psi4wfn_ao_mapping


def lacvps(mol, role):
    '''
    Define the LACVP* basis set.
    lanl2dz for metals and 6-31g* for others.
    '''
    basstrings = {}
    mol.set_basis_all_atoms("6-31g*", role=role)
    mol.set_basis_by_symbol("fe", "lanl2dz", role=role)
    mol.set_basis_by_symbol("co", "lanl2dz", role=role)
    mol.set_basis_by_symbol("cr", "lanl2dz", role=role)
    mol.set_basis_by_symbol("mn", "lanl2dz", role=role)
    mol.set_basis_by_symbol("mo", "lanl2dz", role=role)
    mol.set_basis_by_symbol("tc", "lanl2dz", role=role)
    mol.set_basis_by_symbol("ru", "lanl2dz", role=role)
    mol.set_basis_by_symbol("rh", "lanl2dz", role=role)
    mol.set_basis_by_symbol("I", "lanl2dz", role=role)
    mol.set_basis_by_symbol("Br", "lanl2dz", role=role)
    return basstrings


def get_molecule(xyzfile, charge, spin):
    '''
    Assemble a molecule object from xyzfile, charge and spin.
    '''
    wholetext = "%s %s\n" % (charge, spin)
    if os.path.isfile(xyzfile):
        with open(xyzfile, "r") as fo:
            natoms = int(fo.readline().split()[0])
            fo.readline()
            for ii in range(natoms):
                wholetext += fo.readline()
    wholetext += "\nsymmetry c1\nnoreorient\nnocom\n"
    mol = psi4.geometry("""%s""" % wholetext)
    return mol


def main():
    # from os.path import expanduser
    # home = expanduser("~")
    # psi4_scr = home + '/psi4_scr/'
    psi4_scr = './'
    if not os.path.isdir(psi4_scr):
        os.makedirs(psi4_scr)
    # ---argument parsing---
    parser = argparse.ArgumentParser(description='psi4 dft calculations.')
    parser.add_argument('-c', action="store", type=int, dest='charge')
    parser.add_argument('-spin', action="store", type=int, dest='spin')
    parser.add_argument('-xyz', action="store", type=str, dest='xyzfile')
    parser.add_argument('-molden', action="store", type=str, dest='moldenfile')
    parser.add_argument('-nthread', action="store", type=int, default=4, dest='num_threads')
    parser.add_argument('-memory', action="store", type=str, default='12000 MB', dest='memory')
    parser.add_argument('-ref', action="store", type=str, default='uks', dest='reference')
    args = parser.parse_args()
    moldenfile = args.moldenfile
    xyzfile = args.xyzfile
    charge, spin = args.charge, args.spin
    psi4.set_memory(args.memory)
    psi4.set_num_threads(args.num_threads)

    # ---basic setup---
    filename = "output"
    psi4.core.set_output_file(filename + '.dat', False)
    psi4.qcdb.libmintsbasisset.basishorde['LACVPS'] = lacvps
    psi4.set_options({
        'reference': args.reference,
        "puream": False,
        "DF_SCF_GUESS": False,
        "scf_type": "df",
        "dft_pruning_scheme": "robust",
        "basis": "lacvps",
        "DFT_BASIS_TOLERANCE": 1e-10,
        "INTS_TOLERANCE": 1e-10,
        "PRINT_MOS": False,
        "dft_spherical_points": 590,
        "dft_radial_points": 99,
        "guess": "read", })

    # ----1step scf---
    print("1step scf...")
    mol = get_molecule(xyzfile, charge, spin)
    psi4.set_options({
        "maxiter": 5,
        "D_CONVERGENCE": 1e5,
        "E_CONVERGENCE": 1e5,
        "fail_on_maxiter": False})
    e, wfn = psi4.energy('b3lyp', molecule=mol, return_wfn=True)
    wfn.to_file("wfn-1step.180")

    # -----conversion---
    print("molden2psi4wfn conversion...")
    d_molden = iodata.molden.load_molden(moldenfile)
    restricted = True if any(x in args.reference for x in ["r", "R"]) else False
    Ca, Cb, mapping = tcmolden2psi4wfn_ao_mapping(d_molden, restricted=restricted)
    wfn_minimal_np = np.load("wfn-1step.180.npy", allow_pickle=True)
    wfn_minimal_np[()]['matrix']["Ca"] = Ca
    if not restricted:
        wfn_minimal_np[()]['matrix']["Cb"] = Cb
    else:
        wfn_minimal_np[()]['matrix']["Cb"] = Ca
    np.save("wfn-1step-tc.180.npy", wfn_minimal_np)

    # ----copy wfn file to the right place with a right name---
    print("wfn copying...")
    pid = str(os.getpid())
    targetfile = psi4_scr + filename + '.default.' + pid + '.180.npy'
    shutil.copyfile("wfn-1step-tc.180.npy", targetfile)

    # ---final scf---
    print("final scf...")
    psi4.set_options({
            'SOSCF': False,
            "SOSCF_MAX_ITER": 40,
            })
    psi4.set_options({
        "maxiter": 250,
        "D_CONVERGENCE": 1e-6,
        "E_CONVERGENCE": 1e-6,
        "fail_on_maxiter": True})
    e, wfn = psi4.energy('b3lyp', molecule=mol, return_wfn=True)
    wfn.to_file("wfn.180")
    psi4.molden(wfn, "geo.molden")


if __name__ == "__main__":
    main()
