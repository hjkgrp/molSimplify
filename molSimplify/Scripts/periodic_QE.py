# @file periodic_QE.py
#  Generates Quantum ESPRESSO input files from slab builder
#
#  Written by JP Janet for HJK Group
#
#  Dpt of Chemical Engineering, MIT

from molSimplify.Classes.globalvars import globalvars


###############################


def write_periodic_mol3d_to_qe(mol, cell_vector, path):
    psd = {"Ti": 'Ti.pbe-sp-van_ak.UPF',
           'O': 'O.pbe-van_ak.UPF', 'Si': 'Si.pbe-n-van.UPF'}
    # set global properties

    unique_atoms = mol.getAtomTypes()
    globs = globalvars()
    # print(globs)
    # start writing this file:
    with open(path, 'w') as f:
        f.write("&CONTROL\n")
        f.write('calculation = "relax" \n')
        f.write('prefix = clean\n')
        f.write('pseudo_dir = "/opt/espresso-5.1/pseudo"\n')
        f.write('outdir = "./"\n')
        f.write('wf_collect = .true\n')
        f.write('tprnfor = .true.\n')
        f.write('restart_mode = "from_scratch"\n')
        f.write('nstep = 1000\n')
        f.write("/ \n")
    with open(path, 'a') as f:
        f.write("&SYSTEM\n")
        f.write("ibrav = 0 \n")
        f.write("nat  = " + str(mol.natoms) + "\n")
        f.write("ntyp = " + str(len(unique_atoms)) + "\n")
        f.write("nspin = 1\n")
        f.write('occupations = "smearing"\n')
        f.write('degauss = 0.01\n')
        f.write('ecutwfc = 25.0 \n')
        f.write('ecutrho = 250.0 \n')
        f.write("/ \n")
    with open(path, 'a') as f:
        f.write("&ELECTRONS\n")
        f.write("mixing_beta = 0.4 \n")
        f.write("electron_maxstep = 350 \n")
        f.write("/ \n")
    with open(path, 'a') as f:
        f.write("&IONS\n")
        f.write("ion_dynamics = 'bfgs' \n")
        f.write("/ \n")

    with open(path, 'a') as f:
        f.write("CELL_PARAMETERS {angstrom}\n")
        for cv in cell_vector:
            ss = " ".join(str(e) for e in cv) + "\n"
            f.write(ss)
        f.write(" \n")

    with open(path, 'a') as f:
        f.write("ATOMIC_SPECIES\n")
        for elements in unique_atoms:
            ps_info = ".pbe-van_ak.UPF"
            if str(elements) in list(psd.keys()):
                ps_info = str(psd[str(elements)])
            if len(elements) == 1:
                f.write(str(elements) + "     " + str(globs.amass()
                                                      [elements][0]) + "     " + ps_info + '\n')
            else:
                f.write(str(elements) + "    " + str(globs.amass()
                                                     [elements][0]) + "     " + ps_info + '\n')
    with open(path, 'a') as f:
        pos_list = list()
        write_list = list()
        f.write("ATOMIC_POSITIONS {angstrom}\n")
        for atom in mol.atoms:
            xyz = atom.coords()
            if atom.frozen:
                freeze_vect = [0, 0, 0]
            else:
                freeze_vect = [1, 1, 1]
            pos_list.append(xyz[2])
            write_list.append(
                (atom.sym, xyz[0], xyz[1], xyz[2], freeze_vect[0], freeze_vect[1], freeze_vect[2]))

        sorted_inds = [i[0]
                       for i in sorted(enumerate(pos_list), key=lambda x:x[1])]
        for inds in sorted_inds:
            f.write("%s   %f    %f    %f    %f    %f    %f\n" %
                    write_list[inds])

#                    f.write("%s  %f %f %f %f %f %f\n" % (atom.sym,xyz[0],xyz[1],xyz[2],freeze_vect[0],freeze_vect[1],freeze_vect[2]))
    with open(path, 'a') as f:
        f.write("K_POINTS {automatic}\n")
        f.write("4 4 1 0 0 0\n")
