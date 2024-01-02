# add hydrogens to pdb file

import os
from pymol import cmd
from molSimplify.Classes.protein3D import protein3D
from molSimplify.Classes.mol3D import mol3D

d1 = "Fe-2His_xyz_turned_pdb/"  # directory of pdb files
d2 = 'h_added/'  # directory to add pdb files with hydrogens

for filename in os.listdir(d1):
    if filename[:-4]+"_h.pdb" not in os.listdir(d2):
        p = protein3D()
        p.readfrompdb(d1+filename)
        m = p.findMetal()
        print(m)
        fe = p.atoms[m[0]]  # index of metal, not necessarily Fe
        hoh_indices = []
        for a in p.bonds[fe]:
            if type(p.getMolecule(p.getIndex(a))) == mol3D and p.getMolecule(p.getIndex(a)).name == "HOH":
                hoh_indices.append(str(p.getIndex(a)))
        cmd.load(d1+filename)
        sele = ""
        for i in hoh_indices:
            sele += "(idx " + i + ") or "
        sele = sele[:-4]
        if sele != "":
            cmd.select(sele)
            cmd.alter("sele", "formal_charge=1", quiet=1, space=None)
        cmd.h_add("("+filename[:-4]+")")
        cmd.save(d2+filename)
        cmd.delete("all")
        print(filename)
