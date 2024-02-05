from molSimplify.Classes.monomer3D import monomer3D
from molSimplify.Classes.mol3D import mol3D
import pickle

# filename.txt = txt file with a list of PDB codes
# chosen_confs = folder with pickle files of protein3D objects
# the string "metal" should be replaced with the periodic table symbol of the desired metal
# pdbs = folder of pdb files; can be replaced with a urllib query
# activesites = folder of metal active sites

with open("filename.txt", 'r') as fin:
    text = fin.readlines()
for pdbid in text:
    with open('chosen_confs/' + pdbid + '.pkl', 'rb') as fin:
        p = pickle.load(fin)
    for metal in p.findAtom("metal", False):
        pdb = 'pdbs/' + pdbid
        with open(pdb+'.pdb', 'r') as fin:
            pdbfile = fin.readlines()
        activesite = mol3D()
        ids = []  # atom IDs
        metal_aa3ds = p.getBoundMols(metal, True)
        if metal_aa3ds is None:
            continue
        metal_all = p.getBoundMols(metal)
        metal_aas = []
        for aa3d in metal_aa3ds:
            metal_aas.append(aa3d.three_lc)
        coords = []
        with open('activesites/' + pdbid + "_" + str(metal) + '.pdb', "a") as fout:
            fout.write("HEADER " + pdbid + "_" + str(metal) + "\n")
            ids.append(metal)
            activesite.addAtom(p.atoms[metal], metal)
            fout.write(p.atoms[metal].line)
            coords.append(p.atoms[metal].coords())
            for m in metal_all:
                if type(m) == monomer3D:
                    for (a_id, a) in m.atoms:
                        if a.coords() not in coords:
                            ids.append(a_id)
                            activesite.addAtom(a, a_id)
                            fout.write(a.line)
                            coords.append(a.coords())
                else:
                    for a in m.atoms:
                        if a.coords() not in coords:
                            ids.append(p.getIndex(a))
                            activesite.addAtom(a, p.getIndex(a))
                            fout.write(a.line)
                            coords.append(a.coords())
            for lines in range(0, len(pdbfile)):
                if "CONECT" in pdbfile[lines] and int(pdbfile[lines][6:11]) in ids:
                    fout.write(pdbfile[lines])
