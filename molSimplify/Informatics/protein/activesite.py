# create protein3D

from molSimplify.Classes.protein3D import *
from math import *
import pickle
import os

# in order to mass-generate PDB protein3D objects, the following script can be enclosed in a for loop

pdbid = "1os7" # replace with your desired 4-letter pdb code
p = protein3D()
p.fetch_pdb(pdbid)
p.readMetaData()
p.setEDIAScores()
# the next three lines dump the protein3D object into a pickle file for storage purposes
#file = open('pickles/'+ pdbid +".pkl", 'wb')
#pickle.dump(p, file)
#file.close()
print(len(p.atoms.keys()))
p.autoChooseConf()
print(len(p.atoms.keys()))
# if desired, the chosen conformation of 
#pickle.dump(p, open('chosen_confs_102521/' + pdbid + '.pkl', 'wb'))

# now we extract the active sites of the protein3D object p

from molSimplify.Classes.mol3D import *

# Goal:  extract clusters

# this next set of code may also be enclosed in a for loop for mass production

metal_list = p.findMetal()

for metal in metal_list:
    """
    deal with this - PDB files should not need to be downloaded
    pdb = 'pdbs/' + pdbid
    pdbfile=open(pdb+'.pdb','r').readlines()
    """
    cluster = mol3D()
    ids = [] # atom IDs
    metal_aa3ds = p.getBoundMols(metal, True)
    if metal_aa3ds == None:
        continue
    metal_all = p.getBoundMols(metal)
    metal_aas = []
    for aa3d in metal_aa3ds:
        metal_aas.append(aa3d.three_lc)
    coords = []
    f = open('clusters/' + pdbid + "_" + str(metal) + '.pdb', "a") #"clusters" is the folder to store the active sites
    f.write("HEADER " + pdbid + "_" + str(metal) + "\n")
    ids.append(metal)
    cluster.addAtom(p.atoms[metal], metal)
    f.write(p.atoms[metal].line)
    coords.append(p.atoms[metal].coords())
    for m in fe_all: # loop through all molecules m bound to metal
        if type(m) == AA3D:
            for (a_id, a) in m.atoms:
                if a.coords() not in coords:
                    ids.append(a_id)
                    cluster.addAtom(a, a_id)
                    f.write(a.line)
                    coords.append(a.coords())
        else:
            for a in m.atoms:
                if a.coords() not in coords:
                    ids.append(p.getIndex(a))
                    cluster.addAtom(a, p.getIndex(a))
                    f.write(a.line)
                    coords.append(a.coords())
    for lines in range(0,len(pdbfile)):
        if "CONECT" in pdbfile[lines] and int(pdbfile[lines][6:11]) in ids:
            f.write(pdbfile[lines])
    f.close()

# clean up connectivity in cluster pdb files

with open('clusters/' + pdbid + "_" + str(metal) + '.pdb',"r") as y:
    foo = []
    with open("tmp.pdb","w") as x:
        for lin in y.readlines():
            if "AT" in lin: foo.append(lin[6:11])
            if "CONECT" in lin and lin[6:11] not in foo: lin = ""
            x.write(lin)
    os.replace("tmp.pdb",'clusters/' + pdbid + "_" + str(metal) + '.pdb')
