# Written JP Janet
# for HJK Group
# Dpt of Chemical Engineering, MIT

##########################################################
######## Defines class for postprocessing    #############
########     geometries for octahedral    ################
################      TM complexs          ###############
##########################################################
from molSimplify.Classes.ligand import ligand_breakdown, ligand_assign_consistent
from molSimplify.Scripts.geometry import distance

def getOctBondDistances(mol):
    ## This function gets
    ## ax and equatorial
    ## min and max bond lengths
    liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True)
    ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list, eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list = ligand_assign_consistent(
        mol, liglist, ligdents, ligcons, False, False)
    ax_dist = list()
    eq_dist = list()
    for ax_ligs in ax_con_list:
        tempList = list()
        for conatms in ax_ligs:
            tempList.append(distance(mol.getAtom(mol.findMetal()[0]).coords(), mol.getAtom(conatms).coords()))
        ax_dist.append(tempList)
    for eq_ligs in eq_con_list:
        tempList = list()
        for conatms in eq_ligs:
            tempList.append(distance(mol.getAtom(mol.findMetal()[0]).coords(), mol.getAtom(conatms).coords()))
        eq_dist.append(tempList)
    return ax_dist, eq_dist


def getLigFormulae(mol):
    ## This function gets
    ## ax and equatorial
    ## ligand names for octahedral complexes
    axnames = []
    eqnames = []
    liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True)
    ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list, eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list = ligand_assign_consistent(
        mol, liglist, ligdents, ligcons, False, False)
    for axl in ax_ligand_list:
        axnames.append(axl.mol.make_formula())
    for eql in eq_ligand_list:
        eqnames.append(eql.mol.make_formula())
    return axnames, eqnames


def maximum_ML_dist(mol):
    core = mol.getAtom(mol.findMetal()[0]).coords()
    max_dist = 0
    for atom_inds in mol.getBondedAtomsSmart(mol.findMetal()[0]):
        dist = distance(core, mol.getAtom(atom_inds).coords())
        if (dist > max_dist):
            max_dist = dist
    return max_dist


def maximum_any_dist(mol):
    core = mol.getAtom(mol.findMetal()[0])
    max_dist = 0
    for atoms in mol.getAtoms():
        dist = distance(core.coords(), atoms.coords())
        if (dist > max_dist):
            max_dist = dist
    return max_dist


def minimum_ML_dist(mol):
    core = mol.getAtom(mol.findMetal()[0]).coords()
    min_dist = 1000
    for atom_inds in mol.getBondedAtomsSmart(mol.findMetal()[0]):
        dist = distance(core, mol.getAtom(atom_inds).coords())
        if (dist < min_dist) and (dist > 0):
            min_dist = dist
    return min_dist


def mean_ML_dist(mol):
    core = mol.getAtom(mol.findMetal()[0]).coords()
    mean_dist = 0.0
    for atom_inds in mol.getBondedAtomsSmart(mol.findMetal()[0]):
        dist = distance(core, mol.getAtom(atom_inds).coords())
        mean_dist += float(dist)
    mean_dist = mean_dist / float(len(mol.getBondedAtomsSmart(mol.findMetal()[0])))
    return mean_dist
