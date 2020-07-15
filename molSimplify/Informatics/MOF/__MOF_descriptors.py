from pymatgen.io.cif import CifParser
from molSimplify.Scripts.cellbuilder_tools import *
from molSimplify.Classes import mol3D
from molSimplify.Informatics.autocorrelation import *
from molSimplify.Informatics.misc_descriptors import *
from molSimplify.Informatics.graph_analyze import *
from molSimplify.Informatics.RACassemble import *
import os
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import sparse
import itertools
import networkx as nx
from .PBC_functions import *
import networkx.algorithms.isomorphism as iso
from .atomic import metalslist
import openbabel, pybel
from rdkit import Chem

#########################################################################################
# This MOF RAC generator assumes that pymatgen is installed.                            #
# Pymatgen is used to get the primitive cell.                                           #
#########################################################################################


def get_primitive(datapath, writepath):
    s = CifParser(datapath, occupancy_tolerance=1).get_structures()[0]
    sprim = s.get_primitive_structure()
    sprim.to("cif", writepath)


#########################################################################################
# The RAC functions here average over the different SBUs or linkers present. This is    #
# because one MOF could have multiple different linkers or multiple SBUs, and we need   #
# the vector to be of constant dimension so we can correlate the output property.       #
#########################################################################################


def make_MOF_mc_RACs(SBU,
                     SBU_subgraph,
                     metal_ind,
                     molcif,
                     depth,
                     name,
                     cell,
                     SBU_population,
                     sbupath=False):
    descriptor_list = []
    names = []
    descriptor_names = []
    descriptors = []
    if sbupath:
        sbu_descriptor_path = os.path.dirname(sbupath)
        if os.path.getsize(sbu_descriptor_path + '/mc_descriptors.csv') > 0:
            sbu_descriptors = pd.read_csv(sbu_descriptor_path +
                                          '/mc_descriptors.csv')
        else:
            sbu_descriptors = pd.DataFrame()
    metal_ind_onSBU = SBU.index(metal_ind)
    descriptor_names = []
    descriptors = []
    SBU_mol = mol3D()
    for val in SBU:
        SBU_mol.addAtom(molcif.getAtom(val))
    SBU_mol.graph = SBU_subgraph.todense()

    descriptors = []
    descriptor_names = []
    SBU_mol_cart_coords = np.array([atom.coords() for atom in SBU_mol.atoms])
    SBU_mol_atom_labels = [atom.sym for atom in SBU_mol.atoms]
    SBU_mol_adj_mat = np.array(SBU_mol.graph)
    ###### WRITE THE SBU MOL TO THE PLACE
    xyzname = sbupath + "/" + str(name) + "_sbu_" + str(metal_ind) + ".xyz"
    SBU_mol_fcoords_connected = XYZ_connected(cell, SBU_mol_cart_coords,
                                              SBU_mol_adj_mat)
    writeXYZandGraph(xyzname, SBU_mol_atom_labels, cell,
                     SBU_mol_fcoords_connected, SBU_mol_adj_mat)
    """""" """
    Generate all of the SBU based RACs (full scope, mc)
    """ """"""
    results_dictionary = generate_full_complex_autocorrelations(
        SBU_mol, depth=depth, loud=False, flag_name=False)
    descriptor_names, descriptors = append_descriptors(
        descriptor_names, descriptors, results_dictionary['colnames'],
        results_dictionary['results'], 'f', 'all')
    results_dictionary = generate_metalcenter_autocorrelations(SBU_mol,
                                                               metal_ind_onSBU,
                                                               depth=depth,
                                                               loud=False)
    descriptor_names, descriptors = append_descriptors(
        descriptor_names, descriptors, results_dictionary['colnames'],
        results_dictionary['results'], 'mc', 'all')
    results_dictionary = generate_metalcenter_deltametrics(SBU_mol,
                                                           metal_ind_onSBU,
                                                           depth=depth,
                                                           loud=False)
    descriptor_names, descriptors = append_descriptors(
        descriptor_names, descriptors, results_dictionary['colnames'],
        results_dictionary['results'], 'D_mc', 'all')
    descriptor_names += ['name']
    descriptors += [name]
    metaltype = SBU_mol.atoms[metal_ind_onSBU].sym
    descriptor_names += ['atom_index']
    descriptors += [metal_ind]
    descriptor_names += ['metal_type']
    descriptors += [metaltype]
    mc_x, mc_y, mc_z = SBU_mol_cart_coords[metal_ind_onSBU]
    descriptor_names += ['coordinate_x', 'coordinate_y', 'coordinate_z']
    descriptors += [mc_x, mc_y, mc_z]
    descriptor_names += ['SBU_population']
    descriptors += [SBU_population]
    descriptors == list(descriptors)
    desc_dict = {
        key: descriptors[ii]
        for ii, key in enumerate(descriptor_names)
    }
    sbu_descriptors = sbu_descriptors.append(desc_dict, ignore_index=True)
    sbu_descriptors.to_csv(sbu_descriptor_path + '/mc_descriptors.csv',
                           index=False)
    return descriptor_names, sbu_descriptors


def make_crystalgraph_mc_RACs(molcif, depth, name, cell, sbupath=False):
    descriptor_list = []
    names = []
    descriptor_names = []
    descriptors = []
    if sbupath:
        sbu_descriptor_path = os.path.dirname(sbupath)
        if os.path.getsize(sbu_descriptor_path + '/mc_descriptors.csv') > 0:
            sbu_descriptors = pd.read_csv(sbu_descriptor_path +
                                          '/mc_descriptors.csv')
        else:
            sbu_descriptors = pd.DataFrame()
        """""" """
        Generate all of the SBU based RACs (full scope, mc)
        """ """"""
    results_dictionary = generate_multimetal_autocorrelations(molcif,
                                                              depth=depth,
                                                              loud=False)
    descriptor_names, descriptors = append_descriptors(
        descriptor_names, descriptors, results_dictionary['colnames'],
        results_dictionary['results'], 'mc_CRY', 'all')
    results_dictionary = generate_multimetal_deltametrics(molcif,
                                                          depth=depth,
                                                          loud=False)
    descriptor_names, descriptors = append_descriptors(
        descriptor_names, descriptors, results_dictionary['colnames'],
        results_dictionary['results'], 'D_mc_CRY', 'all')
    results_dictionary = generate_sum_multimetal_autocorrelations(molcif,
                                                                  depth=depth,
                                                                  loud=False)
    descriptor_names, descriptors = append_descriptors(
        descriptor_names, descriptors, results_dictionary['colnames'],
        results_dictionary['results'], 'sum-mc_CRY', 'all')
    results_dictionary = generate_sum_multimetal_deltametrics(molcif,
                                                              depth=depth,
                                                              loud=False)
    descriptor_names, descriptors = append_descriptors(
        descriptor_names, descriptors, results_dictionary['colnames'],
        results_dictionary['results'], 'sum-D_mc_CRY', 'all')
    descriptor_names += ['name']
    descriptors += [name]
    descriptors == list(descriptors)
    desc_dict = {
        key: descriptors[ii]
        for ii, key in enumerate(descriptor_names)
    }
    descriptors.remove(name)
    descriptor_names.remove('name')
    sbu_descriptors = sbu_descriptors.append(desc_dict, ignore_index=True)
    descriptor_list.append(descriptors)
    names = descriptor_names
    sbu_descriptors.to_csv(sbu_descriptor_path + '/mc_descriptors.csv',
                           index=False)
    return descriptor_names, descriptors


def find_unique_subgraphs(BBlist, BB_subgraph, molcif):
    unique_graphs = []
    unique_ids = []
    unique_atom_types = []
    unique_counts = []
    nm = iso.categorical_node_match('atomicSym', "")
    for i, BB in enumerate(BBlist):
        at_types = []
        for at in BB:
            at_types.append(molcif.atoms[at].symbol())
        try:
            adjmat = np.triu(BB_subgraph[i].todense())
        except AttributeError:
            adjmat = np.triu(BB_subgraph[i])

        rows, cols = np.where(adjmat == 1)
        edges = [(a, b) for a, b in zip(rows.tolist(), cols.tolist())]
        nodes = np.arange(len(at_types))
        molgr = make_graph_from_nodes_edges(nodes, edges, at_types)
        new_unique = True
        for ji in range(len(unique_graphs)):
            j = len(unique_graphs) - ji - 1
            try:
                if set(at_types) == unique_atom_types[j]:
                    gr = unique_graphs[j]
                    if nx.is_isomorphic(gr, molgr, node_match=nm):
                        new_unique = False
                        unique_counts[j] += 1
                        break
            except IndexError:
                continue

        if new_unique:
            unique_ids.append(i)
            unique_graphs.append(molgr)
            unique_atom_types.append(set(at_types))
            unique_counts.append(1)

    return unique_ids, unique_counts


def print_parts(BBlist,
                BB_subgraph,
                molcif,
                depth,
                name,
                cell,
                pr="_sbu_",
                sbupath=False,
                checkunique=False):
    if checkunique:
        unique_ids, unique_counts = find_unique_subgraphs(
            BBlist, BB_subgraph, molcif)
    else:
        unique_ids = range(len(BBlist))

    for j, i in enumerate(unique_ids):
        BB = BBlist[i]
        BB_mol = mol3D()
        for val in BB:
            BB_mol.addAtom(molcif.getAtom(val))
        try:
            BB_mol.graph = BB_subgraph[i].todense()
        except AttributeError:
            BB_mol.graph = BB_subgraph[i]

        BB_mol_cart_coords = np.array([atom.coords() for atom in BB_mol.atoms])
        BB_mol_atom_labels = [atom.sym for atom in BB_mol.atoms]
        BB_mol_adj_mat = np.array(BB_mol.graph)
        ###### WRITE THE BB MOL TO THE PLACE ###
        if sbupath and not os.path.exists(sbupath + "/" + str(name) + str(j) +
                                          '.xyz'):
            xyzname = sbupath + "/" + str(name) + pr + str(j) + ".xyz"
            BB_mol_fcoords_connected = XYZ_connected(cell, BB_mol_cart_coords,
                                                     BB_mol_adj_mat)
            writeXYZandGraph(xyzname, BB_mol_atom_labels, cell,
                             BB_mol_fcoords_connected, BB_mol_adj_mat)
            stringXYZ = stringXYZfcoords(BB_mol_atom_labels, cell,
                                         BB_mol_fcoords_connected)


def get_SMILES(BBlist, BB_subgraph, molcif, cell, checkunique=False):
    if checkunique:
        unique_ids, unique_counts = find_unique_subgraphs(
            BBlist, BB_subgraph, molcif)
    else:
        unique_ids = range(len(BBlist))
        unique_counts = list(np.ones(len(unique_ids)))
    SMILES = []
    RDKit_check = True
    for j, i in enumerate(unique_ids):
        BB = BBlist[i]
        BB_mol = mol3D()
        for val in BB:
            BB_mol.addAtom(molcif.getAtom(val))
        try:
            BB_mol.graph = BB_subgraph[i].todense()
        except AttributeError:
            BB_mol.graph = BB_subgraph[i]

        BB_mol_cart_coords = np.array([atom.coords() for atom in BB_mol.atoms])
        BB_mol_atom_labels = [atom.sym for atom in BB_mol.atoms]
        BB_mol_adj_mat = np.array(BB_mol.graph)
        BB_mol_fcoords_connected = XYZ_connected(cell, BB_mol_cart_coords,
                                                 BB_mol_adj_mat)
        stringXYZ = stringXYZfcoords(BB_mol_atom_labels, cell,
                                     BB_mol_fcoords_connected)
        OBmol_xyz = pybel.readstring("xyz", stringXYZ)
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "smi")
        try:
            sm = obConversion.WriteString(OBmol_xyz.OBMol).strip()
            SMILES.append(sm)
        except:
            SMILES.append("OBError")
        ## check the chemisty of linker ##
        obConversion.SetInAndOutFormats("xyz", "mol")
        OBmol_mol = obConversion.WriteString(OBmol_xyz.OBMol)
        m = Chem.MolFromMolBlock(OBmol_mol)
        try:
            sm = Chem.MolToSmiles(m)
        except:
            RDKit_check = False
    return SMILES, unique_counts, RDKit_check


def make_MOF_SBU_RACs(SBUlist,
                      SBU_subgraph,
                      metalstart_atoms_list,
                      molcif,
                      depth,
                      name,
                      cell,
                      anchoring_atoms,
                      sbupath=False,
                      connections_list=False,
                      connections_subgraphlist=False):
    descriptor_list = []
    lc_descriptor_list = []
    lc_names = []
    names = []
    n_sbu = len(SBUlist)
    descriptor_names = []
    descriptors = []
    if sbupath:
        sbu_descriptor_path = os.path.dirname(sbupath)
        if os.path.getsize(sbu_descriptor_path + '/sbu_descriptors.csv') > 0:
            sbu_descriptors = pd.read_csv(sbu_descriptor_path +
                                          '/sbu_descriptors.csv')
        else:
            sbu_descriptors = pd.DataFrame()
        if os.path.getsize(sbu_descriptor_path + '/lc_descriptors.csv') > 0:
            lc_descriptors = pd.read_csv(sbu_descriptor_path +
                                         '/lc_descriptors.csv')
        else:
            lc_descriptors = pd.DataFrame()
    """""" """
    Loop over all SBUs as identified by subgraphs. Then create the mol3Ds for each SBU.
    """ """"""

    for i, (SBU, metal_ind) in enumerate(zip(SBUlist, metalstart_atoms_list)):
        metal_ind_onSBU = SBU.index(metal_ind[0])
        descriptor_names = []
        descriptors = []
        SBU_mol = mol3D()
        for val in SBU:
            SBU_mol.addAtom(molcif.getAtom(val))
        SBU_mol.graph = SBU_subgraph[i].todense()
        """""" """
        For each linker connected to the SBU, find the lc atoms for the lc-RACs.
        """ """"""

        for j, linker in enumerate(connections_list):
            descriptor_names = []
            descriptors = []
            if len(set(SBU).intersection(linker)) > 0:
                temp_mol = mol3D()
                link_list = []
                for jj, val2 in enumerate(linker):
                    if val2 in anchoring_atoms:
                        link_list.append(jj)
                    temp_mol.addAtom(molcif.getAtom(val2))

                temp_mol.graph = connections_subgraphlist[j].todense()
                """""" """
                Generate all of the lc autocorrelations
                """ """"""
                results_dictionary = generate_atomonly_autocorrelations(
                    temp_mol, link_list, loud=False, depth=depth, oct=False)
                descriptor_names, descriptors = append_descriptors(
                    descriptor_names, descriptors,
                    results_dictionary['colnames'],
                    results_dictionary['results'], 'lc', 'all')
                results_dictionary = generate_atomonly_deltametrics(
                    temp_mol, link_list, loud=False, depth=depth, oct=False)
                descriptor_names, descriptors = append_descriptors(
                    descriptor_names, descriptors,
                    results_dictionary['colnames'],
                    results_dictionary['results'], 'D_lc', 'all')
                """""" """
                If functional groups exist (anything that is not C or H, so methyl is missed, also excludes anything lc, so carboxylic metal-coordinating oxygens skipped), 
                compile the list of atoms
                """ """"""
                functional_atoms = []
                for jj in range(len(temp_mol.graph)):
                    if not jj in link_list:
                        if not set({temp_mol.atoms[jj].sym}) & set({"C", "H"}):
                            functional_atoms.append(jj)

                if len(functional_atoms) > 0:
                    results_dictionary = generate_atomonly_autocorrelations(
                        temp_mol,
                        functional_atoms,
                        loud=False,
                        depth=depth,
                        oct=False)
                    descriptor_names, descriptors = append_descriptors(
                        descriptor_names, descriptors,
                        results_dictionary['colnames'],
                        results_dictionary['results'], 'func', 'all')
                    results_dictionary = generate_atomonly_deltametrics(
                        temp_mol,
                        functional_atoms,
                        loud=False,
                        depth=depth,
                        oct=False)
                    descriptor_names, descriptors = append_descriptors(
                        descriptor_names, descriptors,
                        results_dictionary['colnames'],
                        results_dictionary['results'], 'D_func', 'all')
                else:
                    descriptor_names, descriptors = append_descriptors(
                        descriptor_names, descriptors,
                        results_dictionary['colnames'],
                        list(numpy.zeros(int(6 * (depth + 1)))), 'func', 'all')
                    descriptor_names, descriptors = append_descriptors(
                        descriptor_names, descriptors,
                        results_dictionary['colnames'],
                        list(numpy.zeros(int(6 * (depth + 1)))), 'D_func',
                        'all')

                for val in descriptors:
                    if not (type(val) == float
                            or isinstance(val, numpy.float64)):
                        print(
                            'Mixed typing. Please convert to python float, and avoid np float'
                        )
                        sardines
                descriptor_names += ['name']
                descriptors += [name]
                desc_dict = {
                    key2: descriptors[kk]
                    for kk, key2 in enumerate(descriptor_names)
                }
                descriptors.remove(name)
                descriptor_names.remove('name')
                lc_descriptors = lc_descriptors.append(desc_dict,
                                                       ignore_index=True)
                periodic_images = []
                lc_descriptor_list.append(descriptors)
                if j == 0:
                    lc_names = descriptor_names
        averaged_lc_descriptors = np.mean(np.array(lc_descriptor_list), axis=0)
        summed_lc_descriptors = np.sum(np.array(lc_descriptor_list), axis=0)
        lc_descriptors.to_csv(sbu_descriptor_path + '/lc_descriptors.csv',
                              index=False)
        descriptors = []
        descriptor_names = []
        SBU_mol_cart_coords = np.array(
            [atom.coords() for atom in SBU_mol.atoms])
        SBU_mol_atom_labels = [atom.sym for atom in SBU_mol.atoms]
        SBU_mol_adj_mat = np.array(SBU_mol.graph)
        ###### WRITE THE SBU MOL TO THE PLACE
        if sbupath and not os.path.exists(sbupath + "/" + str(name) + str(i) +
                                          '.xyz'):
            xyzname = sbupath + "/" + str(name) + "_sbu_" + str(i) + ".xyz"
            SBU_mol_fcoords_connected = XYZ_connected(cell,
                                                      SBU_mol_cart_coords,
                                                      SBU_mol_adj_mat)
            writeXYZandGraph(xyzname, SBU_mol_atom_labels, cell,
                             SBU_mol_fcoords_connected, SBU_mol_adj_mat)
        """""" """
        Generate all of the SBU based RACs (full scope, mc)
        """ """"""
        results_dictionary = generate_full_complex_autocorrelations(
            SBU_mol, depth=depth, loud=False, flag_name=False)
        descriptor_names, descriptors = append_descriptors(
            descriptor_names, descriptors, results_dictionary['colnames'],
            results_dictionary['results'], 'f', 'all')
        results_dictionary = generate_metalcenter_autocorrelations(
            SBU_mol, metal_ind_onSBU, depth=depth, loud=False)
        descriptor_names, descriptors = append_descriptors(
            descriptor_names, descriptors, results_dictionary['colnames'],
            results_dictionary['results'], 'mc', 'all')
        results_dictionary = generate_metalcenter_deltametrics(SBU_mol,
                                                               metal_ind_onSBU,
                                                               depth=depth,
                                                               loud=False)
        descriptor_names, descriptors = append_descriptors(
            descriptor_names, descriptors, results_dictionary['colnames'],
            results_dictionary['results'], 'D_mc', 'all')

        # results_dictionary = generate_metal_autocorrelations(SBU_mol,depth=depth,loud=False)
        # descriptor_names, descriptors =  append_descriptors(descriptor_names, descriptors, results_dictionary['colnames'],results_dictionary['results'],'mc','all')
        # results_dictionary = generate_metal_deltametrics(SBU_mol,depth=depth,loud=False)
        # descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,results_dictionary['colnames'],results_dictionary['results'],'D_mc','all')
        descriptor_names += ['name']
        descriptors += [name]
        descriptors == list(descriptors)
        desc_dict = {
            key: descriptors[ii]
            for ii, key in enumerate(descriptor_names)
        }
        descriptors.remove(name)
        descriptor_names.remove('name')
        sbu_descriptors = sbu_descriptors.append(desc_dict, ignore_index=True)
        descriptor_list.append(descriptors)
        if i == 0:
            names = descriptor_names
    sbu_descriptors.to_csv(sbu_descriptor_path + '/sbu_descriptors.csv',
                           index=False)
    averaged_SBU_descriptors = np.mean(np.array(descriptor_list), axis=0)
    summed_SBU_descriptors = np.sum(np.array(descriptor_list), axis=0)
    summed_names = ["sum-" + k for k in names]
    summed_lc_names = ["sum-" + k for k in lc_names]
    # if len(names)!=60 and len(lc_names)!=40:
    #     #print(len(names))
    #     sardines2 #This should be 60 units long (60 descriptors) (100 with addition of ligand based descriptors)
    # if len(averaged_SBU_descriptors)!= 60 and len(averaged_lc_descriptors) != 40:
    #     sardines3
    return names, averaged_SBU_descriptors, summed_names, summed_SBU_descriptors, lc_names, averaged_lc_descriptors, summed_lc_names, summed_lc_descriptors


def find_functionalgroups(linkerlist, linker_subgraphlist, molcif, depth):
    nlink = len(linkerlist)
    metal_set = set(molcif.findMetal())
    functionalgroups_list = []
    functionalgroups_subgraphs = []
    for i, linker in enumerate(linkerlist):
        # 1. make mol3D of linker and print it
        linker_mol = mol3D()
        for val in linker:
            linker_mol.addAtom(molcif.getAtom(val))
        linker_mol.graph = linker_subgraphlist[i].todense()
        linker_mol_cart_coords = np.array(
            [atom.coords() for atom in linker_mol.atoms])
        linker_mol_atom_labels = [atom.sym for atom in linker_mol.atoms]
        linker_mol_adj_mat = np.array(linker_mol.graph)

        link_list = []
        lc_atoms = []
        # find the anchoring atoms to metals:
        for jj, linkeratom in enumerate(linker):
            all_bonded_atoms = set(molcif.getBondedAtomsSmart(linkeratom))
            if metal_set.intersection(all_bonded_atoms):
                link_list.append(jj)
                lc_atoms.append(linkeratom)
        """""" """
        If functional groups exist (anything that is not C or H, so methyl is missed, also excludes anything lc, so carboxylic metal-coordinating oxygens skipped), 
        compile the list of atoms
        """ """"""
        functional_atoms = []
        for jj in range(len(linker_mol.graph)):
            if not jj in link_list:
                if not set({linker_mol.atoms[jj].sym}) & set({"C", "H"}):
                    functional_atoms.append(jj)
        if len(functional_atoms) > 0:
            # dumping the xyz file of the functional group graph with 3 extra shells
            functional_atoms_list = [[i] for i in functional_atoms]
            func_list, func_subgraphlist = include_extra_shells(
                copy.deepcopy(functional_atoms_list), [], linker_mol,
                linker_mol.graph)
            for i in range(depth - 1):
                func_list, func_subgraphlist = include_extra_shells(
                    copy.deepcopy(func_list), [], linker_mol, linker_mol.graph)
            for l in range(len(func_list)):
                func_list_molcif = [linker[kk] for kk in func_list[l]]
                functionalgroups_list.append(func_list_molcif)
                functionalgroups_subgraphs.append(func_subgraphlist[l])
    return functionalgroups_list, functionalgroups_subgraphs


def make_MOF_linker_RACs(linkerlist,
                         linker_subgraphlist,
                         molcif,
                         depth,
                         name,
                         cell,
                         linkerpath=False):
    descriptor_list = []
    nlink = len(linkerlist)
    lc_descriptors = []
    lc_names = []
    func_descriptors = []
    func_names = []
    full_descriptors = []
    full_names = []
    metal_set = set(molcif.findMetal())
    if linkerpath:
        linker_descriptor_path = os.path.dirname(linkerpath)
        if os.path.getsize(linker_descriptor_path +
                           '/linker_descriptors.csv') > 0:
            linker_descriptors = pd.read_csv(linker_descriptor_path +
                                             '/linker_descriptors.csv')
        else:
            linker_descriptors = pd.DataFrame()

    for i, linker in enumerate(linkerlist):
        # 1. make mol3D of linker and print it
        linker_mol = mol3D()
        for val in linker:
            linker_mol.addAtom(molcif.getAtom(val))
        linker_mol.graph = linker_subgraphlist[i].todense()
        linker_mol_cart_coords = np.array(
            [atom.coords() for atom in linker_mol.atoms])
        linker_mol_atom_labels = [atom.sym for atom in linker_mol.atoms]
        linker_mol_adj_mat = np.array(linker_mol.graph)
        ###### WRITE THE LINKER MOL TO THE PLACE
        # if linkerpath and not os.path.exists(linkerpath+"/"+str(name)+str(i)+".xyz"):
        #     # linker_mol.writexyz(linkerpath+"/"+str(name)+str(i))
        #     xyzname = linkerpath+"/"+str(name)+"_linker_"+str(i)+".xyz"
        #     linker_mol_fcoords_connected = XYZ_connected(cell , linker_mol_cart_coords , linker_mol_adj_mat )
        #     writeXYZandGraph(xyzname , linker_mol_atom_labels , cell , linker_mol_fcoords_connected,linker_mol_adj_mat)

        link_list = []
        lc_atoms = []
        descriptor_names = []
        descriptors = []
        # find the anchoring atoms to metals:
        for jj, linkeratom in enumerate(linker):
            all_bonded_atoms = set(molcif.getBondedAtomsSmart(linkeratom))
            if metal_set.intersection(all_bonded_atoms):
                link_list.append(jj)
                lc_atoms.append(linkeratom)

        if not len(lc_atoms) > 0:
            print(
                'A linker with no connection to metals! cannot be featurized')
            continue
        else:
            """""" """
            Generate all of the lc autocorrelations
            """ """"""
            results_dictionary = generate_atomonly_autocorrelations(
                linker_mol, link_list, loud=False, depth=depth, oct=False)
            descriptor_names, descriptors = append_descriptors(
                descriptor_names, descriptors, results_dictionary['colnames'],
                results_dictionary['results'], 'lc', 'all')
            results_dictionary = generate_atomonly_deltametrics(linker_mol,
                                                                link_list,
                                                                loud=False,
                                                                depth=depth,
                                                                oct=False)
            descriptor_names, descriptors = append_descriptors(
                descriptor_names, descriptors, results_dictionary['colnames'],
                results_dictionary['results'], 'D_lc', 'all')
            """""" """
            If functional groups exist (anything that is not C or H, so methyl is missed, also excludes anything lc, so carboxylic metal-coordinating oxygens skipped), 
            compile the list of atoms
            """ """"""
            functional_atoms = []
            for jj in range(len(linker_mol.graph)):
                if not jj in link_list:
                    if not set({linker_mol.atoms[jj].sym}) & set({"C", "H"}):
                        functional_atoms.append(jj)

            if len(functional_atoms) > 0:
                results_dictionary = generate_atomonly_autocorrelations(
                    linker_mol,
                    functional_atoms,
                    loud=False,
                    depth=depth,
                    oct=False)
                descriptor_names, descriptors = append_descriptors(
                    descriptor_names, descriptors,
                    results_dictionary['colnames'],
                    results_dictionary['results'], 'func', 'all')
                results_dictionary = generate_atomonly_deltametrics(
                    linker_mol,
                    functional_atoms,
                    loud=False,
                    depth=depth,
                    oct=False)
                descriptor_names, descriptors = append_descriptors(
                    descriptor_names, descriptors,
                    results_dictionary['colnames'],
                    results_dictionary['results'], 'D_func', 'all')
            else:
                descriptor_names, descriptors = append_descriptors(
                    descriptor_names, descriptors,
                    results_dictionary['colnames'],
                    list(numpy.zeros(int(6 * (depth + 1)))), 'func', 'all')
                descriptor_names, descriptors = append_descriptors(
                    descriptor_names, descriptors,
                    results_dictionary['colnames'],
                    list(numpy.zeros(int(6 * (depth + 1)))), 'D_func', 'all')

            for val in descriptors:
                if not (type(val) == float or isinstance(val, numpy.float64)):
                    print(
                        'Mixed typing. Please convert to python float, and avoid np float'
                    )
                    sardines

        # 2. make full linker RACs
        allowed_strings = [
            'electronegativity', 'nuclear_charge', 'ident', 'topology', 'size'
        ]
        labels_strings = ['chi', 'Z', 'I', 'T', 'S']
        colnames = []
        lig_full = list()
        ligand_ac_full = []
        for ii, properties in enumerate(allowed_strings):
            if not list():
                ligand_ac_full = full_autocorrelation(linker_mol, properties,
                                                      depth)
            else:
                ligand_ac_full += full_autocorrelation(linker_mol, properties,
                                                       depth)
            this_colnames = []
            for j in range(0, depth + 1):
                this_colnames.append('f-lig-' + labels_strings[ii] + '-' +
                                     str(j))
            colnames.append(this_colnames)
            lig_full.append(ligand_ac_full)
        lig_full = [item for sublist in lig_full
                    for item in sublist]  #flatten lists
        colnames = [item for sublist in colnames for item in sublist]
        descriptors += lig_full
        descriptor_names += colnames

        # merging all linker descriptors
        descriptor_names += ['name']
        descriptors += [name]
        desc_dict = {
            key2: descriptors[kk]
            for kk, key2 in enumerate(descriptor_names)
        }
        descriptors.remove(name)
        descriptor_names.remove('name')
        linker_descriptors = linker_descriptors.append(desc_dict,
                                                       ignore_index=True)
        descriptor_list.append(descriptors)

    linker_descriptors.to_csv(linker_descriptor_path +
                              '/linker_descriptors.csv',
                              index=False)
    averaged_linker_descriptors = list(
        np.mean(np.array(descriptor_list), axis=0))
    summed_linker_descriptors = list(np.sum(np.array(descriptor_list), axis=0))
    summed_colnames = ["sum-" + k for k in descriptor_names]
    descriptor_names += summed_colnames
    descriptors = averaged_linker_descriptors + summed_linker_descriptors
    return descriptor_names, descriptors


###   def get_ECFP_graphs(linkerlist, linker_subgraphlist,linkerpath, molcif,name,fp_size=1024,fp_rad = 4):
###   #     if linkerpath:
###   #         linker_descriptor_path = os.path.dirname(linkerpath)
###   #         if os.path.getsize(linker_descriptor_path+'/mb_descriptors.csv')>0:
###   #             mb_descriptors = pd.read_csv(linker_descriptor_path+'/mb_descriptors.csv')
###   #         else:
###   #             mb_descriptors = pd.DataFrame()
###       # 1. find unique subgraphs
###       unique_graphs = []
###       unique_ids = []
###       unique_atom_types = []
###       nm = iso.categorical_node_match('atomicNum',"")
###       for i, linker in enumerate(linkerlist):
###           at_types = []
###           for at in linker:
###               at_types.append(molcif.atoms[at].symbol())
###           adjmat = np.triu(linker_subgraphlist[i].todense())
###           rows, cols = np.where(adjmat == 1)
###           edges = [(a,b) for a,b in zip(rows.tolist(), cols.tolist())]
###           atNum = get_atomicNumList(at_types)
###           nodes = np.arange(len(at_types))
###           molgr = make_graph_from_nodes_edges(nodes , edges, atNum)
###           new_unique = True
###           for ji in range(len(unique_graphs)):
###               j = len(unique_graphs) - ji - 1
###               try:
###                   if set(at_types) == unique_atom_types[j]:
###                       gr = unique_graphs [j]
###                       if nx.is_isomorphic(gr,molgr,node_match = nm):
###                           new_unique = False
###                           break
###               except IndexError:
###                   continue
###
###           if new_unique:
###               unique_ids.append(i)
###               unique_graphs.append(molgr)
###               unique_atom_types.append(set(at_types))
###
###       fingerprints = []
###       for i in unique_ids:
###           xyzname = linkerpath+"/"+str(name)+"_linker_"+str(i)+".xyz"
###           fp = get_ECFP_from_XYZ(xyzname,linker_subgraphlist[i].todense(), morgan_radii = fp_rad, bitsize = fp_size, charged_fragments = True, charge = 0,quick = True)
###           fingerprints.append(fp)
###           if i > 0:
###               fullfingerprint = fullfingerprint | fp
###           else:
###               fullfingerprint = fp
###
###       npfp = numpy.zeros((1,))
###       DataStructs.ConvertToNumpyArray(fullfingerprint, npfp)
###       fp_names = ["mbi_%i"%i for i in range(fullfingerprint.GetNumBits()) ]
###       return fp_names,npfp
###
###
###       # 2. for each unique graph: compute ECFP using RDKit and get_ECFP()
###       # 3. use and function for the bit string and return it
###
###   def get_ECFP_from_XYZ(filename,adjmat,morgan_radii = 4, bitsize = 1024,charged_fragments = True, charge = 0,quick = True):
###       # 1. build mol object compatible with RDKit
###       atomicNumList, charge, xyz_coordinates,atom_types = read_xyz_file(filename)
###       mol = xyz2mol(atomicNumList, 0, adjmat, xyz_coordinates, charged_fragments, quick)
###       # 2. compute and return ECFP
###       filename_sdf = filename.replace(".xyz", "")
###       filename_sdf += ".sdf"
###       writer = Chem.SDWriter(filename_sdf)
###       writer.write(mol)
###       smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
###       m = Chem.MolFromSmiles(smiles)
###       smiles = Chem.MolToSmiles(m, isomericSmiles=True)
###       writeXYZcoords_withcomment(filename,atom_types,xyz_coordinates,smiles)
###       bi = {}
###       fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=morgan_radii, nBits=1024 , bitInfo=bi)
###       return fp
###
###
###
###
###   def get_ECFP(molgraph,morgan_radii = 4, bitsize = 1024,charged_fragments = True, charge = 0,quick = True):
###       # takes a molecular graph with edges being adjmat and labels be atom number
###       # since we are working with MOF linkers, we need to hydrogenate them otherwise they count as charged
###       # 1. build mol object compatible with RDKit
###       mol = xyz2mol(atomicNumList, charge, xyz_coordinates, charged_fragments, quick)
###       atomicNumList = nx.get_node_attributes(molgraph,'atomicNum').values()
###       mol = get_proto_mol(atomicNumList)
###       conf = Chem.Conformer(mol.GetNumAtoms())
###       mol.AddConformer(conf)
###       AC = nx.adjacency_matrix(molgraph)
###       BO,atomic_valence_electrons = AC2BO(AC,atomicNumList,charge,charged_fragments,quick)
###       mol = BO2mol(mol,BO, atomicNumList,atomic_valence_electrons,charge,charged_fragments)
###       from rdkit.Chem import Draw
###       Draw.MolToFile(mol, "test.png")
###
###       # 2. compute and return ECFP
###       bi = {}
###       fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=morgan_radii, bitInfo=bi)
###       import inspect
###       ss = inspect.getmembers(fp1, lambda a:not(inspect.isroutine(a)))
###       print(ss)
###       for i in range(fp1.GetNumBits()):
###           if fp1.GetBit(i)==True:
###               print(i)
###
###       sys.exit()
###       from rdkit.Chem import Draw
###       Draw.MolToFile(mol, "test.png")
###       mfp2_svg = Draw.DrawMorganBit(mol, 88, bi)
###       print(mfp2_svg)
###
###       # from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray , GetLength
###
###       # t = ConvertToNumpyArray(fp1)
###       # print(t)

###     if args.sdf:
###         filename = filename.replace(".xyz", "")
###         filename += ".sdf"
###         writer = Chem.SDWriter(filename)
###         writer.write(mol)
###
###     # Canonical hack
###     smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
###     m = Chem.MolFromSmiles(smiles)
###     smiles = Chem.MolToSmiles(m, isomericSmiles=True)
###
###     print(smiles)


def get_MOF_descriptors(data,
                        name,
                        depth,
                        path=False,
                        xyzpath=False,
                        check4ligand=True,
                        check4SBU=True,
                        reg_SBU=False):
    if not path:
        print(
            'Need a directory to place all of the linker, SBU, and ligand objects. Exiting now.'
        )
        sardines
    else:
        if path.endswith('/'):
            path = path[:-1]
        if not os.path.isdir(path + '/ligands'):
            os.mkdir(path + '/ligands')
        if not os.path.isdir(path + '/linkers'):
            os.mkdir(path + '/linkers')
        if not os.path.isdir(path + '/sbus'):
            os.mkdir(path + '/sbus')
        if not os.path.isdir(path + '/xyz'):
            os.mkdir(path + '/xyz')
        if not os.path.isdir(path + '/logs'):
            os.mkdir(path + '/logs')
        if not os.path.exists(path + '/mc_descriptors.csv'):
            with open(path + '/mc_descriptors.csv', 'w') as f:
                f.close()
        if not os.path.exists(path + '/sbu_descriptors.csv'):
            with open(path + '/sbu_descriptors.csv', 'w') as f:
                f.close()
        if not os.path.exists(path + '/linker_descriptors.csv'):
            with open(path + '/linker_descriptors.csv', 'w') as g:
                g.close()
        if not os.path.exists(path + '/lc_descriptors.csv'):
            with open(path + '/lc_descriptors.csv', 'w') as h:
                h.close()
    ligandpath = path + '/ligands'
    linkerpath = path + '/linkers'
    sbupath = path + '/sbus'
    logpath = path + "/logs"
    """""" """
    Input cif file and get the cell parameters and adjacency matrix. If overlap, do not featurize.
    Simultaneously prepare mol3D class for MOF for future RAC featurization (molcif)
    """ """"""

    cpar, allatomtypes, fcoords = readcif(data)
    cell_v = mkcell(cpar)
    cart_coords = fractional2cart(fcoords, cell_v)
    if len(cart_coords) > 2000:
        print("Too large cif file, skip it for now")
        full_names = [0]
        full_descriptors = [0]
        tmpstr = "Failed to featurize %s: large primitive cell\n" % (name)
        write2file(path, "/FailedStructures.log", tmpstr)
        return full_names, full_descriptors
    distance_mat = compute_distance_matrix2(cell_v, cart_coords)
    try:
        adj_matrix = compute_adj_matrix(distance_mat, allatomtypes)
    except NotImplementedError:
        full_names = [0]
        full_descriptors = [0]
        tmpstr = "Failed to featurize %s: atomic overlap\n" % (name)
        write2file(path, "/FailedStructures.log", tmpstr)
        return full_names, full_descriptors

    writeXYZandGraph(xyzpath, allatomtypes, cell_v, fcoords,
                     adj_matrix.todense())
    molcif, _, _, _, _ = import_from_cif(data, True)
    molcif.graph = adj_matrix.todense()
    """""" """
    check number of connected components.
    if more than 1: it checks if the structure is interpenetrated. Fails if no metal in one of the connected components (identified by the graph).
    This includes floating solvent molecules.
    """ """"""

    n_components, labels_components = sparse.csgraph.connected_components(
        csgraph=adj_matrix, directed=False, return_labels=True)
    metal_list = set([at for at in molcif.findMetal()])
    if not len(metal_list) > 0:
        full_names = [0]
        full_descriptors = [0]
        tmpstr = "Failed to featurize %s: no metal found\n" % (name)
        write2file(path, "/FailedStructures.log", tmpstr)
        return full_names, full_descriptors

    for comp in range(n_components):
        # TODO: Here, we can add a flag to print out a cleaned cif file (ie removing solvents)
        # Also, it could be extended to coordinated ligands by checking the number of time crossing PBC
        inds_in_comp = [
            i for i in range(len(labels_components))
            if labels_components[i] == comp
        ]
        if not set(inds_in_comp) & metal_list:
            full_names = [0]
            full_descriptors = [0]
            tmpstr = "Failed to featurize %s: solvent molecules\n" % (name)
            write2file(path, "/FailedStructures.log", tmpstr)
            return full_names, full_descriptors

    if n_components > 1:
        print("structure is interpenetrated")
        tmpstr = "%s found to be an interpenetrated structure\n" % (name)
        write2file(logpath, "/%s.log" % name, tmpstr)
    """""" """
    step 1: metalic part
        removelist = metals (1) + atoms only connected to metals (2) + H connected to (1+2)
        SBUlist = removelist + 1st coordination shell of the metals
    removelist = set()
    Logs the atom types of the connecting atoms to the metal in logpath.
    """ """"""
    metal_list = set([at for at in molcif.findMetal()])
    SBUlist = set()
    [SBUlist.update(set([metal]))
     for metal in molcif.findMetal()]  #Remove all metals as part of the SBU
    [
        SBUlist.update(set(molcif.getBondedAtomsSmart(metal)))
        for metal in molcif.findMetal()
    ]
    removelist = set()
    [removelist.update(set([metal]))
     for metal in molcif.findMetal()]  #Remove all metals as part of the SBU
    for metal in removelist:
        bonded_atoms = set(molcif.getBondedAtomsSmart(metal))
        bonded_atoms_types = set([
            str(allatomtypes[at])
            for at in set(molcif.getBondedAtomsSmart(metal))
        ])
        cn = len(bonded_atoms)
        cn_atom = ",".join([at for at in bonded_atoms_types])
        tmpstr = "atom %i with type of %s found to have %i coordinates with atom types of %s\n" % (
            metal, allatomtypes[metal], cn, cn_atom)
        write2file(logpath, "/%s.log" % name, tmpstr)

    for atom in SBUlist:
        all_bonded_atoms = molcif.getBondedAtomsSmart(atom)
        only_bonded_metal_hydrogen = True
        for val in all_bonded_atoms:
            if not (molcif.getAtom(val).symbol().upper() == 'H'
                    or molcif.getAtom(val).ismetal()):
                only_bonded_metal_hydrogen = False

        if only_bonded_metal_hydrogen:
            removelist.update(set([atom]))

    # more elegant but confusing!
    # [removelist.update(set([atom])) for atom in SBUlist if all((molcif.getAtom(val).ismetal() or molcif.getAtom(val).symbol().upper() == 'H') for val in molcif.getBondedAtomsSmart(atom))]
    """""" """
    adding hydrogens connected to atoms which are only connected to metals. In particular interstitial OH, like in UiO SBU.
    """ """"""
    tobeadded = []
    for atom in removelist:
        for val in molcif.getBondedAtomsSmart(atom):
            if molcif.getAtom(val).symbol().upper() == 'H':
                tobeadded.append(val)
    removelist.update(set(tobeadded))
    """""" """
    At this point:
    The remove list only removes metals and things ONLY connected to metals or hydrogens. 
    Thus the coordinating atoms are double counted in the linker.                         
    
    step 2: organic part
        removelist = linkers are all atoms - the removelist (assuming no bond between 
        organiclinkers)
    """ """"""
    allatoms = set(range(0, adj_matrix.shape[0]))
    linkers = allatoms - removelist
    linker_list, linker_subgraphlist = get_closed_subgraph(
        linkers.copy(), removelist.copy(), adj_matrix)
    connections_list = copy.deepcopy(linker_list)
    connections_subgraphlist = copy.deepcopy(linker_subgraphlist)
    linker_length_list = [len(linker_val) for linker_val in linker_list]

    adjmat = adj_matrix.todense()
    """""" """
    find all anchoring atoms on linkers and ligands (lc identification)
    """ """"""
    anc_atoms = set()
    for linker in linker_list:
        for atom_linker in linker:
            all_bonded_atoms = np.nonzero(adj_matrix[atom_linker, :])[1]
            if set(all_bonded_atoms) & metal_list:
                anc_atoms.add(atom_linker)
    """""" """
    step 3: linker or ligand ?
    checking to find the anchors and #SBUs that are connected to an organic part
    anchor <= 1 -> ligand
    anchor > 1 and #SBU > 1 -> linker
    else: walk over the linker graph and count #crossing PBC
        if #crossing is odd -> linker
        else -> ligand
    """ """"""

    initial_SBU_list, initial_SBU_subgraphlist = get_closed_subgraph(
        removelist.copy(), linkers.copy(), adj_matrix)

    templist = linker_list[:]
    tempgraphlist = linker_subgraphlist[:]
    long_ligands = False
    max_min_linker_length, min_max_linker_length = (0, 100)
    if check4ligand or reg_SBU:
        for ii, atoms_list in reversed(list(
                enumerate(linker_list))):  #Loop over all linker subgraphs
            linkeranchors_list = set()
            linkeranchors_atoms = set()
            sbuanchors_list = set()
            sbu_connect_list = set()
            """""" """
            Here, we are trying to identify what is actually a linker and what is a ligand. 
            To do this, we check if something is connected to more than one SBU. Set to     
            handle cases where primitive cell is small, ambiguous cases are recorded.       
            """ """"""
            for iii, atoms in enumerate(
                    atoms_list):  #loop over all atoms in a linker
                connected_atoms = np.nonzero(adj_matrix[atoms, :])[1]
                for kk, sbu_atoms_list in enumerate(
                        initial_SBU_list):  #loop over all SBU subgraphs
                    for sbu_atoms in sbu_atoms_list:  #Loop over SBU
                        if sbu_atoms in connected_atoms:
                            linkeranchors_list.add(iii)
                            linkeranchors_atoms.add(atoms)
                            sbuanchors_list.add(sbu_atoms)
                            sbu_connect_list.add(kk)  #Add if unique SBUs

            min_length, max_length = linker_length(
                linker_subgraphlist[ii].todense(), linkeranchors_list)
            if len(
                    linkeranchors_list
            ) >= 2:  # linker, and in one ambigous case, could be a ligand.
                if len(
                        sbu_connect_list
                ) >= 2:  #Something that connects two SBUs is certain to be a linker
                    max_min_linker_length = max(min_length,
                                                max_min_linker_length)
                    min_max_linker_length = min(max_length,
                                                min_max_linker_length)
                    continue
                else:
                    # check number of times we cross PBC :
                    # TODO: we still can fail in multidentate ligands!
                    linker_cart_coords=np.array([at.coords() \
                            for at in [molcif.getAtom(val) for val in atoms_list]])
                    linker_adjmat = np.array(linker_subgraphlist[ii].todense())
                    #writeXYZfcoords("sbu_%i.xyz"%iii,atoms,cell,fcoords):
                    pr_image_organic = ligand_detect(cell_v,
                                                     linker_cart_coords,
                                                     linker_adjmat,
                                                     linkeranchors_list)
                    sbu_temp = linkeranchors_atoms.copy()
                    sbu_temp.update({
                        val
                        for val in initial_SBU_list[list(sbu_connect_list)[0]]
                    })
                    sbu_temp = list(sbu_temp)
                    sbu_cart_coords=np.array([at.coords() \
                           for at in [molcif.getAtom(val) for val in sbu_temp]])
                    sbu_adjmat = slice_mat(adj_matrix.todense(), sbu_temp)
                    pr_image_sbu = ligand_detect(
                        cell_v, sbu_cart_coords, sbu_adjmat,
                        set(range(len(linkeranchors_list))))

                    if not (len(np.unique(pr_image_sbu, axis=0)) == 1
                            and len(np.unique(pr_image_organic,
                                              axis=0)) == 1):  # linker
                        max_min_linker_length = max(min_length,
                                                    max_min_linker_length)
                        min_max_linker_length = min(max_length,
                                                    min_max_linker_length)
                        tmpstr = str(name)+','+' Anchors list: '+str(sbuanchors_list) \
                                +','+' SBU connectlist: '+str(sbu_connect_list)+' set to be linker\n'
                        write2file(ligandpath, "/ambiguous.txt", tmpstr)
                        continue

                    else:  #  all anchoring atoms are in the same unitcell -> ligand
                        if check4ligand:
                            removelist.update(
                                set(templist[ii]
                                    ))  # we also want to remove these ligands
                            SBUlist.update(
                                set(templist[ii]
                                    ))  # we also want to remove these ligands
                            linker_list.pop(ii)
                            linker_subgraphlist.pop(ii)
                            tmpstr = str(name)+','+' Anchors list: '+str(sbuanchors_list) \
                                    +','+' SBU connectlist: '+str(sbu_connect_list)+' set to be ligand\n'
                            write2file(ligandpath, "/ambiguous.txt", tmpstr)
                            tmpstr = str(name)+str(ii)+','+' Anchors list: '+ \
                                    str(sbuanchors_list)+','+' SBU connectlist: '+str(sbu_connect_list)+'\n'
                            write2file(ligandpath, "/ligand.txt", tmpstr)

            else:  #definite ligand
                if check4ligand:
                    print(linkeranchors_list)
                    print(templist[ii])
                    write2file(logpath, "/%s.log" % name, "found ligand\n")
                    removelist.update(set(
                        templist[ii]))  # we also want to remove these ligands
                    SBUlist.update(set(
                        templist[ii]))  # we also want to remove these ligands
                    linker_list.pop(ii)
                    linker_subgraphlist.pop(ii)
                    tmpstr = str(name)+','+' Anchors list: '+str(sbuanchors_list) \
                            +','+' SBU connectlist: '+str(sbu_connect_list)+'\n'
                    write2file(ligandpath, "/ligand.txt", tmpstr)


        tmpstr = str(name) + ", (min_max_linker_length,max_min_linker_length): " + \
                    str(min_max_linker_length) + " , " +str(max_min_linker_length) + "\n"
        write2file(logpath, "/%s.log" % name, tmpstr)
        if min_max_linker_length < 3:
            write2file(linkerpath, "/short_ligands.txt", tmpstr)

        if min_max_linker_length > 2:
            # for N-C-C-N ligand ligand
            if max_min_linker_length == min_max_linker_length:
                long_ligands = True
            elif min_max_linker_length > 3:
                long_ligands = True
    """""" """
    In the case of long linkers, add second coordination shell without further checks. In the case of short linkers, start from metal
    and grow outwards using the include_exta_shells function
    """ """"""
    linker_length_list = [len(linker_val) for linker_val in linker_list]
    if not check4SBU:
        # 1.1. find metals and make 2 coord shells of them
        metalic_atoms_list = [[i] for i, at in enumerate(allatomtypes)
                              if at in metalslist]
        SBU_list, SBU_subgraphlist = include_extra_shells(
            copy.deepcopy(metalic_atoms_list), [], molcif, adj_matrix)
        for i in range(depth - 1):
            SBU_list, SBU_subgraphlist = include_extra_shells(
                copy.deepcopy(SBU_list), [], molcif, adj_matrix)
        # truncated_linkers = allatoms - removelist
        # SBU_list, SBU_subgraphlist = get_closed_subgraph(removelist, truncated_linkers, adj_matrix)
        # SBU_list , SBU_subgraphlist = include_extra_shells(SBU_list,SBU_subgraphlist,molcif ,adj_matrix)
        # SBU_list , SBU_subgraphlist = include_extra_shells(SBU_list,SBU_subgraphlist,molcif ,adj_matrix)

    if check4SBU:
        """
        This must not be used for featurisation because T is wrongly assigned for the terminating atoms
        only for printing purpose
        """
        if reg_SBU:
            if min_max_linker_length > 3:
                [
                    [
                        SBUlist.add(val)
                        for val in molcif.getBondedAtomsSmart(zero_first_shell)
                    ] for zero_first_shell in SBUlist.copy()
                ]  #First account for all of the carboxylic acid type linkers, add in the carbons.
                truncated_linkers = allatoms - SBUlist
                SBU_list, SBU_subgraphlist = get_closed_subgraph(
                    SBUlist, truncated_linkers, adj_matrix)

            else:
                print(
                    "regorous SBU detection is not implemented for short linkers"
                )
                full_names = [0]
                full_descriptors = [0]
                tmpstr = "Failed to featurize %s: short ligand for rigorous SBU detection\n" % (
                    name)
                write2file(path, "/FailedStructures.log", tmpstr)
                return full_names, full_descriptors

        elif not min_max_linker_length < 2:  # treating the 2 atom ligands differently! Need caution
            if long_ligands:
                tmpstr = "\nStructure has LONG ligand\n\n"
                write2file(logpath, "/%s.log" % name, tmpstr)
                [
                    [
                        SBUlist.add(val)
                        for val in molcif.getBondedAtomsSmart(zero_first_shell)
                    ] for zero_first_shell in SBUlist.copy()
                ]  #First account for all of the carboxylic acid type linkers, add in the carbons.
            truncated_linkers = allatoms - SBUlist
            SBU_list, SBU_subgraphlist = get_closed_subgraph(
                SBUlist, truncated_linkers, adj_matrix)

            if not long_ligands:
                tmpstr = "\nStructure has SHORT ligand\n\n"
                write2file(logpath, "/%s.log" % name, tmpstr)
                # SBU_list , SBU_subgraphlist = include_extra_shells(SBU_list,SBU_subgraphlist,molcif ,adj_matrix)
                for i in range(depth - 1):
                    SBU_list, SBU_subgraphlist = include_extra_shells(
                        copy.deepcopy(SBU_list), [], molcif, adj_matrix)
        else:
            tmpstr = "strucutre %s has extreamly short ligands, check the outputs\n" % name
            write2file(ligandpath, "/ambiguous.txt", tmpstr)
            tmpstr = "Structure has extreamly short ligands\n"
            write2file(logpath, "/%s.log" % name, tmpstr)
            tmpstr = "Structure has extreamly short ligands\n"
            write2file(logpath, "/%s.log" % name, tmpstr)
            truncated_linkers = allatoms - removelist
            metalic_atoms_list = [[i] for i, at in enumerate(allatomtypes)
                                  if at in metalslist]
            for i in range(depth - 1):
                SBU_list, SBU_subgraphlist = include_extra_shells(
                    copy.deepcopy(SBU_list), [], molcif, adj_matrix)
    """""" """
    For the cases that have a linker subgraph, do the featurization.
    """ """"""
    if len(linker_subgraphlist) >= 1:  #Featurize cases that did not fail
        # try:
        if check4SBU:
            # descriptor_names, descriptors,descriptor_names_summed,descriptors_summed,lc_descriptor_names, lc_descriptors,lc_descriptor_names_summed, lc_descriptors_summed = make_MOF_SBU_RACs(SBU_list, SBU_subgraphlist,metalic_atoms_list, molcif, depth, name , cell_v,anc_atoms, sbupath, connections_list, connections_subgraphlist)
            ### test ##
            print_parts(SBU_list,
                        SBU_subgraphlist,
                        molcif,
                        depth,
                        name,
                        cell_v,
                        pr="_sbu_",
                        sbupath=sbupath,
                        checkunique=True)
            print_parts(linker_list,
                        linker_subgraphlist,
                        molcif,
                        depth,
                        name,
                        cell_v,
                        pr="_linker_",
                        sbupath=linkerpath,
                        checkunique=True)
            functionalgroups_list, functionalgroups_subgraphs = find_functionalgroups(
                linker_list, linker_subgraphlist, molcif, depth)
            print_parts(functionalgroups_list,
                        functionalgroups_subgraphs,
                        molcif,
                        depth,
                        name,
                        cell_v,
                        pr="_func_",
                        sbupath=linkerpath,
                        checkunique=True)
            full_names = [0]
            full_descriptors = [0]
            return full_names, full_descriptors
            #mc_descriptors_names, mc_descriptors  = make_crystalgraph_mc_RACs (molcif, depth, name , cell_v, sbupath)
            # lig_descriptor_names, lig_descriptors = make_MOF_linker_RACs(linker_list, linker_subgraphlist, molcif, depth, name, cell_v, linkerpath)
#            lig_descriptor_names, lig_descriptors , lig_descriptor_names_summed, lig_descriptors_summed = make_MOF_linker_RACs(linker_list, linker_subgraphlist, molcif, depth, name, cell_v, linkerpath)
        else:
            mc_descriptors_names, mc_descriptors = make_crystalgraph_mc_RACs(
                molcif, depth, name, cell_v, sbupath)
            lig_descriptor_names, lig_descriptors = make_MOF_linker_RACs(
                linker_list, linker_subgraphlist, molcif, depth, name, cell_v,
                linkerpath)
            ### dump xyz of the subgraphs ###
            functionalgroups_list, functionalgroups_subgraphs = find_functionalgroups(
                linker_list, linker_subgraphlist, molcif, depth)
            print_parts(functionalgroups_list,
                        functionalgroups_subgraphs,
                        molcif,
                        depth,
                        name,
                        cell_v,
                        pr="_func_",
                        sbupath=linkerpath,
                        checkunique=True)
            print_parts(SBU_list,
                        SBU_subgraphlist,
                        molcif,
                        depth,
                        name,
                        cell_v,
                        pr="_sbu_",
                        sbupath=sbupath,
                        checkunique=True)
            print_parts(linker_list,
                        linker_subgraphlist,
                        molcif,
                        depth,
                        name,
                        cell_v,
                        pr="_linker_",
                        sbupath=linkerpath,
                        checkunique=True)
            lig_SMILES, lig_SMILES_count, RDKit_sanity = get_SMILES(
                linker_list,
                linker_subgraphlist,
                molcif,
                cell_v,
                checkunique=True)


#       """""""
#       mputing mc RACs based on crystal graph
#       """""""
#       Y_names, CRY_descriptors = make_crystalgraph_mc_RACs(molcif, depth, name,cell_v, sbupath)
# FP_names, ECFP_descriptors = get_ECFP_graphs(linker_list, linker_subgraphlist, linkerpath,molcif,name,fp_size=1024,fp_rad = 4)
        full_names = mc_descriptors_names + lig_descriptor_names + [
            "linker-SMILES", "linker-count", "RDKitLinkerSanity", "AtomTypes"
        ]
        full_descriptors = list(mc_descriptors) + list(lig_descriptors) + [
            lig_SMILES
        ] + [lig_SMILES_count] + [RDKit_sanity] + [list(set(allatomtypes))]
        # except:
        #     full_names = [0]
        #     full_descriptors = [0]
    elif len(linker_subgraphlist) == 1:  # this never happens, right?
        print('Suspicious featurization')
        full_names = [1]
        full_descriptors = [1]
    else:
        print('Failed to featurize this MOF.')
        full_names = [0]
        full_descriptors = [0]

    if full_names == [0] and full_descriptors == [0]:
        tmpstr = "Failed to featurize %s\n" % (name)
        write2file(path, "/FailedStructures.log", tmpstr)
    return full_names, full_descriptors


def get_mc_MOF_descriptors(data,
                           name,
                           depth,
                           path=False,
                           xyzpath=False,
                           check4ligand=True,
                           check4SBU=True,
                           reg_SBU=False,
                           check4uniquemetals=False):
    if not path:
        print(
            'Need a directory to place all of the linker, SBU, and ligand objects. Exiting now.'
        )
        sardines
    else:
        if path.endswith('/'):
            path = path[:-1]
        if not os.path.isdir(path + '/ligands'):
            os.mkdir(path + '/ligands')
        if not os.path.isdir(path + '/linkers'):
            os.mkdir(path + '/linkers')
        if not os.path.isdir(path + '/sbus'):
            os.mkdir(path + '/sbus')
        if not os.path.isdir(path + '/xyz'):
            os.mkdir(path + '/xyz')
        if not os.path.isdir(path + '/logs'):
            os.mkdir(path + '/logs')
        if not os.path.exists(path + '/mc_descriptors.csv'):
            with open(path + '/mc_descriptors.csv', 'w') as f:
                f.close()
        if not os.path.exists(path + '/linker_descriptors.csv'):
            with open(path + '/linker_descriptors.csv', 'w') as g:
                g.close()
        if not os.path.exists(path + '/lc_descriptors.csv'):
            with open(path + '/lc_descriptors.csv', 'w') as h:
                h.close()
    ligandpath = path + '/ligands'
    linkerpath = path + '/linkers'
    sbupath = path + '/sbus'
    logpath = path + "/logs"
    """""" """
    Input cif file and get the cell parameters and adjacency matrix. If overlap, do not featurize.
    Simultaneously prepare mol3D class for MOF for future RAC featurization (molcif)
    """ """"""

    cpar, allatomtypes, fcoords = readcif(data)
    cell_v = mkcell(cpar)
    cart_coords = fractional2cart(fcoords, cell_v)

    # 1.0. compute full adjmat
    if len(cart_coords) > 2500:
        print("Too large cif file, skip it for now")
        full_names = [0]
        full_descriptors = [0]
        tmpstr = "Failed to featurize %s: large primitive cell\n" % (name)
        write2file(path, "/FailedStructures.log", tmpstr)
        return full_names, full_descriptors
    # distance_mat = compute_distance_matrix2(cell_v,cart_coords)
    try:
        adj_matrix = compute_adj_matrix(distance_mat, allatomtypes)
    except NotImplementedError:
        full_names = [0]
        full_descriptors = [0]
        tmpstr = "Failed to featurize %s: atomic overlap\n" % (name)
        write2file(path, "/FailedStructures.log", tmpstr)
        return full_names, full_descriptors

    writeXYZandGraph(xyzpath, allatomtypes, cell_v, fcoords,
                     adj_matrix.todense())

    molcif, _, _, _, _ = import_from_cif(data, True)
    molcif.graph = adj_matrix.todense()
    """""" """
    check number of connected components.
    if more than 1: it checks if the structure is interpenetrated. Fails if no metal in one of the connected components (identified by the graph).
    This includes floating solvent molecules.
    """ """"""
    n_components, labels_components = sparse.csgraph.connected_components(
        csgraph=adj_matrix, directed=False, return_labels=True)
    metal_list = set([at for at in molcif.findMetal()])
    if not len(metal_list) > 0:
        full_names = [0]
        full_descriptors = [0]
        tmpstr = "Failed to featurize %s: no metal found\n" % (name)
        write2file(path, "/FailedStructures.log", tmpstr)
        return full_names, full_descriptors

    if n_components > 1:
        print("structure is interpenetrated")
        tmpstr = "%s found to be an interpenetrated structure\n" % (name)
        write2file(logpath, "/%s.log" % name, tmpstr)

    # 1.1. find metals and make 2 coord shells of them
    metalic_atoms_list = [[i] for i, at in enumerate(allatomtypes)
                          if at in metalslist]
    SBU_list, SBU_subgraphlist = include_extra_shells(
        copy.deepcopy(metalic_atoms_list), [], molcif, adj_matrix)
    for i in range(depth):
        SBU_list, SBU_subgraphlist = include_extra_shells(
            copy.deepcopy(SBU_list), [], molcif, adj_matrix)
    SBU_atomtypes_list = []
    for SBU in SBU_list:
        tmplist = [allatomtypes[at] for at in SBU]
        SBU_atomtypes_list.append(tmplist)

    # find unique metal types:
    RAC_population = []
    RAC_metal_start = []
    RAC_SBUs = []
    RAC_SBUs_subgraph = []
    RAC_SBUs_molgraph = []
    RAC_atom_types = []
    nm = iso.categorical_node_match('atomtype', "")
    if check4uniquemetals:
        for metal_id, SBU, at_types, SBU_graph in zip(metalic_atoms_list,
                                                      SBU_list,
                                                      SBU_atomtypes_list,
                                                      SBU_subgraphlist):
            nodes = np.arange(len(at_types))
            SBU_adjmat = upper_triangle(SBU_graph.todense())
            rows, cols = np.where(SBU_adjmat == 1)
            edges = [(a, b) for a, b in zip(rows.tolist(), cols.tolist())]
            molgr = make_graph_from_nodes_edges(nodes, edges, at_types)
            new_unique = True
            for ji in range(len(RAC_SBUs_molgraph)):
                j = len(RAC_SBUs_molgraph) - ji - 1
                try:
                    if set(at_types) == RAC_atom_types[j]:
                        gr = RAC_SBUs_molgraph[j]
                        if nx.faster_could_be_isomorphic(gr, molgr):
                            if nx.fast_could_be_isomorphic(gr, molgr):
                                if nx.is_isomorphic(gr, molgr, node_match=nm):
                                    new_unique = False
                                    RAC_population[j] += 1
                                    break
                except IndexError:
                    continue

            if new_unique:
                RAC_metal_start.append(metal_id[0])
                RAC_population.append(1)
                RAC_SBUs_subgraph.append(SBU_graph)
                RAC_SBUs_molgraph.append(molgr)
                RAC_SBUs.append(SBU)
                RAC_atom_types.append(set(at_types))
    else:
        for metal_id, SBU, at_types, SBU_graph in zip(metalic_atoms_list,
                                                      SBU_list,
                                                      SBU_atomtypes_list,
                                                      SBU_subgraphlist):
            RAC_metal_start.append(metal_id[0])
            RAC_population.append(1)
            RAC_SBUs_subgraph.append(SBU_graph)
            RAC_SBUs.append(SBU)
            RAC_atom_types.append(set(at_types))

    # do featurisation
    full_names = []
    full_descriptors = []
    for i, (metalid, SBU, SBU_graph, SBU_population) in enumerate(
            zip(RAC_metal_start, RAC_SBUs, RAC_SBUs_subgraph, RAC_population)):
        descriptor_names, descriptors = make_MOF_mc_RACs(
            SBU, SBU_graph, metalid, molcif, depth, name, cell_v,
            SBU_population, sbupath)
        full_names = full_names + descriptor_names
        full_descriptors = full_descriptors + list(descriptors)

    full_names = [0]
    full_descriptors = [0]
    return full_names, full_descriptors
