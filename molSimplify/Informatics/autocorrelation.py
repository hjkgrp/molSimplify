import glob
import string
import sys
import os
import numpy as np
import math
import random
import string
import numpy
from molSimplify.Scripts.geometry import *
from molSimplify.Classes.atom3D import *
from molSimplify.Classes.globalvars import globalvars
from molSimplify.Classes.mol3D import*
from molSimplify.Classes.ligand import *

	########### UNIT CONVERSION
HF_to_Kcal_mol = 627.503
def full_autocorrelation(mol,prop,d):
	w = construct_property_vector(mol,prop)
	index_set = range(0,mol.natoms)
	autocorrelation_vector = numpy.zeros(d+1)
	for centers in index_set:
		autocorrelation_vector += autocorrelation(mol,w,centers,d)
	return(autocorrelation_vector)
def atom_only_autocorrelation(mol,prop,d,atomIdx):
	## atomIdx must b either a list of indcies
	## or a single index
	w = construct_property_vector(mol,prop)
	autocorrelation_vector = numpy.zeros(d+1)
	if hasattr(atomIdx, "__len__"):
		for elements in atomIdx:
			autocorrelation_vector += autocorrelation(mol,w,elements,d)
		autocorrelation_vector = np.divide(autocorrelation_vector,len(atomIdx))
	else:
		autocorrelation_vector += autocorrelation(mol,w,atomIdx,d)
	return(autocorrelation_vector)
def metal_only_autocorrelation(mol,prop,d):
	autocorrelation_vector = numpy.zeros(d)
	try:
		metal_ind = mol.findMetal()
		w = construct_property_vector(mol,prop)
		autocorrelation_vector = autocorrelation(mol,w,metal_ind,d)
	except:	
		print('Error, no metal found in mol object!')
		return False
	return(autocorrelation_vector)

	w = construct_property_vector(mol,prop)
	index_set = range(0,mol.natoms)
	deltametric_vector = numpy.zeros(d+1)
	for centers in index_set:
		deltametric_vector += deltametric(mol,w,centers,d)
	return(deltametric_vector)
def atom_only_deltametric(mol,prop,d,atomIdx):
	## atomIdx must b either a list of indcies
	## or a single index
	w = construct_property_vector(mol,prop)
	deltametric_vector = numpy.zeros(d+1)
	if hasattr(atomIdx, "__len__"):
		for elements in atomIdx:
			deltametric_vector += deltametric(mol,w,elements,d)
		deltametric_vector = np.divide(deltametric_vector,len(atomIdx))
	else:
		deltametricvector += deltametric(mol,w,atomIdx,d)
	return(deltametric_vector)
def metal_only_deltametric(mol,prop,d):
	deltametric_vector = numpy.zeros(d)
	try:
		metal_ind = mol.findMetal()
		w = construct_property_vector(mol,prop)
		deltametric_vector = deltametric(mol,w,metal_ind,d)
	except:	
		print('Error, no metal found in mol object!')
		return False
	return(deltametric_vector)
 
def autocorrelation(mol,prop_vec,orig,d):
	## this function returns the autocorrelation 
	## for one atom
	# Inputs:
	#	mol - mol3D class
	#	prop_vec - vector, property of atoms in mol in order of index
	#	orig -  int, zero-indexed starting atom 
	#	d - int, number of hops to travel
	result_vector = numpy.zeros(d+1)
	hopped = 0
	active_set  = set([orig])
	historical_set = set()
	result_vector[hopped] = prop_vec[orig]*prop_vec[orig]
	while hopped < (d):

		hopped += 1
		new_active_set = set()
		for this_atom in active_set:
			## prepare all atoms attached to this connection 
			#print('called in AC')
			this_atoms_neighbors =  mol.getBondedAtomsSmart(this_atom)
			for bound_atoms in this_atoms_neighbors:
				if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
					new_active_set.add(bound_atoms)
		#print('new active set at hop = ' +str(hopped) + ' is ' +str(new_active_set))
		for inds in new_active_set:
			result_vector[hopped] += prop_vec[orig]*prop_vec[inds]
			historical_set.update(active_set)
		active_set = new_active_set
	return(result_vector)
def deltametric(mol,prop_vec,orig,d):
	## this function returns the deltametric 
	## over the whole molecule
	# Inputs:
	#	mol - mol3D class
	#	prop_vec - vector, property of atoms in mol in order of index
	#	orig -  int, zero-indexed starting atom 
	#	d - int, number of hops to travel
	result_vector = numpy.zeros(d+1)
	hopped = 0
	active_set  = set([orig])
	historical_set = set()
	result_vector[hopped] = 0.00
	while hopped < (d):
		hopped += 1
		new_active_set = set()
		for this_atom in active_set:
			## prepare all atoms attached to this connection 
			#print('called in DAC')
			this_atoms_neighbors =  mol.getBondedAtomsSmart(this_atom)
			for bound_atoms in this_atoms_neighbors:
				if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
					new_active_set.add(bound_atoms)
		#print('new active set at hop = ' +str(hopped) + ' is ' +str(new_active_set))
		for inds in new_active_set:
			result_vector[hopped] += prop_vec[orig]-prop_vec[inds]
			historical_set.update(active_set)
		active_set = new_active_set
	return(result_vector)
def construct_property_vector(mol,prop):
	## assigns the value of property
	## for atom i (zero index) in mol
	## to position i in returned vector
	## can be used to create weighted
	## graph representations
	
	#allowed_strings = ['electronegativity','nulcear_charge','ident','coord']
	allowed_strings = ['electronegativity','nuclear_charge','ident','topology','size']
	## note that ident just codes every atom as one, this gives
	## a purely toplogical index. coord gives the number of 
	## connecting atom to attom i (similar to Randic index)

	globs =globalvars() 
	prop_dict = dict()
	w  = numpy.zeros(mol.natoms)
	done = False
	if not prop in allowed_strings:
		print('error, property  ' + str(prop) + ' is not a vaild choice')
		print(' options are  '+ str(allowed_strings))
		return False
	if prop == 'electronegativity':
		prop_dict = globs.endict()
	elif prop == 'size':
		at_keys  = globs.amass().keys()
		for keys in at_keys:
			values = globs.amass()[keys][2]
			prop_dict.update({keys:values})
	elif prop == 'nuclear_charge':
		at_keys  = globs.amass().keys()
		for keys in at_keys:
			values = globs.amass()[keys][1]
			prop_dict.update({keys:values})
	elif prop == 'ident':
		at_keys  = globs.amass().keys()
		for keys in at_keys:
			prop_dict.update({keys:1})
	elif prop == 'topology':
    		for i,atoms in enumerate(mol.getAtoms()):
			#print('atom # ' + str(i) + " symbol =  " + str(atoms.symbol()))
			w[i] = len(mol.getBondedAtomsSmart(i))
		done = True
	if not done:
    		for i,atoms in enumerate(mol.getAtoms()):
			#print('atom # ' + str(i) + " symbol =  " + str(atoms.symbol()))
			w[i] = prop_dict[atoms.symbol()]
	return(w)

def find_ligand_autocorrelations_oct(mol,prop,loud,depth,name=False):
	## this function takes a
	## symmetric (axial == axial,
	## equatorial == equatorial)
	## octahedral complex
	## and returns autocorrelations for 
	## the axial an equatorial ligands
	liglist,ligdents,ligcons = ligand_breakdown(mol)
	ax_ligand_list,eq_ligand_list,ax_natoms_list,eq_natoms_list,ax_con_int_list,eq_con_int_list,ax_con_list,eq_con_list,built_ligand_list=ligand_assign(mol,liglist,ligdents,ligcons,loud,name)
	## count ligands
	n_ax = len(ax_ligand_list)
	n_eq = len(eq_ligand_list)
	## get full ligand AC
	ax_ligand_ac_full = []
	eq_ligand_ac_full = []
	for i in range(0,n_ax):
		if not list(ax_ligand_ac_full):
			ax_ligand_ac_full = full_autocorrelation(ax_ligand_list[i].mol,prop,depth) 
		else:
			ax_ligand_ac_full += full_autocorrelation(ax_ligand_list[i].mol,prop,depth) 
	ax_ligand_ac_full = np.divide(ax_ligand_ac_full,n_ax)
	for i in range(0,n_eq):
		if not list(eq_ligand_ac_full):
			eq_ligand_ac_full = full_autocorrelation(eq_ligand_list[i].mol,prop,depth) 
		else:
			eq_ligand_ac_full += full_autocorrelation(eq_ligand_list[i].mol,prop,depth) 
	eq_ligand_ac_full = np.divide(eq_ligand_ac_full,n_eq)

	## get partial ligand AC
	ax_ligand_ac_con= []
	eq_ligand_ac_con = []
	
	for i in range(0,n_ax):
		if not list(ax_ligand_ac_con):
			ax_ligand_ac_con = atom_only_autocorrelation(ax_ligand_list[i].mol,prop,depth,ax_con_int_list[i]) 
		else:
			ax_ligand_ac_con += atom_only_autocorrelation(ax_ligand_list[i].mol,prop,depth,ax_con_int_list[i]) 
	ax_ligand_ac_con = np.divide(ax_ligand_ac_con,n_ax)
	for i in range(0,n_eq):
		if not list(eq_ligand_ac_con):
			eq_ligand_ac_con = atom_only_autocorrelation(eq_ligand_list[i].mol,prop,depth,eq_con_int_list[i]) 
		else:
			eq_ligand_ac_con += atom_only_autocorrelation(eq_ligand_list[i].mol,prop,depth,eq_con_int_list[i]) 
	eq_ligand_ac_con = np.divide(eq_ligand_ac_con,n_eq)

	#ax_ligand_ac_con = atom_only_autocorrelation(ax_ligand.mol,prop,depth,ax_con_int)
	#eq_ligand_ac_con = atom_only_autocorrelation(eq_ligand.mol,prop,depth,eq_con_int)
	return ax_ligand_ac_full,eq_ligand_ac_full,ax_ligand_ac_con,eq_ligand_ac_con
def find_ligand_deltametrics_oct(mol,prop,loud,depth,name=False):
	## this function takes a
	## octahedral complex
	## and returns deltametrics for 
	## the axial an equatorial ligands
	liglist,ligdents,ligcons = ligand_breakdown(mol)
	ax_ligand_list,eq_ligand_list,ax_natoms_list,eq_natoms_list,ax_con_int_list,eq_con_int_list,ax_con_list,eq_con_list,built_ligand_list=ligand_assign(mol,liglist,ligdents,ligcons,loud,name)
	## count ligands
	n_ax = len(ax_ligand_list)
	n_eq = len(eq_ligand_list)

	## get partial ligand AC
	ax_ligand_ac_con= []
	eq_ligand_ac_con = []
	
	for i in range(0,n_ax):
		if not list(ax_ligand_ac_con):
			ax_ligand_ac_con = atom_only_deltametric(ax_ligand_list[i].mol,prop,depth,ax_con_int_list[i]) 
		else:
			ax_ligand_ac_con += atom_only_deltametric(ax_ligand_list[i].mol,prop,depth,ax_con_int_list[i]) 
	ax_ligand_ac_con = np.divide(ax_ligand_ac_con,n_ax)
	for i in range(0,n_eq):
		if not list(eq_ligand_ac_con):
			eq_ligand_ac_con = atom_only_deltametric(eq_ligand_list[i].mol,prop,depth,eq_con_int_list[i]) 
		else:
			eq_ligand_ac_con += atom_only_deltametric(eq_ligand_list[i].mol,prop,depth,eq_con_int_list[i]) 
	eq_ligand_ac_con = np.divide(eq_ligand_ac_con,n_eq)

	return ax_ligand_ac_con,eq_ligand_ac_con
def generate_all_ligand_autocorrelations(mol,loud,depth=4,name=False):
	result_ax_full = list()
	result_eq_full = list()
	result_ax_con = list()
	result_eq_con = list()
	colnames = []
	allowed_strings = ['electronegativity','nuclear_charge','ident','topology','size']
	labels_strings = ['chi','Z','I','T','S']
	for ii,properties in enumerate(allowed_strings):
		ax_ligand_ac_full,eq_ligand_ac_full,ax_ligand_ac_con,eq_ligand_ac_con = find_ligand_autocorrelations_oct(mol,properties,loud,depth,name)
		this_colnames = []
		for i in range(0,depth+1):
			this_colnames.append(labels_strings[ii] + '-' + str(i))
		colnames.append(this_colnames)
		result_ax_full.append(ax_ligand_ac_full)
		result_eq_full.append(eq_ligand_ac_full)
		result_ax_con.append(ax_ligand_ac_con)
		result_eq_con.append(eq_ligand_ac_con)
	results_dictionary={'colnames':colnames,'result_ax_full':result_ax_full,'result_eq_full':result_eq_full,
                        'result_ax_con':result_ax_con,'result_eq_con':result_eq_con}
	return  results_dictionary
def generate_all_ligand_deltametrics(mol,loud,depth=4,name=False):
	result_ax_full = list()
	result_eq_full = list()
	result_ax_con = list()
	result_eq_con = list()
	colnames = []
	allowed_strings = ['electronegativity','nuclear_charge','ident','topology','size']
	labels_strings = ['chi','Z','I','T','S']
	for ii,properties in enumerate(allowed_strings):
		ax_ligand_ac_con,eq_ligand_ac_con = find_ligand_deltametrics_oct(mol,properties,loud,depth,name)
		this_colnames = []
		for i in range(0,depth+1):
			this_colnames.append(labels_strings[ii] + '-' + str(i))
		colnames.append(this_colnames)
		result_ax_con.append(ax_ligand_ac_con)
		result_eq_con.append(eq_ligand_ac_con)
	results_dictionary={'colnames':colnames,'result_ax_con':result_ax_con,'result_eq_con':result_eq_con}
	return  results_dictionary

def generate_metal_autocorrelations(mol,loud,depth=4):
	result = list()
	colnames = []
	allowed_strings = ['electronegativity','nuclear_charge','ident','topology','size']
	labels_strings = ['chi','Z','I','T','S']
	for ii,properties in enumerate(allowed_strings):
		metal_ac = metal_only_autocorrelation(mol,properties,depth)
		this_colnames = []
		for i in range(0,depth+1):
			this_colnames.append(labels_strings[ii] + '-' + str(i))
		colnames.append(this_colnames)
		result.append(metal_ac)
	results_dictionary={'colnames':colnames,'results':result}
	return  results_dictionary
def generate_metal_deltametrics(mol,loud,depth=4):
	result = list()
	colnames = []
	allowed_strings = ['electronegativity','nuclear_charge','ident','topology','size']
	labels_strings = ['chi','Z','I','T','S']
	for ii,properties in enumerate(allowed_strings):
		metal_ac = metal_only_deltametric(mol,properties,depth)
		this_colnames = []
		for i in range(0,depth+1):
			this_colnames.append(labels_strings[ii] + '-' + str(i))
		colnames.append(this_colnames)
		result.append(metal_ac)
	results_dictionary={'colnames':colnames,'results':result}
	return  results_dictionary
def generate_full_complex_autocorrelations(mol,loud,depth=4):
	result = list()
	colnames = []
	allowed_strings = ['electronegativity','nuclear_charge','ident','topology','size']
	labels_strings = ['chi','Z','I','T','S']
	for ii,properties in enumerate(allowed_strings):
		metal_ac = full_autocorrelation(mol,properties,depth)
		this_colnames = []
		for i in range(0,depth+1):
			this_colnames.append(labels_strings[ii] + '-' + str(i))
		colnames.append(this_colnames)
		result.append(metal_ac)
	results_dictionary={'colnames':colnames,'results':result}
	return  results_dictionary

