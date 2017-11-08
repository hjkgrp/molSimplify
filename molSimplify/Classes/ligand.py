# Written by JP Janet for HJK Group
# Dpt of Chemical Engineering, MIT

##########################################################
######## Defines class of ligands  #######################
########     will be used to postprocess #################
########   DFT results by measuring ligand properties ####
##########################################################



from molSimplify.Classes.mol3D import *
from molSimplify.Classes.atom3D import *
from molSimplify.Scripts.geometry import * 
from collections import Counter

class ligand:
	def __init__(self,master_mol,index_list,dent):
		self.master_mol  = master_mol
		self.index_list = index_list
		self.dent = dent
		self.ext_int_dict = dict() ## store		
								## map betweem
								## int and ext indcies
	def obtain_mol3d(self):
		this_mol = mol3D()
		this_ext_int_dict = dict()
		j = 0
		for i in range(0,self.master_mol.natoms): 
			if i in self.index_list:
				this_mol.addAtom(self.master_mol.getAtom(i))
				this_ext_int_dict.update({i:j})
				j += 1 # keep count of how many are added
		self.mol = this_mol 
		self.ext_int_dict =  this_ext_int_dict
	def obtain_truncation(self,con_atoms,hops):
		self.trunc_mol = mol3D()
		added_list = list()
		for connections in con_atoms:
			hopped = 0
			active_set  = [connections]
			while hopped < hops:
				hopped += 1
				new_active_set = list()
				for this_atom in active_set:
					this_atoms_neighbors =  self.master_mol.getBondedAtoms(this_atom)
					for bound_atoms in this_atoms_neighbors:
						if (bound_atoms in self.index_list) and (bound_atoms not in added_list):
							self.trunc_mol.addAtom(self.master_mol.getAtom(bound_atoms))
							added_list.append(bound_atoms)
					[new_active_set.append(element) for element in this_atoms_neighbors]
				active_set = new_active_set
		return trunc_mol

def ligand_breakdown(mol):
	# this function takes an octahedral 
	# complex and returns ligands
	metal_index = mol.findMetal()
	bondedatoms = mol.getBondedAtomsSmart(metal_index)
	bonded_atom_symbols = [mol.getAtom(i).symbol() for i in bondedatoms]              
	counter = 0
	liglist = []
	ligdents = []
	ligcons = []
	for atom in bondedatoms:
		#print('this atom type is ' + mol.getAtom(atom).symbol())
		#print('conection number ' + str(atom) + " of " + str(bondedatoms))
		fragment = mol.findsubMol(atom,metal_index)
		this_cons = [x for x in fragment if (x in bondedatoms)]
		unique =  True
		for i,unique_ligands in enumerate(liglist):
			if fragment == unique_ligands:
				unique = False
				matched = i
		if unique:
			liglist.append(fragment)
			ligdents.append(1)
			ligcons.append(this_cons)
		else:
			ligdents[matched] += 1
	return liglist,ligdents,ligcons
def ligand_assign(mol,liglist,ligdents,ligcons,loud=False,name=False):
        valid = True
	metal_index = mol.findMetal()
	built_ligand_list  = list()
	lig_natoms_list = list()
	unique_ligands = list()
	ligand_counts  = list()	
	all_ligand_counts = [0,0,0,0,0,0]
	ligand_records = list()
	ax_con_int_list = list()
	eq_con_int_list = list()
	ax_natoms_list = list()
	eq_natoms_list = list()
	n_ligs = len(liglist)
	max_dent = max(ligdents)
	min_dent = min(ligdents)
	if loud:
		print('********************************************')
		print("n_ligs = " + str(n_ligs))
		print("max d = " + str(max_dent))
		print("min_dent = " +  str(min_dent))
		print("ligand list is" + str(liglist))
		print('denticities are  ' + str(ligdents))
	if (max(ligdents) == 4) and (min(ligdents) != 1):
                valid = False
                print('bad denticities: ' + str(ligdents)) 
	if max(ligdents) >= 4:
                valid = False
                print('bad denticities: ' + str(ligdents)) 
	eq_lig_list = list()
	ax_lig_list = list()
	ax_con_list = list()
	eq_con_list = list()
	for i,ligand_indices in enumerate(liglist):
		this_ligand = ligand(mol,ligand_indices,ligdents[i])
		this_ligand.obtain_mol3d()
		built_ligand_list.append(this_ligand)
		lig_natoms_list.append(this_ligand.mol.natoms)
	for j,built_ligs in enumerate(built_ligand_list):
		### test if ligand is unique
		sl =  [ atom.symbol() for atom in built_ligs.mol.getAtoms()]
		if loud:
			print('checking lig ' + str(j) + ' : ' + str(sl))
		unique = 1
		for i,other_sl in enumerate(unique_ligands):
			if sorted(sl) == sorted(other_sl):
				#duplicate
				unique = 0
				ligand_counts[i] +=1
		if unique == 1:
			unique_ligands.append(sl)
			ligand_counts.append(1)
			ligand_records.append(j)
	### loop to bin ligands:
	for j,built_ligs in enumerate(built_ligand_list):
	### test if ligand is unique
		sl =  [ atom.symbol() for atom in built_ligs.mol.getAtoms()]
		unique = 1
		for i,other_sl in enumerate(unique_ligands):
			if sorted(sl) == sorted(other_sl):
				#duplicate
				#print(i,ligand_counts[i])
				all_ligand_counts[j]=ligand_counts[i]

	if loud:
		print('unique ligands' + str(unique_ligands))
		print('ligand counts' +  str(ligand_counts))
		print('ligand records ' + str(ligand_records))	
		print(str(max(ligand_counts)) + ' is the max and min in  ' + str(min(ligand_counts))) 
	n_unique_ligs = len(unique_ligands)
	if (n_ligs == 3) or (n_ligs == 4): # most common case, 
                                           # one/two equitorial and 2 axial mono
                                           # or three bidentate 
		for i,ligs in enumerate(liglist):
			if ligdents[i] == 1 and min_dent == 1:  ## anything with equitorial monos will
			                  ## have higher than 4 n_ligs
				ax_lig_list.append(i)
				if loud:
					print('choosing '+ str(i) + ' as ax based on dent =1')
				ax_con_list.append(ligcons[i])
			if (ligdents[i] >= 2) and (min_dent == 1):
				eq_lig_list.append(i)
				if loud:
					print('choosing lig '+ str(i) + ' as eq based on high dent')
				eq_con_list.append(ligcons[i])
		if (n_ligs == 3) and (min_dent == max_dent):
			if n_unique_ligs == 1: 
				# take any 2, they are all the same
				if loud:
					print('triple bidentate case')
				ax_lig_list.append(0)
				eq_lig_list.append(1)
				eq_lig_list.append(2)
				ax_con_list.append(ligcons[0])
				eq_con_list.append(ligcons[1])
				eq_con_list.append(ligcons[2])
			elif min_dent == 2 and max_dent==2 and n_ligs ==3 and not n_unique_ligs ==1:
				## this is a hetero/bidentate case
				for i,ligs in enumerate(liglist):
					if all_ligand_counts[i] == 2:
						eq_lig_list.append(i)
						eq_con_list.append(ligcons[i])
					elif  all_ligand_counts[i] == 1:
						ax_lig_list.append(i)
						ax_con_list.append(ligcons[i])
	elif (n_ligs == 6): # all mono  case, 
		minz = 500
		maxz = -500
		if loud:
			print('monodentate case')
		allowed = range(0,6)
		not_eq = list()
		for j,built_ligs in enumerate(built_ligand_list):
			this_z = built_ligs.mol.centermass()[2]
			if this_z < minz:
				minz = this_z
				bot_lig = j
				bot_con = ligcons[j]
			if loud:
				print('updating bot axial to ' +str(bot_lig))
			if this_z > maxz:
				maxz = this_z
				top_lig = j
				top_con = ligcons[j]
			if loud:
				print('updating top axial to ' +str(top_lig))
		not_eq.append(bot_lig)
		not_eq.append(top_lig)
		
		allowed = [x for x in allowed  if ((x not in not_eq))]
		if len(allowed) != 4:
			print('error in decomp of monodentate case!',allowed)
		eq_lig_list = allowed
		eq_con_list = [ligcons[i] for i in allowed]
		ax_lig_list = [top_lig,bot_lig] 
		ax_con_list = [top_con,bot_con]
		if loud:
			print('geometric eq_list ' + str(eq_lig_list))
			print('geometric ax_list ' + str(eq_lig_list))
		if (max(ligand_counts) != 4) or (min(ligand_counts) != 2):
			if loud:
				print('not a 4-6 case')
			if (max(ligand_counts) == 6):
				if loud:
					print('6-homoleptic, using geo values')
				#ax=ligand_records[ligand_counts.index(6)]
				#eq_lig=ligand_records[ligand_counts.index(6)]
			else:
				if loud:
					print('monodentates not the same, using geo values ')
					print(ligand_counts)
					print(unique_ligands)
		elif n_unique_ligs == 2:
			if loud:
				print('this is a  4-6 case')
			allowed = range(0,6)
			ax_lig_list = [i for i in allowed if (all_ligand_counts[i] == 2)]
			eq_lig_list = [i for i in allowed if (all_ligand_counts[i] == 4)]
			ax_con_list = [ligcons[i] for i in ax_lig_list]
			eq_con_list = [ligcons[i] for i in eq_lig_list]
			#ax_lig=ligand_records[ligand_counts.index(2)]
			#eq_lig=ligand_records[ligand_counts.index(4)]
	ax_ligand_list = [built_ligand_list[i] for i in ax_lig_list]
	eq_ligand_list = [built_ligand_list[i] for i in eq_lig_list]
	if loud and valid:
		print('lig_nat_list',lig_natoms_list)
		print('eq_liq is ind ',eq_lig_list)
		print('ax_liq is ind ',ax_lig_list)
		print('ax built lig [0] ext ind :' + str(built_ligand_list[ax_lig_list[0]].ext_int_dict.keys())) 
                if len(ax_lig_list)>1:
                        print('ax built lig [1] ext ind :' + str(built_ligand_list[ax_lig_list[1]].ext_int_dict.keys())) 
		print('eq built lig [0] ext ind: '+str(built_ligand_list[eq_lig_list[0]].ext_int_dict.keys())) 
		print('eq_con is '+str((eq_con_list)))
		print('ax_con is '+str((ax_con_list)))
	if name:
		for i,ax_ligand in enumerate(ax_ligand_list):
			if not os.path.isdir('ligands'):
                                os.mkdir('ligands')
			ax_ligand.mol.writexyz('ligands/'+ name+'_'+str(i)+'_ax.xyz')
		for i,eq_ligand in enumerate(eq_ligand_list):
                        if not os.path.isdir('ligands'):
                                os.mkdir('ligands')
			eq_ligand.mol.writexyz('ligands/'+ name+'_'+str(i)+'_eq.xyz')
	for j,ax_con in enumerate(ax_con_list):
		ax_con_int_list.append([built_ligand_list[ax_lig_list[j]].ext_int_dict[i] for i in ax_con]) # convert to interal index
	for j,eq_con in enumerate(eq_con_list):
		eq_con_int_list.append([built_ligand_list[eq_lig_list[j]].ext_int_dict[i] for i in eq_con])# convert to interal index
	if loud:
		print('int eq ' +str(eq_con_int_list))
		print('ext eq ' +str(eq_con_list))
		print('**********************************************')
	for ax_lig in ax_lig_list:
		ax_natoms_list.append(lig_natoms_list[ax_lig])
	for eq_lig in eq_lig_list:
		eq_natoms_list.append(lig_natoms_list[eq_lig])
	return ax_ligand_list,eq_ligand_list,ax_natoms_list,eq_natoms_list,ax_con_int_list,eq_con_int_list,ax_con_list,eq_con_list,built_ligand_list


