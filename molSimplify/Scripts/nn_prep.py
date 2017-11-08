# Written by JP Janet for HJK Group
# Dpt of Chemical Engineering, MIT

##########################################################
####### This script is a collection of helper ############
########  routines for ANN integration in    #############
########           molsimplify               #############
##########################################################

# import custom modules
from molSimplify.Scripts.geometry import *
from molSimplify.Scripts.io import *
from molSimplify.Informatics.decoration_manager import*
from molSimplify.Classes.globalvars import *
from molSimplify.Informatics.graph_analyze import *
from molSimplify.python_nn.ANN import *
import numpy
# import standard modules
import openbabel


def get_bond_order(OBMol,connection_atoms,mol):
    ## informs the ANN
    ## of the highest bond
    ## order in ligand
    # INPUT:
    #   - OBMol:  OBMol class ligand
    # OUTPUT:
    #   - max_bo: int, max bond order

    bond_order_pairs = []

    OBMol.PerceiveBondOrders()
    for atoms in connection_atoms:
            this_neighbourhood = mol.getBondedAtoms(atoms)
            for items in this_neighbourhood:
                    bond_order_pairs.append(tuple([atoms,items]))

    max_bo = 0
    for index_pairs in bond_order_pairs:
            this_bond= OBMol.GetBond(int(index_pairs[0]+1),int(index_pairs[1]+1))
            if this_bond.IsAromatic():
                this_BO = int(2)
            else:
                this_BO = int(this_bond.GetBondOrder())
            if this_BO > max_bo:
               max_bo = this_BO
    return max_bo


def check_ligands(ligs,batlist,dents,tcats):
    ## tests if ligand combination
    ## is compatiable with the ANN
    # INPUT:
    #   - ligs:  list of mol3D class, ligands
    #   - batlist: list of int, occupations 
    #   - dents: list of int, denticity
    #   - tcats: list of int/bool
    # OUTPUT:
    #   - valid: bool
    ## tcats controls
    ## manual overide
    ## of connection atoms

    n_ligs = len(ligs)
    unique_ligands = []
    axial_ind_list =[]
    equitorial_ind_list =[]
    axial_ligs = []
    equitorial_ligs = []
    ax_dent = 0 
    eq_dent  =0
    eq_ligs = []
    eq_tcat = False
    ax_tcat = False
    valid = True
    if  (set(dents) == set([2])):
        print('triple bidentate case\n')
        unique_ligs = []
        ucats = []
        if not(n_ligs) == 3:
                ## something unexpected happened!
                valid = False 
        for i in range(0,n_ligs):
            this_bat = batlist[i]
            this_lig = ligs[i]
            this_dent = dents[i]
            ## mulitple points
            if not (this_lig in unique_ligs):
                    unique_ligs.append(this_lig)
                    ucats.append(tcats[i])
            elif (this_lig in unique_ligs) and not (this_lig in equitorial_ligs) :
                   equitorial_ligs.append(this_lig)
                   eq_dent = this_dent
                   eq_tcat = tcats[i]
        if len(unique_ligs) == 1:
            axial_ligs.append(equitorial_ligs[0])
            ax_dent  = 2
            ax_tcat = eq_tcat
        elif len(unique_ligs) == 2:
            for i,uligs in enumerate(unique_ligs):
                if not (uligs in equitorial_ligs): #only occured once
                    axial_ligs.append(uligs)
                    ax_dent = 2
                    ax_tcat = ucats[i]
        else:
            valid = False
    else:
        for i in range(0,n_ligs):
            this_bat = batlist[i]
            this_lig = ligs[i]
            this_dent = dents[i]
#            print(this_bat,this_lig,this_dent)
            ## mulitple points
            if len(this_bat) == 1:
                if (5 in this_bat) or (6 in this_bat):
                    if not (this_lig in axial_ligs):
                        axial_ligs.append(this_lig)
                        ax_dent = this_dent
                        ax_tcat = tcats[i]
                else:
                    if not (this_lig in equitorial_ligs):
                        equitorial_ligs.append(this_lig)
                        eq_dent = this_dent
                        eq_tcat = tcats[i]
            else:
                if not (this_lig in equitorial_ligs):
                        equitorial_ligs.append(this_lig)
                        eq_dent = this_dent
                        eq_tcat = tcats[i]
    if not (len(axial_ligs) == 1):
        print('axial ligs mismatch: ',axial_ligs,ax_dent)
        valid = False
    if not (len(equitorial_ligs) == 1):
        print('equitorial ligs mismatch: ',equitorial_ligs,eq_dent)
        valid = False
    if valid: # get the index position in ligs
        axial_ind_list = [ligs.index(ax_lig) for ax_lig in axial_ligs]
        equitorial_ind_list = [ligs.index(eq_lig) for eq_lig in equitorial_ligs]
    return valid,axial_ligs,equitorial_ligs,ax_dent,eq_dent,ax_tcat,eq_tcat,axial_ind_list,equitorial_ind_list

def check_metal(metal,oxidation_state):
    supported_metal_dict = {"fe":[2,3],"mn":[2,3],"cr":[2,3],
                            "co":[2,3],"ni":[2]}
    romans={'I':'1','II':'2','III':'3','IV':'4','V':'5','VI':'6'}
    if oxidation_state  in romans.keys():
        oxidation_state= romans[oxidation_state]
    outcome = False
    if metal in supported_metal_dict.keys():
#        print('metal in',supported_metal_dict[metal])
        if int(oxidation_state) in supported_metal_dict[metal]:
            outcome = True
    return outcome,oxidation_state


def get_truncated_kier(ligand,connection_atoms):
        ### three hop truncation
        trunc_mol = obtain_truncation(ligand,connection_atoms,3)
        this_kier = kier(trunc_mol) 
        return this_kier
def get_con_at_type(mol,connection_atoms):
    this_type = ""
    been_set = False
    valid = True
    ## test if the ligand is pi-bonded
    if 'pi' in connection_atoms:
        print('ANN cannot handle Pi bonding (yet)')
        valid = False
        this_type='pi'
    else:
        for atoms in connection_atoms:
            this_symbol = mol.getAtom(atoms).symbol()
            if not (this_symbol == this_type):
                if not been_set:
                    this_type = this_symbol
                else:
                    print('different connection atoms in one ligand')
                    valid = False
        if not this_type in ['C','O','Cl','N','S']:
            valid = False
            print('untrained atom type: ',this_type)
    return valid,this_type


def ANN_preproc(args,ligs,occs,dents,batslist,tcats,licores):
    ### prepares and runs ANN calculation

    ######################
    ANN_reason = False # holder for reason to reject ANN call
    ANN_attributes = dict()
    ######################

    nn_excitation = []
    r = 0
    emsg = list()
    valid = True 
    metal = args.core
    this_metal = metal.lower()
    if len(this_metal) >2 :
        this_metal = this_metal[0:2]
    newligs = []
    newcats = []
    newdents = []
    newdecs = [False]*6
    newdec_inds = [[]]*6
    ANN_trust = False
    count = -1
    for i,lig in enumerate(ligs):
        this_occ = occs[i]
        for j in range(0,int(this_occ)):
            count += 1
            newligs.append(lig)
            newdents.append(dents[i])
            newcats.append(tcats[i])
            if args.decoration:               
                newdecs[count] = (args.decoration[i])
                newdec_inds[count] = (args.decoration_index[i])

    ligs = newligs  
    dents = newdents
    tcats = newcats

    if not args.geometry == "oct":
#        print('nn: geom  is',args.geometry)
#        emsg.append("[ANN] Geometry is not supported at this time, MUST give -geometry = oct")
        valid = False 
        ANN_reason = 'geometry not oct'
    if not args.oxstate:
        emsg.append("\n oxidation state must be given")
        valid = False
        ANN_reason = 'oxstate not given'
    if valid:
        oxidation_state = args.oxstate
        valid, oxidation_state = check_metal(this_metal,oxidation_state)
        ## generate key in descriptor space
        ox = int(oxidation_state)
        spin = args.spin
        if args.debug:
	    print('metal is '+ str(this_metal))
            print('metal validity',valid)
        if not valid:
            emsg.append("\n Oxidation state not available for this metal")
            ANN_reason = 'ox state not avail for metal'
    if valid:
        high_spin,spin_ops = spin_classify(this_metal,spin,ox)
        if not valid:
            emsg.append("\n this spin state not available for this metal")
            ANN_reason = 'spin state not availble for metal'
    if emsg:
        print(str(" ".join( ["ANN messages:"] +   [str(i) for i in emsg] )))
    if valid:
        valid,axial_ligs,equitorial_ligs,ax_dent,eq_dent,ax_tcat,eq_tcat,axial_ind_list,equitorial_ind_list = check_ligands(ligs,batslist,dents,tcats)
        if args.debug:
            print("\n")
            print("ligand validity is  "+str(valid))
            print('Occs')
            print(occs)
            print('Ligands')
            print(ligs)
            print('Dents')
            print(dents)
            print('Bats (backbone atoms)')
            print(batslist)
            print('lig validity',valid)
            print('ax ligs',axial_ligs)
            print('eq ligs',equitorial_ligs)
            print('spin is',spin)
        if not valid:
                ANN_reason  = 'found incorrect ligand symmetry'

    if valid:
            ax_lig3D,r_emsg = lig_load(axial_ligs[0],licores) # load ligand
            if r_emsg:
                    emsg += r_emsg
            ## check decoration index
            if newdecs:
                        if newdecs[axial_ind_list[0]]:
                            #print('decorating ' + str(axial_ligs[0]) + ' with ' +str(newdecs[axial_ind_list[0]]) + ' at sites '  + str(newdec_inds[axial_ind_list[0]]))
                            ax_lig3D = decorate_ligand(args,axial_ligs[0],newdecs[axial_ind_list[0]],newdec_inds[axial_ind_list[0]])
            
            ax_lig3D.convert2mol3D() ## mol3D representation of ligand
            ### eq
            eq_lig3D,r_emsg = lig_load(equitorial_ligs[0],licores) # load ligand
            if newdecs:
                if newdecs[equitorial_ind_list[0]]:
                            #print('decorating ' + str(equitorial_ligs[0]) + ' with ' +str(newdecs[equitorial_ind_list[0]]) + ' at sites '  + str(newdec_inds[equitorial_ind_list[0]]))
                            eq_lig3D = decorate_ligand(args,equitorial_ligs[0],newdecs[equitorial_ind_list[0]],newdec_inds[equitorial_ind_list[0]])
            if r_emsg:
                    emsg += r_emsg
            eq_lig3D.convert2mol3D() ## mol3D representation of ligand
            if ax_tcat:
                    ax_lig3D.cat = ax_tcat
                    if args.debug:
                        print('custom ax connect atom given (0-ind) '+str(ax_tcat))
            if eq_tcat:
                    eq_lig3D.cat = eq_tcat
                    if args.debug:
                        print('custom eq connect atom given (0-ind) '+str(eq_tcat))
    if args.debug:
        print('finished checking ligands, valid is '+str(valid))
    if valid:
        valid,ax_type = get_con_at_type(ax_lig3D,ax_lig3D.cat)
    if valid:
        valid,eq_type = get_con_at_type(eq_lig3D,eq_lig3D.cat)
        if args.debug:
            print('finished con atom types '+ str(ax_type) + ' and ' + str(eq_type))

    if valid:
        eq_ki = get_truncated_kier(eq_lig3D,eq_lig3D.cat)
        ax_ki = get_truncated_kier(ax_lig3D,ax_lig3D.cat)
        eq_EN = get_lig_EN(eq_lig3D,eq_lig3D.cat)
        ax_EN = get_lig_EN(ax_lig3D,ax_lig3D.cat)
        eq_bo = get_bond_order(eq_lig3D.OBMol,eq_lig3D.cat,eq_lig3D)
        ax_bo = get_bond_order(ax_lig3D.OBMol,ax_lig3D.cat,ax_lig3D)

        if axial_ligs[0] in ['carbonyl','cn']:
            ax_bo = 3
        if equitorial_ligs[0] in ['carbonyl','cn']:
            eq_bo = 3
        eq_charge = eq_lig3D.OBMol.GetTotalCharge()
        ax_charge = ax_lig3D.OBMol.GetTotalCharge()

         
        ## preprocess:
        sum_delen  = (2.0)*ax_EN + (4.0)*eq_EN
        if abs(eq_EN) > abs(ax_EN):
                max_delen = eq_EN
        else:
                max_delen = ax_EN
        alpha = 0.2 # default for B3LYP
	if args.exchange:
		try:
			if float(args.exchange) > 1:
				alpha = float(args.exchange)/100 # if given as %
			elif float(args.exchange) <= 1:
				alpha = float(args.exchange)
		except:
			print('cannot case exchange argument as a float, using 20%')
 
        if args.debug:
            print('ax_bo',ax_bo)
            print('eq_bo',eq_bo)
            print('ax_dent',ax_dent)
            print('eq_dent',eq_dent)
            print('ax_charge',ax_charge)
            print('eq_charge',eq_charge)
            print('sum_delen',sum_delen)
            print('max_delen',max_delen)
            print('ax_type',ax_type)
            print('eq_type',eq_type)
            print('ax_ki',ax_ki)
            print('eq_ki',eq_ki)


    if valid:



        nn_excitation = [0,0,0,0,0, # metals co/cr/fe/mn/ni                 #1-5
                   ox,alpha,ax_charge,eq_charge, #ox/alpha/axlig charge/eqlig charge #6-9
                   ax_dent,eq_dent,# ax_dent/eq_dent/ #10-11
                   0,0,0,0, # axlig_connect: Cl,N,O,S #12 -15
                   0,0,0,0, # eqliq_connect: Cl,N,O,S #16-19
                   sum_delen,max_delen, #mdelen, maxdelen #20-21
                   ax_bo,eq_bo, #axlig_bo, eqliq_bo #22-23
                   ax_ki,eq_ki]#axlig_ki, eqliq_kii #24-25
	slope_excitation = [0,0,0,0,0, # metals co/cr/fe/mn/ni                 #1-5
                   ox,ax_charge,eq_charge, #ox/axlig charge/eqlig charge #6-8
                   ax_dent,eq_dent,# ax_dent/eq_dent/ #9-10
                   0,0,0,0, # axlig_connect: Cl,N,O,S #11 -14
                   0,0,0,0, # eqliq_connect: Cl,N,O,S #15-18
                   sum_delen,max_delen, #mdelen, maxdelen #19-20
                   ax_bo,eq_bo, #axlig_bo, eqliq_bo #21-22
                   ax_ki,eq_ki]#axlig_ki, eqliq_kii #23-24
                    
    #print(slope_excitation)
    #print('\n')
    ### discrete variable encodings
    if valid:
        valid,nn_excitation = metal_corrector(nn_excitation,this_metal)
        valid,slope_excitation = metal_corrector(slope_excitation,this_metal)
    #
    if valid:
        valid,nn_excitation = ax_lig_corrector(nn_excitation,ax_type)
        valid,slope_excitation = ax_lig_corrector(slope_excitation,ax_type)

   # print('ax_cor',valid)
  #  print('start eq check')
    if valid:
        valid,nn_excitation = eq_lig_corrector(nn_excitation,eq_type)
        valid,slope_excitation = eq_lig_corrector(slope_excitation,eq_type)
     
  #  print('eq_cor',valid)
    #
    #print(slope_excitation)
    #print('excitations: ')
    #print(nn_excitation)
    #print(slope_excitation)
    
    if valid:
        print("******************************************************************")
        print("************** ANN is engaged and advising on spin ***************")
        print("************** and metal-ligand bond distances    ****************")
        print("******************************************************************")
        if high_spin:
            print('You have selected a high-spin state, s = ' + str(spin))
        else:
            print('You have selected a low-spin state, s = ' + str(spin))
        ## test Euclidean norm to training data distance
        train_dist,best_row = find_eu_dist(nn_excitation)
        ANN_trust = max(0.01,1.0-train_dist)

        ANN_attributes.update({'ANN_dist_to_train':train_dist} )
        ANN_attributes.update({'ANN_closest_train':best_row} )
        print('distance to training data is ' + "{0:.2f}".format(train_dist) + ' ANN trust: ' +str(ANN_trust))
        print(' with closest training row ' + best_row[:-2])
        ANN_trust = 'not set'
        if float(train_dist)< 0.25:
            print('ANN results should be trustworthy for this complex ')
            ANN_trust = 'high'
        elif float(train_dist)< 0.75:
            print('ANN results are probably useful for this complex ')
            ANN_trust  = 'medium'
        elif float(train_dist)< 1.0:
            print('ANN results are fairly far from training data, be cautious ')
            ANN_trust = 'low'
        elif float(train_dist)> 1.0:
            print('ANN results are too far from training data, be cautious ')
            ANN_trust = 'very low'
        ANN_attributes.update({'ANN_trust':ANN_trust})
        ## engage ANN
        delta = 0  

        delta = get_splitting(nn_excitation)
        ## report to stdout
        if delta[0] < 0 and not high_spin:
            if abs(delta[0]) > 5:
                print('warning, ANN predicts a high spin ground state for this complex')
            else:
                print('warning, ANN predicts a near degenerate ground state for this complex')
        if delta[0] >= 0 and high_spin:
            if abs(delta[0]) > 5:
                print('warning, ANN predicts a low spin ground state for this complex')
            else:
                    print('warning, ANN predicts a near degenerate ground state for this complex')
        print("ANN predicts a spin splitting (HS - LS) of " + "{0:.2f}".format(float(delta[0])) + ' kcal/mol')
        ANN_attributes.update({'pred_split_ HS_LS':delta[0]})
        ## reparse to save attributes
        ANN_attributes.update({'This spin':spin})
        if delta[0] < 0 and (abs(delta[0]) > 5):
                ANN_attributes.update({'ANN_ground_state':spin_ops[1]})
        elif delta[0] > 0 and (abs(delta[0]) > 5):
                ANN_attributes.update({'ANN_ground_state':spin_ops[0]})
        else:
                ANN_attributes.update({'ANN_gound_state':'dgen ' + str(spin_ops)})

        r = 0
        if not high_spin:
            r = get_ls_dist(nn_excitation)
        else:
            r = get_hs_dist(nn_excitation)

        print('ANN bond length is predicted to be: '+"{0:.2f}".format(float(r)) + ' angstrom')
        ANN_attributes.update({'ANN_bondl':r[0]})

	### use ANN to predict fucntional sensitivty
        HFX_slope = 0 
        HFX_slope = get_slope(slope_excitation)
        print('Predicted HFX exchange sensitivity is : '+"{0:.2f}".format(float(HFX_slope)) + ' kcal/HFX')
	ANN_attributes.update({'ANN_slope':HFX_slope})
        print("*******************************************************************")
        print("************** ANN complete, saved in record file *****************")
        print("*******************************************************************")

    if not valid and not ANN_reason:
        ANN_reason = ' uncaught rejection (see sdout/stderr)'
    return valid,ANN_reason,ANN_attributes


def ax_lig_corrector(excitation,con_atom_type):
    ax_lig_index_dictionary = {'Cl':11,'F':11,'N':12,'O':13,'S':14}
    ## C is the basic value
    try:
        if not con_atom_type == "C":
            excitation[ax_lig_index_dictionary[con_atom_type]] = 1
        valid = True
    except:
       valid = False
    return valid,excitation

def eq_lig_corrector(excitation,con_atom_type):
    #print('in eliq cor, excitation 1;5'  + str(excitation[1:5]))
    eq_lig_index_dictionary = {'Cl':15,'F':15,'N':16,'O':17,'S':18}
    try:
        if not con_atom_type == "C":
            excitation[eq_lig_index_dictionary[con_atom_type]] = 1
        valid = True
    except:
       valid = False
    #print('end eliq cor, excitation 1;5'  + str(excitation[1:5]))
    return valid,excitation



def metal_corrector(excitation,metal):
    metal_index_dictionary = {'co':0,'cr':1,'fe':2,'mn':3,'ni':4}
    #print(excitation)
    #print('metal is ' + str(metal) +' setting slot ' +str(metal_index_dictionary[metal]) + ' to 1' )
    try:
        excitation[metal_index_dictionary[metal]] += 1
        valid = True
    except:
       valid = False
    
    return valid,excitation
#n = network_builder([25,50,51],"nn_split")

def spin_classify(metal,spin,ox):
    metal_spin_dictionary = {'co':{2:4,3:5},
                              'cr':{2:5,3:4},
                              'fe':{2:5,3:6},
                              'mn':{2:6,3:5},
                              'ni':{2:3}}

    suggest_spin_dictionary = {'co':{2:[2,4],3:[1,5]},
                              'cr':{2:[1,5],3:[2,4]},
                              'fe':{2:[1,5],3:[2,6]},
                              'mn':{2:[2,6],3:[1,5]},
                              'ni':{2:[1,3]}}


    high_spin = False
    if (int(spin) >= int(metal_spin_dictionary[metal][ox])):
        high_spin = True
    spin_ops = suggest_spin_dictionary[metal][ox]
    return high_spin,spin_ops

def get_splitting(excitation):
    print(excitation)
    delta = simple_splitting_ann(excitation)
    return delta
def get_slope(slope_excitation):
    HFX = simple_slope_ann(slope_excitation)    
    return HFX

def get_ls_dist(excitation):
    r = simple_ls_ann(excitation)
    return r

def get_hs_dist(excitation):
    r = simple_hs_ann(excitation)
    return r

