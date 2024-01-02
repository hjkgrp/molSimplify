import json
import ast
import sys
import os
import pandas as pd

### Below is RDKit dependency that this code uses
# from rdkit import Chem
# from rdkit.Chem.rdMolDescriptors import CalcMolFormula
###

'''
functional groups considered
'''
funcs = {
    'C6=CC=CC=C6': 'phenyl',
    'C': 'methyl',
    'F': 'fluoro',
    'C(F)(F)F': 'tetrafluoro',
    'Br': 'bromo',
    'Cl': 'chloro',
    'I': 'iodo',
    'C#N': 'cyano',
    'N': 'amino',
    'O': 'hydroxy',
    'S': 'thiol'
}


def read_synthesized_macrocycles(input_file):
    '''
    takes in a batch of synthesized macrocycles
    '''
    with open(input_file, 'r') as f:
        data = json.load(f)
        print('----', len(data))
        counter = 0
        temp_list_first = []
        for i, row in enumerate(data):
            #### Skip over all contracted rings.
            # print(row)
            # temp = row.replace('null','False')
            # temp_dict = ast.literal_eval(str(temp).strip('\n'))
            temp_dict = ast.literal_eval(str(row).strip('\n'))
            for dictkey in list(temp_dict.keys()):
                if temp_dict[dictkey] == 'False':
                    temp_dict[dictkey] = False
            temp_list_first.append(temp_dict)
    return temp_list_first


def split_at_idx(smiles, idx):
    alphabet_indices = [i for i, val in enumerate(smiles) if val.isalpha()]
    forbidden_end = ['=', '/', '\\', ')', '[']
    if (idx+1) == len(alphabet_indices):
        left = smiles
        right = ''
    else:
        left = smiles[0:alphabet_indices[idx+1]]
        right = smiles[alphabet_indices[idx+1]:]
    while left[-1] in forbidden_end:
        right = left[-1]+right
        left = left[:-1]
    return left, right


def compatibility(func, bridge):
    # not_compatible = ['C', 'O', 'S', 'X', 'x', 'none', 'NONE','N=','P=']
    not_compatible = ['O', 'S', 'X', 'x', 'none', 'NONE', 'N=', 'P=']
    if ((funcs[func] == 'phenyl') and any([bridge == val for val in not_compatible])):
        return False
    elif bridge in ['O', 'S', 'X', 'x', 'none', 'NONE', 'N=', 'P=']:
        return False
    elif bridge in ['N', 'P']:
        # Only allow functional groups at CH bonds
        return False
    else:
        return True


def count_atoms(smiles):
    return len([val for val in smiles if val.isalpha()])


def main(temp_list):
    '''
    to functionalize, you need to start at the first one and move inwards in the order
    of 1 --> 2 --> 3 --> 4
    '''
    double_func = ['C']  # cases that have more than one functionalization
    rounds = 0
    lig_func_list = []
    func_lig_list = []
    functionalized_counter = 0
    for lignum, ligand in enumerate(temp_list):

        used_func_groups = set()
        for func in list(funcs.keys()):
            temp_ligand = ligand.copy()
            rounds += 1
            func_shift = count_atoms(func)
            smicat = temp_ligand['coord_atoms_smicat'].copy()
            smicat_zero = temp_ligand['coord_atoms_zero_index'].copy()
            if 'frag_func_smiles' in list(ligand.keys()):
                smiles = temp_ligand['frag_func_smiles']
                original = temp_ligand['frag_func_smiles']
            else:
                smiles = temp_ligand['macrocycle_smiles']
                original = temp_ligand['macrocycle_smiles']
            if temp_ligand['bridge1'] in double_func:
                bridge1_double_func = True
            else:
                bridge1_double_func = False
            if temp_ligand['bridge2'] in double_func:
                bridge2_double_func = True
            else:
                bridge2_double_func = False
            if temp_ligand['bridge3'] in double_func:
                bridge3_double_func = True
            else:
                bridge3_double_func = False
            inds1 = temp_ligand['bridge1_func'].copy()
            inds2 = temp_ligand['bridge2_func'].copy()
            inds3 = temp_ligand['bridge3_func'][0].copy()
            inds4 = temp_ligand['bridge3_func'][1].copy()
            func_counter = 0
            shifter2 = 0
            shifter3 = 0
            shifter4 = 0
            if (isinstance(inds4, list) and compatibility(func, temp_ligand['bridge3'])):
                for ind in reversed(inds4):
                    left, right = split_at_idx(smiles, ind)
                    if bridge3_double_func:
                        smiles = left+'('+func+')'+'('+func+')'+right
                        func_counter += 2
                        used_func_groups.add(func)
                    else:
                        smiles = left+'('+func+')'+right
                        func_counter += 1
                        used_func_groups.add(func)
            if (isinstance(inds2, list) and compatibility(func, temp_ligand['bridge2'])):
                for ind in reversed(inds2):
                    left, right = split_at_idx(smiles, ind)
                    if bridge2_double_func:
                        smiles = left+'('+func+')'+'('+func+')'+right
                        func_counter += 2
                        shifter4 += 2*func_shift
                        used_func_groups.add(func)
                    else:
                        smiles = left+'('+func+')'+right
                        func_counter += 1
                        shifter4 += func_shift
                        used_func_groups.add(func)
            if (isinstance(inds3, list) and compatibility(func, temp_ligand['bridge3'])):
                for ind in reversed(inds3):
                    left, right = split_at_idx(smiles, ind)
                    if bridge3_double_func:
                        smiles = left+'('+func+')'+'('+func+')'+right
                        func_counter += 2
                        shifter4 += 2*func_shift
                        shifter3 += 2*func_shift
                        used_func_groups.add(func)
                    else:
                        smiles = left+'('+func+')'+right
                        func_counter += 1
                        shifter4 += func_shift
                        shifter3 += func_shift
                        used_func_groups.add(func)
            if (isinstance(inds1, list) and compatibility(func, temp_ligand['bridge1'])):
                for ind in reversed(inds1):
                    left, right = split_at_idx(smiles, ind)
                    if bridge1_double_func:
                        smiles = left+'('+func+')'+'('+func+')'+right
                        func_counter += 2
                        shifter4 += 2*func_shift
                        shifter3 += 2*func_shift
                        shifter2 += 2*func_shift
                        used_func_groups.add(func)
                    else:
                        smiles = left+'('+func+')'+right
                        func_counter += 1
                        shifter4 += func_shift
                        shifter3 += func_shift
                        shifter2 += func_shift
                        used_func_groups.add(func)

            # This is required because the bridging atoms are in the same ring as the coordination atoms and mess that up if not.
            smicat[1] += shifter2
            smicat[2] += shifter3
            smicat[3] += shifter4

            smicat_zero[1] += shifter2
            smicat_zero[2] += shifter3
            smicat_zero[3] += shifter4
            if func_counter > 0:
                m = Chem.MolFromSmiles(smiles)
                if m is not None:
                    functionalized_counter += 1
                    # print(smicat, shifter2, shifter3, shifter4, funcs[func], smiles, temp_ligand['coord_atoms_smicat'])
                    # sard
                    func_canonical = Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
                    temp_ligand['bridge_func_smiles'] = smiles
                    temp_ligand['bridge_func_canonical'] = func_canonical
                    temp_ligand['bridge_func_smiles_formula'] = CalcMolFormula(m)
                    temp_ligand['bridge_func_size'] = count_atoms(smiles)
                    temp_ligand['bridge_func_group'] = funcs[func]
                    temp_ligand['bridge_func_counter'] = func_counter
                    temp_ligand['bridge_func_coord_atoms_smicat'] = smicat
                    temp_ligand['bridge_func_coord_atoms_zero_index'] = smicat_zero
                    temp_ligand['func_name'] = temp_ligand['name']+'_bridge'+str(funcs[func])+str(func_counter)
                    func_lig_list.append(temp_ligand)
                else:
                    temp_ligand['bridge_func_smiles'] = smiles
                    temp_ligand['bridge_func_canonical'] = False
                    temp_ligand['bridge_func_smiles_formula'] = CalcMolFormula(m)
                    temp_ligand['bridge_func_size'] = count_atoms(smiles)
                    temp_ligand['bridge_func_group'] = funcs[func]
                    temp_ligand['bridge_func_counter'] = func_counter
                    temp_ligand['bridge_func_coord_atoms_smicat'] = smicat
                    temp_ligand['bridge_func_coord_atoms_zero_index'] = smicat_zero
                    temp_ligand['func_name'] = temp_ligand['name']+'_bridge'+str(funcs[func])+str(func_counter)
                    if not os.path.exists('failed_bridge_func.csv'):
                        df = pd.DataFrame([temp_ligand])
                        df.to_csv('failed_bridge_func.csv', index=False)
                    else:
                        df = pd.read_csv('failed_bridge_func.csv')
                        new_entry = pd.DataFrame([temp_ligand])
                        new_df = pd.concat([df, new_entry], axis=0)
                        new_df.to_csv('failed_bridge_func.csv', index=False)
                print('counter',lignum, rounds, func_counter)
        temp_dict = {}
        temp_dict['name'] = ligand['name']
        temp_dict['charge'] = ligand['charge']
        temp_dict['func_groups'] = list(used_func_groups)
        lig_func_list.append(temp_dict)

    print('==== FUNCTIONALIZED', functionalized_counter, rounds)
    with open(os.getcwd()+'/bridge_functionalized_synthesized_ligands.json', 'w') as fout:
        json.dump(func_lig_list, fout)
    if not os.path.exists('successful_bridge_functionalizations.csv'):
        df = pd.DataFrame(lig_func_list)
        df.to_csv('successful_bridge_functionalizations.csv', index=False)
    else:
        df = pd.read_csv('successful_bridge_functionalizations.csv')
        new_entry = pd.DataFrame(lig_func_list)
        new_df = pd.concat([df, new_entry], axis=0)
        new_df.to_csv('successful_bridge_functionalizations.csv', index=False)


if __name__ == "__main__":
    input_file = sys.argv[1]
    temp_list = read_synthesized_macrocycles(input_file)
    main(temp_list)
