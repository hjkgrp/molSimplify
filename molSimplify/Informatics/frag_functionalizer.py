import json
import ast
import sys
import os
import pandas as pd

# ## Below is RDKit dependency that this code uses
# from rdkit import Chem
# from rdkit.Chem.rdMolDescriptors import CalcMolFormula
# ##

'''
functional groups considered
'''
funcs = {
    'cccc': 'phenyl',
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
            # ### Skip over all contracted rings.
            # print(row)
            # temp = row.replace('null','False')
            # temp_dict = ast.literal_eval(str(temp).strip('\n'))
            temp_dict = ast.literal_eval(str(row).strip('\n'))
            for dictkey in list(temp_dict.keys()):
                if temp_dict[dictkey] == 'False':
                    temp_dict[dictkey] = False
            temp_list_first.append(temp_dict)
    return temp_list_first


def count_atoms(smiles):
    return len([val for val in smiles if val.isalpha()])


def split_at_idx(smiles, idx):
    alphabet_indices = [i for i, val in enumerate(smiles) if val.isalpha()]
    forbidden_end = ['=', '/', '\\', ')']
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


def main(temp_list):
    '''
    to functionalize, you need to start at the first one and move inwards in the order
    of 1 --> 2 --> 3 --> 4

    only certain rings are compatible with the phthalocyanine style phenyl functionalization.
    '''
    phthalo_compat = ['imidazole2ylidine', 'furan', 'pyrrole', 'phosphole', 'thiophene']
    double_func = []  # We populate this list if we want to doubly functionalize a given carbon on a fragment.
    functionalized_ligands = []
    rounds = 0
    for lignum, ligand in enumerate(temp_list):
        used_func_groups = set()
        for func in list(funcs.keys()):
            temp_ligand = ligand.copy()
            rounds += 1
            smiles = temp_ligand['macrocycle_smiles']
            counter = 5
            func_counter = 0
            if func == 'cccc':
                if temp_ligand['frag1'] in phthalo_compat:
                    inds1 = temp_ligand['frag1_func']
                    inds2 = temp_ligand['frag2_func']
                    left, right = split_at_idx(smiles, inds1[0])
                    if '=' in right:
                        additional = 'C=CC=C'
                    else:
                        additional = '=CC=CC'
                    smiles = left+str(counter)+right+additional+str(counter)
                    counter += 1
                    func_counter += 1
                    left2, right2 = split_at_idx(smiles, inds2[0])
                    new_right2 = right2.split(')', 1)
                    if '=' in new_right2[0]:
                        additional = 'C=CC=C'
                    else:
                        additional = '=CC=CC'
                    smiles = left2+str(counter)+new_right2[0]+additional+str(counter)+')'+new_right2[1]
                    counter += 1
                    func_counter += 1
                    used_func_groups.add(funcs[func])
                if temp_ligand['frag3'] in phthalo_compat:
                    inds3 = temp_ligand['frag3_func']
                    inds4 = temp_ligand['frag4_func']
                    left, right = split_at_idx(smiles, inds3[0])
                    new_right1 = right.split(')', 1)
                    if '=' in new_right1[0]:
                        additional = 'C=CC=C'
                    else:
                        additional = '=CC=CC'
                    smiles = left+str(counter)+new_right1[0]+additional+str(counter)+')'+new_right1[1]
                    counter += 1
                    func_counter += 1
                    left2, right2 = split_at_idx(smiles, inds4[0])
                    new_right2 = right2.split(')', 1)
                    if '=' in new_right2[0]:
                        additional = 'C=CC=C'
                    else:
                        additional = '=CC=CC'
                    smiles = left2+str(counter)+new_right2[0]+additional+str(counter)+')'+new_right2[1]
                    counter += 1
                    func_counter += 1
                    used_func_groups.add(funcs[func])
            else:
                if temp_ligand['frag1'] in double_func:
                    frag1_double_func = True
                else:
                    frag1_double_func = False
                if temp_ligand['frag3'] in double_func:
                    frag2_double_func = True
                else:
                    frag2_double_func = False
                inds1 = temp_ligand['frag1_func']
                inds2 = temp_ligand['frag2_func']
                inds3 = temp_ligand['frag3_func']
                inds4 = temp_ligand['frag4_func']

                func_inds1 = []
                func_inds2 = []
                if isinstance(inds1, list) and len(inds1) > 0:
                    func_inds1 += inds1
                if isinstance(inds2, list) and len(inds2) > 0:
                    func_inds1 += inds2
                if isinstance(inds3, list) and len(inds3) > 0:
                    func_inds2 += inds3
                if isinstance(inds4, list) and len(inds4) > 0:
                    func_inds2 += inds4
                sorted_func_inds1 = sorted(func_inds1)
                sorted_func_inds2 = sorted(func_inds2)
                for ind in reversed(sorted_func_inds1):
                    left, right = split_at_idx(smiles, ind)
                    if frag1_double_func:
                        smiles = left+'('+func+')'+'('+func+')'+right
                        func_counter += 2
                    else:
                        smiles = left+'('+func+')'+right
                        func_counter += 1
                    used_func_groups.add(funcs[func])
                for ind in reversed(sorted_func_inds2):
                    left, right = split_at_idx(smiles, ind)
                    if frag2_double_func:
                        smiles = left+'('+func+')'+'('+func+')'+right
                        func_counter += 2
                    else:
                        smiles = left+'('+func+')'+right
                        func_counter += 1
                    used_func_groups.add(funcs[func])
            if func_counter > 0:
                m = Chem.MolFromSmiles(smiles)
                if m is not None:
                    func_canonical = Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
                    temp_ligand['frag_func_smiles'] = smiles
                    temp_ligand['frag_func_canonical'] = func_canonical
                    temp_ligand['frag_func_smiles_formula'] = CalcMolFormula(m)
                    temp_ligand['frag_func_size'] = count_atoms(smiles)
                    temp_ligand['frag_func_group'] = funcs[func]
                    temp_ligand['frag_func_count'] = func_counter
                    temp_ligand['func_name'] = temp_ligand['name']+'_'+str(funcs[func])+str(func_counter)
                else:
                    temp_ligand['frag_func_smiles'] = smiles
                    temp_ligand['frag_func_canonical'] = func_canonical
                    temp_ligand['frag_func_smiles_formula'] = CalcMolFormula(m)
                    temp_ligand['frag_func_size'] = count_atoms(smiles)
                    temp_ligand['frag_func_group'] = funcs[func]
                    temp_ligand['frag_func_count'] = func_counter
                    if not os.path.exists('failed_frag_func.csv'):
                        df = pd.DataFrame([temp_ligand])
                        df.to_csv('failed_frag_func.csv', index=False)
                    else:
                        df = pd.read_csv('failed_frag_func.csv')
                        new_entry = pd.DataFrame([temp_ligand])
                        new_df = pd.concat([df, new_entry], axis=0)
                        new_df.to_csv('failed_frag_func.csv', index=False)
                print('counter', lignum, rounds)
        temp_dict = {}
        temp_dict['name'] = temp_ligand['name']
        temp_dict['charge'] = temp_ligand['charge']
        temp_dict['func_groups'] = list(used_func_groups)
        if not os.path.exists('successful_functionalizations.csv'):
            df = pd.DataFrame([temp_dict])
            df.to_csv('successful_functionalizations.csv', index=False)
        else:
            df = pd.read_csv('successful_functionalizations.csv')
            new_entry = pd.DataFrame([temp_dict])
            new_df = pd.concat([df, new_entry], axis=0)
            new_df.to_csv('successful_functionalizations.csv', index=False)

    with open(os.getcwd()+'/frag_functionalized_synthesized_ligands.json', 'w') as fout:
        json.dump(functionalized_ligands, fout)


if __name__ == "__main__":
    input_file = sys.argv[1]
    temp_list_first = read_synthesized_macrocycles(input_file)
    '''
    In this section, we disable functionalizations on
    certain fragments after the fact. Here we dont allow
    functionalizations on any pyrans.
    '''
    temp_list = []
    for val in temp_list_first:
        new_dict = val.copy()
        if 'pyran' in new_dict['frag1']:
            new_dict['frag1_func'] = False
        if 'pyran' in new_dict['frag2']:
            new_dict['frag2_func'] = False
        if 'pyran' in new_dict['frag3']:
            new_dict['frag3_func'] = False
        if 'pyran' in new_dict['frag4']:
            new_dict['frag4_func'] = False
        temp_list.append(new_dict)
    main(temp_list)
