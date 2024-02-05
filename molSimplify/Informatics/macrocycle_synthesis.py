import os
from molSimplify.Informatics.fragment_classes import fragment, bridge, tetradentate
import json
import itertools
import pandas as pd

# ## Below is RDKit dependency that this code uses
# from rdkit import Chem
# from rdkit.Chem import AllChem
# ###


def cyclic_equiv(u, v):
    n, i, j = len(u), 0, 0
    if n != len(v):
        return False
    while i < n and j < n:
        k = 1
        while k <= n and u[(i + k) % n] == v[(j + k) % n]:
            k += 1
        if k > n:
            return True
        if u[(i + k) % n] > v[(j + k) % n]:
            i += k
        else:
            j += k
    return False


frag_options = {'pyrrole': [('C1[N-]C=CC=1', 1, (3, 4)),
                            ('C1N=CC=C1', 1, (3, 4)),
                            ('C1[N-]CC=C1', 1, (3, 4))],
                'pyridine': [('C1=NC=CC=C1', 1, (4,))],
                'trimethylamine': [('CN(C)C', 1, ())],
                'dimethylamine': [('CNC', 1, ())],
                'phosphorine': [('C1=PC=CC=C1', 1, (4,))],
                'trimethylphosphine': [('CP(C)C', 1, ())],
                'phosphole': [('C1[P-]C=CC=1', 1, (3, 4)),
                              ('C1P=CC=C1', 1, (3, 4)),
                              ('C1[P-]CC=C1', 1, (3, 4))],
                'dimethylphosphine': [('CPC', 1, ())],
                'pyran': [('C1OC=CCC=1', 1, (4,))],
                'dimethylether': [('COC', 1, ())],
                'furan': [('C1OC=CC=1', 1, (3, 4))],
                'thiopyran': [('C1SC=CCC=1', 1, (4,))],
                'dimethylthioether': [('CSC', 1, ())],
                'thiophene': [('C1SC=CC=1', 1, (3, 4))],
                'imidazole2ylidine': [('N1[C]NC=C1', 1, (3, 4))]}
bridge_options = {'C_double': ['C', (2, 1), (0,)],
                  'N_double': ['N', (2, 1), ()],
                  'P_double': ['P', (2, 1), ()],
                  'C': ['C', (1, 1), (0, 1)],
                  'N': ['N', (1, 1), (0,)],
                  'O': ['O', (1, 1), ()],
                  'S': ['S', (1, 1), ()],
                  'P': ['P', (1, 1), (0,)],
                  'X': ['X', (1, 1), ()]}

monodentates = (list(set([(val[0], val[0]) for val in itertools.combinations(list(frag_options.keys()), r=1)]))
                + list(set([(val[0], val[1]) for val in itertools.combinations(list(frag_options.keys()), r=2)]))
                )
bridges = list(itertools.product(list(bridge_options.keys()),
                                 list(bridge_options.keys()),
                                 list(bridge_options.keys())))
print('---- ', len(bridges)*len(monodentates), ' possible combos being tried ')
ligands_to_dump = []
for i in monodentates:
    # New set of monodentates, cannot be rotational repeats with other monodentates
    checked_list = set()
    for j in bridges:
        temp_dict = {}
        parts_list = [i[0], j[0], i[0], j[2], i[1], j[1], i[1], j[2]]
        if any([cyclic_equiv(parts_list, list(val)) for val in checked_list]):
            print('This is a rotational repeat. Skipping.', parts_list)
            atoms = (bridge_options[j[0]][0]
                     + bridge_options[j[1]][0]
                     + bridge_options[j[2]][0]
                     + frag_options[i[0]][0][0]
                     + frag_options[i[1]][0][0])
            heteroatoms = list(set([val for val in atoms if val.isalpha()])-set('C')-set('X'))
            temp_dict['bridge1'] = j[0]
            temp_dict['bridge2'] = j[1]
            temp_dict['bridge3'] = j[2]
            temp_dict['fragment1'] = i[0]
            temp_dict['fragment2'] = i[0]
            temp_dict['fragment3'] = i[1]
            temp_dict['fragment4'] = i[1]
            temp_dict['heteroatoms'] = heteroatoms
            temp_dict['reason'] = 'rotational_repeat'
            if not os.path.exists('failed_syntheses.csv'):
                df = pd.DataFrame([temp_dict])
                df.to_csv('failed_syntheses.csv', index=False)
            else:
                df = pd.read_csv('failed_syntheses.csv')
                new_entry = pd.DataFrame([temp_dict])
                new_df = pd.concat([df, new_entry], axis=0)
                new_df.to_csv('failed_syntheses.csv', index=False)
            continue
        else:
            checked_list.add(tuple(parts_list))
        atoms = (bridge_options[j[0]][0]
                 + bridge_options[j[1]][0]
                 + bridge_options[j[2]][0]
                 + frag_options[i[0]][0][0]
                 + frag_options[i[1]][0][0])
        heteroatoms = list(set([val for val in atoms if val.isalpha()])-set('C')-set('X'))
        if len(heteroatoms) > 2:
            temp_dict['bridge1'] = j[0]
            temp_dict['bridge2'] = j[1]
            temp_dict['bridge3'] = j[2]
            temp_dict['fragment1'] = i[0]
            temp_dict['fragment2'] = i[0]
            temp_dict['fragment3'] = i[1]
            temp_dict['fragment4'] = i[1]
            temp_dict['heteroatoms'] = heteroatoms
            temp_dict['reason'] = 'heteroatoms'
            if not os.path.exists('failed_syntheses.csv'):
                df = pd.DataFrame([temp_dict])
                df.to_csv('failed_syntheses.csv', index=False)
            else:
                df = pd.read_csv('failed_syntheses.csv')
                new_entry = pd.DataFrame([temp_dict])
                new_df = pd.concat([df, new_entry], axis=0)
                new_df.to_csv('failed_syntheses.csv', index=False)
            continue
        test_frag1 = fragment(i[0], frag_options[i[0]], start_macrocycle=True)
        test_frag2 = fragment(i[0], frag_options[i[0]], ring_closure_ind=2)
        test_frag3 = fragment(i[1], frag_options[i[1]], ring_closure_ind=3)
        test_frag4 = fragment(i[1], frag_options[i[1]], ring_closure_ind=4)
        test_bridge1 = bridge(bridge_options[j[0]][0], bridge_options[j[0]][1], bridge_options[j[0]][2])
        test_bridge2 = bridge(bridge_options[j[1]][0], bridge_options[j[1]][1], bridge_options[j[1]][2])
        test_bridge3 = bridge(bridge_options[j[2]][0], bridge_options[j[2]][1], bridge_options[j[2]][2])
        test_tetra = tetradentate(test_frag1, test_frag2, test_frag3, test_frag4, test_bridge1, test_bridge2, test_bridge3)
        possible_ligands = test_tetra.stitch()
        ligands_to_dump += possible_ligands
        if len(possible_ligands) == 0:
            temp_dict['bridge1'] = j[0]
            temp_dict['bridge2'] = j[1]
            temp_dict['bridge3'] = j[2]
            temp_dict['fragment1'] = i[0]
            temp_dict['fragment2'] = i[0]
            temp_dict['fragment3'] = i[1]
            temp_dict['fragment4'] = i[1]
            temp_dict['heteroatoms'] = heteroatoms
            temp_dict['reason'] = 'no_compatible_combo'
            if not os.path.exists('failed_syntheses.csv'):
                df = pd.DataFrame([temp_dict])
                df.to_csv('failed_syntheses.csv', index=False)
            else:
                df = pd.read_csv('failed_syntheses.csv')
                new_entry = pd.DataFrame([temp_dict])
                new_df = pd.concat([df, new_entry], axis=0)
                new_df.to_csv('failed_syntheses.csv', index=False)
            continue
        else:
            temp_dict['bridge1'] = j[0]
            temp_dict['bridge2'] = j[1]
            temp_dict['bridge3'] = j[2]
            temp_dict['fragment1'] = i[0]
            temp_dict['fragment2'] = i[0]
            temp_dict['fragment3'] = i[1]
            temp_dict['fragment4'] = i[1]
            temp_dict['heteroatoms'] = heteroatoms
            temp_dict['num_macrocycles'] = len(possible_ligands)
            if not os.path.exists('successful_syntheses.csv'):
                df = pd.DataFrame([temp_dict])
                df.to_csv('successful_syntheses.csv', index=False)
            else:
                df = pd.read_csv('successful_syntheses.csv')
                new_entry = pd.DataFrame([temp_dict])
                new_df = pd.concat([df, new_entry], axis=0)
                new_df.to_csv('successful_syntheses.csv', index=False)

with open(os.getcwd()+'/synthesized_ligands.json', 'w') as fout:
    json.dump(ligands_to_dump, fout, indent=2)
