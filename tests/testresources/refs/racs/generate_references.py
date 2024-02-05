import pickle
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.atom3D import atom3D
from molSimplify.Classes.ligand import ligand
from molSimplify.Scripts.io import lig_load
from molSimplify.Informatics.RACassemble import assemble_connectivity_from_parts
from molSimplify.Informatics.lacRACAssemble import get_descriptor_vector


def get_ligand(lig_str):
    lig_mol, _ = lig_load(lig_str)
    lig = ligand(mol3D(), lig_mol.cat, lig_mol.denticity)
    lig.mol = lig_mol
    return lig


def generate_ligand_dict(eq_ligands, ax_ligands):
    eq_ligand_list = []
    for lig_str in eq_ligands:
        eq_ligand_list.append(get_ligand(lig_str))

    ax_ligand_list = []
    for lig_str in ax_ligands:
        ax_ligand_list.append(get_ligand(lig_str))

    ligand_dict = {'eq_ligand_list': eq_ligand_list,
                   'ax_ligand_list': ax_ligand_list,
                   'eq_con_int_list': [h.mol.cat for h in eq_ligand_list],
                   'ax_con_int_list': [h.mol.cat for h in ax_ligand_list]
                   }
    return ligand_dict


def assemble_mol(metal_str, eq_ligands, ax_ligands):
    metal = mol3D()
    metal.addAtom(atom3D(Sym=metal_str))

    ligand_dict = generate_ligand_dict(eq_ligands, ax_ligands)

    mol = assemble_connectivity_from_parts(metal, ligand_dict)
    print(f'assembled mol: {mol.make_formula(latex=False)}')
    return mol


def features_from_mol(mol, ligand_dict):
    names, racs = get_descriptor_vector(mol, custom_ligand_dict=ligand_dict)
    features = dict(zip(names, racs))
    return features


def generate_features(metal_str, eq_ligands, ax_ligands):
    ligand_dict = generate_ligand_dict(eq_ligands, ax_ligands)
    mol = assemble_mol(metal_str, eq_ligands, ax_ligands)
    return features_from_mol(mol, ligand_dict)


def main():
    features = generate_features('Fe', ['carbonyl']*4, ['carbonyl']*2)
    with open('racs_Fe_carbonyl_6.pickle', 'wb') as f:
        pickle.dump(features, f)

    features = generate_features('Mn', ['furan', 'water', 'ammonia', 'furan'],
                                 ['water', 'ammonia'])
    with open('racs_Mn_furan_water_ammonia_furan_water_ammonia.pickle', 'wb') as f:
        pickle.dump(features, f)

    features = generate_features('Co', ['acac', 'en'], ['water', 'hydrogensulfide'])
    with open('racs_Co_acac_en_water_hydrogensulfide.pickle', 'wb') as f:
        pickle.dump(features, f)

    acac = get_ligand('acac')
    bipy = get_ligand('bipy')
    ligand_dict = {
        'ax_ligand_list': [acac, acac],
        'eq_ligand_list': [acac, acac, bipy],
        'ax_con_int_list': [[0], [0]],
        'eq_con_int_list': [[5], [5], [0, 1]]}
    mol = assemble_mol('Cr', ['acac', 'bipy'], ['acac'])
    features = features_from_mol(mol, ligand_dict)
    with open('racs_Cr_acac_acac_bipy.pickle', 'wb') as f:
        pickle.dump(features, f)


if __name__ == '__main__':
    main()
