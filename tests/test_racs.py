import pytest
import pickle
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import ligand
from molSimplify.Informatics.RACassemble import create_OHE
from molSimplify.Informatics.lacRACAssemble import get_descriptor_vector
from molSimplify.Scripts.io import lig_load


@pytest.mark.parametrize('xyz_path, ref_path', [
    ('fe_carbonyl_6.xyz', 'racs_Fe_carbonyl_6.pickle'),
    ('mn_furan_water_ammonia_furan_water_ammonia.xyz',
     'racs_Mn_furan_water_ammonia_furan_water_ammonia.pickle'),
    ('cr_acac_acac_bipy.xyz',
     'racs_Cr_acac_acac_bipy.pickle'),
    ('co_acac_en_water_hydrogensulfide.xyz',
     'racs_Co_acac_en_water_hydrogensulfide.pickle')])
def test_Mn_water2_ammonia_furan2_ammonia(resource_path_root, xyz_path, ref_path):
    xyz_path = resource_path_root / "refs" / "racs" / xyz_path
    mol = mol3D()
    mol.readfromxyz(xyz_path)
    features = mol.get_features()

    ref_path = resource_path_root / "refs" / "racs" / ref_path
    with open(ref_path, 'rb') as fin:
        ref_features = pickle.load(fin)

    assert features.keys() == ref_features.keys()
    for key, val in features.items():
        assert abs(val - ref_features[key]) < 1e-4


def test_six_pyridine_vs_three_bipy(resource_path_root):
    """Up to depth 2 the atom centered racs features for pyr_6 and bipy_3
    should be the same"""
    fe_pyr_6_path = resource_path_root / "refs" / "racs" / "fe_pyr_6.xyz"
    fe_bipy_3_path = resource_path_root / "refs" / "racs" / "fe_bipy_3.xyz"
    fe_pyr_6 = mol3D()
    fe_pyr_6.readfromxyz(fe_pyr_6_path)
    fe_bipy_3 = mol3D()
    fe_bipy_3.readfromxyz(fe_bipy_3_path)

    features_pyr = fe_pyr_6.get_features()
    features_bipy = fe_bipy_3.get_features()

    properties = ['chi', 'Z', 'T', 'S', 'I']
    start_scopes = [('mc', 'all'), ('lc', 'ax'), ('lc', 'eq'),
                    ('D_mc', 'all'), ('D_lc', 'ax'), ('D_lc', 'eq')]

    for start, scope in start_scopes:
        for depth in range(2):
            for prop in properties:
                key = f'{start}-{prop}-{depth}-{scope}'
                assert features_pyr[key] == features_bipy[key]


@pytest.mark.skip('Test fails because molSimplify averages the equatorial '
                  'plane differently for bidentates')
def test_pyr_4_furan_2_vs_bipy_2_bifuran(resource_path_root):
    fe_pyr_4_furan_2_path = resource_path_root / "refs" / "racs" / "fe_pyr_4_furan_2.xyz"
    fe_bipy_2_bifuran_path = resource_path_root / "refs" / "racs" / "fe_bipy_2_bifuran.xyz"
    fe_pyr_4_furan_2 = mol3D()
    fe_pyr_4_furan_2.readfromxyz(fe_pyr_4_furan_2_path)
    fe_bipy_2_bifuran = mol3D()
    fe_bipy_2_bifuran.readfromxyz(fe_bipy_2_bifuran_path)

    # Enforce the same graph orientation as in bidentates:
    def get_ligand(lig_str):
        lig_mol, _ = lig_load(lig_str)
        lig = ligand(mol3D(), lig_mol.cat, lig_mol.denticity)
        lig.mol = lig_mol
        return lig
    pyr = get_ligand('pyr')
    furan = get_ligand('furan')
    eq_ligand_list = [pyr, pyr, pyr, furan]
    ax_ligand_list = [pyr, furan]
    ligand_dict = {'ax_ligand_list': ax_ligand_list,
                   'eq_ligand_list': eq_ligand_list,
                   'ax_con_int_list': [h.mol.cat for h in ax_ligand_list],
                   'eq_con_int_list': [h.mol.cat for h in eq_ligand_list]}
    names, values = get_descriptor_vector(fe_pyr_4_furan_2, ligand_dict)
    features_mono = dict(zip(names, values))
    features_bi = fe_bipy_2_bifuran.get_features()

    properties = ['Z', 'chi', 'T', 'I', 'S']
    start_scopes = [('mc', 'all'), ('lc', 'ax'), ('lc', 'eq'),
                    ('D_mc', 'all'), ('D_lc', 'ax'), ('D_lc', 'eq')]

    for start, scope in start_scopes:
        for depth in range(2):
            for prop in properties:
                key = f'{start}-{prop}-{depth}-{scope}'
                print(key, features_mono[key], features_bi[key])
                assert features_mono[key] == features_bi[key]


def test_create_OHE():
    ohe_names, ohe_values = create_OHE('Cr', '2')
    assert ohe_names == ['ox2', 'ox3', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
    assert ohe_values == [1, 0, 0, 1, 0, 0, 0, 0]

    _, ohe_values = create_OHE('Cr', '3')
    assert ohe_values == [0, 1, 1, 0, 0, 0, 0, 0]

    _, ohe_values = create_OHE('Mn', '2')
    assert ohe_values == [1, 0, 0, 0, 1, 0, 0, 0]

    _, ohe_values = create_OHE('Mn', '3')
    assert ohe_values == [0, 1, 0, 1, 0, 0, 0, 0]

    _, ohe_values = create_OHE('Fe', '2')
    assert ohe_values == [1, 0, 0, 0, 0, 1, 0, 0]

    _, ohe_values = create_OHE('Fe', '3')
    assert ohe_values == [0, 1, 0, 0, 1, 0, 0, 0]

    _, ohe_values = create_OHE('Co', '2')
    assert ohe_values == [1, 0, 0, 0, 0, 0, 1, 0]

    _, ohe_values = create_OHE('Co', '3')
    assert ohe_values == [0, 1, 0, 0, 0, 1, 0, 0]

    _, ohe_values = create_OHE('Ni', '2')
    assert ohe_values == [1, 0, 0, 0, 0, 0, 0, 1]
