import pytest
import pickle
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Informatics.RACassemble import create_OHE


@pytest.mark.parametrize('xyz_path, ref_path', [
    ('fe_carbonyl_6.xyz', 'racs_Fe_carbonyl_6.pickle'),
    ('mn_furan_water_ammonia_furan_water_ammonia.xyz',
     'racs_Mn_furan_water_ammonia_furan_water_ammonia.pickle'),
    ('cr_acac_acac_bipy.xyz',
     'racs_Cr_acac_acac_bipy.pickle'),
    ('co_acac_en_water_hydrogensulfide.xyz',
     'racs_Co_acac_en_water_hydrogensulfide.pickle')])
def test_Mn_water2_ammonia_furan2_ammonia(resource_path_root, xyz_path, ref_path):
    xyz_path = resource_path_root / "inputs" / "xyz_files" / xyz_path
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
    fe_pyr_6_path = resource_path_root / "inputs" / "xyz_files" / "fe_pyr_6.xyz"
    fe_bipy_3_path = resource_path_root / "inputs" / "xyz_files" / "fe_bipy_3.xyz"
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
