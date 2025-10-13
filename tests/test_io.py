import numpy as np
from molSimplify.Scripts.io import (lig_load,
                                    core_load,
                                    printgeoms,
                                    getsubstrates,
                                    readdict_sub,
                                    )
from importlib_resources import files as resource_files


def test_lig_load(resource_path_root):
    lig_file = str(resource_path_root / "inputs" / "io" / "acac.mol2")
    mol, emsg = lig_load(lig_file)
    # Assert that the error message is empty
    assert not emsg
    # Convert to mol3D after loading the OBmol
    mol.convert2mol3D()
    # Load the reference from the ligand folder
    ref, _ = lig_load('acac')

    assert mol.natoms == ref.natoms
    assert all(mol.symvect() == ref.symvect())
    print(mol.coordsvect())
    print(ref.coordsvect())
    np.testing.assert_allclose(mol.coordsvect(), ref.coordsvect())
    assert mol.charge == ref.charge


def test_core_load():
    file = str(resource_files("molSimplify").joinpath("Cores/ferrcore.xyz"))
    core, emsg = core_load(file)
    # Assert that the error message is empty
    assert not emsg
    core.convert2mol3D()
    assert core.make_formula(latex=False, hill_system=True) == "FeC10H9F"

    file = str(resource_files("molSimplify").joinpath("Cores/ferrocene.mol"))
    core, emsg = core_load(file)
    # Assert that the error message is empty
    assert not emsg
    core.convert2mol3D()
    assert core.make_formula(latex=False, hill_system=True) == "FeC10H10"


def test_printgeoms(capsys):
    printgeoms()
    captured = capsys.readouterr()

    ref = (
        "Coordination: 1, geometry: none,\t short name: no\n"
        "Coordination: 2, geometry: linear,\t short name: li\n"
        "Coordination: 3, geometry: trigonal_planar,\t short name: tpl\n"
        "Coordination: 4, geometry: square_planar,\t short name: sqp\n"
        "Coordination: 4, geometry: tetrahedral,\t short name: thd\n"
        "Coordination: 5, geometry: square_pyramidal,\t short name: spy\n"
        "Coordination: 5, geometry: trigonal_bipyramidal,\t short name: tbp\n"
        "Coordination: 6, geometry: octahedral,\t short name: oct\n"
        "Coordination: 6, geometry: trigonal_prismatic,\t short name: tpr\n"
        "Coordination: 7, geometry: pentagonal_bipyramidal,\t short name: pbp\n"
        "Coordination: 8, geometry: square_antiprismatic,\t short name: sqap\n"
        "Coordination: 8, geometry: trigonal_dodecahedral,\t short name: tdhd\n\n"
    )
    assert captured.out == ref


def test_readdict_sub():
    file = resource_files("molSimplify").joinpath("Substrates/substrates.dict")
    sub_dict = readdict_sub(file)
    assert sub_dict["methane"] == ['methane.xyz', 'ch4', '1', ['inter'],
                                   ['N'], ['0', '#', 'BDH', '=', '104.9(0.1)']]
    assert sub_dict["ethane"] == ['ethane.xyz', 'c2h6', '2', ['inter'],
                                  ['B'], ['0', '#', 'BDH', '=', '101.1(0.4)']]


def test_getsubstrates():
    subs = getsubstrates()
    ref = (
        "acetaldehyde acetylene benzene biphenyl bromobenzene cumene "
        "cyclohexene dha diphenylmethane estrogen ethanal ethane ethene "
        "ethylene fluorene formaldehyde formicacid iodobenzene methanal "
        "methane methanoicacid methanol methylazide n-quinolinylbutyramidate "
        "n2 phenyl propane propene propylene propyne tert-butane toluene triazole xanthene"
        )
    assert subs == ref
