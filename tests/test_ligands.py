import pytest
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Scripts.io import readdict, lig_load
from importlib_resources import files as resource_files
from os import listdir
from os.path import isfile, join


path_folder = resource_files("molSimplify").joinpath("Ligands")
path_dict = resource_files("molSimplify").joinpath("Ligands/ligands.dict")
lig_dict = readdict(path_dict)


@pytest.mark.parametrize("lig_name", lig_dict.keys())
def test_ligands_dict(lig_name):
    # Assert that the dict entry has all 6 fields
    assert len(lig_dict[lig_name]) == 6
    lig, emsg = lig_load(lig_name)
    # Assert that the ligand could be loaded
    assert type(lig) is mol3D
    assert emsg == ''
    # Assert that the charge of the loaded ligand equals the
    # charge noted in ligands.dict
    charge = int(lig_dict[lig_name][-1][0])
    assert lig.charge == charge
    connecting_atoms = lig_dict[lig_name][-4]
    group = lig_dict[lig_name][-3]
    if "bidentate" in group:
        assert len(connecting_atoms) == 2
    elif "tridentate" in group:
        assert len(connecting_atoms) == 3
    elif "tetradentate" in group:
        assert len(connecting_atoms) == 4

    # Test that there is no style mistake (additional or missing spaces)
    # in the charge line of .mol files
    if lig_dict[lig_name][0].endswith('.mol'):
        with open(f'{path_folder}/{lig_dict[lig_name][0]}') as fin:
            lines = fin.readlines()
        for line in lines:
            if 'CHG' in line:
                # Should be in the following format:
                # "M  CHGnn8 aaa vvv\n"
                # Where 'n' the number first number after 'CHG' states how
                # many atom/charge pairs (8 characters each) follow.
                n_charges = int(line[6:9])
                # 9 characters including 'n' plus 1 for the newline '/n'
                assert len(line) == 10 + n_charges * 8
                sp = line.split()
                # Recheck that the sum of all charges is the same as the
                # charge listed in ligands.dict
                assert sum([int(s) for s in sp[4::2]]) == charge


def test_no_repeats():
    # This test ensures no key is used more than once in ligands.dict.
    listed_keys = []
    with open(path_dict) as f:
        array_content = f.readlines()
    array_content = [
        line for line in array_content if not line.startswith("#")
    ]  # Ignoring comments.
    listed_keys = [
        line.split(":")[0] for line in array_content
    ]  # Grabbing the ligand names.
    repeated_keys = [
        item for idx, item in enumerate(listed_keys) if item in listed_keys[:idx]
    ]  # The repeated ligand names.
    assert len(repeated_keys) == 0


# Identify the files in the Ligands folder.
my_files = [f for f in listdir(path_folder) if isfile(join(path_folder, f))]
# Keep only the files that end in .mol or .xyz
my_files = [f for f in my_files if f.endswith(".xyz") or f.endswith(".mol")]
# Getting the geometries listed in ligands.dict
geometries = [
    entry[1][0] for entry in list(lig_dict.items())
]  # Requires some ugly indexing to get this done.


@pytest.mark.parametrize("file_name", my_files)
def test_unused_geometries(file_name):
    # This test checks whether any geometry files in the Ligands folder are not used in ligands.dict.
    # This indicates that the geometry file should be removed, as it is clutter.
    assert file_name in geometries
