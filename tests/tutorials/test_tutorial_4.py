import shutil
from molSimplify.Classes.globalvars import globalvars
from molSimplify.Scripts.generator import startgen
from helperFuncs import parse4test, working_directory


def run_db_search(tmp_path, resource_path_root, name):
    # Set the path for the data base file:
    globs = globalvars()
    globs.chemdbdir = str(resource_path_root / "inputs" / "tutorial_4")

    infile = resource_path_root / "inputs" / "in_files" / name
    newinfile, _ = parse4test(infile, tmp_path)
    args = ['main.py', '-i', newinfile]

    with working_directory(tmp_path):
        startgen(args, False, False)


def test_tutorial_4_query(tmp_path, resource_path_root):
    run_db_search(tmp_path, resource_path_root, "tutorial_4_query.in")

    # Compare the generated output file to the reference file.
    with open(f"{tmp_path}/simres.smi", "r") as f:
        output = f.readlines()
    with open(resource_path_root / "refs" / "tutorial" / "tutorial_4" / "simres.smi") as f:
        reference = f.readlines()

    assert output == reference


def test_tutorial_4_dissim(tmp_path, resource_path_root):
    # Copy the results from the query into the working directory.
    shutil.copyfile(resource_path_root / "refs" / "tutorial" / "tutorial_4" / "simres.smi",
                    tmp_path / "simres.smi")

    run_db_search(tmp_path, resource_path_root, "tutorial_4_dissim.in")

    # Compare the generated output file to the reference file.
    with open(f"{tmp_path}/dissimres.smi", "r") as f:
        output = f.readlines()
    with open(resource_path_root / "refs" / "tutorial" / "tutorial_4" / "dissimres.smi") as f:
        reference = f.readlines()

    assert output == reference


def test_tutorial_4_human(tmp_path, resource_path_root):
    run_db_search(tmp_path, resource_path_root, "tutorial_4_human.in")

    # Compare the generated output file to the reference file.
    with open(f"{tmp_path}/simres.smi", "r") as f:
        output = f.readlines()
    with open(resource_path_root / "refs" / "tutorial" / "tutorial_4" / "simres_human.smi") as f:
        reference = f.readlines()

    assert output == reference
