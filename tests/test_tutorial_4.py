import shutil
from molSimplify.Classes.globalvars import globalvars
from molSimplify.Scripts.generator import startgen
from helperFuncs import parse4test, working_directory


def run_db_search(tmpdir, resource_path_root, name):
    # Set the path for the data base file:
    globs = globalvars()
    globs.chemdbdir = str(resource_path_root / "inputs")

    infile = resource_path_root / "inputs" / name
    newinfile, _ = parse4test(infile, tmpdir)
    args = ['main.py', '-i', newinfile]

    with working_directory(tmpdir):
        startgen(args, False, False)


def test_tutorial_4_query(tmpdir, resource_path_root):
    run_db_search(tmpdir, resource_path_root, "tutorial_4_query.in")

    # Compare the generated output file to the reference file
    with open(f"{tmpdir}/simres.smi", "r") as f:
        output = f.readlines()
    with open(resource_path_root / "refs" / "tutorial_4" / "simres.smi") as f:
        reference = f.readlines()

    assert output == reference


def test_tutorial_4_dissim(tmpdir, resource_path_root):
    # Copy the results from the query into the working directory
    shutil.copyfile(resource_path_root / "refs" / "tutorial_4" / "simres.smi",
                    tmpdir / "simres.smi")

    run_db_search(tmpdir, resource_path_root, "tutorial_4_dissim.in")

    # Compare the generated output file to the reference file
    with open(f"{tmpdir}/dissimres.smi", "r") as f:
        output = f.readlines()
    with open(resource_path_root / "refs" / "tutorial_4" / "dissimres.smi") as f:
        reference = f.readlines()

    assert output == reference


def test_tutorial_4_human(tmpdir, resource_path_root):
    run_db_search(tmpdir, resource_path_root, "tutorial_4_human.in")

    # Compare the generated output file to the reference file
    with open(f"{tmpdir}/simres.smi", "r") as f:
        output = f.readlines()
    with open(resource_path_root / "refs" / "tutorial_4" / "simres_human.smi") as f:
        reference = f.readlines()

    assert output == reference
