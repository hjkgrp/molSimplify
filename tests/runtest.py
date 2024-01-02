from molSimplify.Scripts.generator import startgen


def runtest(resource_path_root):
    infile = resource_path_root / "inputs" / "example_1_noff.in"
    args = ['main.py', '-i', infile]
    startgen(args, False, False)
    infile = resource_path_root / "inputs" / "example_1.in"
    args = ['main.py', '-i', infile]
    startgen(args, False, False)
