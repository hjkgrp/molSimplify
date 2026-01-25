#!/usr/bin/env python
"""Generate reference files for tutorial_7 decoration tests.

Run from the repository root with molSimplify dependencies installed, e.g.:
    python tests/generate_tutorial_7_decoration_refs.py
    python tests/generate_tutorial_7_decoration_refs.py tutorial_7_decoration_multi
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import random
import shutil

import numpy as np

from molSimplify.Scripts.generator import startgen

from helperFuncs import parse4test, working_directory


def main():
    import tempfile
    name = (sys.argv[1] if len(sys.argv) > 1 else "tutorial_7_decoration").strip()
    resource_path_root = REPO_ROOT / "tests" / "testresources"
    tmp_path = Path(tempfile.mkdtemp(prefix=name + "_"))
    infile = resource_path_root / "inputs" / "in_files" / f"{name}.in"
    if not infile.exists():
        print(f"Input file not found: {infile}")
        sys.exit(1)

    random.seed(31415)
    np.random.seed(31415)
    newinfile, myjobdir = parse4test(infile, tmp_path)
    args = ["main.py", "-i", newinfile]
    with working_directory(tmp_path):
        startgen(args, False, False)
    jobdir = Path(myjobdir)
    ref_dir = resource_path_root / "refs" / "tutorial"
    ref_dir.mkdir(parents=True, exist_ok=True)

    for ext, src_name in [(".xyz", f"{name}.xyz"), (".report", f"{name}.report")]:
        src = jobdir / src_name
        if src.exists():
            dst = ref_dir / (name + ext)
            shutil.copy(src, dst)
            print(f"Wrote {dst}")
        else:
            print(f"Missing output: {src}")

    # qcgen writes terachem_input or name.in when -name is set
    qc_src = jobdir / "terachem_input"
    if not qc_src.exists():
        qc_src = jobdir / (name + ".in")
    if qc_src.exists():
        dst = ref_dir / (name + ".qcin")
        shutil.copy(qc_src, dst)
        print(f"Wrote {dst}")
    else:
        print(f"Missing QC output: {qc_src}")

    print(f"Refs for {name} are updated.")


if __name__ == "__main__":
    main()
