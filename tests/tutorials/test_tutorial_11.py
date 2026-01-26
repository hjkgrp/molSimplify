"""Tutorial 11: load octahedral.xyz, call IsOct(), compare output to reference."""

import json

import helperFuncs as hp
from molSimplify.Classes.mol3D import mol3D


def test_tutorial_11_isoct(resource_path_root):
    """Tutorial 11: octahedral.readfromxyz('octahedral.xyz'); octahedral.IsOct().

    Checks that the return value of IsOct (flag_oct, flag_list, dict_oct_info)
    matches the reference for the tutorial octahedral.xyz (Fe(NH3)6).
    """
    xyz_path = resource_path_root / "inputs" / "in_files" / "tutorial_11" / "octahedral.xyz"
    ref_path = resource_path_root / "refs" / "tutorial" / "tutorial_11" / "ref.json"

    assert xyz_path.exists(), f"Input {xyz_path} missing"
    assert ref_path.exists(), f"Reference {ref_path} missing"

    octahedral = mol3D()
    octahedral.readfromxyz(str(xyz_path))
    flag_oct, flag_list, dict_oct_info = octahedral.IsOct()

    with open(ref_path) as f:
        ref = json.load(f)

    assert flag_oct == ref["flag_oct"], (
        f"IsOct flag_oct: got {flag_oct}, reference {ref['flag_oct']}"
    )
    ref_flag_list = ref["flag_list"]
    if ref_flag_list is None:
        assert flag_list is None or flag_list == "None", (
            f"IsOct flag_list: got {flag_list!r}, expected None or 'None'"
        )
    else:
        assert flag_list == ref_flag_list, (
            f"IsOct flag_list: got {flag_list}, reference {ref_flag_list}"
        )

    thresh = 1e-9
    assert hp.comparedict(ref["dict_oct_info"], dict_oct_info, thresh), (
        f"IsOct dict_oct_info does not match reference (thresh={thresh})"
    )
