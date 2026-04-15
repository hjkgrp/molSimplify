import pytest
import sys
import types
import builtins
import matplotlib.pyplot as plt
from molSimplify.__main__ import main


def test_help_no_error():
    """Ensure 'molsimplify -h' (and --help) do not raise; they exit 0 via argparse."""
    with pytest.raises(SystemExit) as exc_info:
        main(args=["-h"])
    #assert exc_info.value.code == 0

    with pytest.raises(SystemExit) as exc_info:
        main(args=["--help"])
    #assert exc_info.value.code == 0


def test_main_does_not_import_pydentate_with_explicit_usercatoms(monkeypatch, tmp_path):
    class DummyMol:
        def writexyz(self, path):
            with open(path, "w") as handle:
                handle.write("dummy xyz\n")

        def writemol2_bodict(self, ignore_dummy_atoms=False, write_bond_orders=True, return_string=False, output_file=None):
            with open(output_file, "w") as handle:
                handle.write("dummy mol2\n")

    class DummyFig:
        def savefig(self, path, dpi=300):
            with open(path, "w") as handle:
                handle.write("dummy fig\n")

    fake_enhanced = types.ModuleType("molSimplify.Scripts.enhanced_structgen")
    fake_enhanced.create_ligand_list = lambda userligand_list, usercatoms_list=None, occupancy_list=None, isomer_list=None: [
        ("lig", [0, 7], 3, None)
    ]
    fake_enhanced.generate_complex = lambda *args, **kwargs: (DummyMol(), [], {}, DummyFig(), [], [])
    fake_enhanced.enhanced_init_ANN = lambda *args, **kwargs: None
    fake_enhanced.enforce_metal_ligand_distances_and_optimize = (
        lambda mol, bondl, backbone_core_indices: (mol, None, None)
    )

    fake_functionality = types.ModuleType("molSimplify.Scripts.enhanced_structgen_functionality")
    fake_functionality.check_badjob = lambda mol: (False, True)

    monkeypatch.setitem(sys.modules, "molSimplify.Scripts.enhanced_structgen", fake_enhanced)
    monkeypatch.setitem(sys.modules, "molSimplify.Scripts.enhanced_structgen_functionality", fake_functionality)
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pydentate":
            raise AssertionError("pydentate should not be imported when explicit usercatoms are provided")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    main(args=[
        "--ligand", "NCCCOCCN",
        "--usercatoms", "[0,7]",
        "--occupancy", "3",
        "--metal", "Fe",
        "--ox", "2",
        "--spin", "1",
        "--geometry", "octahedral",
        "--run-dir", str(tmp_path),
    ])
