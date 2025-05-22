import pytest
import numpy as np
import os
from molSimplify.Informatics.MOF.PBC_functions import readcif
from molSimplify.Informatics.MOF.linker_rotation import rotate_and_write

@pytest.mark.parametrize(
    ("cif_name", "rotation_angle"),
    [
        ("UiO-66", 45), # Zr, BDC, 45 degrees
        ("UiO-66", 135), # Zr, BDC, 90 degrees
        ("UiO-66", 270), # Zr, BDC, 135 degrees
        ("UiO-67", 45), # Zr, BPDC, 45 degrees
        ("MIL-53", 45) # Al, BDC, 45 degrees
    ])

def test_linker_rotation(resource_path_root, tmp_path, cif_name, rotation_angle):
    input_cif = os.path.join(resource_path_root, "inputs", "cif_files", f"{cif_name}.cif")

    rotate_and_write(
        input_cif=input_cif,
        path2write=tmp_path,
        rot_angle=rotation_angle,
        is_degree=True
    )

    new_cif_name = f"{cif_name}_rot_{rotation_angle:.2f}.cif"
    generated_cif = str(tmp_path / new_cif_name)
    reference_cif = str(resource_path_root / "refs" / "informatics" / "mof" / "cif" / new_cif_name)

    cpar1, atom_types1, fcoords1 = readcif(generated_cif)
    cpar2, atom_types2, fcoords2 = readcif(reference_cif)

    assert np.allclose(cpar1, cpar2, atol=1e-3)
    assert atom_types1 == atom_types2
    assert np.allclose(fcoords1, fcoords2, atol=1e-3)
