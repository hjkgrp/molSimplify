import helperFuncs as hp
from molSimplify.__main__ import main


def test_tutorial_09_drawmode(tmp_path, resource_path_root):
    """Run `molsimplify -core zncat -drawmode` and check that zncat.svg matches the reference."""
    with hp.working_directory(tmp_path):
        main(args=["-core", "zncat", "-drawmode"])
    svg_path = tmp_path / "zncat.svg"
    assert svg_path.exists(), f"Expected vector graphic {svg_path} to exist"
    generated = svg_path.read_text()
    assert "<svg" in generated or "svg" in generated.lower(), "Expected zncat.svg to be valid SVG"
    ref_svg = resource_path_root / "refs" / "tutorial" / "tutorial_09" / "zncat.svg"
    assert ref_svg.exists(), f"Reference {ref_svg} missing"
    ref_content = ref_svg.read_text()
    assert generated == ref_content, (
        f"zncat.svg does not match reference {ref_svg}. "
        "Regenerate the reference if the change is intentional."
    )


def test_tutorial_09_part_one(tmp_path, resource_path_root):
    testName = "tutorial_09_part_one"
    threshMLBL = 0.1
    threshLG = 2.0
    threshOG = 2.0
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtest(
        tmp_path, resource_path_root, testName, threshMLBL, threshLG, threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
