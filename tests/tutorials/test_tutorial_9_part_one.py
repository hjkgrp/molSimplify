import re

import helperFuncs as hp
from molSimplify.__main__ import main


def _normalize_svg_for_compare(svg_str: str) -> str:
    """Normalize SVG for order-independent, platform-tolerant comparison.
    Sorts element lines and rounds coordinates to avoid CI float differences.
    """
    lines = [ln.strip() for ln in svg_str.strip().splitlines() if ln.strip()]

    def round_nums_in_line(line: str) -> str:
        def sub(m):
            try:
                return f"{float(m.group(0)):.4f}"
            except ValueError:
                return m.group(0)
        return re.sub(r"-?\d+\.?\d*", sub, line)

    normalized = sorted(round_nums_in_line(ln) for ln in lines)
    return "\n".join(normalized)


def test_tutorial_9_drawmode(tmp_path, resource_path_root):
    """Run `molsimplify -core zncat -drawmode` and check that zncat.svg matches the reference."""
    with hp.working_directory(tmp_path):
        main(args=["legacy", "-core", "zncat", "-drawmode"])
    svg_path = tmp_path / "zncat.svg"
    assert svg_path.exists(), f"Expected vector graphic {svg_path} to exist"
    generated = svg_path.read_text()
    assert "<svg" in generated or "svg" in generated.lower(), "Expected zncat.svg to be valid SVG"
    ref_svg = resource_path_root / "refs" / "tutorial" / "tutorial_9" / "zncat.svg"
    assert ref_svg.exists(), f"Reference {ref_svg} missing"
    ref_content = ref_svg.read_text()
    gen_norm = _normalize_svg_for_compare(generated)
    ref_norm = _normalize_svg_for_compare(ref_content)
    assert gen_norm == ref_norm, (
        f"zncat.svg does not match reference {ref_svg} (after normalizing element order and "
        "numerics). Regenerate the reference if the change is intentional."
    )


def test_tutorial_9_part_one(tmp_path, resource_path_root):
    testName = "tutorial_9_part_one"
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
