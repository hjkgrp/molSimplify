import pytest
from helperFuncs import compare_report_new
from molSimplify.__main__ import main


#def test_main_empty(tmp_path, resource_path_root):
#    main(args=[f"-rundir {tmp_path}"])
#    compare_report_new(
#        tmp_path / "fe_oct_2_water_6_s_5" / "fe_oct_2_water_6_s_5_conf_1" /
#        "fe_oct_2_water_6_s_5_conf_1.report",
#        resource_path_root / "refs" / "test_cli" /
#        "fe_oct_2_water_6_s_5_conf_1.report")


def test_help_no_error():
    """Ensure 'molsimplify -h' (and --help) do not raise, e.g. UnboundLocalError on argparse."""
    main(args=["-h"])
    main(args=["--help"])
    # If we get here, neither call raised.


@pytest.mark.skip("Test for help not working yet.")
def test_help(capsys):
    main(args=["--help",])
    captured = capsys.readouterr()
    print(captured.out)
    assert "Welcome to molSimplify. Only basic usage is described here." in captured.out
