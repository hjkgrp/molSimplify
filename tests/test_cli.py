import pytest
from molSimplify.__main__ import main


def test_help_no_error():
    """Ensure 'molsimplify -h' (and --help) do not raise; they exit 0 via argparse."""
    with pytest.raises(SystemExit) as exc_info:
        main(args=["-h"])
    assert exc_info.value.code == 0

    with pytest.raises(SystemExit) as exc_info:
        main(args=["--help"])
    assert exc_info.value.code == 0