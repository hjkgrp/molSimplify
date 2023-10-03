import os
from molSimplify.Scripts.inparse import (parseinputfile, checkinput,
                                         parseall,
                                         parseinputs_basic,
                                         parseinputs_advanced,
                                         check_ligloc)
from argparse import ArgumentParser, Namespace
from pkg_resources import resource_filename, Requirement


def test_parseinputfile_empty():
    defaults = {'skipANN': False, 'oldANN': False,
                'dbvdent': False, 'dbvconns': False,
                'dbvhyb': False, 'dbvlinks': False,
                'rprompt': False, 'rundir': f'{os.getcwd()}/Runs'}

    args = Namespace()
    parseinputfile(args, inputfile_str=' ')

    # Assert defaults are set
    assert args.__dict__ == defaults


def test_parseinputs_basic(monkeypatch):
    # Monkeypatch is used to change sys.argv parsed by the Argumentparser.
    monkeypatch.setattr('sys.argv', ['molsimplify'])
    parser = ArgumentParser()
    args = parseinputs_basic(parser)
    defaults = dict(coord=False, core=None, ff='uff', ff_final_opt=None,
                    ffoption='N', geo=False, geometry=False, keepHs=None,
                    lig=None, ligloc=False, ligocc=False, multiplicity=None,
                    oxstate=None, rundir=False, skipANN=None, smicat=False,
                    spin=None, spinmultiplicity=None, custom_data_dir=None)
    assert args.__dict__ == defaults


def test_parseinputs_advanced(monkeypatch):
    # Monkeypatch is used to change sys.argv parsed by the Argumentparser.
    monkeypatch.setattr('sys.argv', ['molsimplify'])
    parser = ArgumentParser()
    args = parseinputs_advanced(parser)
    defaults = dict(MLbonds=False, antigeoisomer=None, calccharge=True,
                    charge=None, decoration=False, decoration_index=False,
                    distort='0', genall=False, isomers=None, langles=False,
                    ligalign=False, nconfs='1', oldANN=None, pangles=False,
                    reportonly=None, scoreconfs=False, stereos=None)
    assert args.__dict__ == defaults


def test_checkinput(monkeypatch):
    # Monkeypatch is used to change sys.argv parsed by the Argumentparser.
    monkeypatch.setattr('sys.argv', ['molsimplify'])
    parser = ArgumentParser()
    # Runs all parsers and populates the args Namespace
    args = parseall(parser)
    checkinput(args, calctype='base')
    # Test a few defaults:
    assert args.core == ['Fe']
    assert args.oxstate == '2'
    assert args.spin == '5'


def test_parseinputfile_inputfile_kwarg():
    """Test that both methods of calling parseinputfile
    (with an args.i file or inputfile_str kwarg) yield the same result"""
    infile = resource_filename(Requirement.parse(
        "molSimplify"), "tests/inputs/example_1.in")

    args1 = Namespace(i=infile)
    parseinputfile(args1)

    with open(infile, 'r') as fin:
        lines = fin.read()
    args2 = Namespace()
    parseinputfile(args2, inputfile_str=lines)
    # Add 'i' argument for comparison
    args2.i = infile

    assert args1 == args2


def test_check_ligloc():
    assert check_ligloc('-ligloc True') is True
    assert check_ligloc('-ligloc true') is True
    assert check_ligloc('-ligloc False') is False
    assert check_ligloc('-ligloc false') is False
    # Test passing in backbone atoms
    assert check_ligloc('-ligloc [[1,2,3],[4,5],[6]]') == [[1, 2, 3], [4, 5], [6]]
    assert check_ligloc('-ligloc [[1, 2, 5], [3, 4, 6]]') == [[1, 2, 5], [3, 4, 6]]
    assert check_ligloc('-ligloc [[12, 4], [2,17], [3, 4]]') == [[12, 4], [2, 17], [3, 4]]
