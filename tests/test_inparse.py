import os
from molSimplify.Scripts.inparse import (parseinputfile, checkinput,
                                         parseall,
                                         parseinputs_basic,
                                         parseinputs_advanced)
from argparse import ArgumentParser, Namespace


def test_parseinputfile_empty():
    defaults = {'skipANN': False, 'oldANN': False,
                'dbvdent': False, 'dbvconns': False,
                'dbvhyb': False, 'dbvlinks': False,
                'rprompt': False, 'rundir': f'{os.getcwd()}/Runs',
                'overwrite': False,}

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
                    ligalign=False, nconfs='1', no_tabs=False, oldANN=None, pangles=False,
                    reportonly=None, scoreconfs=False, stereos=None,
                    jobmanager=False)
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


def test_parseinputfile_inputfile_kwarg(resource_path_root):
    """Test that both methods of calling parseinputfile
    (with an args.i file or inputfile_str kwarg) yield the same result"""
    infile = resource_path_root / "inputs" / "example_1.in"

    args1 = Namespace(i=infile)
    parseinputfile(args1)

    with open(infile, 'r') as fin:
        lines = fin.read()
    args2 = Namespace()
    parseinputfile(args2, inputfile_str=lines)
    # Add 'i' argument for comparison
    args2.i = infile

    assert args1 == args2
