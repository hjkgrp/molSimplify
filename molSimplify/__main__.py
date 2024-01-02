# @file __main__.py
# Gateway script to rest of program
#
# Written by Tim Ioannidis for HJK Group
#
# Dpt of Chemical Engineering, MIT

# !/usr/bin/env python
'''
    Copyright 2017 Kulik Lab @ MIT

    This file is part of molSimplify.
    molSimplify is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published
    by the Free Software Foundation, either version 3 of the License,
    or (at your option) any later version.

    molSimplify is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with molSimplify. If not, see http://www.gnu.org/licenses/.
'''
# fix OB bug: https://github.com/openbabel/openbabel/issues/1983
import sys
import argparse
if not ('win' in sys.platform):
    flags = sys.getdlopenflags()
if not ('win' in sys.platform):
    sys.setdlopenflags(flags)

from molSimplify.Scripts.inparse import (parseinputs_advanced, parseinputs_slabgen,
                                         parseinputs_db, parseinputs_inputgen,
                                         parseinputs_postproc, parseinputs_random,
                                         parseinputs_binding, parseinputs_tsgen,
                                         parseinputs_customcore, parseinputs_naming,
                                         parseinputs_ligdict, parseinputs_basic,
                                         parseCLI)
from molSimplify.Scripts.generator import startgen
from molSimplify.Classes.globalvars import globalvars
from molSimplify.utils.tensorflow import tensorflow_silence

globs = globalvars()
# Basic help description string
DescString_basic = (
    'Welcome to molSimplify. Only basic usage is described here.\n'
    'For help on advanced modules, please refer to our documentation at '
    'molsimplify.mit.edu or provide additional commands to -h, as below:\n'
    '-h advanced: advanced structure generation help\n'
    '-h slabgen: slab builder help\n'
    # '-h chainb: chain builder help\n'
    '-h autocorr: automated correlation analysis help\n'
    '-h db: database search help\n'
    '-h inputgen: quantum chemistry code input file generation help\n'
    '-h postproc: post-processing help\n'
    '-h random: random generation help\n'
    '-h binding: binding species (second molecule) generation help\n'
    '-h customcore: custom core functionalization help\n'
    '-h tsgen: transition state generation help\n'
    '-h naming: custom filename help\n'
    '-h liganddict: ligands.dict help\n'
)
# Advanced help description string
DescString_advanced = 'Printing advanced structure generation help.'
# Slab builder help description string
DescString_slabgen = 'Printing slab builder help.'
# Chain builder help description string
DescString_chainb = 'Printing chain builder help.'
# Automated correlation analysis description string
DescString_autocorr = 'Printing automated correlation analysis help.'
# Database search help description string
DescString_db = 'Printing database search help.'
# Input file generation help description string
DescString_inputgen = 'Printing quantum chemistry code input file generation help.'
# Post-processing help description string
DescString_postproc = 'Printing post-processing help.'
# Random generation help description string
DescString_random = 'Printing random generation help.'
# Binding species placement help description string
DescString_binding = 'Printing binding species (second molecule) generation help.'
# Transition state generation help description string
DescString_tsgen = 'Printing transition state generation help.'
# Ligand replacement help description string
DescString_customcore = 'Printing ligand replacement help.'
# Custom file naming help description string
DescString_naming = 'Printing custom filename help.'
# Ligand dictionary help description string
DescString_ligdict = 'Printing ligand dictionary help.'


try:
    from PyQt5.QtWidgets import QApplication
    from molSimplify.Classes.mGUI import mGUI
    qtflag = True
except ImportError:
    qtflag = False


# Main function
#  @param args Argument namespace
def main(args=None):
    # issue a call to test TF, this is needed to keep
    # ordering between openbabel and TF calls consistent
    # on some sytems
    if globs.testTF():
        print('TensorFlow connection successful')
        tensorflow_silence()
    else:
        print('TensorFlow connection failed')

    if args is None:
        args = sys.argv[1:]
    # ## run GUI by default ###
    gui = True
    if len(args) == 0 and not qtflag:
        print("\nGUI not supported since PyQt5 can not be loaded. Please use commandline version.\n")
        exit()
    ####################################
    # ## print help ###
    elif '-h' in args or '-H' in args or '--help' in args:
        if 'advanced' in args:
            parser = argparse.ArgumentParser(description=DescString_advanced)
            parseinputs_advanced(parser)
        if 'slabgen' in args:
            parser = argparse.ArgumentParser(description=DescString_slabgen)
            parseinputs_slabgen(parser)
        #    elif 'chainb' in args:
        #        parser = argparse.ArgumentParser(description=DescString_chainb)
        #        parseinputs_chainb(parser)
        #    elif 'autocorr' in args:
        #        parser = argparse.ArgumentParser(description=DescString_autocorr)
        #        parseinputs_autocorr(parser)
        elif 'db' in args:
            parser = argparse.ArgumentParser(description=DescString_db)
            parseinputs_db(parser)
        elif 'inputgen' in args:
            parser = argparse.ArgumentParser(description=DescString_inputgen)
            parseinputs_inputgen(parser)
        elif 'postproc' in args:
            parser = argparse.ArgumentParser(description=DescString_postproc)
            parseinputs_postproc(parser)
        elif 'random' in args:
            parser = argparse.ArgumentParser(description=DescString_random)
            parseinputs_random(parser)
        elif 'binding' in args:
            parser = argparse.ArgumentParser(description=DescString_binding)
            parseinputs_binding(parser)
        elif 'tsgen' in args:
            parser = argparse.ArgumentParser(description=DescString_tsgen)
            parseinputs_tsgen(parser)
        elif 'customcore' in args:
            parser = argparse.ArgumentParser(description=DescString_customcore)
            parseinputs_customcore(parser)
        elif 'naming' in args:
            parser = argparse.ArgumentParser(description=DescString_naming)
            parseinputs_naming(parser)
        elif 'liganddict' in args:
            # The formatter class allows for the display of new lines.
            parser = argparse.ArgumentParser(description=DescString_ligdict,
                                             formatter_class=argparse.RawTextHelpFormatter)
            parseinputs_ligdict(parser)
        else:
            # print basic help
            parser = argparse.ArgumentParser(description=DescString_basic,
                                             formatter_class=argparse.RawDescriptionHelpFormatter)
            parseinputs_basic(parser)
        exit()
    # ## run with gui ###
    elif gui and len(args) == 0:
        print('molSimplify is starting!')
        # ## create main application
        app = QApplication(sys.argv)  # main application
        _ = mGUI(app)  # main GUI class
        app.processEvents()
        app.exec_()
    # ## if input file is specified run without GUI ###
    elif '-i' in args:
        print('Input file detected, reading arguments from input file')
        print('molSimplify is starting!')
        gui = False
        # run from commandline
        startgen(sys.argv, False, gui)
    # ## grab from commandline arguments ###
    else:
        print('No input file detected, reading arguments from commandline')
        print('molSimplify is starting!')
        gui = False
        # create input file from commandline
        infile = parseCLI([_f for _f in args if _f])
        args = ['main.py', '-i', infile]
        startgen(args, False, gui)


if __name__ == '__main__':
    main()
