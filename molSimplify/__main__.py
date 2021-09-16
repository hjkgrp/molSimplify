## @file __main__.py
#  Gateway script to rest of program
#
#  Written by Tim Ioannidis for HJK Group
#
#  Dpt of Chemical Engineering, MIT

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
import sys, argparse, os, platform, shutil
if not('win' in sys.platform):
    flags = sys.getdlopenflags()
from openbabel import openbabel
if not('win' in sys.platform):
    sys.setdlopenflags(flags)

from .Scripts.inparse import *
from .Scripts.generator import *
from molSimplify.Classes.globalvars import *

globs = globalvars()
## Basic help description string
DescString_basic = 'Welcome to molSimplify. Only basic usage is described here.\n'
DescString_basic += 'For help on advanced modules, please refer to our documentation at molsimplify.mit.edu or provide additional commands to -h, as below:\n'
DescString_basic += '-h advanced: advanced structure generation help\n'
DescString_basic += '-h slabgen: slab builder help\n'
# DescString_basic += '-h chainb: chain builder help\n'
DescString_basic += '-h autocorr: automated correlation analysis help\n'
DescString_basic += '-h db: database search help\n'
DescString_basic += '-h inputgen: quantum chemistry code input file generation help\n'
DescString_basic += '-h postproc: post-processing help\n'
DescString_basic += '-h random: random generation help\n'
DescString_basic += '-h binding: binding species (second molecule) generation help\n'
DescString_basic += '-h customcore: custom core functionalization help\n'
DescString_basic += '-h tsgen: transition state generation help\n'
DescString_basic += '-h naming: custom filename help\n'
## Advanced help description string
DescString_advanced = 'Printing advanced structure generation help.'
## Slab builder help description string
DescString_slabgen = 'Printing slab builder help.'
## Chain builder help description string
DescString_chainb = 'Printing chain builder help.'
## Automated correlation analysis description string
DescString_autocorr = 'Printing automated correlation analysis help.'
## Database search help description string
DescString_db = 'Printing database search help.'
## Input file generation help description string
DescString_inputgen = 'Printing quantum chemistry code input file generation help.'
## Post-processing help description string
DescString_postproc = 'Printing post-processing help.'
## Random generation help description string
DescString_random = 'Printing random generation help.'
## Binding species placement help description string
DescString_binding = 'Printing binding species (second molecule) generation help.'
## Transition state generation help description string
DescString_tsgen = 'Printing transition state generation help.'
## Ligand replacement help description string
DescString_customcore = 'Printing ligand replacement help.'
## Custom file naming help description string
DescString_naming = 'Printing custom filename help.'

def tensorflow_silence():
    ## thanks to
    # stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
    try:
        from tensorflow.compat.v1 import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['KMP_WARNINGS'] = '0'

        def deprecated(date, instructions, warn_once=False):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass


try:
    import PyQt5
    from PyQt5.QtGui import *
    from molSimplify.Classes.mGUI import *

    qtflag = True
except ImportError:
    qtflag = False
    pass


## Main function
#  @param args Argument namespace
def main(args=None):
    ## issue a call to test TF, this is needed to keep
    ## ordering between openbabel and TF calls consistent
    ## on some sytems
    if globs.testTF():
        print('TensorFlow connection successful')
        tensorflow_silence()
    else:
        print('TensorFlow connection failed')

    if args is None:
        args = sys.argv[1:]
    ### run GUI by default ###
    args = sys.argv[1:]
    gui = True
    cmd = False
    if len(args)==0 and not qtflag:
        print("\nGUI not supported since PyQt5 can not be loaded. Please use commandline version.\n")
        exit()
    ####################################
    ### print help ###
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
        else:
            # print basic help
            parser = argparse.ArgumentParser(description=DescString_basic,
                                             formatter_class=argparse.RawDescriptionHelpFormatter)
            parseinputs_basic(parser)
        exit()
    ### run with gui ###
    elif gui and len(args) == 0:
        print('molSimplify is starting!')
        ### create main application
        app = QApplication(sys.argv)  # main application
        gui = mGUI(app)  # main GUI class
        app.processEvents()
        app.exec_()
    ### if input file is specified run without GUI ###
    elif '-i' in args:
        print('Input file detected, reading arguments from input file')
        print('molSimplify is starting!')
        gui = False
        # run from commandline
        emsg = startgen(sys.argv, False, gui)
    ### grab from commandline arguments ###
    else:
        print('No input file detected, reading arguments from commandline')
        print('molSimplify is starting!')
        gui = False
        # create input file from commandline
        infile = parseCLI([_f for _f in args if _f])
        args = ['main.py','-i',infile]
        emsg = startgen(args,False,gui)
if __name__ == '__main__':
    main()
