# @file postproc.py
#  Main postprocessing driver
#
#  Written by Tim Ioannidis for HJK Group
#
#  Dpt of Chemical Engineering, MIT

import glob
import os
import shutil
import time

from molSimplify.Scripts.postmold import (moldpost)
from molSimplify.Scripts.postmwfn import (deloc,
                                          getcharges,
                                          getcubes,
                                          getwfnprops,
                                          globalvars,
                                          mybash)
from molSimplify.Scripts.postparse import (gampost,
                                           nbopost,
                                           terapost)


# Check if multiwfn exists
#  @param mdir Multiwfn directory
#  @return bool


def checkmultiwfn(mdir):
    if not glob.glob(mdir):
        return False
    else:
        return True

# Main postprocessing driver
#  @param rundir Runs directory
#  @param args Namespace of arguments
#  @param globs Global variables


def postproc(rundir, args, globs):
    globs = globalvars()
    if args.gui:
        from molSimplify.Classes.mWidgets import mQDialogInf
        choice = mQDialogInf(
            'Post processing', 'Parsing the results will take a while..Please be patient. Start?')
        choice.setParent(args.gui.pWindow)
    # locate output files
    pdir = args.postdir if args.postdir else globs.rundir
    cmd = "find '"+pdir+"' -name *out"
    t = mybash(cmd)
    resf = t.splitlines()
    logfile = pdir+"/post.log"
    if not os.path.isdir(pdir):
        print(('\nSpecified directory '+pdir+' does not exist..\n\n'))
        if args.gui:
            args.gui.iWtxt.setText(
                '\nSpecified directory '+pdir+' does not exist.\n\n'+args.gui.iWtxt.toPlainText())
        return
    with open(logfile, 'a') as flog:
        flog.write('\n\n\n##### Date: ' +
                   time.strftime('%m/%d/%Y %H:%M')+'#####\n\n')
        # run summary report
        if args.pres:
            print('\nGetting runs summary..\n\n')
            flog.write('\nGetting runs summary..\n\n')
            if args.gui:
                args.gui.iWtxt.setText(
                    '\nGetting runs summary..\n\n'+args.gui.iWtxt.toPlainText())
            terapost(resf, pdir, args.gui, flog)
            gampost(resf, pdir, args.gui, flog)
        # run nbo analysis
        if args.pnbo:
            print('\nGetting NBO summary..\n\n')
            flog.write('\nGetting NBO summary..\n\n')
            if args.gui:
                args.gui.iWtxt.setText(
                    '\nGetting NBO summary..\n\n'+args.gui.iWtxt.toPlainText())
            nbopost(resf, pdir, args.gui, flog)
        # locate molden files
        cmd = "find "+"'"+pdir+"'"+" -name *molden"
        t = mybash(cmd)
        molf = t.splitlines()
        # parse molecular orbitals
        if args.porbinfo:
            print('\nGetting MO information..\n\n')
            flog.write('\nGetting MO information..\n\n')
            if args.gui:
                args.gui.iWtxt.setText(
                    '\nGetting MO information..\n\n'+args.gui.iWtxt.toPlainText())
            if not os.path.isdir(pdir+'/MO_files'):
                os.mkdir(pdir+'/MO_files')
            moldpost(molf, pdir, args.gui, flog)
        # calculate delocalization indices
        if args.pdeloc:
            print('\nCalculating delocalization indices..\n\n')
            flog.write('\nCalculating delocalization indices..\n\n')
            if args.gui:
                args.gui.iWtxt.setText(
                    '\nCalculating delocalization indices..\n\n'+args.gui.iWtxt.toPlainText())
            if not os.path.isdir(pdir+'/Deloc_files'):
                os.mkdir(pdir+'/Deloc_files')
            deloc(molf, pdir, args.gui, flog)
        # calculate charges
        if args.pcharge:
            print('\nCalculating charges..\n\n')
            flog.write('\nCalculating charges..\n\n')
            if args.gui:
                args.gui.iWtxt.setText(
                    '\nCalculating charges..\n\n'+args.gui.iWtxt.toPlainText())
            if not os.path.isdir(pdir+'/Charge_files'):
                os.mkdir(pdir+'/Charge_files')
            getcharges(molf, pdir, args.gui, flog)
        # parse wavefunction
        if args.pwfninfo:
            print('\nCalculating wavefunction properties..\n\n')
            flog.write('\nCalculating wavefunction properties..\n\n')
            if args.gui:
                args.gui.iWtxt.setText(
                    '\nCalculating wavefunction properties..\n\n'+args.gui.iWtxt.toPlainText())
            if not os.path.isdir(pdir+'/Wfn_files'):
                os.mkdir(pdir+'/Wfn_files')
            if not os.path.isdir(pdir+'/Cube_files'):
                os.mkdir(pdir+'/Cube_files')
            getcubes(molf, pdir, args.gui, flog)
            getwfnprops(molf, pdir, args.gui, flog)
            if not args.pgencubes and os.path.isdir(pdir+'/Cube_files'):
                shutil.rmtree(pdir+'/Cube_files')
        # generate cube files
        if args.pgencubes:
            print('\nGenerating cube files..\n\n')
            flog.write('\nGenerating cube files..\n\n')
            if args.gui:
                args.gui.iWtxt.setText(
                    '\nGenerating cube files..\n\n'+args.gui.iWtxt.toPlainText())
            if not os.path.isdir(pdir+'/Cube_files'):
                os.mkdir(pdir+'/Cube_files')
            getcubes(molf, pdir, args.gui, flog)
