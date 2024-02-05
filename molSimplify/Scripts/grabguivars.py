# @file grabguivars.py
#  Grabs arguments from GUI input
#
#  Written by Tim Ioannidisfor HJK Group
#
#  Dpt of Chemical Engineering, MIT

import re
import time
from molSimplify.Classes.globalvars import globalvars

# Check true or false
#  @param arg String to be checked
#  @return bool


def checkTrue(arg):
    if 'n' in arg.lower() or '0' in arg.lower() or 'f' in arg.lower():
        return False
    else:
        return True

# Collects ligand-related arguments
#  @param gui GUI flag
#  @return ligand-related arguments (ligs,ligoccs,lcats,kHs,MLb,lang,lname)


def getligands(gui):
    ligs, ligoccs, lcats, kHs, MLb, lang, lname = '', '', '', '', '', '', ''
    for ii in range(0, 7):
        if len(gui.lig[ii].currentText().replace(' ', '')) > 0:
            ligs += gui.lig[ii].currentText().replace(' ', '')+','
            ligoccs += str(gui.ligocc[ii].value())+','
            kHs += gui.ligH[ii].currentText()+','
            lcats += gui.ligconn[ii].text().replace(' ', '')+'/'
            MLb += gui.ligML[ii].text().replace(' ', '')+','
            lang += gui.ligan[ii].text().replace(' ', '')+','
            lname += gui.lignam[ii].text().replace(' ', '')+','
    # check for empty
    ligs = ligs.rsplit(',', 1)[0]
    lgtxt = ligoccs.replace(',', '')
    if len(lgtxt) > 0:
        ligoccs = ligoccs.rsplit(',', 1)[0]
    else:
        ligoccs = ''
    lgtxt = kHs.replace(',', '')
    if len(lgtxt) > 0:
        kHs = kHs.rsplit(',', 1)[0]
    else:
        kHs = ''
    lgtxt = lcats.replace('/', '')
    if len(lgtxt) > 0:
        lcats = lcats.rsplit('/', 1)[0]
    else:
        lcats = ''
    lgtxt = MLb.replace(',', '')
    if len(lgtxt) > 0:
        MLb = MLb.rsplit(',', 1)[0]
    else:
        MLb = ''
    lgtxt = lang.replace(',', '')
    if len(lgtxt) > 0:
        lang = lang.rsplit(',', 1)[0]
    else:
        lang = ''
    lgtxt = lname.replace(',', '')
    if len(lgtxt) > 0:
        lname = lname.rsplit(',', 1)[0]
    else:
        lname = ''
    return ligs, ligoccs, lcats, kHs, MLb, lang, lname


# Fills in GUI variables from ligand-related arguments
#  @param gui GUI flag
#  @param ligs,ligoccs,lcats,kHs,MLb,lang,lname ligand-related arguments
def setligands(gui, ligs, ligoccs, lcats, kHs, MLb, lang, lname):
    ligs = [_f for _f in re.split(',|\t|&', ligs) if _f]
    ligoccs = [_f for _f in re.split(',|\t|&', ligoccs) if _f]
    lcats = [_f for _f in re.split('/|\t|&', lcats) if _f]
    kHs = [_f for _f in re.split(',|\t|&', kHs) if _f]
    MLb = [_f for _f in re.split(',|\t|&', MLb) if _f]
    lang = [_f for _f in re.split(',|\t|&', lang) if _f]
    lname = [_f for _f in re.split(',|\t|&', lname) if _f]
    for ii in range(0, len(ligs)):
        gui.lig[ii].setCurrentText(ligs[ii])
        if len(ligoccs) > ii and ligoccs[ii] != '':
            gui.ligocc[ii].setValue(int(ligoccs[ii]))
        if len(lcats) > ii:
            gui.ligconn[ii].setText(lcats[ii])
        if len(kHs) > ii:
            if checkTrue(kHs[ii]):
                gui.ligH[ii].setCurrentText('yes')
            else:
                gui.ligH[ii].setCurrentText('no')
        if len(MLb) > ii:
            gui.ligML[ii].setText(MLb[ii])
        if len(lang) > 0:
            gui.ligan[ii].setText(lang[ii])
        if len(lname) > 0:
            gui.lignam[ii].setText(lname[ii])
        if ii < 7:
            gui.ligadd[ii].hide()
            gui.lig[ii+1].setDisabled(False)
            gui.ligocc[ii+1].setDisabled(False)
            gui.ligconn[ii+1].setDisabled(False)
            gui.ligH[ii+1].setDisabled(False)
            gui.ligML[ii+1].setDisabled(False)
            gui.ligan[ii+1].setDisabled(False)
            gui.lignam[ii+1].setDisabled(False)
            gui.ligadd[ii+1].show()
            gui.ligadd[ii+1].setDisabled(False)

# Writes options to input file with arbitrary name
#  @param args Namespace of arguments
#  @param fname Input file name


def writeinputc(args, fname):
    with open(fname, 'w') as f:
        f.write("# Input file generated from GUI options\n")
        for key, val in list(args.items()):
            if len(val) > 0:
                vals = val.splitlines()
                for v in vals:
                    f.write(key+' '+v+'\n')

# Writes postprocessing options to input file
#  @param args Namespace of arguments
#  @param fname Input file name


def writeinputp(args, fname):
    with open(fname, 'w') as f:
        f.write("# Input file generated from GUI options\n")
        for key, val in list(args.items()):
            f.write(key+' '+val+'\n')

# Writes options to input file with fixed name
#  @param args Namespace of arguments


def writeinputf(args):
    with open(args['-rundir']+'/geninput.inp', 'w') as f:
        f.write("# Input file generated from GUI options\n")
        with open(args['-rundir']+'/molSimp.log', 'a') as ff:
            dd = "# Input file generated from GUI options at " + \
                time.strftime('%m/%d/%Y %H:%M')+'\n'
            ff.write(dd)
            for key, val in list(args.items()):
                if len(val) > 0:
                    vals = val.splitlines()
                    for v in vals:
                        f.write(key+' '+v+'\n')
                        ff.write(key+' '+v+'\n')

# Grabs all GUI options
#  @param gui GUI flag
#  @return Namespace of arguments


def grabguivars(gui):
    # list with arguments
    args = dict()
    ### general structure generation options ###
    args['-core'] = gui.etcore.currentText()
    args['-ccatoms'] = gui.etccat.text()
    ligs, ligoccs, lcats, kHs, MLb, lang, lname = getligands(gui)
    noligs = len(ligs.split(','))-1
    args['-lig'] = ligs
    if len(ligoccs) > noligs:
        args['-ligocc'] = ligoccs
    if len(MLb) > noligs:
        args['-MLbonds'] = MLb
    if len(lang) > noligs:
        args['-pangles'] = lang
    if len(kHs) > noligs:
        args['-keepHs'] = kHs
    if len(lcats) > noligs:
        args['-smicat'] = lcats
    if len(lname) > noligs:
        args['-sminame'] = lname
    if gui.replig.getState():
        args['-replig'] = '1'
    if gui.ligfloc.getState():
        args['-ligloc'] = '1'
    if gui.ligfalign.getState():
        args['-ligalign'] = '1'
    if gui.chkgenall.getState():
        args['-genall'] = '1'
    args['-coord'] = gui.dcoord.currentText()
    args['-geometry'] = gui.dcoordg.currentText()
    if gui.randomchk.getState():
        args['-lignum'] = gui.etlignum.currentText()
        args['-liggrp'] = gui.etliggrp.currentText()
        args['-ligctg'] = gui.etligctg.currentText()
        args['-rgen'] = gui.etrgen.text()
        if gui.randkHs.getState():
            args['-rkHs'] = 'yes'
    args['-distort'] = str(gui.sdist.value())
    args['-oxstate'] = gui.doxs.currentText()
    args['-spin'] = gui.dspin.currentText()
    args['-rundir'] = gui.etrdir.text()
    rdir = args['-rundir']
    if rdir[1] == '/':
        args['-rundir'] = rdir[:-1]
    args['-suff'] = gui.etsuff.text()
    ### binding molecule options ###
    if gui.chkM.getState():
        args['-bind'] = gui.etbind.currentText()
        args['-bcharge'] = gui.etchbind.text()
        args['-nbind'] = gui.etnbind.text()
        if gui.chsep.getState():
            args['-bsep'] = 'yes'
        args['-nambsmi'] = gui.etbsmi.text()
        args['-maxd'] = gui.etplacemax.text()
        args['-mind'] = gui.etplacemin.text()
        args['-place'] = gui.dmolp.currentText()
        args['-bphi'] = gui.etplacephi.text()
        args['-btheta'] = gui.etplacetheta.text()
        args['-bref'] = gui.etmaskbind.text()
    ### force field optimization ###
    if gui.chkFF.getState():
        args['-ff'] = gui.dff.currentText()
        args['-ffoption'] = gui.dffba.currentText()
    ### Quantum Chemistry options ###
    if gui.chkI.getState():
        args['-qccode'] = gui.qcode.currentText()
        if args['-qccode'].lower() in 'terachem':
            args['-charge'] = gui.etqctch.text()
            if gui.chch.getState():
                args['-calccharge'] = 'yes'
            args['-spin'] = gui.etqctspin.text()
            print((gui.qctcalc.currentText()))
            args['-runtyp'] = gui.qctcalc.currentText()
            args['-method'] = gui.etqctmethod.text()
            args['-basis'] = gui.etqctbasis.text()
            args['-dispersion'] = gui.qctsel.currentText()
            args['-qoption'] = gui.qceditor.toPlainText()
        elif args['-qccode'].lower() in 'gamess':
            args['-charge'] = gui.etqcgch.text()
            if gui.chch.getState():
                args['-calccharge'] = 'yes'
            args['-spin'] = gui.etqcgspin.text()
            args['-runtyp'] = gui.qcgcalc.currentText()
            args['-method'] = gui.etqcgmethod.text()
            args['-gbasis'] = gui.etqcgbasis.text()
            args['-ngauss'] = gui.etqcngauss.text()
            args['-npfunc'] = gui.etqcnpfunc.text()
            args['-ndfunc'] = gui.etqcndfunc.text()
            args['-sysoption'] = gui.qcgedsys.toPlainText()
            args['-ctrloption'] = gui.qcgedctrl.toPlainText()
            args['-scfoption'] = gui.qcgedscf.toPlainText()
            args['-statoption'] = gui.qcgedstat.toPlainText()
        elif args['-qccode'].lower() in 'qchem':
            args['-charge'] = gui.etqcQch.text()
            if gui.chch.getState():
                args['-calccharge'] = 'yes'
            args['-spin'] = gui.etqcQspin.text()
            args['-runtyp'] = gui.qcQcalc.currentText()
            args['-basis'] = gui.etqcQbasis.text()
            args['-remoption'] = gui.qcQeditor.toPlainText()
            args['-exchange'] = gui.etqcQex.text()
            args['-correlation'] = gui.etqcQcor.text()
            if gui.chQun:
                args['-unrestricted'] = '1'
    ### jobscript options ###
    if gui.chkJ.getState():
        args['-jsched'] = gui.scheduler.currentText()
        args['-jname'] = gui.etjname.text()
        args['-memory'] = gui.etjmem.text()
        args['-wtime'] = gui.etjwallt.text()
        args['-queue'] = gui.etjqueue.text()
        args['-gpus'] = gui.etjgpus.text()
        args['-cpus'] = gui.etjcpus.text()
        args['-modules'] = gui.etjmod.text()
        args['-joption'] = gui.etjopt.toPlainText()
        args['-jcommand'] = gui.jcomm.toPlainText()
    # extra flag to control duplication prompt
    args['-rprompt'] = 'True'
    writeinputf(args)
    return args

# Grabs GUI options for terachem input files
#  @param gui GUI flag
#  @return Namespace of arguments of related arguments (charge, spin, runtyp, method, basis, dispersion, qoption, rprompt)


def grabguivarstc(gui):
    globs = globalvars()
    # list with arguments
    args = dict()
    args['-charge'] = gui.etqctch.text()
    args['-spin'] = gui.etqctspin.text()
    args['-runtyp'] = gui.qctcalc.currentText()
    args['-method'] = gui.etqctmethod.text()
    args['-basis'] = gui.etqctbasis.text()
    args['-dispersion'] = gui.qctsel.currentText()
    args['-qoption'] = gui.qceditor.toPlainText()
    # extra flag to control duplication prompt
    args['-rprompt'] = 'True'
    ### write input file ###
    writeinputc(args, globs.installdir+'/Data/.tcdefinput.inp')
    return args

# Grabs GUI options for GAMESS input files
#  @param gui GUI flag
#  @return Namespace of arguments of related arguments (charge, spin, runtyp, method, gbasis, ngauss, npfunc, ndfunc, sysoption, ctrloption, scfoption, statoption, rprompt)


def grabguivarsgam(gui):
    globs = globalvars()
    # list with arguments
    args = dict()
    args['-charge'] = gui.etqcgch.text()
    args['-spin'] = gui.etqcgspin.text()
    args['-runtyp'] = gui.qcgcalc.currentText()
    args['-method'] = gui.etqcgmethod.text()
    args['-gbasis'] = gui.etqcgbasis.text()
    args['-ngauss'] = gui.etqcngauss.text()
    args['-npfunc'] = gui.etqcnpfunc.text()
    args['-ndfunc'] = gui.etqcndfunc.text()
    args['-sysoption'] = gui.qcgedsys.toPlainText()
    args['-ctrloption'] = gui.qcgedctrl.toPlainText()
    args['-scfoption'] = gui.qcgedscf.toPlainText()
    args['-statoption'] = gui.qcgedstat.toPlainText()
    # extra flag to control duplication prompt
    args['-rprompt'] = 'True'
    ### write input file ###
    writeinputc(args, globs.installdir+'/Data/.gamdefinput.inp')
    return args

# Grabs GUI options for QChem input files
#  @param gui GUI flag
#  @return Namespace of arguments of related arguments (charge, spin, runtyp, basis, remoption, exchange, correlation, unrestricted, rprompt)


def grabguivarsqch(gui):
    globs = globalvars()
    # list with arguments
    args = dict()
    args['-charge'] = gui.etqcQch.text()
    args['-spin'] = gui.etqcQspin.text()
    args['-runtyp'] = gui.qcQcalc.currentText()
    args['-basis'] = gui.etqcQbasis.text()
    args['-remoption'] = gui.qcQeditor.toPlainText()
    args['-exchange'] = gui.etqcQex.text()
    args['-correlation'] = gui.etqcQcor.text()
    if gui.chQun:
        args['-unrestricted'] = '1'
    ### write input file ###
    # extra flag to control duplication prompt
    args['-rprompt'] = 'True'
    writeinputc(args, globs.installdir+'/Data/.qchdefinput.inp')
    return args

# Grabs GUI options for jobscripts
#  @param gui GUI flag
#  @return Namespace of arguments of related arguments (jname, memory, wtime, queue, gpus, cpus, modules, joption, jcommand, rprompt)


def grabguivarsjob(gui):
    globs = globalvars()
    # list with arguments
    args = dict()
    args['-jname'] = gui.etjname.text()
    args['-memory'] = gui.etjmem.text()
    args['-wtime'] = gui.etjwallt.text()
    args['-queue'] = gui.etjqueue.text()
    args['-gpus'] = gui.etjgpus.text()
    args['-cpus'] = gui.etjcpus.text()
    args['-modules'] = gui.etjmod.text()
    args['-joption'] = gui.etjopt.toPlainText()
    args['-jcommand'] = gui.jcomm.toPlainText()
    # set prompt option
    args['-rprompt'] = 'True'
    ### write input file ###
    writeinputc(args, globs.installdir+'/Data/.jobdefinput.inp')
    return args

# Grabs GUI options for DB search
#  @param gui GUI flag
#  @return Namespace of arguments of related arguments (dbsim, dbcatoms, dbresults, dboutputf, dbbase, dbsmarts, dbfinger, dbatoms, dbbonds, dbarbonds, dbsbonds, dbmw)


def grabdbguivars(gui):
    args = dict()
    ### database search options ###
    args['-dbsim'] = gui.etcDBsmi.text()
    args['-dbcatoms'] = gui.etcDBcatoms.text()
    args['-dbresults'] = gui.etcDBnres.text()
    args['-dboutputf'] = gui.etcDBoutf.text()+gui.cDBdent.currentText()
    args['-dbbase'] = gui.cDBsel.currentText()
    args['-dbsmarts'] = gui.etcDBsmarts.text()
    args['-dbfinger'] = gui.cDBsf.currentText()
    args['-dbatoms'] = gui.etcDBsatoms0.text()+'<'+gui.etcDBsatoms1.text()
    args['-dbbonds'] = gui.etcDBsbonds0.text()+'<'+gui.etcDBsbonds1.text()
    args['-dbarbonds'] = gui.etcDBsabonds0.text()+'<'+gui.etcDBsabonds1.text()
    args['-dbsbonds'] = gui.etcDBsbondss0.text()+'<'+gui.etcDBsbondss1.text()
    args['-dbmw'] = gui.etcDBmw0.text()+'<'+gui.etcDBmw1.text()
    if len(args['-dboutputf'].replace(' ', '')) == 4:
        args['-dboutputf'] = ''
    # check input
    if args['-dbatoms'] == '<':
        args['-dbatoms'] = ''
    if args['-dbbonds'] == '<':
        args['-dbbonds'] = ''
    if args['-dbarbonds'] == '<':
        args['-dbarbonds'] = ''
    if args['-dbsbonds'] == '<':
        args['-dbsbonds'] = ''
    if args['-dbmw'] == '<':
        args['-dbmw'] = ''
    rdir = gui.etrdir.text()
    if rdir[-1] == '/':
        rdir = rdir[:-1]
    args['-rundir'] = rdir
    # extra flag to control duplication prompt
    args['-rprompt'] = 'True'
    ### write input file ###
    writeinputc(args, rdir+'/dbinput.inp')
    return args

# Grabs GUI options for postprocessing
#  @param gui GUI flag


def grabguivarsP(gui):
    args = dict()
    ### post-processing options ###
    args['-postp'] = ''
    args['-postdir'] = gui.etpdir.text()
    args['-postqc'] = gui.pqcode.currentText()
    rdir = args['-postdir']
    if rdir[-1] == '/':
        rdir = rdir[:-1]
    if gui.psum.getState() == 1:
        args['-pres'] = ''
    if gui.pwfnav.getState() == 1:
        args['-pwfninfo'] = ''
    if gui.pch.getState() == 1:
        args['-pcharge'] = ''
    if gui.pcub.getState() == 1:
        args['-pgencubes'] = ''
    if gui.porbs.getState() == 1:
        args['-porbinfo'] = ''
    if gui.pdeloc.getState() == 1:
        args['-pdeloc'] = ''
    if gui.pnbo.getState() == 1:
        args['-pnbo'] = ''
    # extra flag to control duplication prompt
    args['-rprompt'] = 'True'
    ### write input file ###
    writeinputp(args, rdir+'/postproc.inp')

# Reads terachem input options from input file
#  @param gui GUI flag
#  @param fname input file name


def loadfrominputtc(gui, fname):
    with open(fname, 'r') as f:
        s = f.read()
    s = [_f for _f in s.splitlines() if _f]
    gui.qceditor.setText('')
    ### general structure generation options ###
    for ss in s:
        st = ss.split(None, 1)
        if '-charge' == st[0]:
            gui.etqctch.setText(st[-1])
        if '-spin' == st[0]:
            gui.etqctspin.setText(st[-1])
        if '-runtyp' == st[0]:
            gui.qctcalc.setCurrentText(st[-1])
        if '-method' == st[0]:
            gui.etqctmethod.setText(st[-1])
        if '-basis' == st[0]:
            gui.etqctbasis.setText(st[-1])
        if '-dispersion' == st[0]:
            gui.qctsel.setCurrentText(st[-1])
        if '-qoption' == st[0]:
            gui.qceditor.setText(gui.qceditor.toPlainText()+st[-1]+'\n')
        if '-charge' == st[0]:
            gui.etqctch.setText(st[-1])
        if '-runtyp' == st[0]:
            gui.qctcalc.setCurrentText(st[-1])

# Reads GAMESS input options from input file
#  @param gui GUI flag
#  @param fname input file name


def loadfrominputgam(gui, fname):
    with open(fname, 'r') as f:
        s = f.read()
    s = [_f for _f in s.splitlines() if _f]
    gui.qcgedsys.setText('')
    gui.qcgedctrl.setText('')
    gui.qcgedscf.setText('')
    gui.qcgedstat.setText('')
    ### general structure generation options ###
    for ss in s:
        st = ss.split(None, 1)
        ### Quantum Chemistry options ###
        if '-charge' == st[0]:
            gui.etqcgch.setText(st[-1])
        if '-spin' == st[0]:
            gui.etqcgspin.setText(st[-1])
        if '-runtyp' == st[0]:
            gui.qcgcalc.setCurrentText(st[-1])
        if '-method' == st[0]:
            gui.etqcgmethod.setText(st[-1])
        if '-gbasis' == st[0]:
            gui.etqcgbasis.setText(st[-1])
        if '-ngauss' == st[0]:
            gui.etqcngauss.setText(st[-1])
        if '-npfunc' == st[0]:
            gui.etqcnpfunc.setText(st[-1])
        if '-ndfunc' == st[0]:
            gui.etqcndfunc.setText(st[-1])
        if '-sysoption' == st[0]:
            gui.qcgedsys.setText(gui.qcgedsys.toPlainText()+st[-1]+'\n')
        if '-ctrloption' == st[0]:
            gui.qcgedctrl.setText(gui.qcgedctrl.toPlainText()+st[-1]+'\n')
        if '-scfoption' == st[0]:
            gui.qcgedscf.setText(gui.qcgedscf.toPlainText()+st[-1]+'\n')
        if '-statoption' == st[0]:
            gui.qcgedstat.setText(gui.qcgedstat.toPlainText()+st[-1]+'\n')

# Reads QChem input options from input file
#  @param gui GUI flag
#  @param fname input file name


def loadfrominputqch(gui, fname):
    with open(fname, 'r') as f:
        s = f.read()
    s = [_f for _f in s.splitlines() if _f]
    gui.qcQeditor.setText('')
    ### general structure generation options ###
    for ss in s:
        st = ss.split(None, 1)
        ### Quantum Chemistry options ###
        if '-basis' == st[0]:
            gui.etqcQbasis.setText(st[-1])
        if '-charge' == st[0]:
            gui.etqcQch.setText(st[-1])
        if '-spin' == st[0]:
            gui.etqcQspin.setText(st[-1])
        if '-runtyp' == st[0]:
            gui.qcQcalc.setCurrentText(st[-1])
        if '-remoption' == st[0]:
            gui.qcQeditor.setText(gui.qcQeditor.toPlainText()+st[-1]+'\n')
        if '-exchange' == st[0]:
            gui.etqcQex.setText(st[-1])
        if '-correlation' == st[0]:
            gui.etqcQcor.setText(st[-1])
        if '-unrestricted' == st[0]:
            gui.chQun.setChecked(True)

# Reads jobscript options from input file
#  @param gui GUI flag
#  @param fname input file name


def loadfrominputjob(gui, fname):
    with open(fname, 'r') as fin:
        s = fin.read()
    s = [_f for _f in s.splitlines() if _f]

    gui.etjopt.setText('')
    gui.jcomm.setText('')
    ### general structure generation options ###
    for ss in s:
        st = ss.split(None, 1)
        ### jobscript options ###
        if '-jname' == st[0]:
            gui.etjname.setText(st[-1])
        if '-memory' == st[0]:
            gui.etjmem.setText(st[-1])
        if '-wtime' == st[0]:
            gui.etjwallt.setText(st[-1])
        if '-queue' == st[0]:
            gui.etjqueue.setText(st[-1])
        if '-gpus' == st[0]:
            gui.etjgpus.setText(st[-1])
        if '-cpus' == st[0]:
            gui.etjcpus.setText(st[-1])
        if '-modules' == st[0]:
            gui.etjmod.setText(st[-1])
        if '-joption' == st[0]:
            gui.etjopt.setText(gui.etjopt.toPlainText()+st[-1]+'\n')
        if '-jcommand' == st[0]:
            gui.jcomm.setText(gui.jcomm.toPlainText()+st[-1]+'\n')

# Reads all input options from input file
#  @param gui GUI flag
#  @param fname input file name


def loadfrominputfile(gui, fname):
    with open(fname, 'r') as f:
        s = f.read()
    s = [_f for _f in s.splitlines() if _f]
    # re-initialize GUI
    # gui.initGUI(gui.app)
    db = False
    ligs, ligoccs, lcats, kHs, MLb, lang, lname = '', '', '', '', '', '', ''
    # reset edit boxes
    gui.qceditor.setText('')
    gui.qcgedsys.setText('')
    gui.qcgedctrl.setText('')
    gui.qcgedscf.setText('')
    gui.qcgedstat.setText('')
    gui.qcQeditor.setText('')
    gui.etjopt.setText('')
    gui.jcomm.setText('')
    ### general structure generation options ###
    for ss in s:
        st = ss.split(None, 1)
        if '-lig' not in st[0]:
            st[-1] = st[-1].split('#')[0]
        if '-core' == st[0]:
            gui.etcore.setCurrentText(st[-1])
        if '-lig' == st[0]:
            ligs = st[-1]
        if '-ligocc' == st[0]:
            ligoccs = st[-1]
        if '-replig' == st[0]:
            if checkTrue(st[-1]):
                gui.replig.setChecked(True)
        if '-ligloc' == st[0]:
            if checkTrue(st[-1]):
                gui.ligfloc.setChecked(True)
        if '-ligalign' == st[0]:
            if checkTrue(st[-1]):
                gui.ligfalign.setChecked(True)
        if '-genall' == st[0]:
            if checkTrue(st[-1]):
                gui.chkgenall.setChecked(True)
        if '-MLbonds' == st[0]:
            Mlb = st[-1]  # noqa F841 variable never used
        if '-distort' == st[0]:
            gui.sdist.setValue(int(st[-1]))
        if '-pangles' == st[0]:
            lang = st[-1]
        if '-ccatoms' == st[0]:
            gui.etccat.setText(st[-1])
        if '-coord' == st[0]:
            gui.dcoord.setCurrentText(st[-1])
        if '-geometry' == st[0]:
            gui.dcoordg.setCurrentText(st[-1])
        if '-lignum' == st[0]:
            gui.etlignum.setCurrentText(st[-1])
        if '-liggrp' == st[0]:
            gui.etliggrp.setCurrentText(st[-1])
        if '-ligctg' == st[0]:
            gui.etligctg.setCurrentText(st[-1])
        if '-rkHs' == st[0]:
            if checkTrue(st[-1]):
                gui.randkHs.setChecked(True)
        if '-rgen' == st[0]:
            gui.etrgen.setText(st[-1])
            gui.randomchk.setChecked(True)
        if '-keepHs' == st[0]:
            kHs = st[-1]
        if '-smicat' == st[0]:
            lcats = st[-1]
        if '-sminame' == st[0]:
            lname = st[-1]
        if '-oxstate' == st[0]:
            gui.doxs.setCurrentText(st[-1])
        if '-rundir' == st[0]:
            gui.etrdir.setText(st[-1])
        if '-suff' == st[0]:
            gui.etsuff.setText(st[-1])
        ### binding molecule options ###
        if '-bind' == st[0]:
            gui.chkM.setChecked(True)
            gui.enableemol()
            gui.etbind.setCurrentText(st[-1])
        if '-bsep' == st[0]:
            gui.chsep.setChecked(True)
        if '-bcharge' == st[0]:
            gui.etchbind.setText(st[-1])
        if '-nbind' == st[0]:
            gui.etnbind.setText(st[-1])
        if '-nambsmi' == st[0]:
            gui.etbsmi.setText(st[-1])
        if '-maxd' == st[0]:
            gui.etplacemax.setText(st[-1])
        if '-mind' == st[0]:
            gui.etplacemin.setText(st[-1])
        if '-place' == st[0]:
            gui.dmolp.setCurrentText(st[-1])
        if '-bphi' == st[0]:
            gui.etplacephi.setText(st[-1])
        if '-btheta' == st[0]:
            gui.etplacetheta.setText(st[-1])
        if '-bref' == st[0]:
            gui.etmaskbind.setText(st[-1])
        ### force field optimization ###
        ff = False  # noqa F841 variable never used
        if '-ff' == st[0]:
            gui.chkFF.setChecked(True)
            gui.enableffinput()
            gui.dff.setCurrentText(st[-1])
            ff = True  # noqa F841 variable never used
        if '-ffoption' == st[0]:
            sopt = st[-1].split('&')
            b, a = False, False
            for sopts in sopt:
                ssopt = sopts.replace(' ', '')
                if ssopt[0].lower() == 'b':
                    b = True
                elif ssopt[0].lower() == 'a':
                    a = True
            if b and not a:
                idx = 0
            elif not b and a:
                idx = 1
            elif b and a:
                idx = 2
            else:
                idx = 2
            gui.dffba.setCurrentIndex(idx)
        ### Quantum Chemistry options ###
        if '-qccode' == st[0]:
            gui.chkI.setChecked(True)
            gui.enableqeinput()
            gui.qcode.setCurrentText(st[-1])
        if '-calccharge' == st[0]:
            gui.chch.setChecked(True)
        if '-basis' == st[0]:
            gui.etqctbasis.setText(st[-1])
            gui.etqcQbasis.setText(st[-1])
        if '-dispersion' == st[0]:
            gui.qctsel.setCurrentText(st[-1])
        if '-qoption' == st[0]:
            gui.qceditor.setText(gui.qceditor.toPlainText()+st[-1]+'\n')
        if '-charge' == st[0]:
            gui.etqcgch.setText(st[-1])
            gui.etqctch.setText(st[-1])
            gui.etqcQch.setText(st[-1])
        if '-spin' == st[0]:
            gui.etqcgspin.setText(st[-1])
            gui.etqctspin.setText(st[-1])
            gui.etqcQspin.setText(st[-1])
        if '-runtyp' == st[0]:
            gui.qcgcalc.setCurrentText(st[-1])
            gui.qctcalc.setCurrentText(st[-1])
            gui.qcQcalc.setCurrentText(st[-1])
        if '-method' == st[0]:
            gui.etqctmethod.setText(st[-1])
            gui.etqcgmethod.setText(st[-1])
        if '-gbasis' == st[0]:
            gui.etqcgbasis.setText(st[-1])
        if '-ngauss' == st[0]:
            gui.etqcngauss.setText(st[-1])
        if '-npfunc' == st[0]:
            gui.etqcnpfunc.setText(st[-1])
        if '-ndfunc' == st[0]:
            gui.etqcndfunc.setText(st[-1])
        if '-sysoption' == st[0]:
            gui.qcgedsys.setText(gui.qcgedsys.toPlainText()+st[-1]+'\n')
        if '-ctrloption' == st[0]:
            gui.qcgedctrl.setText(gui.qcgedctrl.toPlainText()+st[-1]+'\n')
        if '-scfoption' == st[0]:
            gui.qcgedscf.setText(gui.qcgedscf.toPlainText()+st[-1]+'\n')
        if '-statoption' == st[0]:
            gui.qcgedstat.setText(gui.qcgedstat.toPlainText()+st[-1]+'\n')
        if '-remoption' == st[0]:
            gui.qcQeditor.setText(gui.qcQeditor.toPlainText()+st[-1]+'\n')
        if '-exchange' == st[0]:
            gui.etqcQex.setText(st[-1])
        if '-correlation' == st[0]:
            gui.etqcQcor.setText(st[-1])
        if '-unrestricted' == st[0]:
            gui.chQun.setChecked(True)
        ### jobscript options ###
        if '-jsched' == st[0]:
            gui.chkJ.setChecked(True)
            gui.enableqeinput()
            gui.scheduler.setCurrentText(st[-1])
        if '-jname' == st[0]:
            gui.etjname.setText(st[-1])
        if '-memory' == st[0]:
            gui.etjmem.setText(st[-1])
        if '-wtime' == st[0]:
            gui.etjwallt.setText(st[-1])
        if '-queue' == st[0]:
            gui.etjqueue.setText(st[-1])
        if '-gpus' == st[0]:
            gui.etjgpus.setText(st[-1])
        if '-cpus' == st[0]:
            gui.etjcpus.setText(st[-1])
        if '-modules' == st[0]:
            gui.etjmod.setText(st[-1])
        if '-joption' == st[0]:
            gui.etjopt.setText(gui.etjopt.toPlainText()+st[-1]+'\n')
        if '-jcommand' == st[0]:
            gui.jcomm.setText(gui.jcomm.toPlainText()+st[-1]+'\n')
        ### database search options ###
        if '-dbsim' == st[0]:
            gui.etcDBsmi.setText(st[-1])
            db = True
        if '-dbcatoms' == st[0]:
            gui.etcDBcatoms.setText(st[-1])
        if '-dbresults' == st[0]:
            gui.etcDBnres.setText(st[-1])
            db = True
        if '-dboutputf' == st[0]:
            ssp = st[-1].split('.')
            gui.etcDBoutf.setText(ssp[0])
            gui.cDBdent.setCurrentText('.'+ssp[-1])
        if '-dbbase' == st[0]:
            gui.cDBsel.setCurrentText(st[-1])
            db = True
        if '-dbsmarts' == st[0]:
            gui.etcDBsmarts.setText(st[-1])
        if '-dbfinger' == st[0]:
            gui.cDBsf.setCurrentText(st[-1])
        if '-dbatoms' == st[0]:
            ssp = st[-1].split('<')
            gui.etcDBsatoms0.setText(ssp[0])
            gui.etcDBsatoms1.setText(ssp[-1])
        if '-dbbonds' == st[0]:
            ssp = st[-1].split('<')
            gui.etcDBsbonds0.setText(ssp[0])
            gui.etcDBsbonds1.setText(ssp[-1])
        if '-dbarbonds' == st[0]:
            ssp = st[-1].split('<')
            gui.etcDBsabonds0.setText(ssp[0])
            gui.etcDBsabonds1.setText(ssp[-1])
        if '-dbsbonds' == st[0]:
            ssp = st[-1].split('<')
            gui.etcDBsbondss0.setText(ssp[0])
            gui.etcDBsbondss1.setText(ssp[-1])
        if '-dbmw' == st[0]:
            ssp = st[-1].split('<')
            gui.etcDBmw0.setText(ssp[0])
            gui.etcDBmw1.setText(ssp[-1])
    setligands(gui, ligs, ligoccs, lcats, kHs, MLb, lang, lname)
    if db:
        gui.searchDBW()
