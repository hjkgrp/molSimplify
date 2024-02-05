# @file mGUI.py
#  Contains main GUI class and routines.
#
#  Written by Tim Ioannidis for HJK Group
#
#  Modified by JP Janet
#
#  Dpt of Chemical Engineering, MIT

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import (QWidget, QGridLayout, QStackedLayout, QAction,
                             QFileDialog, QMessageBox, QPushButton,
                             QDesktopWidget)
from PyQt5.QtGui import QPalette, QFont, QIcon
from molSimplify.Classes.mWidgets import (mQMainWindow, mQPixmap, mQLabel,
                                          mQLineEdit, mQLineEditL, mQTextEdit,
                                          mQCheckBox, mQComboBox, mQSpinBox,
                                          mQPushButton, mQSlider, mQDialogWarn,
                                          mSvgWidget, relresize, center)
from molSimplify.Classes.globalvars import globalvars, mybash
from molSimplify.Scripts.generator import startgen
from molSimplify.Scripts.grabguivars import (grabguivars, grabguivarsqch,
                                             grabguivarstc, grabguivarsgam,
                                             grabguivarsjob, grabguivarsP,
                                             grabdbguivars, loadfrominputfile,
                                             loadfrominputtc, loadfrominputgam,
                                             loadfrominputqch, writeinputc,
                                             loadfrominputjob)
from molSimplify.Scripts.io import (readdict, getgeoms, getcores, getmcores,
                                    getbcores, getlicores, getslicores,
                                    getligs, getbinds, getligroups,
                                    lig_load, copy_to_custom_path)
from molSimplify.Scripts.addtodb import (addtoldb, addtocdb, addtobdb,
                                         removefromDB)
import sys
import os
import shutil
import unicodedata
import glob
import tempfile
import re
from importlib_resources import files as resource_files
import xml.etree.ElementTree as ET

try:
    from openbabel import openbabel  # version 3 style import
except ImportError:
    import openbabel  # fallback to version 2

# Main GUI class


class mGUI():
    getgeoms()

    def __init__(self, app):
        # build gui
        self.app = app
        self.initGUI(app)
        cwd = os.getcwd()
        globs = globalvars()  # global variables
        globs.rundir = os.path.join(cwd, 'Runs')
        if not os.path.exists(globs.rundir):
            print((globs.rundir))
            os.makedirs(globs.rundir)
    # builds the gui

    def initGUI(self, app):
        '''
        ######################
        ### build main GUI ###
        ######################
        '''
        # ## check for configuration file ###
        cwd = os.getcwd()
        globs = globalvars()  # global variables
        globs.rundir = os.path.join(cwd, 'Runs')
        if not os.path.exists(globs.rundir):
            os.makedirs(globs.rundir)
        # overX = True if 'localhost' in os.environ['DISPLAY'].lower() else False # detect running over X
#        configfile = False if not glob.glob(homedir+'/.molSimplify') else True
#        if not configfile:
#            self.wwindow = mQMainWindow()
#            self.wwindow.resize(0.5,0.5)
#            QMessageBox.information(self.wwindow, 'Setup', 'It looks like the '
#                                    'configuration file "~/.molSimplify" does '
#                                    'not exist!Please follow the next steps to '
#                                    'configure the file.')
#            QMessageBox.information(self.wwindow, 'Installation directory',
#                                    "Please select the top installation "
#                                    "directory for the program.")
#            instdir = QFileDialog.getExistingDirectory(self.wwindow, "Select Directory")
#            f = open(homedir+'/.molSimplify','w')
#            if len(instdir) > 1:
#                f.write("INSTALLDIR="+instdir+'\n')
        # ## end set-up configuration file ###
        # main window widget
        self.wmwindow = mQMainWindow()
        self.wmwindow.show()
        self.wmain = QWidget()
        self.wmwindow.setWindowTitle("molSimplify")
        self.wmwindow.setMinimumSize(1000, 700)
        # set background color
        p = QPalette()
        p.setColor(QPalette.Background, Qt.white)
        self.wmwindow.setPalette(p)
        # ## main grid layout ###
        self.grid = QGridLayout()
        # ## stacked layouts ###
        self.sgrid = QStackedLayout()
        self.sgrid.setStackingMode(1)
        self.sgrid.addWidget(self.wmain)
        # set manually size of rows/columns
        self.grid.setRowMinimumHeight(0, 15)
        self.grid.setRowMinimumHeight(2, 90)
        self.grid.setRowMinimumHeight(3, 15)
        self.grid.setRowMinimumHeight(4, 50)
        ####################################
        # ## create menubar and callbacks ###
        menubar = self.wmwindow.menuBar()
        menu0 = menubar.addMenu('&File')
        menu1 = menubar.addMenu('&Load')
        menu2 = menubar.addMenu('&Help')
        exitAction = QAction('&Exit', self.wmwindow)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.qexit)
        menu0.addAction(exitAction)
        saveAction = QAction('&Save As..', self.wmwindow)
        saveAction.setShortcut('Ctrl+S')
        saveAction.setStatusTip('Save current input settings')
        saveAction.triggered.connect(self.qsaveinput)
        menu0.addAction(saveAction)
        loadAction = QAction('&Load', self.wmwindow)
        loadAction.setShortcut('Ctrl+O')
        loadAction.setStatusTip('Load input file')
        loadAction.triggered.connect(self.qloadinput)
        menu1.addAction(loadAction)
        helpAction = QAction('&About', self.wmwindow)
        helpAction.setShortcut('Ctrl+H')
        helpAction.setStatusTip('Show program information')
        helpAction.triggered.connect(self.qshowhelp)
        menu2.addAction(helpAction)
        # ## place title top ###
        f = resource_files("molSimplify").joinpath("icons/logo.png")
        clogo = mQPixmap(f)
        self.grid.addWidget(clogo, 2, 8, 1, 10)
        self.txtdev = mQLabel('Developed by Kulik group @ MIT', '', 'c', 16)
        self.grid.addWidget(self.txtdev, 19, 8, 2, 10)
        # ##################################################
        # ##################################################
        # ######### STRUCTURE GENERATION INPUTS ############
        # ##################################################
        # ##################################################
        # ## title ###
        self.txtgpar = mQLabel('Structure specification', '', 'c', 20)
        self.grid.addWidget(self.txtgpar, 4, 0, 2, 14)
        # ## core structure specification ###
        ctip = 'Core of structure'
        self.rtcore = mQLabel('Core:', ctip, 'C', 12)
        f = QFont("Helvetica", 14, 75)
        self.rtcore.setFont(f)
        self.grid.addWidget(self.rtcore, 7, 2, 1, 1)
        cores = list(sorted(getmcores()))
        self.etcore = mQLineEditL('', ctip, 'l', 12, cores)
        self.grid.addWidget(self.etcore, 7, 3, 1, 3)
        # ## Connection atoms for core ###
        ctip = 'Specify connection atoms for core if using SMILES or custom cores (default: 1)'
        self.rtccat = mQLabel('Core connections:', ctip, 'C', 12)
        self.rtccat.setWordWrap(True)
        self.grid.addWidget(self.rtccat, 7, 6, 1, 2)
        self.etccat = mQLineEdit('', ctip, 'l', 12)
        self.grid.addWidget(self.etccat, 7, 8, 1, 2)
        # ## replace option ###
        ctip = 'Replace ligand at specified connection point'
        self.replig = mQCheckBox('replace', ctip, 12)
        self.grid.addWidget(self.replig, 7, 10, 1, 4)
        # coordination
        ctip = 'Number of ligands connected to the metal.'
        self.rcoord = mQLabel('Coordination:', ctip, 'Cr', 12)
        self.grid.addWidget(self.rcoord, 8, 2, 1, 2)
        coords, geomnames, geomshorts, geomgroups = getgeoms()
        qcav = sorted(list(set(coords)))
        self.dcoord = mQComboBox(qcav, ctip, 12)
        self.dcoord.setCurrentIndex(5)
        self.dcoord.currentIndexChanged.connect(self.matchgeomcoord)
        self.grid.addWidget(self.dcoord, 8, 4, 1, 1)
        # geometry of coordination
        self.dcoordg = mQComboBox('', '', 12)
        self.dcoordg.setCurrentIndex(0)
        self.matchgeomcoord()
        self.grid.addWidget(self.dcoordg, 8, 5, 1, 2)
        # add new coordination
        ctip = 'Add geometry'
        self.butaddg = mQPushButton('Add\ngeometry', ctip, 12)
        self.butaddg.clicked.connect(self.addgeom)
        self.grid.addWidget(self.butaddg, 8, 7, 1, 2)
        # add new coordination
        ctip = 'View geometry with atom labels'
        self.butvg = mQPushButton('View\ngeometry', ctip, 12)
        self.butvg.clicked.connect(self.viewgeom)
        self.grid.addWidget(self.butvg, 8, 9, 1, 2)
        # ############################################
        # ################# LIGANDS ##################
        # ## ligands tables ###
        ctip0 = 'Ligand(s) to be used'
        self.rtligh = mQLabel('Ligand', ctip0, 'c', 12)  # ligand header
        f = QFont("Helvetica", 12, 75)
        self.rtligh.setFont(f)
        ctip1 = 'Occurrence of corresponding ligand(s)'
        self.rtligocch = mQLabel(
            'Frequency', ctip1, 'c', 12)  # occurrence header
        ctip2 = 'Connection atom(s) of ligands (default: 1).'
        # connection atom header
        self.rtsmicath = mQLabel('Connections', ctip2, 'c', 12)
        # keep Hs header
        ctip3 = 'Do not remove hydrogens while connecting ligand to core. default True'
        self.keepHh = mQLabel('keep\nHs', ctip3, 'c', 12)  # occurrence header
        ctip4 = 'Custom bond length for M-L in Angstrom'
        # custom metal ligand bond length header
        self.MLbondsh = mQLabel('M-L\nbond', ctip4, 'c', 12)
        ctip5 = 'Custom angles for connection points (polar theta, azimuthal phi) in degrees separated with /. e.g 10/20'
        # custom angles for distortion header
        self.canglesh = mQLabel('Angle', ctip5, 'c', 12)
        ctip6 = 'Name of ligand'
        # name of ligand header
        self.nameligh = mQLabel('Name', ctip6, 'c', 12)
        ctip7 = 'Force ligand order and disable smart reordering'
        self.ligfloc = mQCheckBox('Force location', ctip7, 12)
        ctip8 = 'Ligand smart alignment. Aligns first the bulky ligands.'
        self.ligfalign = mQCheckBox('Smart alignment', ctip8, 12)
        self.ligfalign.setChecked(True)
        # add to layout
        self.grid.setColumnMinimumWidth(0, 150)
        self.grid.addWidget(self.rtligh, 9, 0, 1, 2)
        self.grid.addWidget(self.rtligocch, 9, 2, 1, 2)
        self.grid.addWidget(self.rtsmicath, 9, 4, 1, 2)
        self.grid.addWidget(self.keepHh, 9, 6, 1, 1)
        self.grid.addWidget(self.MLbondsh, 9, 7, 1, 2)
        self.grid.addWidget(self.canglesh, 9, 9, 1, 1)
        self.grid.addWidget(self.nameligh, 9, 10, 1, 1)
        self.grid.addWidget(self.ligfalign, 13, 15, 1, 2)
        self.grid.addWidget(self.ligfloc, 13, 17, 1, 2)
        # ligands #
        licores = list(sorted(getlicores()))
        self.lig = []
        self.ligocc = []
        self.ligconn = []
        self.ligH = []
        self.ligML = []
        self.ligan = []
        self.lignam = []
        self.ligadd = []
        for ii in range(0, 8):
            # ligands #
            self.lig.append(mQLineEditL('', ctip0, 'l', 12, licores))
            self.grid.addWidget(self.lig[ii], 10+ii, 0, 1, 2)  # add to layout
            # occurrences #
            self.ligocc.append(mQSpinBox(ctip1))
            self.grid.addWidget(self.ligocc[ii], 10+ii, 2, 1, 2)
            # connections #
            self.ligconn.append(mQLineEdit('', ctip2, 'l', 12))
            self.grid.addWidget(self.ligconn[ii], 10+ii, 4, 1, 2)
            # keep Hydrogens #
            self.ligH.append(mQComboBox(['yes', 'no'], ctip3, 12))
            self.grid.addWidget(self.ligH[ii], 10+ii, 6, 1, 1)
            # ML bond lengths #
            self.ligML.append(mQLineEdit('', ctip4, 'l', 12))
            self.grid.addWidget(self.ligML[ii], 10+ii, 7, 1, 2)
            # custom angles #
            self.ligan.append(mQLineEdit('', ctip5, 'l', 12))
            self.grid.addWidget(self.ligan[ii], 10+ii, 9, 1, 1)
            # ligand names #
            self.lignam.append(mQLineEdit('', ctip6, 'l', 12))
            self.grid.addWidget(self.lignam[ii], 10+ii, 10, 1, 1)
            # ligand buttons #
            if ii < 7:
                ctip = 'Add new ligand.'
                self.ligadd.append(mQPushButton('+', ctip, 12))
                self.grid.addWidget(self.ligadd[ii], 10+ii, 13, 1, 1)
                self.ligadd[ii].setDisabled(True)
                self.ligadd[ii].hide()
            if ii > 0:
                self.lig[ii].setDisabled(True)  # disable
                self.ligocc[ii].setDisabled(True)
                self.ligconn[ii].setDisabled(True)
                self.ligH[ii].setDisabled(True)
                self.ligML[ii].setDisabled(True)
                self.ligan[ii].setDisabled(True)
                self.lignam[ii].setDisabled(True)
            self.ligadd[0].setDisabled(False)
            self.ligadd[0].show()
        # callbacks
        self.ligadd[0].clicked.connect(self.addlig0)  # callback
        self.ligadd[1].clicked.connect(self.addlig1)  # callback
        self.ligadd[2].clicked.connect(self.addlig2)  # callback
        self.ligadd[3].clicked.connect(self.addlig3)  # callback
        self.ligadd[4].clicked.connect(self.addlig4)  # callback
        self.ligadd[5].clicked.connect(self.addlig5)  # callback
        self.ligadd[6].clicked.connect(self.addlig6)  # callback
        #############################################
        # List molecules  #
        ctip = 'Generate 2D ligand representation.'
        self.butList = mQPushButton('List molecules', ctip, 12)
        self.butList.clicked.connect(self.listmols)
        self.grid.addWidget(self.butList, 18, 3, 1, 2)
        # Draw ligand button #
        ctip = 'Generate 2D ligand representation.'
        self.butDrl = mQPushButton('Draw ligands', ctip, 12)
        self.butDrl.clicked.connect(self.drawligs_svg)
        self.grid.addWidget(self.butDrl, 18, 1, 1, 2)
        # Local database button #
        ctip = 'Add core/ligand/binding species to local database.'
        self.butADB = mQPushButton('Add to local DB', ctip, 12)
        self.butADB.clicked.connect(self.enableDB)
        self.grid.addWidget(self.butADB, 18, 5, 1, 3)
        # #################################################
        # #################################################
        # ########## GENERAL PARAMETERS INPUTS ############
        # #################################################
        # #################################################
        self.txtgp = mQLabel('General parameters', '', 'c', 20)
        self.grid.addWidget(self.txtgp, 4, 13, 2, 8)
        # random generation
        ctip = 'Enable random generation.'
        self.randomchk = mQCheckBox('Random generation', ctip, 12)
        self.randomchk.stateChanged.connect(self.enablerandom)
        self.grid.addWidget(self.randomchk, 7, 15, 1, 2)
        # charge calculation
        ctip = 'Calculate charge based on ox state and ligands'
        self.chch = mQCheckBox('Calculate\ncharge', ctip, 12)
        self.chch.setDisabled(True)
        self.grid.addWidget(self.chch, 7, 17, 1, 1)
        # keep Hs
        ctip = 'Keep hydrogens for random generation.'
        self.randkHs = mQCheckBox('Keep\nHs', ctip, 12)
        self.grid.addWidget(self.randkHs, 7, 18, 1, 1)
        self.randkHs.setDisabled(True)
        # number of random generated structures
        ctip = 'Number of structures to be randomly generated.'
        self.rtrgen = mQLabel('Structures:', ctip, 'Cr', 12)
        self.etrgen = mQLineEdit('', ctip, 'l', 12)
        self.grid.addWidget(self.rtrgen, 8, 15, 1, 1)
        self.grid.addWidget(self.etrgen, 8, 16, 1, 1)
        self.rtrgen.setDisabled(True)
        self.etrgen.setDisabled(True)
        # number of different ligands to use
        ctip = 'For random generation: total number of different ligands including specified ones.'
        self.rtlignum = mQLabel('Different\nligands:', ctip, 'Cr', 12)
        qcav = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self.etlignum = mQComboBox(qcav, ctip, 12)
        self.grid.addWidget(self.rtlignum, 8, 17, 1, 1)
        self.grid.addWidget(self.etlignum, 8, 18, 1, 1)
        self.rtlignum.setDisabled(True)
        self.etlignum.setDisabled(True)
        # group of different ligands
        ctip = 'For random generation: select random ligands from group.'
        self.rtliggrp = mQLabel('Ligand\ngroup:', ctip, 'Cr', 12)
        cwd = os.getcwd()
        globs = globalvars()
        globs.rundir = os.path.join(cwd, 'Runs')
        if globs.custom_path:
            f = globs.custom_path + "/Ligands/ligands.dict"
        else:
            f = str(resource_files("molSimplify").joinpath("Ligands/ligands.dict"))
        qcav0 = getligroups(readdict(f))
        qcav = [_f for _f in qcav0.split(' ') if _f]
        self.etliggrp = mQComboBox(qcav, ctip, 12)
        self.grid.addWidget(self.rtliggrp, 9, 15, 1, 1)
        self.grid.addWidget(self.etliggrp, 9, 16, 1, 1)
        self.rtliggrp.setDisabled(True)
        self.etliggrp.setDisabled(True)
        # ligand category
        ctip = 'For random generation: select random ligands from category.\nOptions are all'
        ctip += '"build" for building complexes, "functionalize" for functionalizing'
        self.rtligctg = mQLabel('Ligand\ncategory:', ctip, 'Cr', 12)
        qcav = ['all', 'build', 'functionalize']
        self.etligctg = mQComboBox(qcav, ctip, 12)
        self.grid.addWidget(self.rtligctg, 9, 17, 1, 1)
        self.grid.addWidget(self.etligctg, 9, 18, 1, 1)
        self.rtligctg.setDisabled(True)
        self.etligctg.setDisabled(True)
        # oxidation state
        ctip = 'Metal Oxidation state'
        self.roxstate = mQLabel('Ox State:', ctip, 'Cr', 12)
        qcav = ['0', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']
        self.doxs = mQComboBox(qcav, ctip, 12)
        self.doxs.setCurrentIndex(0)
        self.grid.addWidget(self.roxstate, 10, 15, 1, 1)
        self.grid.addWidget(self.doxs, 10, 16, 1, 1)
        # spin state
        ctip = 'System spin multiplicity'
        self.rspstate = mQLabel('Spin:', ctip, 'Cr', 12)
        qcav = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self.dspin = mQComboBox(qcav, ctip, 12)
        self.dspin.setCurrentIndex(0)
        self.grid.addWidget(self.rspstate, 10, 17, 1, 1)
        self.grid.addWidget(self.dspin, 10, 18, 1, 1)
        # force field optimization
        ctip = 'Perform Force Field optimization'
        self.chkFF = mQCheckBox('FF optimize', ctip, 12)
        self.chkFF.stateChanged.connect(self.enableffinput)
        self.grid.addWidget(self.chkFF, 11, 15, 1, 1)
        # generate all
        ctip = 'Generate structure with and without optimization.'
        self.chkgenall = mQCheckBox('Generate\nall', ctip, 12)
        self.chkgenall.stateChanged.connect(self.disableffinput)
        self.grid.addWidget(self.chkgenall, 12, 15, 1, 1)
        # perform optimization
        ctip = 'Select Force Field'
        qcav = ['MMFF94', 'UFF', 'gchemical', 'GAFF']
        self.dff = mQComboBox(qcav, ctip, 12)
        self.dff.setCurrentIndex(0)
        self.dff.setDisabled(True)
        self.grid.addWidget(self.dff, 11, 16, 1, 1)
        # optimize before or after
        ctip = 'Optimize before or after building the structure'
        qcav = ['Before', 'After', 'Before & After']
        self.dffba = mQComboBox(qcav, ctip, 12)
        self.dffba.setDisabled(True)
        self.dffba.setCurrentIndex(2)
        self.grid.addWidget(self.dffba, 11, 17, 1, 2)
        # create distortion slider
        ctip = 'Percent random distortion from default coordination geometry.'
        self.distper = mQLabel('Distort:0%', ctip, 'Cr', 12)
        self.sdist = mQSlider(ctip)
        self.sdist.valueChanged.connect(self.sliderChanged)
        self.grid.addWidget(self.distper, 12, 17, 1, 1)
        self.grid.addWidget(self.sdist, 12, 18, 1, 1)
        # jobs dir #
        ctip = 'Top directory for job folders.'
        self.rtrdir = mQLabel('Jobs dir:', ctip, 'Cr', 12)
        self.grid.addWidget(self.rtrdir, 14, 15, 1, 1)
        self.etrdir = mQLineEdit(globs.rundir, ctip, 'l', 12)
        self.grid.addWidget(self.etrdir, 14, 16, 1, 2)
        # button for browsing rundir
        ctip = 'Browse running directory.'
        self.butpbrdir = mQPushButton('Browse..', ctip, 12)
        self.butpbrdir.clicked.connect(self.dirload)
        self.grid.addWidget(self.butpbrdir, 14, 18, 1, 1)
        # suffix
        ctip = 'Suffix for job directories.'
        self.rtsuff = mQLabel('Suffix:', ctip, 'Cr', 12)
        self.etsuff = mQLineEdit('', ctip, 'l', 12)
        self.grid.addWidget(self.rtsuff, 15, 15, 1, 1)
        self.grid.addWidget(self.etsuff, 15, 16, 1, 2)
        # structure generation
        ctip = 'Generate structures'
        self.butGen = mQPushButton('Generate', ctip, 18)
        self.butGen.clicked.connect(self.runGUI)
        self.grid.addWidget(self.butGen, 17, 15, 2, 2)
        # post-processing setup
        ctip = 'Setup post-processing'
        self.butPost = mQPushButton('Post-process', ctip, 16)
        self.butPost.clicked.connect(self.setupp)
        self.grid.addWidget(self.butPost, 17, 17, 2, 2)
        # Search DB button #
        ctip = 'Search for ligands in chemical databases.'
        self.searchDB = mQPushButton('Search DB', ctip, 16)
        self.searchDB.clicked.connect(self.searchDBW)
        self.grid.addWidget(self.searchDB, 17, 20, 2, 2)
        # ##################################################
        # ##################################################
        # ########## ADDITIONAL MOLECULE INPUTS ############
        # ##################################################
        # ##################################################
        # generate edit texts for additional molecule
        self.txtamol = mQLabel('Additional molecule', '', 'c', 20)
        self.txtamol.setDisabled(True)
        self.grid.addWidget(self.txtamol, 4, 19, 2, 5)
        # additional molecule
        ctip = 'Place additional molecule'
        self.chkM = mQCheckBox('Extra molecule', ctip, 12)
        self.chkM.stateChanged.connect(self.enableemol)
        self.grid.addWidget(self.chkM, 7, 20, 1, 3)
        # name of binding species
        ctip = 'Binding species'
        self.rtbind = mQLabel('Molecule:', ctip, 'Cr', 12)
        bicores = list(sorted(getbcores()))
        self.etbind = mQLineEditL('', ctip, 'l', 12, bicores)
        self.rtbind.setDisabled(True)
        self.etbind.setDisabled(True)
        self.grid.addWidget(self.rtbind, 8, 20, 1, 1)
        self.grid.addWidget(self.etbind, 8, 21, 1, 2)
        # name of binding molecule from SMILES
        ctip = 'Name of binding molecule using SMILES'
        self.rtbsmi = mQLabel('Name:', ctip, 'Cr', 12)
        self.etbsmi = mQLineEdit('', ctip, 'l', 12)
        self.rtbsmi.setDisabled(True)
        self.etbsmi.setDisabled(True)
        self.grid.addWidget(self.rtbsmi, 9, 20, 1, 1)
        self.grid.addWidget(self.etbsmi, 9, 21, 1, 2)
        # separate in xyz file
        ctip = 'Separate molecules in xyz or input file with ------'
        self.chsep = mQCheckBox('separate', ctip, 12)
        self.chsep.setDisabled(True)
        self.grid.addWidget(self.chsep, 9, 23, 1, 2)
        # number of binding conformations to generate
        ctip = 'Number of different conformations to be generated'
        self.rtnbind = mQLabel('Conformations:', ctip, 'Cr', 12)
        self.etnbind = mQLineEdit('', ctip, 'l', 12)
        self.rtnbind.setDisabled(True)
        self.etnbind.setDisabled(True)
        self.grid.addWidget(self.rtnbind, 10, 20, 1, 1)
        self.grid.addWidget(self.etnbind, 10, 21, 1, 1)
        # charge of binding species
        ctip = 'Charge of binding species'
        self.rtchbind = mQLabel('Charge:', ctip, 'Cr', 12)
        self.etchbind = mQLineEdit('', ctip, 'l', 12)
        self.rtchbind.setDisabled(True)
        self.etchbind.setDisabled(True)
        self.grid.addWidget(self.rtchbind, 10, 22, 1, 1)
        self.grid.addWidget(self.etchbind, 10, 23, 1, 1)
        # min/max distance
        ctip = 'Specify placing minimum/maximum distance (in A)'
        self.rtplace = mQLabel('Distance:', ctip, 'Cr', 12)
        ctip = 'Minimum distance between the two molecules. 0 corresponds the marginally non-overlapping configuration'
        self.etplacemin = mQLineEdit('', ctip, 'l', 12)
        ctip = 'Maximum distance between the two molecules. 0 corresponds the marginally non-overlapping configuration'
        self.etplacemax = mQLineEdit('', ctip, 'l', 12)
        self.rtplace.setDisabled(True)
        self.etplacemin.setDisabled(True)
        self.etplacemax.setDisabled(True)
        self.grid.addWidget(self.rtplace, 11, 20, 1, 1)
        self.grid.addWidget(self.etplacemin, 11, 21, 1, 1)
        self.grid.addWidget(self.etplacemax, 11, 22, 1, 1)
        # mask for atom/center of mass reference
        ctip = ('Reference atoms in extra molecules to be used for placement '
                '(e.g. 1,2 or 1-6 or COM or Fe) Default COM (center mass)')
        # self.rtmaskbind = mQLabel('Reference:',ctip,'r',14)
        self.etmaskbind = mQLineEdit('COM', ctip, 'l', 12)
        # self.rtmaskbind.setDisabled(True)
        self.etmaskbind.setDisabled(True)
        # self.grid.addWidget(self.rtmaskbind,11,23,1,1)
        self.grid.addWidget(self.etmaskbind, 11, 23, 1, 1)
        # angle/orientation
        ctip = 'Specify placement type or angle. Angle overwrites placement.'
        self.rtplacea = mQLabel('Angle:', ctip, 'Cr', 12)
        ctip = 'Azimouthal angle phi from 0 to 180'
        self.etplacephi = mQLineEdit('', ctip, 'l', 12)
        ctip = 'Polar angle theta from 0 to 360'
        self.etplacetheta = mQLineEdit('', ctip, 'l', 12)
        self.rtplacea.setDisabled(True)
        self.etplacephi.setDisabled(True)
        self.etplacetheta.setDisabled(True)
        self.grid.addWidget(self.rtplacea, 12, 20, 1, 1)
        self.grid.addWidget(self.etplacephi, 12, 21, 1, 1)
        self.grid.addWidget(self.etplacetheta, 12, 22, 1, 1)
        # placement of extr molecule
        ctip = 'Orientation for placing additional molecule'
        qcav = ['', 'axial', 'equatorial']
        self.dmolp = mQComboBox(qcav, ctip, 12)
        self.dmolp.setDisabled(True)
        self.dmolp.currentIndexChanged.connect(self.checkaxial)
        self.grid.addWidget(self.dmolp, 12, 23, 1, 1)
        # input file generation
        ctip = 'Generate input files'
        self.chkI = mQCheckBox('Input files', ctip, 12)
        self.chkI.stateChanged.connect(self.enableqeinput)
        self.grid.addWidget(self.chkI, 13, 20, 1, 2)
        # jobscript generation
        ctip = 'Generate jobscripts'
        self.chkJ = mQCheckBox('Jobscripts', ctip, 12)
        self.chkJ.stateChanged.connect(self.enablejinput)
        self.grid.addWidget(self.chkJ, 13, 22, 1, 2)
        # input for QC calculation
        ctip = 'Enter input for Quantum Chemistry calculations'
        self.butQc = mQPushButton('Enter QC input', ctip, 12)
        self.butQc.setDisabled(True)
        self.butQc.clicked.connect(self.qcinput)
        ctip = 'Select QC code'
        qcav = ['TeraChem', 'GAMESS', 'QChem']
        self.qcode = mQComboBox(qcav, ctip, 12)
        self.qcode.setDisabled(True)
        self.grid.addWidget(self.butQc, 14, 20, 1, 2)
        self.grid.addWidget(self.qcode, 15, 20, 1, 2)
        # input for jobscripts
        ctip = 'Enter input for jobscript files'
        self.butJob = mQPushButton('Enter job input', ctip, 12)
        self.butJob.setDisabled(True)
        self.butJob.clicked.connect(self.jobenable)
        ctip = 'Select job scheduler'
        qcav = ['SGE', 'SLURM']
        self.scheduler = mQComboBox(qcav, ctip, 12)
        self.scheduler.setDisabled(True)
        self.grid.addWidget(self.butJob, 14, 22, 1, 2)
        self.grid.addWidget(self.scheduler, 15, 22, 1, 2)
        # quit button
        ctip = 'Quit program'
        self.butQ = mQPushButton('Quit', ctip, 14)
        self.butQ.clicked.connect(self.qexit)
        self.grid.addWidget(self.butQ, 17, 23, 1, 1)
        # ###############################################
        # ###############################################
        # ###############################################
        # #########################
        # ## information window ###
        # #########################
        self.iWind = QWidget()
        self.iWind.setWindowTitle('Running')
        self.iWtxt = mQTextEdit('Program started..', 'l', 14)
        self.iWtxt.setParent(self.iWind)
        self.sgrid.addWidget(self.iWind)
        # ######################################
        # ## create terachem-qc input window ###
        # ######################################
        self.qctWindow = QWidget()  # TC QC window
        self.qctgrid = QGridLayout()
        self.qctWindow.setWindowTitle('Terachem Input')
        self.sgrid.addWidget(self.qctWindow)  # add to stacked grid
        self.qctWindow.setPalette(p)  # set background color
        self.qctWindow.setLayout(self.qctgrid)  # set layout

        f = resource_files("molSimplify").joinpath("icons/petachem.png")
        c0 = mQPixmap(f)
        self.qctgrid.addWidget(c0, 0, 2, 1, 1)
        # top text
        self.txtqc = mQLabel('   TeraChem Input', '', 'C', 18)
        self.qctgrid.addWidget(self.txtqc, 0, 3, 1, 2)
        # text for specifying charge
        ctip = 'Charge of the system, default: 0'
        self.rtqctch = mQLabel('Charge:', ctip, 'r', 14)
        self.etqctch = mQLineEdit('0', ctip, 'l', 14)
        self.qctgrid.addWidget(self.rtqctch, 1, 1, 1, 1)
        self.qctgrid.addWidget(self.etqctch, 1, 2, 1, 1)
        # text for specifying spin state
        ctip = 'Spin multiplicity of the system, default: 1'
        self.rtqctspin = mQLabel('Spin:', ctip, 'r', 14)
        self.etqctspin = mQLineEdit('1', ctip, 'l', 14)
        self.qctgrid.addWidget(self.rtqctspin, 1, 3, 1, 1)
        self.qctgrid.addWidget(self.etqctspin, 1, 4, 1, 1)
        # drop menu for selecting type of calculation
        qcav = ['energy', 'minimize', 'gradient', 'ts']
        ctip = 'Specify calculation type, default: minimize'
        self.rtqctcalc = mQLabel('Calculation:', ctip, 'r', 14)
        self.qctcalc = mQComboBox(qcav, ctip, 14)
        self.qctgrid.addWidget(self.rtqctcalc, 2, 1, 1, 1)
        self.qctgrid.addWidget(self.qctcalc, 2, 2, 1, 1)
        # text for specifying electronic structure method
        ctip = 'Select electronic structure method, default: ub3lyp'
        self.rtqctmethod = mQLabel('Method:', ctip, 'r', 14)
        self.etqctmethod = mQLineEdit('ub3lyp', ctip, 'l', 14)
        self.qctgrid.addWidget(self.rtqctmethod, 2, 3, 1, 1)
        self.qctgrid.addWidget(self.etqctmethod, 2, 4, 1, 1)
        # text for specifying basis set
        ctip = 'Select basis set, default: lacvp_s'
        self.rtqctbasis = mQLabel('Basis:', ctip, 'r', 14)
        self.etqctbasis = mQLineEdit('lacvps_ecp', ctip, 'l', 14)
        self.qctgrid.addWidget(self.rtqctbasis, 3, 1, 1, 1)
        self.qctgrid.addWidget(self.etqctbasis, 3, 2, 1, 1)
        # drop menu for selecting dispersion
        qcav = ['yes', 'no', 'd2', 'd3']
        ctip = 'Select dispersion correction'
        self.rtcdisp = mQLabel('Dispersion:', ctip, 'r', 14)
        self.qctsel = mQComboBox(qcav, ctip, 14)
        self.qctgrid.addWidget(self.rtcdisp, 3, 3, 1, 1)
        self.qctgrid.addWidget(self.qctsel, 3, 4, 1, 1)
        self.qctsel.setCurrentIndex(1)
        # editor for additional input
        ctip = 'Specify additional input here'
        self.rtqctadd = mQLabel('Additional input:', ctip, 'Cr', 14)
        self.qceditor = mQTextEdit('', 'l', 14)
        self.qctgrid.addWidget(self.rtqctadd, 4, 2, 1, 1)
        self.qctgrid.addWidget(self.qceditor, 4, 3, 1, 2)
        # button for addition
        ctip = 'make default'
        self.butqctlf = mQPushButton('Make default', ctip, 14)
        self.butqctlf.clicked.connect(self.qctdef)
        self.qctgrid.addWidget(self.butqctlf, 5, 1, 2, 1)
        # button for addition
        ctip = 'Submit input for Quantum Chemistry'
        self.butqctSub = mQPushButton('Submit', ctip, 14)
        self.butqctSub.clicked.connect(self.qretmain)
        self.qctgrid.addWidget(self.butqctSub, 5, 2, 2, 1)
        # button for return
        ctip = 'Return to main menu'
        self.butqctRet = mQPushButton('Return', ctip, 14)
        self.butqctRet.clicked.connect(self.qretmain)
        self.qctgrid.addWidget(self.butqctRet, 5, 4, 2, 1)
        # load defaults if existing
        if glob.glob(str(globs.custom_path)+'/Data/.tcdefinput.inp'):
            loadfrominputtc(self, str(globs.custom_path) +
                            '/Data/.tcdefinput.inp')
        # ####################################
        # ## create gamess-qc input window ###
        # ######################################
        self.qcgWindow = QWidget()  # TC QC window
        self.qcggrid = QGridLayout()
        self.qcgWindow.setWindowTitle('GAMESS Input')
        self.sgrid.addWidget(self.qcgWindow)  # add to stacked grid
        self.qcgWindow.setPalette(p)  # set background color
        self.qcgWindow.setLayout(self.qcggrid)  # set layout
        # top text
        self.txtqcg = mQLabel('GAMESS Input', '', 'c', 18)
        self.qcggrid.addWidget(self.txtqcg, 0, 0, 1, 4)
        # text for specifying charge
        ctip = 'Charge of the system, default: 0'
        self.rtqcgch = mQLabel('Charge:', ctip, 'r', 14)
        self.etqcgch = mQLineEdit('0', ctip, 'l', 14)
        self.qcggrid.addWidget(self.rtqcgch, 1, 0, 1, 1)
        self.qcggrid.addWidget(self.etqcgch, 1, 1, 1, 1)
        # text for specifying spin state
        ctip = 'Spin multiplicity of the system, default: 1'
        self.rtqcgspin = mQLabel('Spin:', ctip, 'r', 14)
        self.etqcgspin = mQLineEdit('1', ctip, 'l', 14)
        self.qcggrid.addWidget(self.rtqcgspin, 1, 2, 1, 1)
        self.qcggrid.addWidget(self.etqcgspin, 1, 3, 1, 1)
        # drop menu for selecting type of calculation
        qcav = ['energy', 'minimize', 'ts']
        ctip = 'Specify calculation type, default: minimize'
        self.rtqcgcalc = mQLabel('Calculation:', ctip, 'r', 14)
        self.qcgcalc = mQComboBox(qcav, ctip, 14)
        self.qcggrid.addWidget(self.rtqcgcalc, 2, 0, 1, 1)
        self.qcggrid.addWidget(self.qcgcalc, 2, 1, 1, 1)
        # text for specifying electronic structure method
        ctip = 'Select electronic structure method, default: ub3lyp'
        self.rtqcgmethod = mQLabel('Method:', ctip, 'r', 14)
        self.etqcgmethod = mQLineEdit('ub3lyp', ctip, 'l', 14)
        self.qcggrid.addWidget(self.rtqcgmethod, 2, 2, 1, 1)
        self.qcggrid.addWidget(self.etqcgmethod, 2, 3, 1, 1)
        # text for specifying basis set
        ctip = 'Select GBASIS input, default: 6'
        self.rtqcgbasis = mQLabel('GBASIS:', ctip, 'r', 14)
        self.etqcgbasis = mQLineEdit('6', ctip, 'l', 14)
        self.qcggrid.addWidget(self.rtqcgbasis, 3, 0, 1, 1)
        self.qcggrid.addWidget(self.etqcgbasis, 3, 1, 1, 1)
        # text for specifying basis set
        ctip = 'Select NGAUSS input, default: N31'
        self.rtqcngauss = mQLabel('NGAUSS:', ctip, 'r', 14)
        self.etqcngauss = mQLineEdit('N31', ctip, 'l', 14)
        self.qcggrid.addWidget(self.rtqcngauss, 3, 2, 1, 1)
        self.qcggrid.addWidget(self.etqcngauss, 3, 3, 1, 1)
        # text for specifying polarization functions
        ctip = 'Select NPFUNC input, default: 0'
        self.rtqcnpfunc = mQLabel('NPFUNC:', ctip, 'r', 14)
        self.etqcnpfunc = mQLineEdit('', ctip, 'l', 14)
        self.qcggrid.addWidget(self.rtqcnpfunc, 4, 0, 1, 1)
        self.qcggrid.addWidget(self.etqcnpfunc, 4, 1, 1, 1)
        # text for specifying polarization functions
        ctip = 'Select NDFUNC input, default: 0'
        self.rtqcndfunc = mQLabel('NDFUNC:', ctip, 'r', 14)
        self.etqcndfunc = mQLineEdit('', ctip, 'l', 14)
        self.qcggrid.addWidget(self.rtqcndfunc, 4, 2, 1, 1)
        self.qcggrid.addWidget(self.etqcndfunc, 4, 3, 1, 1)
        # editor for additional input
        ctip = 'Specify additional input for block SYS here'
        self.rtqcgadd1 = mQLabel('SYS input:', ctip, 'Cr', 14)
        self.qcgedsys = mQTextEdit('', 'l', 12)
        self.qcggrid.addWidget(self.rtqcgadd1, 5, 0, 1, 1)
        self.qcggrid.addWidget(self.qcgedsys, 5, 1, 1, 1)
        # editor for additional input
        ctip = 'Specify additional input for block CTRL here'
        self.rtqcgadd1 = mQLabel('CTRL input:', ctip, 'Cr', 14)
        self.qcgedctrl = mQTextEdit('', 'l', 12)
        self.qcggrid.addWidget(self.rtqcgadd1, 5, 2, 1, 1)
        self.qcggrid.addWidget(self.qcgedctrl, 5, 3, 1, 1)
        # editor for additional input
        ctip = 'Specify additional input for block SCF here'
        self.rtqcgadd3 = mQLabel('SCF input:', ctip, 'Cr', 14)
        self.qcgedscf = mQTextEdit('', 'l', 12)
        self.qcggrid.addWidget(self.rtqcgadd3, 6, 0, 1, 1)
        self.qcggrid.addWidget(self.qcgedscf, 6, 1, 1, 1)
        # editor for additional input
        ctip = 'Specify additional input for block STAT here'
        self.rtqcgadd4 = mQLabel('STAT input:', ctip, 'Cr', 14)
        self.qcgedstat = mQTextEdit('', 'l', 12)
        self.qcggrid.addWidget(self.rtqcgadd4, 6, 2, 1, 1)
        self.qcggrid.addWidget(self.qcgedstat, 6, 3, 1, 1)
        # button for addition
        ctip = 'make default'
        self.butqcglf = mQPushButton('Make default', ctip, 14)
        self.butqcglf.clicked.connect(self.qcgdef)
        self.qcggrid.addWidget(self.butqcglf, 7, 0, 1, 1)
        # button for addition
        ctip = 'Submit input for Quantum Chemistry'
        self.butqcgSub = mQPushButton('Submit', ctip, 14)
        self.butqcgSub.clicked.connect(self.qretmain)
        self.qcggrid.addWidget(self.butqcgSub, 7, 1, 1, 2)
        # button for return
        ctip = 'Return to main menu'
        self.butqcgRet = mQPushButton('Return', ctip, 14)
        self.butqcgRet.clicked.connect(self.qretmain)
        self.qcggrid.addWidget(self.butqcgRet, 7, 3, 1, 1)
        self.qcggrid.setRowMinimumHeight(7, 30)
        # load defaults if existing
        if glob.glob(str(globs.custom_path)+'/Data/.gamdefinput.inp'):
            loadfrominputgam(self, str(globs.custom_path))
        # ######################################
        # ### create Qchem-qc input window #####
        # ######################################
        self.qcQWindow = QWidget()  # TC QC window
        self.qcQgrid = QGridLayout()
        self.qcQWindow.setWindowTitle('QChem Input')
        self.sgrid.addWidget(self.qcQWindow)  # add to stacked grid
        self.qcQWindow.setPalette(p)  # set background color
        self.qcQWindow.setLayout(self.qcQgrid)  # set layout
        # top text
        self.txtQqc = mQLabel('QChem Input', '', 'c', 18)
        self.qcQgrid.addWidget(self.txtQqc, 0, 0, 1, 5)
        # text for specifying charge
        ctip = 'Charge of the system, default: 0'
        self.rtqcQch = mQLabel('Charge:', ctip, 'r', 14)
        self.etqcQch = mQLineEdit('0', ctip, 'l', 14)
        self.qcQgrid.addWidget(self.rtqcQch, 1, 0, 1, 1)
        self.qcQgrid.addWidget(self.etqcQch, 1, 1, 1, 1)
        # text for specifying spin state
        ctip = 'Spin multiplicity of the system, default: 1'
        self.rtqcQspin = mQLabel('Spin:', ctip, 'r', 14)
        self.etqcQspin = mQLineEdit('1', ctip, 'l', 14)
        self.qcQgrid.addWidget(self.rtqcQspin, 1, 2, 1, 1)
        self.qcQgrid.addWidget(self.etqcQspin, 1, 3, 1, 1)
        # additional molecule
        ctip = 'Unrestricted calculation?'
        self.chQun = mQCheckBox('Unrestricted', ctip, 14)
        self.chQun.setChecked(True)
        self.qcQgrid.addWidget(self.chQun, 1, 4, 1, 1)
        # drop menu for selecting type of calculation
        qcav = ['energy', 'minimize', 'ts']
        ctip = 'Specify calculation type, default: minimize'
        self.rtqcQcalc = mQLabel('Calculation:', ctip, 'r', 14)
        self.qcQcalc = mQComboBox(qcav, ctip, 14)
        self.qcQgrid.addWidget(self.rtqcQcalc, 2, 0, 1, 1)
        self.qcQgrid.addWidget(self.qcQcalc, 2, 1, 1, 1)
        # text for specifying basis set
        ctip = 'Select basis set, default: lanl2dz'
        self.rtqcQbasis = mQLabel('Basis:', ctip, 'r', 14)
        self.etqcQbasis = mQLineEdit('lanl2dz', ctip, 'l', 14)
        self.qcQgrid.addWidget(self.rtqcQbasis, 2, 2, 1, 1)
        self.qcQgrid.addWidget(self.etqcQbasis, 2, 3, 1, 1)
        # text for specifying exchange
        ctip = 'Select exchange, default: b3lyp'
        self.rtqcQex = mQLabel('Exchange:', ctip, 'r', 14)
        self.etqcQex = mQLineEdit('b3lyp', ctip, 'l', 14)
        self.qcQgrid.addWidget(self.rtqcQex, 3, 0, 1, 1)
        self.qcQgrid.addWidget(self.etqcQex, 3, 1, 1, 1)
        # text for specifying electronic structure method
        ctip = 'Select correlation, default: none'
        self.rtqcQcor = mQLabel('Correlation:', ctip, 'r', 14)
        self.etqcQcor = mQLineEdit('none', ctip, 'l', 14)
        self.qcQgrid.addWidget(self.rtqcQcor, 3, 2, 1, 1)
        self.qcQgrid.addWidget(self.etqcQcor, 3, 3, 1, 1)
        # editor for additional input
        ctip = 'Specify additional input here'
        self.rtqcQadd = mQLabel('Additional input:', ctip, 'Cr', 14)
        self.qcQeditor = mQTextEdit('', 'l', 12)
        self.qcQgrid.addWidget(self.rtqcQadd, 4, 1, 1, 1)
        self.qcQgrid.addWidget(self.qcQeditor, 4, 2, 1, 2)
        # button for addition
        ctip = 'make default'
        self.butqcQlf = mQPushButton('Make default', ctip, 14)
        self.butqcQlf.clicked.connect(self.qcqdef)
        self.qcQgrid.addWidget(self.butqcQlf, 5, 0, 1, 1)
        # button for addition
        ctip = 'Submit input for Quantum Chemistry'
        self.butqcQSub = mQPushButton('Submit', ctip, 14)
        self.butqcQSub.clicked.connect(self.qretmain)
        self.qcQgrid.addWidget(self.butqcQSub, 5, 2, 1, 2)
        # button for return
        ctip = 'Return to main menu'
        self.butqcQRet = mQPushButton('Return', ctip, 14)
        self.butqcQRet.clicked.connect(self.qretmain)
        self.qcQgrid.addWidget(self.butqcQRet, 5, 4, 1, 1)
        # load defaults if existing
        if glob.glob(str(globs.custom_path)+'/Data/.qchdefinput.inp'):
            loadfrominputqch(self, str(globs.custom_path) +
                             '/Data/.qchdefinput.inp')
        # ####################################
        # ## create jobscript input window ###
        # ####################################
        self.jWindow = QWidget()  # jobscript window
        self.jgrid = QGridLayout()
        self.jWindow.setWindowTitle('Jobscript parameters')
        self.sgrid.addWidget(self.jWindow)  # add to stacked grid
        self.jWindow.setPalette(p)  # set background color
        self.jWindow.setLayout(self.jgrid)  # set layout
        f1 = resource_files("molSimplify").joinpath("icons/sge.png")
        f2 = resource_files("molSimplify").joinpath("icons/slurm.png")
        c1 = mQPixmap(f1)
        c2 = mQPixmap(f2)
        self.jgrid.addWidget(c1, 1, 2, 1, 1)
        self.jgrid.addWidget(c2, 1, 4, 1, 1)
        # top text
        self.txtj = mQLabel('Jobscript Input', '', 'r', 18)
        self.jgrid.addWidget(self.txtj, 0, 2, 1, 2)
        # text for main job identifier
        ctip = 'Job main identifier, e.g. feII'
        self.rtjname = mQLabel('Job name:', ctip, 'r', 14)
        self.etjname = mQLineEdit('myjob', ctip, 'l', 14)
        self.jgrid.addWidget(self.rtjname, 2, 1, 1, 1)
        self.jgrid.addWidget(self.etjname, 2, 2, 1, 1)
        # text for specifying spin state
        ctip = 'Queue to use, e.g. gpus'
        self.rtjqueue = mQLabel('Queue:', ctip, 'r', 14)
        self.etjqueue = mQLineEdit('', ctip, 'l', 14)
        self.jgrid.addWidget(self.rtjqueue, 2, 3, 1, 1)
        self.jgrid.addWidget(self.etjqueue, 2, 4, 1, 1)
        # text for specifying wall time
        ctip = 'Wall time request, e.g. 48'
        self.rtjwallt = mQLabel('Wall time:', ctip, 'r', 14)
        self.etjwallt = mQLineEdit('48h', ctip, 'l', 14)
        self.jgrid.addWidget(self.rtjwallt, 3, 1, 1, 1)
        self.jgrid.addWidget(self.etjwallt, 3, 2, 1, 1)
        # text for specifying memory
        ctip = 'Memory request, e.g. 10G'
        self.rtjmem = mQLabel('Memory:', ctip, 'r', 14)
        self.etjmem = mQLineEdit('10G', ctip, 'l', 14)
        self.jgrid.addWidget(self.rtjmem, 3, 3, 1, 1)
        self.jgrid.addWidget(self.etjmem, 3, 4, 1, 1)
        # text for specifying charge
        ctip = 'Number of CPUs requested, default: 0'
        self.rtjcpus = mQLabel('CPUs:', ctip, 'r', 14)
        self.etjcpus = mQLineEdit('', ctip, 'l', 14)
        self.jgrid.addWidget(self.rtjcpus, 4, 1, 1, 1)
        self.jgrid.addWidget(self.etjcpus, 4, 2, 1, 1)
        # text for specifying spin state
        ctip = 'Number of GPUs requested, default: 0'
        self.rtjgpus = mQLabel('GPUs:', ctip, 'r', 14)
        self.etjgpus = mQLineEdit('', ctip, 'l', 14)
        self.jgrid.addWidget(self.rtjgpus, 4, 3, 1, 1)
        self.jgrid.addWidget(self.etjgpus, 4, 4, 1, 1)
        # text for modules to be loaded
        ctip = 'Modules to be loaded, e.g. terachem, openmpi'
        self.rtjmod = mQLabel('Modules:', ctip, 'r', 14)
        self.etjmod = mQLineEdit('', ctip, 'l', 14)
        self.jgrid.addWidget(self.rtjmod, 5, 1, 1, 1)
        self.jgrid.addWidget(self.etjmod, 5, 2, 1, 1)
        # editor for additional input
        ctip = 'Specify additional input for initial options, e.g. -j y'
        self.rtjadd1 = mQLabel('Options:', ctip, 'r', 14)
        self.etjopt = mQTextEdit('', 'l', 12)
        self.jgrid.addWidget(self.rtjadd1, 6, 1, 1, 1)
        self.jgrid.addWidget(self.etjopt, 6, 2, 1, 1)
        # editor for additional input
        ctip = 'Specify additional commands, e.g. mkdir $WORKDIR/test'
        self.rtjadd2 = mQLabel('Commands:', ctip, 'Cr', 14)
        self.jcomm = mQTextEdit('', 'l', 12)
        self.jgrid.addWidget(self.rtjadd2, 6, 3, 1, 1)
        self.jgrid.addWidget(self.jcomm, 6, 4, 1, 1)
        # button for addition
        ctip = 'make default'
        self.butqcJlf = mQPushButton('Make default', ctip, 14)
        self.butqcJlf.clicked.connect(self.jobdef)
        self.jgrid.addWidget(self.butqcJlf, 7, 1, 1, 1)
        # button for addition
        ctip = 'Submit input for Quantum Chemistry'
        self.butqcgSub = mQPushButton('Submit', ctip, 14)
        self.butqcgSub.clicked.connect(self.qretmain)
        self.jgrid.addWidget(self.butqcgSub, 7, 2, 1, 2)
        # button for return
        ctip = 'Return to main menu'
        self.butqcgRet = mQPushButton('Return', ctip, 14)
        self.butqcgRet.clicked.connect(self.qretmain)
        self.jgrid.addWidget(self.butqcgRet, 7, 4, 1, 1)
        # load defaults if existing
        if glob.glob(str(globs.custom_path)+'/Data/.jobdefinput.inp'):
            loadfrominputjob(self, str(globs.custom_path)+'/Data/.jobdefinput.inp')
        # #########################################
        # ## create local DB interaction window ###
        # #########################################
        self.DBWindow = QWidget()  # DB Window
        self.DBlgrid = QGridLayout()
        self.sgrid.addWidget(self.DBWindow)
        self.DBWindow.setPalette(p)
        self.DBWindow.setLayout(self.DBlgrid)
        self.DBWindow.setWindowTitle('Insert/remove to/from Database')
        # top text
        self.txtdb = mQLabel('Database Update', '', 'c', 18)
        self.DBlgrid.addWidget(self.txtdb, 0, 0, 1, 4)
        # drop menu for selecting type
        ctip = 'Select what type of molecule you want to add/remove to/from the database'
        self.rtDBsel = mQLabel('Select type:', '', 'r', 14)
        qcav = ['core', 'ligand', 'binding']
        self.DBsel = mQComboBox(qcav, ctip, 14)
        self.DBsel.setCurrentIndex(1)
        self.DBlgrid.addWidget(self.rtDBsel, 1, 1, 1, 1)
        self.DBlgrid.addWidget(self.DBsel, 1, 2, 1, 1)
        # text for selecting type
        ctip = 'Type SMILES string for molecule'
        self.rtDBsmi = mQLabel('SMILES or file:', '', 'r', 14)
        self.etDBsmi = mQLineEdit('', ctip, 'l', 14)
        self.DBlgrid.addWidget(self.rtDBsmi, 2, 1, 1, 1)
        self.DBlgrid.addWidget(self.etDBsmi, 2, 2, 1, 1)
        # text for specifying name
        ctip = 'Type name for molecule'
        self.rtDBname = mQLabel('Name:', '', 'r', 14)
        self.etDBname = mQLineEdit('', ctip, 'l', 14)
        self.DBlgrid.addWidget(self.rtDBname, 3, 1, 1, 1)
        self.DBlgrid.addWidget(self.etDBname, 3, 2, 1, 1)
        # text for specifying category
        ctip = 'Type groups for ligand'
        self.rtDBgrps = mQLabel('Groups:', '', 'r', 14)
        self.etDBgrps = mQLineEdit('', ctip, 'l', 14)
        self.DBlgrid.addWidget(self.rtDBgrps, 4, 1, 1, 1)
        self.DBlgrid.addWidget(self.etDBgrps, 4, 2, 1, 1)
        # text for specifying groups
        ctip = 'Type category for ligand. It can be used for building complexes("build")\n'
        ctip += 'functionalizing existing cores("functionalize") or both.'
        self.rtDBctg = mQLabel('Category:', '', 'r', 14)
        self.etDBctg = mQComboBox(['all', 'build', 'functionalize'], ctip, 14)
        self.DBlgrid.addWidget(self.rtDBctg, 5, 1, 1, 1)
        self.DBlgrid.addWidget(self.etDBctg, 5, 2, 1, 1)
        # checkboxes for FF optimization
        ctip = 'Force-field optimize in isolation'
        self.lFFb = mQCheckBox('FF before', ctip, 14)
        self.DBlgrid.addWidget(self.lFFb, 6, 1, 1, 1)
        self.lFFb.setChecked(True)
        # checkboxes for FF optimization
        ctip = 'Force-field optimize in molecule'
        self.lFFa = mQCheckBox('FF after', ctip, 14)
        self.DBlgrid.addWidget(self.lFFa, 6, 2, 1, 1)
        self.lFFa.setChecked(True)
        # drop menu for denticity
        ctip = 'Type denticity for SMILES molecule'
        self.rtDBsmident = mQLabel('Denticity:', '', 'r', 14)
        qcav = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self.DBdent = mQComboBox(qcav, ctip, 14)
        self.DBdent.setCurrentIndex(0)
        self.DBlgrid.addWidget(self.rtDBsmident, 7, 1, 1, 1)
        self.DBlgrid.addWidget(self.DBdent, 7, 2, 1, 1)
        self.DBsel.currentIndexChanged.connect(
            self.dbchange)  # error if not here
        # text for typing input connection atoms
        ctip = 'Type indices for connection atoms, default: 1'
        self.rtDBsmicat = mQLabel('Catoms:', ctip, 'r', 14)
        self.etDBsmicat = mQLineEdit('', ctip, 'l', 14)
        self.DBlgrid.addWidget(self.rtDBsmicat, 8, 1, 1, 1)
        self.DBlgrid.addWidget(self.etDBsmicat, 8, 2, 1, 1)
        # button for addition
        ctip = 'Load molecule from file'
        self.butDBAlf = mQPushButton('Load file..', ctip, 14)
        self.butDBAlf.clicked.connect(self.qDBload)
        self.DBlgrid.addWidget(self.butDBAlf, 9, 0, 1, 1)
        # button for addition
        ctip = 'Add new molecule to database'
        self.butDBAub = mQPushButton('Add', ctip, 14)
        self.butDBAub.clicked.connect(self.qaddDB)
        self.DBlgrid.addWidget(self.butDBAub, 9, 1, 1, 1)
        # button for removal
        ctip = 'Remove molecule from database'
        self.butDBDub = mQPushButton('Remove', ctip, 14)
        self.butDBDub.clicked.connect(self.qdelDB)
        self.DBlgrid.addWidget(self.butDBDub, 9, 2, 1, 1)
        # button for return
        ctip = 'Return to main menu'
        self.butDBRet = mQPushButton('Return', ctip, 14)
        self.butDBRet.clicked.connect(self.qretmain)
        self.DBlgrid.addWidget(self.butDBRet, 9, 3, 1, 1)
        # ############################################
        # ## create chemical DB interaction window ###
        # ############################################
        self.cDBWindow = QWidget()  # DB Window
        self.cDBgrid = QGridLayout()
        self.sgrid.addWidget(self.cDBWindow)
        self.cDBWindow.setPalette(p)
        self.cDBWindow.setLayout(self.cDBgrid)
        self.cDBWindow.setWindowTitle('Chemical Database Search')
        ctip = 'Specify screening options or similarity search.'
        # top text
        self.txtcdb = mQLabel('Database Search', ctip, 'c', 18)
        self.cDBgrid.addWidget(self.txtcdb, 0, 0, 1, 9)
        # text for reference
        self.rtcDBref = mQLabel('Similarity search', '', 'c', 18)
        self.cDBgrid.addWidget(self.rtcDBref, 1, 0, 1, 3)
        # text for selecting type
        ctip = 'Type SMILES string or molecule name for reference molecule in similarity search'
        self.rtcDBsmi = mQLabel('SMILES:', ctip, 'r', 14)
        self.etcDBsmi = mQLineEdit('', ctip, 'l', 14)
        self.cDBgrid.addWidget(self.rtcDBsmi, 2, 0, 1, 1)
        self.cDBgrid.addWidget(self.etcDBsmi, 2, 1, 1, 2)
        # button for loading from file
        ctip = 'Load reference molecule from file'
        self.rtcDBAlf = mQLabel('From file:', ctip, 'r', 14)
        self.butcDBAlf = mQPushButton('Load..', ctip, 14)
        self.butcDBAlf.clicked.connect(self.qcDBload)
        self.cDBgrid.addWidget(self.rtcDBAlf, 3, 0, 1, 1)
        self.cDBgrid.addWidget(self.butcDBAlf, 3, 1, 1, 2)
        # connection atoms for smarts/smiles
        ctip = 'Specify the connection atoms in SMARTS/SMILES. Default: 1'
        self.rtcDBcatoms = mQLabel('Conn atoms:', ctip, 'r', 14)
        self.etcDBcatoms = mQLineEdit('1', ctip, 'l', 14)
        self.cDBgrid.addWidget(self.rtcDBcatoms, 4, 0, 1, 1)
        self.cDBgrid.addWidget(self.etcDBcatoms, 4, 1, 1, 1)
        # how many molecules to return
        ctip = 'Specify the number of results you want.'
        self.rtcDBres = mQLabel('Results:', ctip, 'r', 14)
        self.etcDBnres = mQLineEdit('', ctip, 'l', 14)
        self.cDBgrid.addWidget(self.rtcDBres, 5, 0, 1, 1)
        self.cDBgrid.addWidget(self.etcDBnres, 5, 1, 1, 1)
        # text for output file
        ctip = 'Please type in output file'
        self.rtcDBsoutf = mQLabel('Output file:', ctip, 'r', 14)
        self.etcDBoutf = mQLineEdit('simres', ctip, 'l', 14)
        self.cDBgrid.addWidget(self.rtcDBsoutf, 6, 0, 1, 1)
        self.cDBgrid.addWidget(self.etcDBoutf, 6, 1, 1, 1)
        # drop menu for output file
        qcav = ['.smi']  # ,'.mol','.sdf']
        self.cDBdent = mQComboBox(qcav, ctip, 14)
        self.cDBgrid.addWidget(self.cDBdent, 6, 2, 1, 1)
        # text for screening options
        self.rtcDBsc = mQLabel('Screening options', ctip, 'c', 18)
        self.cDBgrid.addWidget(self.rtcDBsc, 1, 6, 1, 4)
        # text for selecting database
        self.rtcDBsel = mQLabel('Select database:', ctip, 'c', 14)
        self.cDBgrid.addWidget(self.rtcDBsel, 2, 5, 1, 2)
        # get existing databases
        ctip = 'Select the database you want to use. Please note that fastsearch indexes DBs allow for much faster screening.'
        dbdir = globs.chemdbdir
        dbs0 = glob.glob(dbdir+"/*.sdf")
        dbs1 = [d.rsplit('/', 1)[-1] for d in dbs0]
        dbs = [d.split('.', 1)[0] for d in dbs1]
        self.cDBsel = mQComboBox(dbs, ctip, 14)
        self.cDBgrid.addWidget(self.cDBsel, 2, 7, 1, 2)
        # text for SMARTS pattern
        ctip = 'Type SMARTS pattern for matching'
        self.rtcDBsmarts = mQLabel('SMARTS:', ctip, 'r', 14)
        self.etcDBsmarts = mQLineEdit('', ctip, 'l', 14)
        self.cDBgrid.addWidget(self.rtcDBsmarts, 3, 5, 1, 2)
        self.cDBgrid.addWidget(self.etcDBsmarts, 3, 7, 1, 2)
        ctip = 'Select molecular fingerprint'
        # text for selecting fingerprint
        self.rtcDBsf = mQLabel('Select Fingerprint:', ctip, 'r', 14)
        self.cDBgrid.addWidget(self.rtcDBsf, 4, 5, 1, 2)
        # select fingerprint
        opts = ['FP2', 'FP3', 'FP4', 'MACCS']
        self.cDBsf = mQComboBox(opts, ctip, 14)
        self.cDBgrid.addWidget(self.cDBsf, 4, 7, 1, 2)
        # get options for screening
        ctip = 'Specify minimum and maximum values for filters.'
        self.rtcDBmin = mQLabel('min', ctip, 'c', 14)
        self.rtcDBmax = mQLabel('max', ctip, 'c', 14)
        self.cDBgrid.addWidget(self.rtcDBmin, 6, 7, 1, 1)
        self.cDBgrid.addWidget(self.rtcDBmax, 6, 8, 1, 1)
        ctip = 'Total number of atoms.'
        self.rtcDBat0 = mQLabel('atoms:', ctip, 'r', 14)
        self.cDBgrid.addWidget(self.rtcDBat0, 7, 5, 1, 2)
        ctip = 'Minimum number of atoms.'
        self.etcDBsatoms0 = mQLineEdit('', ctip, 'l', 14)
        self.cDBgrid.addWidget(self.etcDBsatoms0, 7, 7, 1, 1)
        ctip = 'Maximum number of atoms.'
        self.etcDBsatoms1 = mQLineEdit('', ctip, 'l', 14)
        self.cDBgrid.addWidget(self.etcDBsatoms1, 7, 8, 1, 1)
        ctip = 'Total number of bonds.'
        self.rtcDBb0 = mQLabel('bonds:', ctip, 'r', 14)
        self.cDBgrid.addWidget(self.rtcDBb0, 8, 5, 1, 2)
        ctip = 'Minimum number of total bonds.'
        self.etcDBsbonds0 = mQLineEdit('', ctip, 'l', 14)
        self.cDBgrid.addWidget(self.etcDBsbonds0, 8, 7, 1, 1)
        ctip = 'Maximum number of total bonds.'
        self.etcDBsbonds1 = mQLineEdit('', ctip, 'l', 14)
        self.cDBgrid.addWidget(self.etcDBsbonds1, 8, 8, 1, 1)
        ctip = 'Total number of aromatic.'
        self.rtcDBba0 = mQLabel('aromatic bonds:', ctip, 'r', 14)
        self.cDBgrid.addWidget(self.rtcDBba0, 9, 5, 1, 2)
        ctip = 'Minimum number of aromatic bonds.'
        self.etcDBsabonds0 = mQLineEdit('', ctip, 'l', 14)
        self.cDBgrid.addWidget(self.etcDBsabonds0, 9, 7, 1, 1)
        ctip = 'Maximum number of aromatic bonds.'
        self.etcDBsabonds1 = mQLineEdit('', ctip, 'l', 14)
        self.cDBgrid.addWidget(self.etcDBsabonds1, 9, 8, 1, 1)
        ctip = 'Total number of single bonds.'
        self.rtcDBs0 = mQLabel('single bonds:', ctip, 'r', 14)
        self.cDBgrid.addWidget(self.rtcDBs0, 10, 5, 1, 2)
        ctip = 'Minimum number of single bonds.'
        self.etcDBsbondss0 = mQLineEdit('', ctip, 'l', 14)
        self.cDBgrid.addWidget(self.etcDBsbondss0, 10, 7, 1, 1)
        ctip = 'Maximum number of single bonds.'
        self.etcDBsbondss1 = mQLineEdit('', ctip, 'l', 14)
        self.cDBgrid.addWidget(self.etcDBsbondss1, 10, 8, 1, 1)
        ctip = 'Molecular weight.'
        self.rtcDBbtmw0 = mQLabel('MW:', ctip, '', 14)
        self.cDBgrid.addWidget(self.rtcDBbtmw0, 11, 5, 1, 2)
        ctip = 'Minimum molecular weight.'
        self.etcDBmw0 = mQLineEdit('', ctip, 'l', 14)
        self.cDBgrid.addWidget(self.etcDBmw0, 11, 7, 1, 1)
        ctip = 'Maximum molecular weight.'
        self.etcDBmw1 = mQLineEdit('', ctip, 'l', 14)
        self.cDBgrid.addWidget(self.etcDBmw1, 11, 8, 1, 1)
        # aspirin icon
        f = resource_files("molSimplify").joinpath("icons/chemdb.png")
        c = mQPixmap(f)
        relresize(c, c, 0.4)
        self.cDBgrid.addWidget(c, 7, 0, 4, 2)
        # button for addition
        ctip = 'Search database:'
        self.butcDBAub = mQPushButton('Search', ctip, 14)
        self.butcDBAub.clicked.connect(self.qaddcDB)
        self.cDBgrid.addWidget(self.butcDBAub, 11, 0, 1, 2)
        # button for addition
        ctip = 'Draw results'
        self.butcDBd0 = mQPushButton('Draw', ctip, 14)
        self.butcDBd0.clicked.connect(self.drawres)
        self.cDBgrid.addWidget(self.butcDBd0, 9, 2, 1, 1)
        # button for return
        ctip = 'Return to main menu'
        self.butcDBRet = mQPushButton('Return', ctip, 14)
        self.butcDBRet.clicked.connect(self.qretmain)
        self.cDBgrid.addWidget(self.butcDBRet, 11, 2, 1, 1)
        # ##########################
        # ## post-process window ###
        # ##########################
        self.pWindow = QWidget()  # Post Window
        self.pgrid = QGridLayout()
        self.sgrid.addWidget(self.pWindow)
        self.pWindow.setPalette(p)
        self.pWindow.setLayout(self.pgrid)
        self.pWindow.setWindowTitle('Post-processing')
        self.pgrid.setRowMinimumHeight(0, 50)
        self.pgrid.setRowMinimumHeight(2, 30)
        # top text
        self.txtp = mQLabel('Post-processing setup', '', 'c', 18)
        self.pgrid.addWidget(self.txtp, 0, 0, 1, 4)
        # input for jobscripts
        ctip = 'Select top jobs directory'
        self.butpd = mQPushButton('Select directory', ctip, 16)
        self.butpd.clicked.connect(self.pdload)
        self.pgrid.addWidget(self.butpd, 1, 0, 1, 1)
        self.etpdir = mQLineEdit(globs.rundir, ctip, 'l', 16)
        self.pgrid.addWidget(self.etpdir, 1, 1, 1, 1)
        # select qc code
        ctip = 'Select QC code'
        qcav = ['TeraChem', 'GAMESS']
        self.rpcode = mQLabel('QC code:', '', 'c', 16)
        self.pgrid.addWidget(self.rpcode, 1, 2, 1, 1)
        self.pqcode = mQComboBox(qcav, ctip, 14)
        self.pgrid.addWidget(self.pqcode, 1, 3, 1, 1)
        # general summary
        ctip = 'Generate results summary'
        self.psum = mQCheckBox('Summary', ctip, 16)
        self.pgrid.addWidget(self.psum, 3, 1, 1, 1)
        # metal charges
        ctip = 'Calculate metal charge'
        self.pch = mQCheckBox('Charge', ctip, 16)
        self.pgrid.addWidget(self.pch, 3, 2, 1, 1)
        # wavefunction properties
        ctip = 'Generate average properties of the wavefunction'
        self.pwfnav = mQCheckBox('Wavefunction', ctip, 16)
        self.pgrid.addWidget(self.pwfnav, 4, 1, 1, 1)
        # cube files
        ctip = 'Generate cubefiles'
        self.pcub = mQCheckBox('Cubes', ctip, 16)
        self.pgrid.addWidget(self.pcub, 4, 2, 1, 1)
        # molecular orbital information
        ctip = 'Molecular Orbital information'
        self.porbs = mQCheckBox('MO', ctip, 16)
        self.pgrid.addWidget(self.porbs, 5, 1, 1, 1)
        # NBO analysis
        ctip = 'NBO analysis'
        self.pnbo = mQCheckBox('NBO', ctip, 16)
        self.pgrid.addWidget(self.pnbo, 5, 2, 1, 1)
        # d-orbital information
        # ctip = 'd-orbital information'
        # self.pdorbs = mCheck(self.pWindow,0.55,0.525,0.2,0.1,'d-orbitals',ctip,16)
        # delocalization indices
        ctip = 'Localization and delocalization indices'
        self.pdeloc = mQCheckBox('Deloc', ctip, 16)
        self.pgrid.addWidget(self.pdeloc, 6, 1, 1, 1)
        # button for addition
        ctip = 'Run post-processing'
        self.butpR = mQPushButton('Run', ctip, 16)
        self.pgrid.addWidget(self.butpR, 7, 1, 1, 2)
        self.butpR.clicked.connect(self.postprocGUI)
        # button for return
        ctip = 'Return to main menu'
        self.butpret = mQPushButton('Return', ctip, 16)
        self.butpret.clicked.connect(self.qretmain)
        self.pgrid.addWidget(self.butpret, 7, 3, 1, 2)
        # c1p = mPic(self.pWindow,globs.installdir+'/icons/wft1.png',0.04,0.7,0.2)
        f = resource_files("molSimplify").joinpath("icons/wft3.png")
        c3p = mQPixmap(f)
        self.pgrid.addWidget(c3p, 3, 0, 4, 1)
        # c2p = mPic(self.pWindow,globs.installdir+'/icons/wft2.png',0.04,0.035,0.2)
        # #################################
        # ## create add geometry window ###
        # #################################
        self.geWindow = QWidget()  # Geometry window
        self.gegrid = QGridLayout()
        self.sgrid.addWidget(self.geWindow)
        self.geWindow.setPalette(p)
        self.geWindow.setLayout(self.gegrid)
        self.geWindow.setWindowTitle('Add new geometry')
        # top text
        self.txtg = mQLabel('New geometry', '', 'c', 18)
        self.gegrid.addWidget(self.txtg, 0, 0, 1, 4)
        # text for specifying name
        ctip = 'Type name of geometry'
        self.rtgname = mQLabel('Name:', '', 'r', 14)
        self.etgname = mQLineEdit('', ctip, 'l', 14)
        self.gegrid.addWidget(self.rtgname, 1, 0, 1, 1)
        self.gegrid.addWidget(self.etgname, 1, 1, 1, 1)
        # text for specifying short name
        ctip = 'Type short identifier of geometry (2-4 letters)'
        self.rtgshort = mQLabel('Short identifier:', '', 'r', 14)
        self.etgshort = mQLineEdit('', ctip, 'l', 14)
        self.gegrid.addWidget(self.rtgshort, 2, 0, 1, 1)
        self.gegrid.addWidget(self.etgshort, 2, 1, 1, 1)
        # xyz file
        ctip = 'Load xyz file with geometry.'
        self.rtgf = mQLabel('Filename:', '', 'r', 14)
        self.etgf = mQLineEdit('', ctip, 'l', 14)
        self.gegrid.addWidget(self.rtgf, 3, 0, 1, 1)
        self.gegrid.addWidget(self.etgf, 3, 1, 1, 1)
        # png file
        ctip = 'Load png file with chemdraw for geometry.'
        self.rtgfc = mQLabel('Chemdraw file:', '', 'r', 14)
        self.etgfc = mQLineEdit('', ctip, 'l', 14)
        self.gegrid.addWidget(self.rtgfc, 4, 0, 1, 1)
        self.gegrid.addWidget(self.etgfc, 4, 1, 1, 1)
        # load file
        self.butge = mQPushButton('Load geometry file..', ctip, 14)
        self.butge.clicked.connect(self.qgeomload)
        self.gegrid.addWidget(self.butge, 5, 0, 1, 1)
        # load file
        self.butge2 = mQPushButton('Load Chemdraw file..', ctip, 14)
        self.butge2.clicked.connect(self.qgeomload2)
        self.gegrid.addWidget(self.butge2, 5, 1, 1, 1)
        # button for addition
        ctip = 'Add new geometry to database'
        self.butgAub = mQPushButton('Add', ctip, 14)
        self.butgAub.clicked.connect(self.qaddg)
        self.gegrid.addWidget(self.butgAub, 6, 0, 1, 1)
        # button for removal
        ctip = 'Remove geometry from database'
        self.butgDub = mQPushButton('Remove', ctip, 14)
        self.butgDub.clicked.connect(self.qdelg)
        self.gegrid.addWidget(self.butgDub, 6, 1, 1, 1)
        # button for return
        ctip = 'Return to main menu'
        self.butgRet = mQPushButton('Return', ctip, 14)
        self.butgRet.clicked.connect(self.qretmain)
        self.gegrid.addWidget(self.butgRet, 7, 1, 1, 1)
        # ###############################
        # ## create ligands 2D window ###
        # ###############################
        self.lwindow = QWidget()
        self.lwindow.setPalette(p)
        self.lgrid = QGridLayout()
        self.lwindow.setLayout(self.lgrid)
        # self.sgrid.addWidget(self.lwindow)
        self.lwindow.setWindowTitle('Ligands 2D')
        self.c1p = QWidget()
        # ###################
        # ###################
        # ###################
        # ## Run main GUI ###
        # ###################
        # ###################
        # ###################
        self.wmain.setLayout(self.grid)
        self.wmwindow.setCentralWidget(self.wmain)
        self.sgrid.setCurrentWidget(self.wmain)
        self.wmain.showMaximized()
        self.wmwindow.showMaximized()
        # resize other windows
        relresize(self.iWind, self.wmwindow, 0.7)
        relresize(self.iWtxt, self.iWind, 1.0)

    def update_ligands_binds(self):
        # this function updates the list
        # of ligand and binding species
        ctip0 = 'Ligand(s) to be used'
        self.rtligh = mQLabel('Ligand', ctip0, 'c', 12)  # ligand header
        f = QFont("Helvetica", 12, 75)
        self.rtligh.setFont(f)
        # add to layout
        self.grid.addWidget(self.rtligh, 9, 0, 1, 2)
        # ligands #
        licores = list(sorted(getlicores()))
        self.lig = []
        self.ligocc = []
        self.ligconn = []
        self.ligH = []
        self.ligML = []
        self.ligan = []
        self.lignam = []
        self.ligadd = []
        for ii in range(0, 8):
            # ligands #
            self.lig.append(mQLineEditL('', ctip0, 'l', 12, licores))
            self.grid.addWidget(self.lig[ii], 10+ii, 0, 1, 2)  # add to layout
            # ligand buttons #
        #############################################

        '''
        #############################
        ### Callbacks for buttons ###
        #############################
        '''
        # ##################################
        # ##### Add new ligands input ######
        # ##################################

    def addlig0(self):
        idx = 0
        txt0 = self.lig[idx].currentText().replace(' ', '')
        if len(txt0) == 0:
            mQDialogWarn('No ligand specified',
                         'Please specify a ligand before adding another one.')
        else:
            self.ligadd[idx].setDisabled(True)
            self.ligadd[idx].hide()
            self.ligadd[idx+1].setDisabled(False)
            self.ligadd[idx+1].show()
            self.lig[idx+1].setDisabled(False)
            self.ligocc[idx+1].setDisabled(False)
            self.ligconn[idx+1].setDisabled(False)
            self.ligH[idx+1].setDisabled(False)
            self.ligML[idx+1].setDisabled(False)
            self.ligan[idx+1].setDisabled(False)
            self.lignam[idx+1].setDisabled(False)

    def addlig1(self):
        idx = 1
        txt0 = self.lig[idx].currentText().replace(' ', '')
        if len(txt0) == 0:
            mQDialogWarn('No ligand specified',
                         'Please specify a ligand before adding another one.')
        else:
            self.ligadd[idx].setDisabled(True)
            self.ligadd[idx].hide()
            self.ligadd[idx+1].setDisabled(False)
            self.ligadd[idx+1].show()
            self.lig[idx+1].setDisabled(False)
            self.ligocc[idx+1].setDisabled(False)
            self.ligconn[idx+1].setDisabled(False)
            self.ligH[idx+1].setDisabled(False)
            self.ligML[idx+1].setDisabled(False)
            self.ligan[idx+1].setDisabled(False)
            self.lignam[idx+1].setDisabled(False)

    def addlig2(self):
        idx = 2
        txt0 = self.lig[idx].currentText().replace(' ', '')
        if len(txt0) == 0:
            mQDialogWarn('No ligand specified',
                         'Please specify a ligand before adding another one.')
        else:
            self.ligadd[idx].setDisabled(True)
            self.ligadd[idx].hide()
            self.ligadd[idx+1].setDisabled(False)
            self.ligadd[idx+1].show()
            self.lig[idx+1].setDisabled(False)
            self.ligocc[idx+1].setDisabled(False)
            self.ligconn[idx+1].setDisabled(False)
            self.ligH[idx+1].setDisabled(False)
            self.ligML[idx+1].setDisabled(False)
            self.ligan[idx+1].setDisabled(False)
            self.lignam[idx+1].setDisabled(False)

    def addlig3(self):
        idx = 3
        txt0 = self.lig[idx].currentText().replace(' ', '')
        if len(txt0) == 0:
            mQDialogWarn('No ligand specified',
                         'Please specify a ligand before adding another one.')
        else:
            self.ligadd[idx].setDisabled(True)
            self.ligadd[idx].hide()
            self.ligadd[idx+1].setDisabled(False)
            self.ligadd[idx+1].show()
            self.lig[idx+1].setDisabled(False)
            self.ligocc[idx+1].setDisabled(False)
            self.ligconn[idx+1].setDisabled(False)
            self.ligH[idx+1].setDisabled(False)
            self.ligML[idx+1].setDisabled(False)
            self.ligan[idx+1].setDisabled(False)
            self.lignam[idx+1].setDisabled(False)

    def addlig4(self):
        idx = 4
        txt0 = self.lig[idx].currentText().replace(' ', '')
        if len(txt0) == 0:
            mQDialogWarn('No ligand specified',
                         'Please specify a ligand before adding another one.')
        else:
            self.ligadd[idx].setDisabled(True)
            self.ligadd[idx].hide()
            self.ligadd[idx+1].setDisabled(False)
            self.ligadd[idx+1].show()
            self.lig[idx+1].setDisabled(False)
            self.ligocc[idx+1].setDisabled(False)
            self.ligconn[idx+1].setDisabled(False)
            self.ligH[idx+1].setDisabled(False)
            self.ligML[idx+1].setDisabled(False)
            self.ligan[idx+1].setDisabled(False)
            self.lignam[idx+1].setDisabled(False)

    def addlig5(self):
        idx = 5
        txt0 = self.lig[idx].currentText().replace(' ', '')
        if len(txt0) == 0:
            mQDialogWarn('No ligand specified',
                         'Please specify a ligand before adding another one.')
        else:
            self.ligadd[idx].setDisabled(True)
            self.ligadd[idx].hide()
            self.ligadd[idx+1].setDisabled(False)
            self.ligadd[idx+1].show()
            self.lig[idx+1].setDisabled(False)
            self.ligocc[idx+1].setDisabled(False)
            self.ligconn[idx+1].setDisabled(False)
            self.ligH[idx+1].setDisabled(False)
            self.ligML[idx+1].setDisabled(False)
            self.ligan[idx+1].setDisabled(False)
            self.lignam[idx+1].setDisabled(False)

    def addlig6(self):
        idx = 6
        txt0 = self.lig[idx].currentText().replace(' ', '')
        if len(txt0) == 0:
            mQDialogWarn('No ligand specified',
                         'Please specify a ligand before adding another one.')
        else:
            self.ligadd[idx].setDisabled(True)
            self.ligadd[idx].hide()
            self.ligadd[idx+1].setDisabled(False)
            self.ligadd[idx+1].show()
            self.lig[idx+1].setDisabled(False)
            self.ligocc[idx+1].setDisabled(False)
            self.ligconn[idx+1].setDisabled(False)
            self.ligH[idx+1].setDisabled(False)
            self.ligML[idx+1].setDisabled(False)
            self.ligan[idx+1].setDisabled(False)
            self.lignam[idx+1].setDisabled(False)

    def checkaxial(self):
        if self.dmolp.currentIndex() > 0:
            self.etplacephi.setText('')
            self.etplacephi.setDisabled(True)
            self.etplacetheta.setText('')
            self.etplacetheta.setDisabled(True)
        else:
            self.etplacephi.setDisabled(False)
            self.etplacetheta.setDisabled(False)
        # #############################
        # ## Local Database window ####
        # #############################
        # load molecule from file

    def qDBload(self):
        name = QFileDialog.getOpenFileName(
            self.DBWindow, 'Open File', '.', "Molecule files *.xyz, *.mol (*.xyz *.mol)")
        if name[0] != '':
            self.etDBsmi.setText(os.path.relpath(name[0]))
        # enable add to database interface

    def enableDB(self):
        self.DBWindow.setWindowModality(2)
        self.DBWindow.show()
        self.sgrid.setCurrentWidget(self.DBWindow)
        # callback for addition button, adds to database

    def qaddDB(self):
        coption = self.DBsel.currentText()
        smimol = self.etDBsmi.text()
        sminame = self.etDBname.text()
        smident = self.DBdent.currentText()
        smicat = self.etDBsmicat.text()
        smigrps = self.etDBgrps.text()
        smictg = self.etDBctg.currentText()
        globs = globalvars()
        if not globs.custom_path:  # this will be false unless set
            choice = QMessageBox.question(self.cDBWindow, 'Custom path needed',
                                          'In order to add to the database, you'
                                          ' need to select a local path for '
                                          'molSimplify to store data files. '
                                          'Would you like to configure it now?',
                                          QMessageBox.Yes, QMessageBox.No)
            if choice == QMessageBox.Yes:
                QMessageBox.information(
                    self.cDBWindow, 'Custom path needed',
                    "Please select a local, user-writable location.")
                new_path = QFileDialog.getExistingDirectory(
                    self.cDBWindow, 'Select a writable directory.')
                if len(new_path) > 0:
                    globs.add_custom_path(new_path)
                    globs.custom_path = new_path
                    copy_to_custom_path()  # this function lives in scripts/io.py
        # instdir = globs.installdir
        if self.lFFb.getState() and self.lFFa.getState():
            ffopt = 'BA'
        elif self.lFFb.getState() and not self.lFFa.getState():
            ffopt = 'B'
        elif not self.lFFb.getState() and self.lFFa.getState():
            ffopt = 'A'
        else:
            ffopt = 'N'
        if smictg == 'all':
            smictg = 'build functionalize'
        if smimol == '' or sminame == '':
            _ = QMessageBox.warning(
                self.DBWindow, 'Error', 'Please specify molecule and name!')
        else:
            # add to database
            emsg = ''
            if 'ligand' in coption:
                emsg = addtoldb(smimol, sminame, smident,
                                smicat, smigrps, smictg, ffopt)
            elif 'core' in coption:
                emsg = addtocdb(smimol, sminame, smicat)
            elif 'bind' in coption:
                emsg = addtobdb(smimol, sminame)
            if emsg:
                _ = QMessageBox.warning(self.DBWindow, 'Error', emsg)
            else:
                _ = QMessageBox.information(
                    self.DBWindow, 'Add', 'Successfully added to the database!')
        self.update_ligands_binds()
        # callback for removal button, removes from db

    def qdelDB(self):
        globs = globalvars()
        if not globs.custom_path:  # this will be false unless set
            choice = QMessageBox.question(self.cDBWindow, 'Custom path needed',
                                          'In order to add to the database, you'
                                          ' need to select a local path for '
                                          'molSimplify to store data files. '
                                          'Would you like to configure it now?',
                                          QMessageBox.Yes, QMessageBox.No)
            if choice == QMessageBox.Yes:
                QMessageBox.information(
                    self.cDBWindow, 'Custom path needed',
                    "Please select a local, user-writable location.")
                new_path = QFileDialog.getExistingDirectory(
                    self.cDBWindow, 'Select a writable directory.')
                if len(new_path) > 0:
                    globs.add_custom_path(new_path)
                    copy_to_custom_path()  # this function lives in scripts/io.py

        coption = self.DBsel.currentIndex()
        sminame = self.etDBname.text()
        if sminame == '':
            _ = QMessageBox.warning(
                self.DBWindow, 'Error', 'Please specify name!')
        else:
            # remove from database
            emsg = removefromDB(sminame, int(coption))
            if emsg:
                _ = QMessageBox.warning(self.DBWindow, 'Error', emsg)
            else:
                _ = QMessageBox.information(
                    self.DBWindow, 'Remove', 'Removed from the database!')
        # db type change

    def dbchange(self):
        ci = self.DBsel.currentIndex()
        if ci == 1:
            self.rtDBsmident.setDisabled(False)
            self.DBdent.setDisabled(False)
            self.rtDBsmicat.setDisabled(False)
            self.etDBsmicat.setDisabled(False)
            self.etDBgrps.setDisabled(False)
            self.etDBgrps.setDisabled(False)
            self.lFFb.setDisabled(False)
            self.lFFa.setDisabled(False)
        elif ci == 0:
            self.rtDBsmident.setDisabled(True)
            self.DBdent.setDisabled(True)
            self.rtDBsmicat.setDisabled(False)
            self.etDBsmicat.setDisabled(False)
            self.etDBgrps.setDisabled(True)
            self.etDBgrps.setDisabled(True)
            self.lFFb.setDisabled(True)
            self.lFFa.setDisabled(True)
        else:
            self.rtDBsmident.setDisabled(True)
            self.DBdent.setDisabled(True)
            self.rtDBsmicat.setDisabled(True)
            self.etDBsmicat.setDisabled(True)
            self.etDBgrps.setDisabled(True)
            self.etDBgrps.setDisabled(True)
            self.lFFb.setDisabled(True)
            self.lFFa.setDisabled(True)
        # perform post-processing

    def postprocGUI(self):
        rdir = self.etpdir.text()
        if rdir[-1] == '/':
            rdir = rdir[:-1]
        defaultparams = ['main.py', '-i', rdir+'/postproc.inp']
        grabguivarsP(self)
        self.pWindow.close()
        self.sgrid.setCurrentWidget(self.iWind)
        emsg = startgen(defaultparams, True, self)
        if not emsg:
            QMessageBox.information(
                self.pWindow, 'DONE', 'Your results are ready..')
        # ############################
        # ### Add geometry window ####
        # ############################
        # load molecule from file

    def qgeomload(self):
        name = QFileDialog.getOpenFileName(
            self.DBWindow, 'Open File', '.', "Molecule files *.xyz (*.xyz)")
        if name[0] != '':
            self.etgf.setText(os.path.relpath(name[0]))
        # load png from file

    def qgeomload2(self):
        name = QFileDialog.getOpenFileName(
            self.DBWindow, 'Open File', '.', "Geometry image files *.png (*.png)")
        if name[0] != '':
            self.etgfc.setText(os.path.relpath(name[0]))
        # enable add new geometry

    def addgeom(self):
        self.geWindow.setWindowModality(2)
        self.geWindow.show()
        # callback for addition button, adds to database

    def qaddg(self):
        globs = globalvars()
        if not globs.custom_path:  # this will be false unless set
            choice = QMessageBox.question(self.cDBWindow, 'Custom path needed',
                                          'In order to add to the database, you'
                                          ' need to select a local path for '
                                          'molSimplify to store data files. '
                                          'Would you like to configure it now?',
                                          QMessageBox.Yes, QMessageBox.No)
            if choice == QMessageBox.Yes:
                QMessageBox.information(
                    self.cDBWindow, 'Custom path needed', "Please select a local, user-writable location.")
                new_path = QFileDialog.getExistingDirectory(
                    self.cDBWindow, 'Select a writable directory.')
                if len(new_path) > 0:
                    globs.add_custom_path(new_path)
                    copy_to_custom_path()  # this funciton lives in scripts/io.py
        gname = self.etgname.text().lower()
        gname = gname.replace(' ', '_')
        gfile = self.etgf.text()
        if gname == '' or gfile == '':
            _ = QMessageBox.warning(
                self.geWindow, 'Error', 'Please specify name and xyz file!')
        elif not glob.glob(gfile):
            _ = QMessageBox.warning(
                self.geWindow, 'Error', 'XYZ file '+gfile+' does not exist!')
        else:
            globs = globalvars()
            if globs.custom_path:
                f = globs.custom_path + "/Data/coordinations.dict"
            else:
                f = str(resource_files("molSimplify").joinpath("Data/coordinations.dict"))
            with open(f, 'r') as fl:
                s = fl.read().splitlines()
            if gname.lower() in s:
                _ = QMessageBox.warning(
                    self.geWindow, 'Add', 'Coordination '+gname+' already exists.')
            return

            gshort = self.etgshort.text()
            cfile = self.etgfc.text()
            # get geometry from xyz file
            with open(gfile, 'r') as f:
                snew = f.read().splitlines()
            dent = int(snew[0])-1
            xyzl = ''
            for ii in range(0, dent+1):
                li = [_f for _f in re.split(' |\t', snew[2+ii]) if _f]
                xyzl += li[1]+' '+li[2]+' '+li[3]+'\n'
            # write new entry in coordinations.dict
            s.append(str(dent)+': '+gname+' '+gshort)
            ssort = [_f for _f in list(sorted(s[1:])) if _f]
            globs = globalvars()
            if globs.custom_path:
                f = globs.custom_path + "/Data/coordinations.dict"
            else:
                print('Error, please set custom file path')
                QMessageBox.warning(
                    self.wmain, 'Error, please set custom file path (dd to database)')
            with open(f, 'w') as fl:
                fl.write(s[0]+'\n')
                for ss in ssort:
                    fl.write(ss+'\n')
            # write new backbone file
            if globs.custom_path:
                f = globs.custom_path + "/Data/"
            else:
                print('Error, please set custom file path')
                QMessageBox.warning(
                    self.wmain, 'Error, please set custom file path (dd to database)')
            with open(f+gshort+'.dat', 'w') as fl:
                fl.write(xyzl)
            # write png file
            if glob.glob(cfile):
                f = str(resource_files("molSimplify").joinpath("icons/geoms/"))
                shutil.copy2(cfile, f+gshort+'.png')
            choice = QMessageBox.information(
                self.geWindow, 'Add', 'Successfully added to the database!')
        self.matchgeomcoord()

        # callback for removal button, removes from db
    def qdelg(self):
        globs = globalvars()
        if not globs.custom_path:  # this will be false unless set
            choice = QMessageBox.question(self.cDBWindow, 'Custom path needed',
                                          'In order to add to the database, you'
                                          ' need to select a local path for '
                                          'molSimplify to store data files. '
                                          'Would you like to configure it now?',
                                          QMessageBox.Yes, QMessageBox.No)
            if choice == QMessageBox.Yes:
                QMessageBox.information(
                    self.cDBWindow, 'Custom path needed', "Please select a local, user-writable location.")
                new_path = QFileDialog.getExistingDirectory(
                    self.cDBWindow, 'Select a writable directory.')
                if len(new_path) > 0:
                    globs.add_custom_path(new_path)
                    copy_to_custom_path()  # this funciton lives in scripts/io.py

        globs = globalvars()
        gname = self.etgname.text().lower()
        gname = gname.replace(' ', '_')
        gshort = self.etgshort.text()
        if gname == '' and gshort == '':
            _ = QMessageBox.warning(
                self.geWindow, 'Error', 'Please specify geometry name!')
        else:
            if globs.custom_path:
                f = globs.custom_path + "/Data/coordinations.dict"
            else:
                f = str(resource_files("molSimplify").joinpath("Data/coordinations.dict"))
            with open(f, 'r') as fl:
                s = fl.read()
            if gname.lower() not in s and gshort.lower() not in s:
                _ = QMessageBox.warning(
                    self.geWindow, 'Remove', 'Coordination '+gname+' does not exist.')
                return
            # remove entry from coordinations.dict
            snew = ''
            srem = ''
            for ss in s.splitlines():
                sl = [_f for _f in ss.split(' ') if _f]
                if gname.lower() != sl[1] and gshort.lower() != sl[2]:
                    snew += ss+'\n'
                else:
                    srem = [_f for _f in ss.split(' ')[-1] if _f]
            if globs.custom_path:
                f = globs.custom_path + "/Data/coordinations.dict"
            else:
                f = str(resource_files("molSimplify").joinpath("Data/coordinations.dict"))
            with open(f, 'w') as fl:
                fl.write(snew)
            # remove file
            if glob.glob(str(globs.custom_path)+'/Data/'+srem+'.dat'):
                os.remove(str(globs.custom_path)+'/Data/'+srem+'.dat')
            _ = QMessageBox.information(
                self.geWindow, 'Remove', 'Successfully removed from the database!')
        self.matchgeomcoord()
    # ############################
    # ## Chem Database window ####
    # ############################
    # load molecule from file

    def qcDBload(self):
        name = QFileDialog.getOpenFileName(
            self.DBWindow, 'Open File', '.', "Molecule files *.xyz, *.mol *.sdf *.smi (*.xyz *.mol *.sdf *.smi)")
        if name[0] != '':
            self.etcDBsmi.setText(os.path.relpath(name[0]))
    # enable add to database interface

    def searchDBW(self):
        globs = globalvars()
        self.cDBWindow.setWindowModality(2)
        self.cDBWindow.show()
        writef = False
        # instdir = globs.installdir
        mwfn = globs.multiwfn
        cdbdir = globs.chemdbdir
        if not os.path.isdir(globs.chemdbdir):
            choice = QMessageBox.question(self.cDBWindow, 'Database setup',
                                          'It looks like the Chemical Database'
                                          ' directory is not configured or '
                                          'does not exist. Would you like '
                                          'to configure it now?',
                                          QMessageBox.Yes, QMessageBox.No)
            if choice == QMessageBox.Yes:
                QMessageBox.information(
                    self.cDBWindow, 'Chem DB', "Please select the directory containing chemical databases.")
                cdbdir = QFileDialog.getExistingDirectory(
                    self.cDBWindow, 'Select the directory containing chemical databases.')
                if len(cdbdir) > 0:
                    writef = True
        if writef:
            with open(globs.homedir+'/.molSimplify', 'w') as f:
                #  f.write("INSTALLDIR="+instdir+'\n')
                f.write("CHEMDBDIR="+cdbdir+'\n')
                if len(mwfn) > 1:
                    f.write("MULTIWFN="+mwfn[0]+'\n')
            # get existing databases
            globsnew = globalvars()
            dbdir = globsnew.chemdbdir
            dbs0 = glob.glob(dbdir+"/*.sdf")
            dbs1 = [d.rsplit('/', 1)[-1] for d in dbs0]
            dbs = [d.split('.', 1)[0] for d in dbs1]
            for d in dbs:
                self.cDBsel.addItem(d)
        # callback for database search

    def qaddcDB(self):
        """Collects all the info and passes it to molSimplify."""
        rdir = self.etrdir.text()
        if rdir[-1] == '/':
            rdir = rdir[:-1]
        # create running dir if not existing
        if not os.path.isdir(rdir):
            os.mkdir(rdir)
        grabdbguivars(self)
        defaultparams = ['main.py', '-i', rdir+'/dbinput.inp']
        emsg = startgen(defaultparams, True, self)
        if not emsg:
            QMessageBox.information(
                self.cDBWindow, 'DONE', 'Search is done..')
        self.sgrid.setCurrentWidget(self.cDBWindow)
        # db type change

    def cdbchange(self):
        ci = self.DBsel.currentIndex()
        if (ci == 1):
            self.rtDBsmident.setDisabled(False)
            self.DBdent.setDisabled(False)
            self.rtDBsmicat.setDisabled(False)
            self.etDBsmicat.setDisabled(False)
        else:
            self.rtDBsmident.setDisabled(True)
            self.DBdent.setDisabled(True)
            self.rtDBsmicat.setDisabled(True)
            self.etDBsmicat.setDisabled(True)
        # ###################
        # ## Main window ####
        # ###################
        # run generation

    def runGUI(self):
        # collects all the info and passes it to molSimplify #
        rdir = self.etrdir.text()
        if rdir[-1] == '/':
            rdir = rdir[:-1]
        # create running dir if not existing
        if not os.path.isdir(rdir):
            try:
                os.mkdir(rdir)
            except FileExistsError:
                emsg = 'Directory '+rdir+' could not be created. Check your input.\n'
                QMessageBox.critical(self.wmain, 'Problem', emsg)
                return
        # check for extra molecule
        if self.chkM.getState() and self.etbind.currentText() == '':
            emsg = ('You checked the extra molecule box but specified\n'
                    'no extra molecule. No extra molecule will be generated.\n')
            QMessageBox.warning(self.wmain, 'Warning', emsg)
        # get parameters
        grabguivars(self)
        defaultparams = ['main.py', '-i', rdir+'/geninput.inp']
        self.iWind.hide()
        self.sgrid.setCurrentWidget(self.iWind)
        self.iWind.show()
        msgBox = QMessageBox()
        if self.randomchk.getState():
            msgBox.setText(
                "Random generation initiated. This process might take some time.")
            msgBox.setIcon(1)
            msgBox.setInformativeText('Please be patient. OK?')
            msgBox.setWindowTitle('Running..')
            msgBox.exec_()
        # do the generation
        emsg = startgen(defaultparams, True, self)
        if not emsg:
            QMessageBox.information(
                self.wmain, 'Done', 'Structure generation terminated successfully!')
        else:
            QMessageBox.warning(self.wmain, 'Problem', emsg)
        # list molecules

    def listmols(self):
        coreslist = getcores()
        liglist = getligs()
        bindlist = getbinds()
        msg = 'Available cores in the database:\n'+coreslist
        msg += '\n\nAvailable ligands in the database:\n'+liglist
        msg += '\n\nAvailable extra molecules in the database:\n'+bindlist+'\n'
        QMessageBox.information(self.wmain, 'List of molecules', msg)
        # draw ligands

    def drawligs_svg(self):
        """Draw the ligands entered into the GUI as SVGs in a pop-up window"""
        # collects all the info and passes it to molSimplify #
        args = grabguivars(self)
        # processes ligand arguments
        if len(args['-lig']) < 1:
            mQDialogWarn('Warning', 'No ligands are specified.')
            return False
        else:
            rows = self.lgrid.rowCount()
            # Clear existing widgets in layout
            if rows > 1:
                for i in reversed(list(range(self.lgrid.count()))):
                    self.lgrid.itemAt(i).widget().setParent(None)

            args['-lig'] = args['-lig'].replace(' ', '')
            lls = args['-lig'].split(',')
            liglist = []
            # check if multiple ligands in .smi file
            for li in lls:
                if '.smi' in li:
                    with open(li, 'r') as f:
                        smis = [_f for _f in f.read().splitlines() if _f]
                    liglist += smis
                else:
                    liglist.append(li)
            # Get known ligand dictionaries
            licores = getlicores()
            simpleligs = getslicores()
            ligs = []
            for li in liglist:
                # check in simple dictionary
                if li in list(simpleligs.keys()):
                    li = simpleligs[li][0]
                if isinstance(li, str):
                    ll = unicodedata.normalize(
                        'NFKD', li).encode('ascii', 'ignore')
                else:
                    ll = li
                # load ligands as molecules
                lig, emsg = lig_load(ll, licores)
                if emsg:
                    mQDialogWarn('Error', emsg)
                else:
                    ligs.append(lig.OBMol)
            if len(ligs) == 0:
                return
            else:
                for i, pmol in enumerate(ligs):
                    # use openbabel to convert to labeled SVG
                    obConversion = openbabel.OBConversion()
                    obConversion.SetOutFormat("svg")
                    obConversion.AddOption("i", obConversion.OUTOPTIONS, "")
                    # return the svg with atom labels as a string
                    svgstr = obConversion.WriteString(pmol)
                    # unpacked nested svg as in pybel._repr_svg_
                    namespace = "http://www.w3.org/2000/svg"
                    ET.register_namespace("", namespace)
                    tree = ET.fromstring(svgstr)
                    svg = tree.find(
                        "{{{ns}}}g/{{{ns}}}svg".format(ns=namespace))
                    newsvg = ET.tostring(svg).decode("utf-8")
                    # write unpacked svg to temp file
                    filedes, filename = tempfile.mkstemp()
                    with open(filename, "w") as svg_file:
                        svg_file.write(newsvg)
                    # Read the temp file into an SvgWidget
                    self.svgwidget = mSvgWidget(filename)
                    # Add the svg to the window
                    self.lgrid.addWidget(self.svgwidget, 0, i)
                    # Cleanup temp files
                    os.close(filedes)
                    os.remove(filename)
                # Build close button
                self.lwindow.setWindowTitle('Ligands 2D')
                self.lwclose = QPushButton('Close')
                self.lwclose.clicked.connect(self.qcloseligs)
                # Add close button to window
                self.lgrid.addWidget(self.lwclose, 1, 0)
                self.lwindow.show()

    def viewgeom(self):
        # get geometry
        geom = self.dcoordg.currentText()
        gfname = str(resource_files("molSimplify").joinpath(f"icons/geoms/{geom}.png"))
        if glob.glob(gfname):
            rows = self.lgrid.rowCount()
            # Clear existing widgets in layout
            if rows > 1:
                for i in reversed(list(range(self.lgrid.count()))):
                    self.lgrid.itemAt(i).widget().setParent(None)
            self.c1p = mQPixmap(gfname)
            self.lgrid.addWidget(self.c1p, 0, 0)
            # button for closing window
            ctip = 'Close current window'
            self.lwclose = mQPushButton('Close', ctip, 14)
            self.lwclose.clicked.connect(self.qcloseligs)
            self.lgrid.addWidget(self.lwclose, 1, 0)
            self.lwindow.setWindowTitle('Geometry:'+self.dcoordg.currentText())
            self.lwindow.show()
            center(self.lwindow)
        else:
            mQDialogWarn('Warning', 'No file '+gfname+' exists..')
    # draw results from db search

    def drawres(self):
        # collects all the info and passes it to molSimplify #
        rdir = self.etrdir.text()
        if rdir[-1] == '/':
            rdir = rdir[:-1]
        # creat running dir if not existing
        if not os.path.isdir(rdir):
            os.mkdir(rdir)
        outf = rdir+'/'+self.etcDBoutf.text()
        outf = outf.replace(' ', '')+'.smi'
        if not glob.glob(outf):
            mQDialogWarn('Warning', 'No database results file in '+rdir)
            return False
        else:
            lls = [outf]
            liglist = []
            # check if multiple ligands in .smi file
            for li in lls:
                if '.smi' in li:
                    with open(li, 'r') as f:
                        smis = [_f for _f in f.read().splitlines() if _f]
                    liglist += smis
                else:
                    liglist.append(li)
            licores = getlicores()
            ligs = []
            for li in liglist:
                if isinstance(li, str):
                    ll = unicodedata.normalize(
                        'NFKD', li).encode('ascii', 'ignore')
                else:
                    ll = li

                lig, emsg = lig_load(ll, licores)
                if not emsg:
                    ligs.append(lig.OBMol)
            if len(ligs) == 0:
                return
            else:
                for i, pmol in enumerate(ligs):
                    # use openbabel to convert to labeled SVG
                    obConversion = openbabel.OBConversion()
                    obConversion.SetOutFormat("svg")
                    obConversion.AddOption("i", obConversion.OUTOPTIONS, "")
                    # return the svg with atom labels as a string
                    svgstr = obConversion.WriteString(pmol)
                    # unpacked nested svg as in pybel._repr_svg_
                    namespace = "http://www.w3.org/2000/svg"
                    ET.register_namespace("", namespace)
                    tree = ET.fromstring(svgstr)
                    svg = tree.find(
                        "{{{ns}}}g/{{{ns}}}svg".format(ns=namespace))
                    newsvg = ET.tostring(svg).decode("utf-8")
                    # write unpacked svg to temp file
                    filedes, filename = tempfile.mkstemp()
                    with open(filename, "w") as svg_file:
                        svg_file.write(newsvg)
                    # Read the temp file into an SvgWidget
                    self.svgwidget = mSvgWidget(filename)
                    # Add the svg to the window
                    self.lgrid.addWidget(self.svgwidget, 0, i)
                    # Cleanup temp files
                    os.close(filedes)
                    os.remove(filename)
                # Build close button
                self.lwindow.setWindowTitle('Ligands 2D')
                self.lwclose = QPushButton('Close')
                self.lwclose.clicked.connect(self.qcloseligs)
                # Add close button to window
                self.lgrid.addWidget(self.lwclose, 1, 0)
                self.lwindow.show()
    # enable random input

    def enablerandom(self):
        if self.randomchk.isChecked():
            self.rtrgen.setDisabled(False)
            self.etrgen.setDisabled(False)
            self.rtlignum.setDisabled(False)
            self.etlignum.setDisabled(False)
            self.rtliggrp.setDisabled(False)
            self.etliggrp.setDisabled(False)
            self.rtligctg.setDisabled(False)
            self.etligctg.setDisabled(False)
            self.randkHs.setDisabled(False)
            if len(self.etrgen.text()) == 0:
                self.etrgen.setText('1')
        else:
            self.rtrgen.setDisabled(True)
            self.etrgen.setDisabled(True)
            self.rtlignum.setDisabled(True)
            self.etlignum.setDisabled(True)
            self.rtliggrp.setDisabled(True)
            self.etliggrp.setDisabled(True)
            self.rtligctg.setDisabled(True)
            self.etligctg.setDisabled(True)
            self.randkHs.setDisabled(True)
    # generate all enable FF input

    def disableffinput(self):
        if self.chkgenall.isChecked():
            self.dff.setDisabled(True)
            self.dffba.setDisabled(True)
            self.chkFF.state = True
            self.chkFF.setDisabled(True)
        else:
            self.dff.setDisabled(False)
            self.dffba.setDisabled(False)
            self.chkFF.setDisabled(False)
            self.enableffinput()
    # enable FF input

    def enableffinput(self):
        if self.chkFF.isChecked():
            self.chkFF.state = False
            self.dff.setDisabled(False)
            self.dffba.setDisabled(False)
        else:
            self.dff.setDisabled(True)
            self.dffba.setDisabled(True)
            self.chkFF.state = True
    # load file for editing

    def qloadinput(self):
        name = QFileDialog.getOpenFileName(self.wmain, 'Open File')[0]
        if name != '':
            loadfrominputfile(self, name)
    # load directory

    def dirload(self):
        name = QFileDialog.getExistingDirectory(self.wmain, 'Select Directory')
        if len(name) > 0 and name[0] != '':
            self.etrdir.setText(name)
    # save as input file

    def qsaveinput(self, gui):
        name = QFileDialog.getSaveFileName(
            self.wmain, 'Save as..', '.', "Input files * (*)")[0]
        if name != '':
            varsg = grabguivars(self)
            writeinputc(varsg, name)
    # show help menu

    def qshowhelp(self):
        globs = globalvars()
        QMessageBox.information(self.wmain, 'About', globs.about)

    def getscreensize(self):
        screenShape = QDesktopWidget().screenGeometry()
        width = int(screenShape.width())
        height = int(screenShape.height())
        return [width, height]
    # slider changed value

    def sliderChanged(self, val):
        self.distper.setText('Distort:'+str(val)+'%')
    # match index with coordination

    def matchgeomcoord(self):
        # get current index
        dc = self.dcoord.currentIndex()
        coords, geomnames, geomshorts, geomgroups = getgeoms()
        qcav = geomgroups
        ctip = geomnames
        # empty the box
        for i in range(0, self.dcoordg.count()):
            self.dcoordg.removeItem(0)
        qc = qcav[dc]
        # add to box
        for i, t in enumerate(qc):
            # File f is never actually used? RM 2022/02/17
            f = str(resource_files("molSimplify").joinpath(f"icons/geoms/{t}.png"))  # noqa F841
            self.dcoordg.addItem(QIcon(t), t)
        self.dcoordg.setIconSize(QSize(60, 60))
        # set default geometry
        self.dcoordg.setCurrentIndex(0)
        # get global index
        elem = [i for i, s in enumerate(
            geomshorts) if s in self.dcoordg.currentText()]
        ct = ''
        for ii in range(elem[0], elem[0]+len(geomgroups[dc])):
            ct += ctip[ii]+', '
        ct = ct[:-2]
        # set correct tooltip
        self.dcoordg.setToolTip(ct)
    # enable extra molecule input

    def enableemol(self):
        if self.chkM.isChecked():
            self.chkM.state = False
            self.txtamol.setDisabled(False)
            self.rtbind.setDisabled(False)
            self.etbind.setDisabled(False)
            self.rtbsmi.setDisabled(False)
            self.etbsmi.setDisabled(False)
            self.rtnbind.setDisabled(False)
            self.etnbind.setDisabled(False)
            if self.etnbind.text() == '':
                self.etnbind.setText('1')
            self.rtplace.setDisabled(False)
            self.etplacemin.setDisabled(False)
            self.etplacemax.setDisabled(False)
            self.dmolp.setDisabled(False)
            self.rtchbind.setDisabled(False)
            self.etchbind.setDisabled(False)
            self.rtplacea.setDisabled(False)
            self.etplacephi.setDisabled(False)
            self.etplacetheta.setDisabled(False)
            # self.rtmaskbind.setDisabled(False)
            self.etmaskbind.setDisabled(False)
            self.chsep.setDisabled(False)
        else:
            self.txtamol.setDisabled(True)
            self.rtbind.setDisabled(True)
            self.etbind.setDisabled(True)
            self.rtbsmi.setDisabled(True)
            self.etbsmi.setDisabled(True)
            self.rtnbind.setDisabled(True)
            self.etnbind.setDisabled(True)
            self.rtplace.setDisabled(True)
            self.etplacemin.setDisabled(True)
            self.etplacemax.setDisabled(True)
            self.dmolp.setDisabled(True)
            self.chkM.state = True
            self.rtchbind.setDisabled(True)
            self.etchbind.setDisabled(True)
            self.rtplacea.setDisabled(True)
            self.etplacephi.setDisabled(True)
            self.etplacetheta.setDisabled(True)
            # self.rtmaskbind.setDisabled(True)
            self.etmaskbind.setDisabled(True)
            self.chsep.setDisabled(True)
    # ####################
    # ### QEt/g input ####
    # ####################
    # callback for QE input

    def qcinput(self):
        if self.chch.getState():
            self.etqcgch.setDisabled(True)
            self.etqctch.setDisabled(True)
            self.etqcQch.setDisabled(True)
        if self.qcode.currentIndex() == 0:  # generate terachem input
            self.qctWindow.setWindowModality(2)
            self.qctWindow.show()
        elif self.qcode.currentIndex() == 1:  # generate GAMESS input
            self.qcgWindow.setWindowModality(2)
            self.qcgWindow.show()
        elif self.qcode.currentIndex() == 2:  # generate Qchem input
            self.qcQWindow.setWindowModality(2)
            self.qcQWindow.show()
    # make default button callback

    def jobdef(self):
        grabguivarsjob(self)
        QMessageBox.information(
            self.wmain, 'Done', 'The current settings are the default ones now.')

    def qctdef(self):
        grabguivarstc(self)
        QMessageBox.information(
            self.wmain, 'Done', 'The current settings are the default ones now.')

    def qcgdef(self):
        grabguivarsgam(self)
        QMessageBox.information(
            self.wmain, 'Done', 'The current settings are the default ones now.')

    def qcqdef(self):
        grabguivarsqch(self)
        QMessageBox.information(
            self.wmain, 'Done', 'The current settings are the default ones now.')

    def qcgload(self):
        name = QFileDialog.getOpenFileName(
            self.qctWindow, 'Open File', '.', "GAMESS input files")
        if name[0] != '':
            with open(name[0], 'r') as f:
                self.qcgWindow.molf = f.read()
    # enable QE input

    def enableqeinput(self):
        if self.chkI.isChecked():
            self.chkI.state = False
            self.qcode.setDisabled(False)
            self.butQc.setDisabled(False)
            self.chch.setDisabled(False)
        else:
            self.qcode.setDisabled(True)
            self.butQc.setDisabled(True)
            self.chch.setDisabled(True)
            self.chkI.state = True
    # ########################
    # ### jobscript input ####
    # ########################
    # enable Jobscript input

    def enablejinput(self):
        if self.chkJ.isChecked():
            self.chkJ.state = False
            self.scheduler.setDisabled(False)
            self.butJob.setDisabled(False)
        else:
            self.scheduler.setDisabled(True)
            self.butJob.setDisabled(True)
            self.chkJ.state = True

    def jobenable(self):
        self.jWindow.setWindowModality(2)
        self.jWindow.show()
    # load file

    def jload(self):
        name = QFileDialog.getOpenFileName(
            self.qctWindow, 'Open File', '.', "Jobscript files")
        if name[0] != '':
            with open(name[0], 'r') as f:
                self.jWindow.molf = f.read()
    # ##############################
    # ### post-processing input ####
    # ##############################
    # enable Post processing input

    def setupp(self):
        self.pWindow.setWindowModality(2)
        self.pWindow.show()
        # check if Multiwfn exists
        globs = globalvars()
        cwd = os.getcwd()
        os.chdir(globs.homedir)
        inputtxt = '0\n0\n'
        with open('input1', 'w') as f:
            f.write(inputtxt)
        writef = False
        mwfn = globs.multiwfn
        cdbdir = globs.chemdbdir
        if not os.path.isfile(globs.multiwfn[1:-1]):
            choice = QMessageBox.question(self.pWindow, 'Multiwfn setup',
                                          'It looks like the Multiwfn '
                                          'executable is not configured '
                                          'or does not exist. Would you '
                                          'like to configure it now?',
                                          QMessageBox.Yes, QMessageBox.No)
            if choice == QMessageBox.Yes:
                QMessageBox.information(
                    self.pWindow, 'Multiwfn', "Please select the Multiwfn executable.")
                mwfn = QFileDialog.getOpenFileName(
                    self.pWindow, 'Select the Multiwfn executable.', '.')
                if len(mwfn[0]) > 1:
                    writef = True
        if writef:
            with open(globs.homedir+'/.molSimplify', 'w') as f:
                # f.write("INSTALLDIR="+instdir+'\n')
                if len(cdbdir) > 0:
                    f.write("CHEMDBDIR="+cdbdir+'\n')
                if len(mwfn) > 0:
                    f.write("MULTIWFN="+mwfn[0]+"\n")
        newglobs = globalvars()
        com = newglobs.multiwfn
        tt = mybash(com + '< input1')
        os.remove('input1')
        if 'Multifunctional Wavefunction Analyzer' not in tt:
            self.pch.setDisabled(True)
            self.pwfnav.setDisabled(True)
            self.pcub.setDisabled(True)
            self.pdeloc.setDisabled(True)
        os.chdir(cwd)
    # load directory

    def pdload(self):
        name = QFileDialog.getExistingDirectory(
            self.pWindow, 'Select Directory')
        if len(name) > 0 and name[0] != '':
            self.etpdir.setText(name)
    # ##############################
    # ##### general callbacks ######
    # ##############################
    # ## exit application ###

    def qexit(self):
        choice = QMessageBox.question(self.wmain, 'Exit', 'Are you sure you want to quit?',
                                      QMessageBox.Yes, QMessageBox.No)
        if choice == QMessageBox.Yes:
            sys.exit()
        else:
            pass
    # hide current widget #

    def qhide(self):
        self.hide()
    # close ligands window

    def qcloseligs(self):
        self.lwindow.hide()
    # return to main window #

    def qretmain(self):
        # hide all windows
        self.qctWindow.hide()
        self.qcgWindow.hide()
        self.qcQWindow.hide()
        self.pWindow.hide()
        self.jWindow.hide()
        self.DBWindow.hide()
        self.cDBWindow.hide()
        self.geWindow.hide()
