# Written by Tim Ioannidis for HJK Group
# Dpt of Chemical Engineering, MIT

####################################################
#########   Defines class of global    #############
########   variables that are shared   #############
##########    within the program       #############
####################################################

import os, inspect, glob, platform, sys, subprocess
from math import sqrt 

# atoms dictionary contains atomic mass, atomic number, covalent radius, data from http://www.webelements.com/ (last accessed May 13th 2015)
amassdict = {'X':(1.0,0,0.77),'H':(1.0079,1,0.37),'B':(10.83,5,0.85),'C':(12.0107,6,0.77),'N':(14.0067,7,0.75),'O':(15.9994,8,0.73),
             'F':(18.9984,9,0.71),'Na':(22.99,11,1.55),'Mg':(24.30,12,1.39),'Al':(26.98,13,1.26),'Si':(28.08,14,1.16),
             'P':(30.9738,15,1.06),'S':(32.065,16,1.02),'Cl':(35.453,17,0.99),'K':(39.10,19,1.96),'Ca':(40.08,20,1.71),
             'Sc':(44.96,21,1.7),'Ti':(47.867,22,1.36),'V':(50.94,23,1.22),'Cr':(51.9961,24,1.27),'Mn':(54.938,25,1.39),
             'Fe':(55.84526,26,1.25),'Ni':(58.4934,28,1.21),'Co':(58.9332,27,1.26),'Cu':(63.546,29,1.38),'Zn':(65.39,30,1.31),
             'Ga':(69.72,31,1.24),'Ge':(72.63,32,1.21),'As':(74.92,33,1.21),'Se':(78.96,34,1.16),'Br':(79.904,35,1.14),
             'Rb':(85.47,37,2.10),'Sr':(87.62,38,1.85),'Y':(88.91,39,1.63),'Zr':(91.22,40,1.54),'Nb':(92.91,41,1.47),
             'Mo':(95.96,42,1.38),'Ru':(101.1,44,1.25),'Rh':(102.9,45,1.25),'Pd':(106.4,46,1.20),'Ag':(107.9,47,1.28),
             'In':(114.8,49,1.42),'Sn':(118.7,50,1.40),'I':(126.9,53,1.33),'Pt':(195.1,78,1.23),'Au':(197.0,79,1.24)}

# list of metals
metalslist = ['Sc','SC','scandium','Ti','TI','titanium','V','vanadium','Cr','CR','chromium','Mn','MN','manganese','Fe','FE','iron','Co','CO',
            'cobalt','Ni','NI','nickel','Cu','CU','copper','Zn','ZN','zinc','Y','yttrium','Zr','ZR','zirconium','Nb','NB','niobium','Mo','MO',
            'molybdenum','Tc','TC','technetium','Ru','RU','ruthenium','Rh','RH','rhodium','Pd','PD','palladium','Ag','AG','silver','Cd','CD',
            'cadmium','Lu','LU','lutetium','Hf','HF','hafnium','Ta','TA','tantalum','W','tungsten','Re','RE','rhenium','Os','OS','osmium',
            'Ir','IR','iridium','Pt','PT','platinum','Au','AU','gold','Hg','HG','mercury']

# list of elements sorted by atomic number
elementsbynum=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca',
                    'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
                    'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe',
		    'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf',
		    'Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th',
		    'Pa','U','Np','Pu', 'Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh',
		    'Hs','Mt','Ds','Rg','Cn','Uut','Fl','Uup','Lv','Uus','Uuo']
########################################
### module for running bash commands ###
########################################
def mybash(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = []
    while True:
        line = p.stdout.readline()
        stdout.append(line)        
        if line == '' and p.poll() != None:
            break
    return ''.join(stdout)

class globalvars:
    def __init__(self):
        ###### PROGRAM NAME ######
        self.PROGRAM = 'molSimplify'
        ###### About message #####
        s = 'molSimplify v1.0\n\nFreely distributed under the GNU GPL license.\n\n'
        s += 'Copyright 2016 Kulik Lab @ MIT\n\n'
        s += 'Developed by: Efthymios Ioannidis (timis@mit.edu)\n\n'
        s += 'Contributions by:\n\tHeather J. Kulik (hjkulik@mit.edu)\n'
        s += '\tTerry Gani (terryg@mit.edu)\n'
        s += '\n slab builder extension by JP Janet (jpjanet@mit.edu)\n'
        self.about = s
        ###### GET INFORMATION ######
        runfromcmd, Linux, OSX = False, False, False
        ### check if running through commandline ###
        if sys.stdin.isatty():
            # running through command line
            runfromcmd = True
        else:
            runfromcmd = False
        ### get running os ###
        if platform.system().lower() in 'linux':
            Linux = True
        elif platform.system().lower() in 'darwin':
            OSX = True
        self.osx = OSX
        ### get cwd
        cfile = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
        cdir2 = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
        cdir = cdir2.rsplit('/',1)[0]
        cdir2 = cdir
        homedir = os.path.expanduser("~")
        # create default molSimplify for mac
        if OSX and not glob.glob(homedir+'/.'+self.PROGRAM) and not runfromcmd:
            txt = 'INSTALLDIR=/Applications/'+self.PROGRAM+'.app/Contents/Resources\n'
            f = open(homedir+'/.'+self.PROGRAM,'w')
            f.write(txt)
            f.close()
        self.chemdbdir = ''
        self.multiwfn = ''
        ###### check for ~/.molSimplify ######
        if glob.glob(homedir+'/.'+self.PROGRAM):
            f = open(homedir+'/.'+self.PROGRAM,'r')
            s = filter(None,f.read().splitlines())
            d = dict()
            for ss in s:
                sp = filter(None,ss.split('='))
                d[sp[0]] = sp[1]
            if 'INSTALLDIR' in d.keys():
                self.installdir = d['INSTALLDIR']
            else:
                self.installdir = cdir
            if 'CHEMDBDIR' in d.keys():
                self.chemdbdir = d['CHEMDBDIR']
            if 'MULTIWFN' in d.keys():
                self.multiwfn = "'"+d['MULTIWFN']+"'"
        else:
            self.installdir = cdir
        # global settings
        self.homedir = homedir
        self.nosmiles = 0 # number of smiles ligands
        self.rundir = homedir+'/Runs/'# Jobs directory
        self.generated = 0 
        self.debug = True # additional output for debuggin
    def amass(self):
        return amassdict
    def metals(self):
        return metalslist
    def elementsbynum(self):
        return elementsbynum
