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
	     'Tc':(98.9,43,1.56),'Cd':(112.4,48,1.48),'La':(138.9,57,1.69),'Hf':(178.5,72,1.50),'Ta':(180.9,73,1.38),
             'W':(183.8,74,1.46),'Re':(186.2,75,1.59),'Os':(190.2,76,1.28),'Ir':(192.2,77,1.37),'Hg':(200.6,80,1.49),
             'In':(114.8,49,1.42),'Sn':(118.7,50,1.40),'I':(126.9,53,1.33),'Pt':(195.1,78,1.23),'Au':(197.0,79,1.24)}

# list of metals
metalslist = ['Sc','SC','scandium','Ti','TI','titanium','V','vanadium','Cr','CR','chromium','Mn','MN','manganese','Fe','FE','iron','Co','CO',
            'cobalt','Ni','NI','nickel','Cu','CU','copper','Zn','ZN','zinc','Y','yttrium','Zr','ZR','zirconium','Nb','NB','niobium','Mo','MO',
            'molybdenum','Tc','TC','technetium','Ru','RU','ruthenium','Rh','RH','rhodium','Pd','PD','palladium','Ag','AG','silver','Cd','CD',
            'cadmium','La','LA','lanthanum','Hf','HF','hafnium','Ta','TA','tantalum','W','tungsten','Re','RE','rhenium','Os','OS','osmium',
            'Ir','IR','iridium','Pt','PT','platinum','Au','AU','gold','Hg','HG','mercury']

# list of transition metals and d-electron count
mtlsdlist = {'sc':1,'ti':2,'v':3,'cr':4,'mn':5,'fe':6,'ni':7,'co':8,'cu':9,'zn':10,'y':1,'zr':2,'nb':3,
             'mo':4,'tc':5,'ru':6,'rh':7,'pd':8,'ag':9,'cd':10,'hf':1,'ta':2,'w':3,'re':4,'os':5,'ir':6,
             'pt':8,'au':9,'hg':10}

# list of default spins for each d-electron count (make this metal/oxidation state specific)
defaultspins = {0:'1',1:'2',2:'3',3:'4',4:'5',5:'6',6:'5',7:'4',8:'3',9:'2',10:'1'}

# list of elements sorted by atomic number
elementsbynum=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca',
                    'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
                    'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe',
		    'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf',
		    'Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th',
		    'Pa','U','Np','Pu', 'Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh',
		    'Hs','Mt','Ds','Rg','Cn','Uut','Fl','Uup','Lv','Uus','Uuo']

## Electronegativity (Pauling) by atom symbol
endict =     { "H" : 2.20, "Li": 0.98, "Be": 1.57, "B" : 2.04, "C" : 2.55, "N" : 3.04, "O" : 3.44,
     "F" : 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P" : 2.19, "S" : 2.58,
     "Cl": 3.16, "K" : 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V" : 1.63, "Cr": 1.66,
    "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65,  "Ga": 1.81,
    "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Mo": 2.16, "Tc": 2.10, "Rh": 2.28,
    "Pd": 2.20, "Ag": 1.93,"Cd": 1.69, "In": 1.78, "Sb": 2.05, "I":  2.66, "Cs": 0.79, 
    "Y":1.22, "Zr":1.33, "Nb":1.60, "Ru":2.20, "La":1.10, "Hf":1.30, "Ta":1.50, "W":2.36, "Re":1.90}

# roman numerals
romans={'I':'1','II':'2','III':'3','IV':'4','V':'5','VI':'6','VII':'7','VIII':'8'}

# backbone combinations
bbcombs_mononuc = dict()
bbcombs_mononuc['one'] = [[1]]
bbcombs_mononuc['li'] = [[1],[2]]
bbcombs_mononuc['oct'] = [[1,2,3,4,5,6], # 6-dentate
				 [1,2,3,4,5],[1,2,3,4,6],[1,2,3,5,6],[1,2,4,5,6], # 5-dentate
				 [1,3,4,5,6],[2,3,4,5,6], # 5-dentate
				 [1,2,3,4],[2,5,4,6],[1,5,3,6], # 4-dentate
				 [1,2,3],[1,4,2],[1,4,3],[1,5,3],[1,6,3],[2,3,4], # 3-dentate
				 [2,5,4],[2,6,4],[5,4,6],[5,1,6],[5,2,6],[5,3,6], # 3-dentate
				 [1,2],[1,4],[1,5],[1,6],[2,3],[2,5], # 2-dentate
				 [2,6],[3,5],[3,6],[4,5],[4,6],[3,4], # 2-dentate
				 [1],[2],[3],[4],[5],[6]] # 1-dentate
bbcombs_mononuc['pbp'] = [[1,2,3,4,5,6],[1,2,3,4,6], # 6/5-dentate
				  [1,2,3,5], # 4-dentate
				  [1,2,3],[1,2,4],[2,1,5],[3,1,6],[5,6,3],[2,6,5], # 3-dentate
				  [1,2],[2,3],[3,4],[4,5],[1,7],[2,6],[5,7],[3,6], # 2-dentate
				  [1],[2],[3],[4],[5],[6],[7]] # 1-dentate
bbcombs_mononuc['spy'] = [[1,2,3,4,5],[1,2,3,4],[1,2,3],[2,3,4],[3,4,1],[4,1,2],
				 [1,2],[1,4],[2,3],[3,4],[4,5],[2,5],[3,5],[1,5],[1],[2],[3],[4],[5]]
bbcombs_mononuc['sqp'] = [[1,4,2,3],[1,2,3],[2,3,4],[3,4,1],[4,1,2],[1,2],[1,4],[2,3],[3,4],
				  [1],[2],[3],[4]]
bbcombs_mononuc['tbp'] = [[1,2,3,4,5],[1,3,4,5],[3,2,4],[4,5,3],[5,1,3],[4,5],[5,3],[3,4],
				 [1,4],[1,5],[1,3],[2,4],[2,5],[2,3],[1],[2],[3],[4],[5]]
bbcombs_mononuc['thd'] = [[1,2,3,4],[3,2,4],[2,4,1],[4,1,3],[2,4],[4,3],[3,2],[1,3],[1,4],[2,4],[1],[2],[3],[4]]
bbcombs_mononuc['tpl'] = [[1,2,3],[1,2],[2,3],[1,3],[1],[2],[3]]
bbcombs_mononuc['tpr'] = [[1,2,3,4,5,6],[1,2,3,4,5],[1,2,5,4],[5,2,3,6],[1,4,6,3],[1,2,3],[3,6,5],
				 [2,3],[2,5],[5,6],[6,4],[4,1],[1],[2],[3],[4],[5],[6]]

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
        s = '\nmolSimplify v1.3.3x\nFreely distributed under the GNU GPL license.\n'
        s += 'Copyright 2017 Kulik Lab @ MIT\n'
        s += 'Developed by: Efthymios Ioannidis (timis@mit.edu)\n'
        s += 'Contributions by:\n\tHeather J. Kulik (hjkulik@mit.edu)\n'
        s += '\t Terry Gani (terryg@mit.edu)\n'
        s += '\t JP Janet (jpjanet@mit.edu)\n'
        s += 'E. I. Ioannidis, T. Z. H. Gani, H. J. Kulik. J. Comput. Chem. 2016, 37, 2106-2117.\n'
        s += 'J.P. Janet, Q. Zhao, E.I. Ioannidis, H.J. Kulik. Mol. Simul. 2017,43(5-6), 327-345.\n'
        s += 'J.P. Janet, T. Z. H. Gani, A. H. Steeves, E. I. Ioannidis, H. J. Kulik. Ind. Eng. Chem. Res. 2017, 56(17), 4898-4910.\n'
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
	self.custom_path = False
        ###### check for ~/.molSimplify ######
        if glob.glob(homedir+'/.'+self.PROGRAM):
            f = open(homedir+'/.'+self.PROGRAM,'r')
            s = filter(None,f.read().splitlines())
            d = dict()
            for ss in s:
                sp = filter(None,ss.split('='))
                try:
                    d[sp[0]] = sp[1]
                except:
                    pass
                #self.installdir = cdir  ## this is should be unused
            if 'CHEMDBDIR' in d.keys():
                self.chemdbdir = d['CHEMDBDIR']
            if 'MULTIWFN' in d.keys():
                self.multiwfn = "'"+d['MULTIWFN']+"'"
	    if 'CUSTOM_DATA_PATH' in d.keys():
		self.custom_path = d['CUSTOM_DATA_PATH']
        else:
            self.installdir = cdir
            f = open(homedir+'/.'+self.PROGRAM,'w')
            f.write('CHEMDBDIR=\n')
            f.close()

        # global settings
        self.homedir = homedir
        self.nosmiles = 0 # number of smiles ligands
        self.rundir = homedir+'/Runs/'# Jobs directory
        self.generated = 0 
        self.debug = False # additional output for debuggin
        self.remHsmarts = ["O=CN","O=CO","n","N=CN","nN"] # remove Hs from connecting atoms within these SMARTS patterns
        # default geometries for each coordination number if none specified
        self.defaultgeometry = {8:('sqap','square_antiprismatic'),7:('pbp','pentagonal_bipyramidal'),6:('oct','octahedral'),5:('tbp','trigonal bipyramidal'),4:('thd','tetrahedral'),3:('trigonal planar','tpl'),2:('linear','li'),1:('one','one')}
        self.defaultoxstate = {'au':'I','gold':'I','scandium':'III','sc':'III','ti':'IV','titanium':'IV'}
        self.linearbentang = 45 # bent "linear" angle in degrees, e.g., in Fe(III)-superoxo or a bent nitrosyl
    def amass(self):
        return amassdict
    def metals(self):
        return metalslist
    def elementsbynum(self):
        return elementsbynum
    def endict(self):
        return endict
    def add_custom_path(self,path):
    	# this function  is used to
    	# record the cutsom path in 
    	# the ~./molsimplify fil
        homedir = os.path.expanduser("~")
     	f = open(homedir+'/.'+self.PROGRAM,'a')
        f.write('CUSTOM_DATA_PATH='+str(path)+"\n")
        f.close()


