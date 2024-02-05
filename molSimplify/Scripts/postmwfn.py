# @file postmwfn.py
#  Post-processes Multiwfn output
#
#  Written by Tim Ioannidis for HJK Group
#
#  Dpt of Chemical Engineering, MIT

import math
import glob
import os

from numpy import cross, dot
from math import sqrt

from molSimplify.Classes.globalvars import (globalvars,
                                            mybash)


# Gets distance between points
#  @param R1 Point 1
#  @param R2 Point 2
#  @return Distance


def distance(R1, R2):
    d = 0.0
    d += pow(R1[0]-R2[0], 2)
    d += pow(R1[1]-R2[1], 2)
    d += pow(R1[2]-R2[2], 2)
    return sqrt(d)

# Gets subset of string between two substrings
#  @param s String to be split
#  @param first First substring
#  @param last Last substring
#  @return Subset of string between substrings


def find_between(s, first, last):
    # returns string between first and last substrings
    s = s.split(first, 1)
    if len(s) > 1:
        s = s[1].split(last, 1)
        return s[0]
    else:
        return ""


# List of metals
metals = {'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
          'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Pt': 78, 'Au': 79, 'In': 49}

# Gets distance of Cartesian point to origin
#  @param v Cartesian point
#  @return Distance


def radial(v):
    rsq = v[1]*v[1]
    rsq += v[2]*v[2]
    rsq += v[3]*v[3]
    return math.sqrt(rsq)

# Integrate grid
#  @param den Array of points
#  @param dV Volume of differential element
#  @return Integrated quantity


def calc(den, dV):
    integral = 0  # initialize integral
    # loop over all integrating cubes
    for dval in den:
        # Sum values
        integral += dval[0]
    integral = integral*dV
    return integral

# Calculate spreads and averages of density and ELF
#  @param den Array of densities
#  @param ELF Array of ELF values
#  @param totel Total number of electrons
#  @param dV Volume of differential element
#  @return Spreads and averages (mean, stdev, skewness)


def spreadcalc(den, ELF, totel, dV):
    Rm = 0  # mean distance = av(r*rho)
    Sp = 0  # spread = standard dev
    Sk = 0  # skewness
    Em = 0  # mean ELF
    ESD = 0  # stdv ELF
    ESk = 0  # skewness ELF
    for i, dval in enumerate(den):
        # sum values
        r = radial(dval)
        Rm += dval[0]*r
    Rm = dV*Rm/totel  # normalize
    # calculate spread
    for dval in den:
        r = radial(dval)
        Sp += dval[0]*math.pow(r-Rm, 2)
    Sp = math.sqrt(dV*Sp/totel)
    # calculate skewness
    for dval in den:
        r = radial(dval)
        Sk += dval[0]*math.pow((r-Rm)/Sp, 3)
    Sk = dV*Sk/totel
    for i, dval in enumerate(den):
        # sum values
        Em += dval[0]*ELF[i][0]
    Em = dV*Em/totel  # normalize
    # calculate spread
    for i, dval in enumerate(den):
        ESD += dval[0]*math.pow(ELF[i][0]-Em, 2)
    ESD = math.sqrt(dV*ESD/totel)
    # calculate skewness
    for i, dval in enumerate(den):
        ESk += dval[0]*math.pow((ELF[i][0]-Em)/ESD, 3)
    ESk = dV*ESk/totel
    return Rm, Sp, Sk, Em, ESD, ESk

# Calculate HELP function from cube
#  @param den Array of points
#  @param ELF ELF values
#  @param dV Volume of differential element
#  @return Integrated HELP


def calcHELP(den, ELF, dV):
    if len(den) != len(ELF):
        exit('ELF and den cube files have not the same dimensions..Exiting.')
    integral = 0  # initialize integral
    # loop over all integrating cubes
    for i, dval in enumerate(den):
        if (dval[0] > 0.001 and ELF[i][0] >= 0.5):
            # Summ values
            integral += dval[0]
    integral = integral*dV
    return integral

# Parse cube file
#  @param cubef Name of cube file
#  @return Array of data, volume of differential element


def parsecube(cubef):
    # open and read cube file
    with open(cubef, 'r') as f:
        s = f.read().splitlines()
    params = []
    for i in range(2, 6):
        ss = [_f for _f in s[i].split(' ') if _f]
        for sss in ss:
            params.append(sss)
    # get parameters
    natoms = int(params[0])
    n1 = int(params[4])
    n2 = int(params[8])
    n3 = int(params[12])
    # origin and direction vectors
    r_origin = [float(params[1]), float(params[2]),
                float(params[3])]       # UNITS in bohr
    r1 = [float(params[5]), float(params[6]), float(params[7])]
    r2 = [float(params[9]), float(params[10]), float(params[11])]
    r3 = [float(params[13]), float(params[14]), float(params[15])]
    # Calculate differential vector using |(a x b)*c|
    dV = dot(cross(r1, r2), r3)
    # loop over atoms
    atoms = [[0 for x in range(5)] for x in range(natoms)]
    # check for metal (WITH ECPs YOU NEED TO CORRECT ATOMIC NUMBER inside MOLDEN!)
    mfound = False
    rmetal = [atoms[0][2], atoms[0][3], atoms[0][4]]  # metal coordinates
    for i in range(6, 6+natoms):
        ss = [_f for _f in s[i].split(' ') if _f]
        for j, sss in enumerate(ss):
            # each atom contains ATOMIC_NO, Charge, X, Y, Z
            atoms[i-6][j] = float(sss)
        if (21 <= atoms[i-6][0] <= 30) or (39 <= atoms[i-6][0] <= 48) or (72 <= atoms[i-6][0] <= 80):
            rmetal = [atoms[i-6][2], atoms[i-6][3],
                      atoms[i-6][4]]  # metal coordinates
            mfound = True
            break
    # Check for ECPs
    if not mfound:
        for i in range(6, 6+natoms):
            ss = [_f for _f in s[i].split(' ') if _f]
            for j, sss in enumerate(ss):
                # each atom contains ATOMIC_NO, Charge, X, Y, Z
                atoms[i-6][j] = float(sss)
            if (11 <= atoms[i-6][0] <= 20):
                rmetal = [atoms[i-6][2], atoms[i-6][3],
                          atoms[i-6][4]]  # metal coordinates
                break
    # read density values
    den = [[0.0 for x in range(5)] for x in range(n1*n2*n3)]
    offset = 6+natoms
    summing = 0
    fmax = 0
    # get wfc values
    sv = ''
    for li in range(offset, len(s)):
        sv += '  '.join(s[li].split('\n'))
    val = [_f for _f in sv.split(' ') if _f]
    for i in range(0, n1):
        for j in range(0, n2):
            for k in range(0, n3):
                idx = i*(n2*n3)+j*n3 + k  # global index
                X = r_origin[0] + r1[0]*i + r2[0]*j + r3[0]*k  # X value
                Y = r_origin[1] + r1[1]*i + r2[1]*j + r3[1]*k  # Y value
                Z = r_origin[2] + r1[2]*i + r2[2]*j + r3[2]*k  # Z value
                # each element contains density, x, y, z
                if(abs(float(val[idx])) < 1.0E-10):
                    den[idx][0] = 0.0
                else:
                    den[idx][0] = float(val[idx])
                    summing += den[idx][0]
                if abs(den[idx][0]) > fmax:
                    fmax = abs(den[idx][0])
                den[idx][1] = X - rmetal[0]  # with reference to Metal
                den[idx][2] = Y - rmetal[1]
                den[idx][3] = Z - rmetal[2]
    return den, dV

# Calculate wavefunction properties
#  @param denf Density cube file
#  @param elff ELF cube file
#  @return Integrated HELP, number of electrons, spreads


def wfncalc(denf, elff):
    [den, dV] = parsecube(denf)
    totel = calc(den, dV)
    [ELF, dV] = parsecube(elff)
    svars = spreadcalc(den, ELF, totel, dV)
    HELPval = calcHELP(den, ELF, dV)
    return HELPval, totel, svars

# Computes alpha and beta densities from total and spin density cubes
#  @param cubefTOT Density cube file
#  @param cubefSPIN Spin density cube file


def cubespin(cubefTOT, cubefSPIN):
    # open and read cube files
    [denT, n, pre] = readden(cubefTOT)
    [denS, n, pre1] = readden(cubefSPIN)
    # compute the difference
    for i in range(0, len(denT)):
        t = denT[i][0]
        s = denS[i][0]
        denT[i][0] = (t + s)/2.0
        denS[i][0] = (t - s)/2.0
    denA = denT  # assign by reference
    denB = denS  # assign by reference
    j = 0
    with open('denalpha.cub', 'w') as f:
        f.write(pre)
        for i in range(0, len(denA)):
            f.write("%.5e" % denA[i][0] + '  ')
            j += 1
            if (j % 6 == 0 or (i+1) % n == 0):
                f.write('\n')
                j = 0
    with open('denbeta.cub', 'w') as f:
        f.write(pre)
        for i in range(0, len(denB)):
            f.write("%.5e" % denB[i][0] + '  ')
            j += 1
            if (j % 6 == 0 or (i+1) % n == 0):
                f.write('\n')
                j = 0

# Parse density cube file
#  @param inputf Density cube file
#  @return Density array, parameters, additional info


def readden(inputf):
    with open(inputf, 'r') as f:
        s = f.read().splitlines()
    params = []
    for i in range(2, 6):
        ss = [_f for _f in s[i].split(' ') if _f]
        for sss in ss:
            params.append(sss)
    # get parameters
    natoms = int(params[0])
    n1 = int(params[4])
    n2 = int(params[8])
    n3 = int(params[12])
    # origin and direction vectors
    r_origin = [float(params[1]), float(params[2]),
                float(params[3])]		# UNITS in bohr
    r1 = [float(params[5]), float(params[6]), float(params[7])]
    r2 = [float(params[9]), float(params[10]), float(params[11])]
    r3 = [float(params[13]), float(params[14]), float(params[15])]
    atoms = [[0 for x in range(5)] for x in range(natoms)]
    # check for metal (WITH ECPs YOU NEED TO CORRECT ATOMIC NUMBER in MOLDEN!)
    for i in range(6, 6+natoms):
        ss = [_f for _f in s[i].split(' ') if _f]
        for j, sss in enumerate(ss):
            # each atom contains ATOMIC_NO, Charge, X, Y, Z
            atoms[i-6][j] = float(sss)
    # read density values
    den = [[0.0 for x in range(5)] for x in range(n1*n2*n3)]
    offset = 6+natoms
    summing = 0
    fmax = 0
    pre = ''
    for i in range(0, offset):
        pre += s[i] + '\n'
    # get wfc values
    sv = ''
    for li in range(offset, len(s)):
        sv += '  '.join(s[li].split('\n'))
    val = [_f for _f in sv.split(' ') if _f]
    for i in range(0, n1):
        for j in range(0, n2):
            for k in range(0, n3):
                idx = i*(n2*n3)+j*n3 + k  # global index
                X = r_origin[0] + r1[0]*i + r2[0]*j + r3[0]*k  # X value
                Y = r_origin[1] + r1[1]*i + r2[1]*j + r3[1]*k  # Y value
                Z = r_origin[2] + r1[2]*i + r2[2]*j + r3[2]*k  # Z value
                # each element contains density, x, y, z
                den[idx][0] = float(val[idx])
                summing += den[idx][0]
                if abs(den[idx][0]) > fmax:
                    fmax = abs(den[idx][0])
                den[idx][1] = X
                den[idx][2] = Y
                den[idx][3] = Z
    return den, n1, pre

# Use Multiwfn to generate cube files from molden files
#  @param molf Input molden file
#  @param folder Folder containing runs
#  @param gui GUI flag
#  @param flog Log filename


def getcubes(molf, folder, gui, flog):
    # get Multiwfn
    globs = globalvars()
    Multiwfn = globs.multiwfn
    # analyze results
    flog.write(
        "##################### Generating cube files ######################\n")
    print("##################### Generating cube files ######################\n")
    # loop over folders
    for numi, resf in enumerate(molf):
        resd = os.path.relpath(resf, folder)
        resd = resd.split('.molden')[0]
        resd = resd.replace('/', '_')
        cubedir = folder+'/Cube_files/'
        flog.write('Processing '+resf+'\n')
        print(('Processing ', resf))
        if gui:
            gui.iWtxt.setText('Processing '+resf+'\n'+gui.iWtxt.toPlainText())
            gui.app.processEvents()
        #################################################
        ### generate density cube ###
        inputtxt = '5\n1\n3\n2\n'
        with open('input1', 'w') as f:
            f.write(inputtxt)
        com = Multiwfn+' ' + "'" + resf + "'"+' < input1'
        if not glob.glob(cubedir+resd+'-density.cub'):
            _ = mybash(com)
            os.remove('input1')
            if glob.glob('density.cub'):
                os.rename('density.cub', cubedir+resd+'-density.cub')
        #################################################
        inputtxt = '5\n9\n3\n2\n'  # generate ELF
        with open('input1', 'w') as f:
            f.write(inputtxt)
        com = Multiwfn+' ' + "'" + resf + "'"+' < input1'
        if not glob.glob(cubedir+resd+'-ELF.cub'):
            _ = mybash(com)
            os.remove('input1')
            if glob.glob('ELF.cub'):
                os.rename('ELF.cub', cubedir+resd+'-ELF.cub')
        #################################################
        inputtxt = '5\n5\n3\n2\n'  # generate spin density
        with open('input1', 'w') as f:
            f.write(inputtxt)
        com = Multiwfn+' ' + "'" + resf + "'" + ' < input1'
        if not glob.glob(cubedir+resd+'-spindensity.cub'):
            _ = mybash(com)
            os.remove('input1')
            if glob.glob('spindensity.cub'):
                os.rename('spindensity.cub', cubedir+resd+'-spindensity.cub')
        #################################################
        if not glob.glob(cubedir+resd+'-denalpha.cub'):
            cubespin(cubedir+resd+'-density.cub', cubedir+resd +
                     '-spindensity.cub')  # generate spin densities
            if glob.glob('denalpha.cub'):
                os.rename('denalpha.cub', cubedir+resd+'-denalpha.cub')
                os.rename('denbeta.cub', cubedir+resd+'-denbeta.cub')

# Calculate wavefunction properties from molden file
#  @param molf Input molden file
#  @param folder Folder containing runs
#  @param gui GUI flag
#  @param flog Log filename


def getwfnprops(molf, folder, gui, flog):
    # analyze results
    flog.write(
        "##################### Getting wavefunction properties ######################\n")
    print("##################### Getting wavefunction properties ######################\n")
    header = "\nFolder                                                  HELP  (%)        Rav       RSD       RSk      ELFav     ELFSD      ELFSk\n"
    header += "-----------------------------------------------------------------------------------------------------------------------------------------\n"
    # loop over folders
    txt = []
    for numi, resf in enumerate(molf):
        resd = os.path.relpath(resf, folder)
        resd = resd.split('.molden')[0]
        resdp = resd
        resd = resd.replace('/', '_')
        #################################################
        print(('Processing ', resf))
        flog.write('Processing '+resf+'\n')
        if gui:
            gui.iWtxt.setText('Processing '+resd+'\n'+gui.iWtxt.toPlainText())
            gui.app.processEvents()
        wfndir = folder+'/Wfn_files/'
        outfile1 = wfndir+resd+'-HELP.txt'
        outfile2 = wfndir+resd+'-denELF.txt'
        if not glob.glob(outfile1) or not glob.glob(outfile2):
            cubedir = folder + '/Cube_files/'
            denf = cubedir+resd+'-density.cub'
            elff = cubedir+resd+'-ELF.cub'
            HELPpop, totel, svars = wfncalc(denf, elff)  # calculate HELP
            with open(outfile1, 'w') as f:
                f.write('HELP: '+str(HELPpop)+' '+str(100*HELPpop/totel)+' %\n')
            with open(outfile2, 'w') as f:
                f.write('Rav: '+str(svars[0])+' RSD: ' +
                        str(svars[1])+' RSk: '+str(svars[2]))
                f.write(' ELFav: '+str(svars[3])+' ELF_SD: ' +
                        str(svars[4])+' ELF_Sk: '+str(svars[5])+'\n')
        #################################################
        with open(outfile1, 'r') as f:
            sf = f.read()
        with open(outfile2, 'r') as f:
            sff = f.read()
        HELPpop = float([_f for _f in sf.split(None) if _f][1])
        HELPper = float([_f for _f in sf.split(None) if _f][-2])
        sf = [_f for _f in sff.split(None) if _f]
        svars = [float(sf[1]), float(sf[3]), float(sf[5]),
                 float(sf[7]), float(sf[9]), float(sf[11])]
        sr = []
        for i, ss in enumerate(svars):
            sr.append("{:10.5f}".format(ss))
        txt.append(resdp.ljust(50)+"{:10.3f}".format(HELPpop).ljust(10)+' ('+"{:4.2f}".format(HELPper)+'%)' +
                   sr[0].ljust(10)+sr[1].ljust(10)+sr[2].ljust(10)+sr[3].ljust(10)+sr[4].ljust(11)+sr[5].ljust(10)+'\n')
    text = sorted(txt)
    with open(folder+'/wfnprops.txt', 'w') as f:
        f.write(header+''.join(text))

# Calculate charges from molden file
#  @param molf Input molden file
#  @param folder Folder containing runs
#  @param gui GUI flag
#  @param flog Log filename


def getcharges(molf, folder, gui, flog):
    # get Multiwfn
    globs = globalvars()
    Multiwfn = globs.multiwfn
    # analyze results
    header = "\nFolder                                                    Hirshfeld    VDD     Mulliken\n"
    header += "-----------------------------------------------------------------------------------------\n"
    txt = []
    for numi, resf in enumerate(molf):
        resd = os.path.relpath(resf, folder)
        resd = resd.split('.molden')[0]
        resdp = resd
        resd = resd.replace('/', '_')
        # get metal index in molden file
        with open(resf, 'r') as f:
            ss = f.read()
        ss = find_between(ss, 'Atoms', 'GTO')
        ss = ss.splitlines()
        midx = 0  # default
        for sl in ss:
            sll = sl.split(None)
            for met in metals:
                if len(sll) > 1 and met in sll[0]:
                    midx = int(sll[1])-1
                    break
        #################################################
        outfile1 = folder+'/Charge_files/'+resd+'-chH.txt'
        print(('Processing ', resdp))
        flog.write('Processing '+resdp+'\n')
        if gui:
            gui.iWtxt.setText('Processing '+resf+'\n'+gui.iWtxt.toPlainText())
            gui.app.processEvents()
        # Run multiwfn
        if not glob.glob(outfile1):
            inputtxt = '7\n1\n1\n'  # Hirschfeld
            with open('input1', 'w') as f:
                f.write(inputtxt)
            com = Multiwfn+' ' + "'" + resf + "'" + " < input1 > '"+outfile1+"'"
            _ = mybash(com)
            os.remove('input1')
        with open(outfile1, 'r') as f:
            ss = f.read()
        s = find_between(ss, ' of atom     1', 'Atomic dipole')
        hirsch = 'NA'
        if len(s.splitlines()) > midx+1:
            hirscht = s.splitlines()[midx]
            hirschll = hirscht.split(None)
            if len(hirschll) > 2:
                hirsch = "{:5.3f}".format(float(hirschll[-1]))
        #################################################
        outfile2 = folder+'/Charge_files/'+resd+'-chV.txt'
        if not glob.glob(outfile2):
            inputtxt = '7\n2\n1\n'  # VDD
            with open('input1', 'w') as f:
                f.write(inputtxt)
            com = Multiwfn+' '+"'" + resf + "'" + " < input1 > '"+outfile2+"'"
            _ = mybash(com)
            os.remove('input1')
        with open(outfile2, 'r') as f:
            ss = f.read()
        s = find_between(ss, 'VDD charge', 'dipole')
        vdd = 'NA'
        if len(s.splitlines()) > midx+1:
            vddt = s.splitlines()[midx]
            vddll = vddt.split(None)
            if len(vddll) > 2:
                vdd = "{:5.3f}".format(float(vddll[-1]))
        #################################################
        outfile3 = folder+'/Charge_files/'+resd+'-chM.txt'
        if not glob.glob(outfile3):
            inputtxt = '7\n5\n1\n'  # Mulliken
            with open('input1', 'w') as f:
                f.write(inputtxt)
            com = Multiwfn+' ' + "'" + resf + "'" + " < input1 > '"+outfile3+"'"
            _ = mybash(com)
            os.remove('input1')
        with open(outfile3, 'r') as f:
            ss = f.read()
        s = find_between(ss, 'Population of atoms', 'Total net')
        mull = 'NA'
        if len(s.splitlines()) > midx+2:
            mullt = s.splitlines()[midx+2]
            mulll = mullt.split(None)
            if len(mulll) > 2:
                mull = "{:5.3f}".format(float(mulll[-1]))
        txt.append(resdp.ljust(60)+hirsch.ljust(10) +
                   vdd.ljust(10)+mull.ljust(10)+'\n')
    text = sorted(txt)
    with open(folder+'/charges.txt', 'w') as f:
        f.write(header+''.join(text))

# Calculate delocalization indices from molden file
#  @param molf Input molden file
#  @param folder Folder containing runs
#  @param gui GUI flag
#  @param flog Log filename


def deloc(molf, folder, gui, flog):
    # get multiwfn exec
    globs = globalvars()
    Multiwfn = globs.multiwfn
    # get results files
    header = "\nFile                                                                No Attr     Loc-indx     Tot-Deloc\n"
    header += "-------------------------------------------------------------------------------------------------------\n"
    text = []
    indexin = []
    mindices = []
    skipm = False
    # loop over molden files
    for numi, resf in enumerate(molf):
        indxst = ''
        resd = resf.rsplit('/', 1)[0]
        moln = resf.rsplit('/', 1)[-1]
        moln = moln.split('.molden')[0]
        resd = os.path.relpath(resd, folder)
        resd = resd.replace('/', '_')+'_'+moln
        print(resd)
        outfile = folder+'/Deloc_files/'+resd+'-deloc.txt'
        print(('Processing  '+resd+' and writing output to '+outfile))
        flog.write('Processing  '+resd+'\n')
        if gui:
            gui.iWtxt.setText('Processing '+resd+'\n'+gui.iWtxt.toPlainText())
            gui.app.processEvents()
        # get coordinates of metal
        with open(resf, 'r') as f:
            sm = f.read().splitlines()
        # get indices of heavy elements
        found = False
        for met in metals:
            if not found:
                ml = [line for line in sm if met in line]
            if len(ml) > 0 and not found:
                if 'Title' in ml[0] and len(ml) > 1:
                    mlll = ml[1].split(None)
                else:
                    mlll = ml[0].split(None)
                if len(mlll) > 2:
                    found = True
        if len(ml) == 0:
            print(
                'WARNING:No metal found, defaulting to 1st atom for relative properties..')
            skipm = True
            ml = [sm[4]]
            mlll = [_f for _f in ml[0].split(None) if _f]
        mindex = mlll[1]
        mindices.append(mindex)
        mcoords = [float(mlll[3]), float(mlll[4]), float(mlll[5])]
        parse, skipc = False, False
        # Run multiwfn
        if not glob.glob(outfile):
            inputtxt = '17\n1\n1\n2\n4\n'
            with open('input0', 'w') as f:
                f.write(inputtxt)
            com = Multiwfn+' ' + "'" + resf + "'" + " < input0 > '"+outfile+"'"
            tt = mybash(com)
            print(tt)
            # check if seg fault
            skipc = False
            if 'Segmentation' in tt or 'core dumped' in tt:
                skipc = True
            # read outputfile and check
            with open(outfile, 'r') as f:
                ssf = f.read()
            # if error redo
            if ('Note: There are attractors having very low ' in ssf or 'Hint' in ssf):
                inputtxt = '17\n1\n1\n2\n3\n4\n'
                with open('input0', 'w') as f:
                    f.write(inputtxt)
                tt = mybash(com)
                print(tt)
            os.remove('input0')
        # check if good
        with open(outfile, 'r') as f:
            ssf = f.read()
        parse = False
        if 'Total localization' in ssf:
            parse = True
        ################
        # parse output #
        ################
        if not skipc and parse:
            with open(outfile, 'r') as f:
                s = f.read()
            # get basins
            if ('The attractors after clustering' in s):
                st = find_between(s, 'Index      Average', 'The number')
                sst = [_f for _f in st.splitlines()[1:] if _f]
            else:
                st = find_between(s, 'Attractor', 'Detecting')
                sst = [_f for _f in st.splitlines()[1:] if _f]
            basidx = []  # list with metal attractors
            attlist = []  # List with neighboring attractors
            att_thrsd = 2.4  # Angstrom
            att_range = 1.0  # Angstrom
            for line in sst:
                xyzcoords = line.split(None)
                if (len(xyzcoords) > 3):
                    xyzc = [float(xyzcoords[1]), float(
                        xyzcoords[2]), float(xyzcoords[3])]
                    if distance(mcoords, xyzc) < att_range:
                        basidx.append(xyzcoords[0])
                        indxst += mindex+'\n'
                    elif (distance(mcoords, xyzc) > att_range and distance(mcoords, xyzc) < att_thrsd):
                        attlist.append(int(xyzcoords[0]))
            # get total number of basins
            if not skipm:
                dd = int(find_between(
                    s, 'matrix for basin', '...').split(None)[-1])
                # get delocalization matrix
                sdel = find_between(
                    s, 'Total delocalization index matrix', 'Total localization')
                sdeloc = sdel.splitlines()[1:]
                deloc = []
                for bidx in basidx:
                    [a, b] = divmod(int(bidx)-1, 5)
                    ll = sdeloc[a*(dd+1)+1:(a+1)*(dd+1)]
                    for ii, l in enumerate(ll):
                        if (ii+1 in attlist):
                            lf = [_f for _f in l.split(None) if _f]
                            deloc.append(float(lf[b+1]))
                    # get localization index
                s = find_between(s, 'Total localization index:', '==')
                loc = []
                for bidx in basidx:
                    ss = s.split(bidx+':')[-1]
                    ss = ss.split(None)
                    loc.append(float(ss[0]))
                if (len(attlist) == 0):
                    tdeloc = 0.0
                else:
                    tdeloc = sum(deloc)/len(attlist)
                tloc = sum(loc)
                tt = resd.ljust(70)+str(len(basidx)).ljust(6)+"{:10.3f}".format(
                    tloc).ljust(14)+"{:10.3f}".format(tdeloc).ljust(10)+'\n'
                text.append(tt)
            indexin.append(indxst)
    # sort alphabetically and print
    text = sorted(text)
    with open(folder+'/deloc_res.txt', 'w') as f:
        f.write(header+''.join(text))
    print("\n##################### Deloc indices are ready ######################\n")
