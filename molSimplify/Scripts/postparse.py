# @file postparse.py
#  Post-processes quantum chemistry output files
#
#  Written by Tim Ioannidis for HJK Group
#
#  Dpt of Chemical Engineering, MIT

import time
import os


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

# Parse unrestricted NBO output
#  @param s String containing NBO output
#  @return NBO output including d-band center and d occupancies


def nbo_parser_unrestricted(s):
    res = []  # list of output: [atom],[charge],[average % metal-centered hyb in NBOs],
    # [average %d character in NBOs], [average LV orb occup], [average %d char in LVs]
    # averages with respect to occupations
    ''' Charge parsing '''
    # get nbo charge
    scharge = s.split('Charge')[1].split('* Total *')[0]
    # get atoms and summary of NP analysis
    scharge = scharge.splitlines()[2:-2]
    MET = False
    # find metal charge
    for line in scharge:  # loop over atoms
        if line.split(None)[0] in metals:  # find metal in atoms
            res.append(line.split(None)[0])
            res.append(line.split(None)[2])
            MET = True
    # if no metal is found, default to first atom and get charge
    if len(res) == 0:
        res.append(scharge[0].split(None)[0])
        res.append(scharge[0].split(None)[2])
    if MET:
        ''' NLMO %NBO parsing '''
        # get metal d occupation
        sstxt = find_between(s, 'Natural Electron Configuration', '********')
        smet = find_between(sstxt, res[0], '\n')
        doccup = find_between(smet, '3d(', ')')
        # get metal d-band center
        sstxt = [_f for _f in find_between(
            s, 'NATURAL POPULATIONS', 'Summary').splitlines() if _f]
        dband = 0.0
        for ll in sstxt:
            la = [_f for _f in ll.split(None) if _f]
            if len(la) > 4:
                if(la[1] == res[0] and la[4] == 'Val(' and la[5] == '3d)'):
                    dband += 0.2*float(la[7])
        # split output for alpha beta electrons
        salpha = s.split('Alpha spin orbitals')[1]
        sbeta = salpha.split('Beta  spin orbitals')[1]
        salpha = salpha.split('Beta  spin orbitals')[0]
        # parse alpha
        # output list with [occup,%metal hyb,%d in metal hyb]
        [alphares] = spinnbo(salpha, res[0])
        # anlmo = spinnlmo(salpha,res[0]) # output list with [occup,%metal hyb,%d in metal hyb]
        anlmo = [0.0, 0.0, 0.0]
        # parse beta
        # output list with [occup,%metal hyb,%d in metal hyb]
        [betares] = spinnbo(sbeta, res[0])
        # bnlmo = spinnlmo(sbeta,res[0]) # output list with [occup,%metal hyb,%d in metal hyb]
        bnlmo = [0.0, 0.0, 0.0]
        # average results from alpha, beta
        hyb = 0.0
        dper = 0.0
        occtot = 0.0
        for entry in alphares:
            hyb += float(entry[0])*0.01*float(entry[1])
            dper += float(entry[0])*1e-4*float(entry[1])*float(entry[2])
            occtot += float(entry[0])
        for entry in betares:
            hyb += float(entry[0])*0.01*float(entry[1])
            dper += float(entry[0])*1e-4*float(entry[1])*float(entry[2])
            occtot += float(entry[0])
        # normalize with respect to total occupation
        if occtot > 0:
            res.append(hyb/occtot)
            res.append(dper/occtot)
        else:
            res.append(0.0)
            res.append(0.0)
        if (anlmo[0]+bnlmo[0]) > 0:
            # combine nlmo data
            nlmotot = (anlmo[1]+bnlmo[1])/(anlmo[0]+bnlmo[0])
            res.append(nlmotot)
        else:
            res.append(0.0)
    else:
        res += [0.0, 0.0, 0.0, 0.0, 0.0]
        doccup = 0.0
        dband = 0.0
    return res, doccup, dband

# Parse restricted NBO output
#  @param s String containing NBO output
#  @return NBO output


def nbo_parser_restricted(s):
    res = []  # list of output: [atom],[charge],[average % metal-centered hyb in NBOs],
    # [average %d character in NBOs]
    # averages with respect to occupations
    ''' Charge parsing '''
    # get nbo charge
    scharge = s.split('Charge')[1].split('* Total *')[0]
    # get atoms and summary of NP analysis
    scharge = scharge.splitlines()[2:-2]
    MET = False
    # find metal charge
    for line in scharge:  # loop over atoms
        if line.split(None)[0] in metals:  # find metal in atoms
            res.append(line.split(None)[0])
            res.append(line.split(None)[2])
            MET = True
    if len(res) == 0:
        res.append(scharge[0].split(None)[0])
        res.append(scharge[0].split(None)[2])
    if MET:
        ''' NLMO %NBO parsing '''
        [totres, tLV] = spinnbo(
            s, res[0])  # output list with [occup,%metal hyb,%d in metal hyb]
        # output list with [occup,%metal hyb,%d in metal hyb]
        nlmores = spinnlmo(s, res[0])
        # average results from alpha, beta
        hyb = 0.0
        dper = 0.0
        occtot = 0.0
        for entry in totres:
            hyb += float(entry[0])*0.01*float(entry[1])
            dper += float(entry[0])*1e-4*float(entry[1])*float(entry[2])
            occtot += float(entry[0])
    else:
        # if no metal do not perform analysis
        hyb = 0.0
        occtot = 1.0
        dper = 0.0
        nlmores = [1.0, 0.0]
    # normalize with respect to total occupation
    res.append(hyb/occtot)
    res.append(dper/occtot)
    res.append(0.01*nlmores[1]/nlmores[0])
    return res

# Get information on orbitals with metal character
#  @param s String containing NBO output
#  @param metal Element name of metal
#  @return Orbital information


def spinnbo(s, metal):
    sNBO = []
    # get molecular orbitals containing metal
    ss = s.split('NATURAL BOND ORBITAL')[1]
    ss2 = ss.split('DIRECTIONALITY')
    if len(ss2) > 1:
        slines = ss[1].splitlines()
    else:
        slines = ss.splitlines()
    for i, line in enumerate(slines):
        if ((metal in line) and ('BD' in line)):
            # get NBO occupation
            occup = find_between(line.split(None)[1], '(', ')')
            # get block of NBO
            no = line.split(None)[0].strip(' ')
            # join rest of text to search
            ttt = ''
            for lt in slines[i:i+20]:
                ttt += lt + '\n'
            # find metal centered hybrid inside NBO block
            sa = find_between(
                ttt, no+' (', str(int(no[0:2])+1)+'. (').splitlines()
            ml = [ll for ll in sa[1:] if metal in ll]  # metal centered hybrid
            if len(ml) > 0:
                perhyb = find_between(ml[0], '(', '%').strip(
                    ' ')  # get % of metal hyb in NBO
                perds = ml[0].rsplit('(', 1)
                perd = perds[-1].split('%)')[0]  # get d orbital character
                # append result
                sNBO.append([occup, perhyb, perd])
    return [sNBO]

# Get unrestricted NLMO information
#  @param s String containing NBO output
#  @param metal Element name of metal
#  @return NLMO information


def spinnlmo(s, metal):
    # get NLMOs containing metal
    ss = s.split('NATURAL LOCALIZED MOLECULAR ORBITAL (NLMO) ANALYSIS')[
        1].splitlines()
    nlmoccup = 0.0
    pernlmo = 0.0
    for line in ss:
        if ((metal in line) and ('BD' in line)):
            # get NBO occupation
            nlmoccup += float(find_between(line.split(None)
                                           [1], '(', ')').strip(' '))
            # get %NBO in NLMO
            pernlmo += 0.01*float(find_between(line, ')', '%').strip(' '))
    nlmo = [nlmoccup, pernlmo]
    return nlmo

# Parse NBO output
#  @param resfiles Files to be post-processed
#  @param folder Folder containing runs
#  @param gui GUI flag
#  @param flog Log file name


def nbopost(resfiles, folder, gui, flog):
    t = time.strftime('%c')
    headern = "Date: " + t + \
        "\nHere are the current results for runs in folder '"+folder+"'\n"
    headern += "\nFolder                                           Metal  MCharge  AvhybNBO  AvDorbNBO   AvNLMO   Doccup    Dband-center\n"
    headern += "----------------------------------------------------------------------------------------------------------------------------------------\n"
    textnbo = []
    for numi, resf in enumerate(resfiles):
        resfp = os.path.relpath(resf, folder)
        resd = resf.rsplit('/', 1)[0]
        resfold = os.path.relpath(resd, folder)
        print(('Processing ', resfp))
        flog.write('Processing '+resfp+'\n')
        if gui:
            gui.iWtxt.setText('Processing '+resfp+'\n'+gui.iWtxt.toPlainText())
            gui.app.processEvents()
        with open(resf) as f:
            s = f.read()
        # split output into QC and nbo parts
        ssp = ' N A T U R A L   A T O M I C   O R B I T A L'
        if ssp in s:
            snbo = s.split(ssp)[-1]  # get nbo output
            # check if unrestricted
            if 'Beta  spin orbitals' in s:
                nbores, doccup, dband = nbo_parser_unrestricted(snbo)
                tt = resfold.ljust(50) + nbores[0].ljust(6)+"{:6.4f}".format(
                    float(nbores[1])).ljust(10)+"{:6.4f}".format(nbores[2]).ljust(10)
                tt += "{:6.4f}".format(nbores[3]).ljust(10) + \
                    "{:6.5f}".format(nbores[4]).ljust(12)
                tt += "{:4.2f}".format(float(doccup)).ljust(12) + \
                    "{:4.2f}".format(dband).ljust(10)+'\n'
            else:
                nbores = nbo_parser_restricted(snbo)
                tt = resfold.ljust(50) + nbores[0].ljust(6)+"{:6.4f}".format(
                    float(nbores[1])).ljust(10)+"{:6.4f}".format(nbores[2]).ljust(10)
                tt += "{:6.4f}".format(nbores[3]).ljust(10)+'\n'
            textnbo.append(tt)
    textnbo = sorted(textnbo)
    with open(folder+'/nbo.txt', 'w') as f:
        f.write(headern+''.join(textnbo))

# Parse terachem output
#  @param resfiles Files to be post-processed
#  @param folder Folder containing runs
#  @param gui GUI flag
#  @param flog Log file name


def terapost(resfiles, folder, gui, flog):
    t = time.strftime('%c')
    flog.write(
        '################## Calculating results summary ##################\n\n')
    header = "Date: " + t + "\nHere are the current results for runs in folder '"+folder+"'\n"
    header += "\nFolder                                            Compound    Method  %HF  Restricted   Optim  Converged  NoSteps   Spin   S^2   Charge    Energy(au)   Time(s)\n"
    header += "------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
    # loop over folders
    text = []
    for numi, resf in enumerate(resfiles):
        resfp = os.path.relpath(resf, folder)
        resd = resf.rsplit('/', 1)[0]
        resfold = os.path.relpath(resd, folder)
        print(('Processing ', resfp))
        flog.write('Processing '+resfp+'\n')
        if gui:
            gui.iWtxt.setText('Processing '+resfp+'\n'+gui.iWtxt.toPlainText())
        with open(resf) as f:
            s = f.read()
        # split output into terachem and nbo parts
        stera = s  # get tera output
        if 'TeraChem' in s:
            ss = stera.splitlines()
            # get simulation parameters
            comp = [line for line in ss if 'XYZ coordinates' in line][0].rsplit(
                None, 1)[-1]
            comp = comp.split('.xyz')[0]  # compound name
            spin = [line for line in ss if 'Spin multiplicity:' in line][0].rsplit(
                None, 1)[-1]  # spin mult
            # SPIN S-SQUARED
            ssq = [_f for _f in [
                line for line in ss if 'SPIN S-SQUARED' in line] if _f]
            if (len(ssq) > 0):
                ssqs = [_f for _f in ssq[-1].split(')') if _f]
                ssq = ssqs[-1].split(None)[-1]
            else:
                ssq = 'NA'
            tcharge = [line for line in ss if 'Total charge:' in line]
            if len(tcharge) > 0:
                tcharge = tcharge[0].rsplit(None, 1)[-1]  # total charge
            else:
                tcharge = 'ERROR'
            UR = [line for line in ss if 'Wavefunction:' in line][0].rsplit(
                None, 1)[-1][0]  # U/R (Un)restricted
            smethod = [line for line in ss if 'Method:' in line][0].split(
                None)  # functional
            method = smethod[1]
            if (smethod[-2] == 'dispersion'):
                method += '-D'
            hfx = [line for line in ss if 'Hartree-Fock exact exchange:' in line]
            if len(hfx) > 0:
                hfx = hfx[0].rsplit(None, 1)[-1]  # HFX
            else:
                hfx = '0.0'
            # optimization?
            optim = [line for line in ss if 'RUNNING GEOMETRY' in line]
            optim = 'Y' if len(optim) > 0 else 'N'
            # get results
            en = [line for line in ss if 'FINAL ENERGY:' in line]  # energy
            en = en[-1].rsplit(None, 2)[-2] if len(en) > 0 else 'NaN'
            if (optim == 'Y'):
                conv = [line for line in ss if 'Converged!' in line]
                conv = 'Y' if len(conv) > 0 else 'N'
                if conv == 'Y':
                    lnosteps = [
                        line for line in ss if 'Number of steps' in line][-1]
                    nosteps = [_f for _f in lnosteps.split(None) if _f][-1]
                else:
                    nosteps = 'NA'
            else:
                conv = 'NA'
                nosteps = 'NA'
            timet = [line for line in ss if 'Total processing time:' in line]
            if (len(timet) > 0):
                tm = timet[0].rsplit(None, 2)[-2]
            else:
                tm = 'NOT DONE'
            # construct string record of results
            tt = resfold.ljust(50)+comp.ljust(12)+method.ljust(7)+("{:3.0f}".format(100*float(hfx))+'%').ljust(
                10)+UR.ljust(10)+optim.ljust(8)+conv.ljust(9)+nosteps.ljust(10)+spin.ljust(5)+ssq.ljust(9)
            tt += tcharge.ljust(6) + \
                "{:10.6f}".format(float(en)).ljust(18)+tm+'\n'
            text.append(tt)
    # sort alphabetically and print
    text = sorted(text)
    if len(text) > 0:
        with open(folder+'/tera-results.txt', 'w') as f:
            f.write(header+''.join(text))

# Parse GAMESS output
#  @param resfiles Files to be post-processed
#  @param folder Folder containing runs
#  @param gui GUI flag
#  @param flog Log file name


def gampost(resfiles, folder, gui, flog):
    t = time.strftime('%c')
    header = "Date: " + t + "\nHere are the current results for runs in folder '"+folder+"'\n"
    header += "\nFolder                                            Method   %MGGA   %LDA  Optim  Converged  NoSteps   S-SQ   Spin   Charge    Energy(au)      Time(MIN)\n"
    header += "--------------------------------------------------------------------------------------------------------------------------------------------------------\n"
    # loop over folders
    text = []
    flog.write(
        '################## Calculating results summary ##################\n\n')
    for numi, resf in enumerate(resfiles):
        error = False
        resfp = os.path.relpath(resf, folder)
        resd = resf.rsplit('/', 1)[0]
        resfold = os.path.relpath(resd, folder)
        if gui:
            gui.iWtxt.setText('Processing '+resfp+'\n'+gui.iWtxt.toPlainText())
            gui.app.processEvents()
        print(('Processing '+resfp))
        flog.write('Processing '+resfp+'\n')
        if len(resfold) > 1:
            resfold = resfold[-2]
        else:
            resfold = resfold[-1]
        with open(resf) as f:
            s = f.read()
        if 'GAMESS' in s:
            sgam = s  # get gamess output
            ''' Parse gamess output '''
            ss = sgam.splitlines()
            # get simulation parameters
            comp = resfold.split('/', 1)[-1]
            spins = [line for line in ss if 'SPIN MULTIPLICITY' in line]
            if len(spins) > 0:
                spin = spins[0].rsplit(None, 1)[-1]  # spin mult
            tcharge = [line for line in ss if 'CHARGE OF MOLECULE' in line]
            if len(tcharge) > 0:
                tcharge = tcharge[0].rsplit(None, 1)[-1]  # total charge
            else:
                tcharge = 'ERROR'
            smethod = [line for line in ss if 'DFTTYP=' in line][0].split(
                'DFTTYP=')  # functional
            if len(smethod) > 0:
                method = smethod[-1].split(None, 1)[0]
                if (smethod[-2] == 'dispersion'):
                    method += '-D'
            errmsgs = [line for line in ss if 'semget return an error' in line]
            if len(errmsgs) > 0:
                error = True
            aplda = [line for line in ss if 'APLDA=' in line]
            if len(aplda) > 0:
                aplda = aplda[0].split('APLDA=', 1)[1]
                aplda = aplda.split(None, 1)[0]
            else:
                aplda = 'NA'
            apgga = [line for line in ss if 'APGGA=' in line]
            if len(apgga) > 0:
                apgga = apgga[0].split('APGGA=', 1)[1]
                apgga = apgga.split(None, 1)[0]
            else:
                apgga = 'NA'
            # SPIN S-SQUARED
            ssq = [_f for _f in [line for line in ss if 'S-SQUARED' in line] if _f]
            if (len(ssq) > 0):
                ssqs = [_f for _f in ssq[-1].split(')') if _f]
                ssq = ssqs[-1].split(None)[-1]
            else:
                ssq = 'NA'
            # optimization?
            optim = [line for line in ss if 'OPTIMIZE' in line]
            optim = 'Y' if len(optim) > 0 else 'N'
            # get results
            en = [line for line in ss if 'FINAL U' in line]  # energy
            en = en[-1].split(None)[4] if len(en) > 0 else 'NaN'
            conv = 'NA'
            if (optim == 'Y'):
                conv = [
                    line for line in ss if 'EQUILIBRIUM GEOMETRY LOCATED' in line]
                conv = 'Y' if len(conv) > 0 else 'N'
                if conv == 'Y':
                    lnosteps = [line for line in ss if 'NSERCH:' in line][-1]
                    nosteps = [_f for _f in lnosteps.split(None) if _f][1]
                else:
                    nosteps = 'NA'
            else:
                conv = 'NA'
                nosteps = 'NA'
            timet = [line for line in ss if 'TOTAL WALL CLOCK TIME' in line]
            if (len(timet) > 0):
                tm = timet[-1].split('=', 1)[-1]
                tm = tm.split('SECONDS', 1)[0]
                tm = "{:10.1f}".format(float(tm)/60.0)  # convert to MIN
            else:
                tm = 'NOT DONE'
            # construct string record of results
            if not error:
                tt = resfold.ljust(50)+method.ljust(9) + \
                    apgga.ljust(8)+aplda.ljust(8)
                tt += optim.ljust(9)+conv.ljust(9) + \
                    nosteps.ljust(8)+ssq.ljust(9)+spin.ljust(8)
                tt += tcharge.ljust(6) + \
                    "{:10.6f}".format(float(en)).ljust(14)+tm+'\n'
            else:
                tt = resfold.ljust(40)+comp.ljust(12) + \
                    'ERROR in the calculation\n'
            text.append(tt)
    # sort alphabetically and print
    text = sorted(text)
    if len(text) > 0:
        with open(folder+'/gam-results.txt', 'w') as f:
            f.write(header+''.join(text))
