import os
import glob
import copy
import numpy as np
import subprocess
import pandas as pd
import shutil
from molSimplify.job_manager.classes import resub_history, textfile
from ast import literal_eval
import json


def try_float(obj):
    # Converts an object to a floating point if possible
    try:
        floating_point = float(obj)
    except ValueError:
        floating_point = obj
    except TypeError:
        floating_point = obj
    return floating_point


def convert_to_absolute_path(path):
    if path[0] != '/':
        path = os.path.join(os.getcwd(), path)

    return path


def read_outfile(outfile_path, short_ouput=False, long_output=True):
    ## Reads TeraChem and ORCA outfiles
    #  @param outfile_path complete path to the outfile to be read, as a string
    #  @return A dictionary with keys finalenergy,s_squared,s_squared_ideal,time
    output = textfile(outfile_path)
    output_type = output.wordgrab(['TeraChem', 'ORCA'], ['whole_line', 'whole_line'])
    # print("output_type: ", output_type)
    for counter, match in enumerate(output_type):
        if match[0]:
            break
        if counter == 1:
            if 'nohup' in outfile_path:
                print('Warning! Nohup file caught in outfile processing')
                print(outfile_path)
                counter = 0
            elif 'smd.out' in outfile_path:
                print('Warning! SMD file caught in outfile processing')
                print(outfile_path)
                counter = 0
            elif ('atom' in outfile_path) and ('ORCA' in output_type):
                print('Density fitting output caught in outfile processing')
                print(outfile_path)
                counter = 0
            else:
                print('.out file type not recognized for file: ' + outfile_path)
                return_dict = {'name': None, 'charge': None, 'finalenergy': None,
                               'time': None, 's_squared': None, 's_squared_ideal': None,
                               'finished': False, 'min_energy': None, 'scf_error': False,
                               'thermo_grad_error': False, 'solvation_energy': None, 'optimization_cycles': None,
                               'thermo_vib_energy': None, 'thermo_vib_free_energy': None, 'thermo_suspect': None,
                               'orbital_occupation': None, 'oscillating_scf_error': False}
                return return_dict

    output_type = ['TeraChem', 'ORCA'][counter]

    name = None
    finished = False
    charge = None
    finalenergy = None
    min_energy = None
    s_squared = None
    s_squared_ideal = None
    scf_error = False
    time = None
    thermo_grad_error = False
    implicit_solvation_energy = None
    geo_opt_cycles = None
    thermo_vib = None
    thermo_vib_f = None
    thermo_suspect = None
    orbital_occupation = None
    oscillating_scf_error = False

    name = os.path.split(outfile_path)[-1]
    name = name.rsplit('.', 1)[0]
    if output_type == 'TeraChem':

        charge = output.wordgrab(['charge:'], [2], first_line=True)[0]
        if charge:
            charge = int(charge)
        if not short_ouput:
            (finalenergy, s_squared, s_squared_ideal, time, thermo_grad_error,
             implicit_solvation_energy, geo_opt_cycles,
             thermo_vib, thermo_vib_f, thermo_suspect) = output.wordgrab(
                ['FINAL', 'S-SQUARED:', 'S-SQUARED:', 'processing',
                 'Maximum component of gradient is too large',
                 'C-PCM contribution to final energy:',
                 'Optimization Cycle', 'Thermal vibrational energy',
                 'Thermal vibrational free energy',
                 'Thermochemical Analysis is Suspect'],
                [2, 2, 4, 3, 0, 4, 3, 7, 10, 0], last_line=True)
        if short_ouput:
            s_squared, s_squared_ideal, thermo_grad_error = output.wordgrab(
                ['S-SQUARED:', 'S-SQUARED:', 'Maximum component of gradient is too large'],
                [2, 4, 0], last_line=True)
        oscillating_scf = get_scf_progress(outfile_path)
        if oscillating_scf:
            oscillating_scf_error = True
        else:
            oscillating_scf_error = False
        if thermo_grad_error:
            thermo_grad_error = True
        else:
            thermo_grad_error = False
        if thermo_suspect:
            thermo_suspect = True
        else:
            thermo_suspect = False

        if s_squared_ideal:
            s_squared_ideal = float(s_squared_ideal.strip(')'))
        if implicit_solvation_energy:
            implicit_solvation_energy = try_float(implicit_solvation_energy.split(':')[-1])

        min_energy = output.wordgrab('FINAL', 2, min_value=True)[0]

        is_finished = output.wordgrab(['finished:'], 'whole_line', last_line=True)[0]
        if is_finished:  # for full optimization
            if is_finished[0] == 'Job' and is_finished[1] == 'finished:':
                finished = True

        is_finished = output.wordgrab(['processing'], 'whole_line', last_line=True)[0]
        if is_finished:  # for hydrogen optimization
            if is_finished[0] == 'Total' and is_finished[1] == 'processing':
                finished = True

        is_scf_error = output.wordgrab('DIIS', 5, matching_index=True)[0]
        if is_scf_error[0]:
            is_scf_error = [output.lines[i].split() for i in is_scf_error]
        else:
            is_scf_error = []
        if type(is_scf_error) == list and len(is_scf_error) > 0:
            for scf in is_scf_error:
                if ('failed' in scf) and ('converge' in scf) and ('iterations,' in scf) and ('ADIIS' in scf):
                    scf = scf[5]
                    scf = int(scf.split('+')[0])
                    if scf > 5000:
                        scf_error = [True, scf]
        if long_output:
            nbo_start, nbo_end = output.wordgrab(['NATURAL POPULATIONS:  Natural atomic orbital occupancies',
                                                  'Summary of Natural Population Analysis:'], 'whole_line',
                                                 matching_index=True, first_line=True)
            if nbo_start and nbo_end:
                nbo_lines = output.lines[nbo_start:nbo_end]
                nbo_lines = [line for line in nbo_lines if len(line.split()) > 0]  # filter out empty lines
                nbo_lines = [line for line in nbo_lines if line.split()[0].isdigit()]  # filter only results lines
                nbo_lines = [line for line in nbo_lines if line.split()[4] == 'Val(']  # filter only valence orbitals

                if len(nbo_lines) > 0:
                    orbital_occupation = dict()
                    for line in nbo_lines:
                        key = line.split()[1] + '_' + line.split()[2] + '_' + line.split()[3]
                        if key in orbital_occupation.keys():
                            raise Exception(outfile_path + ' ' + key + ': Same key found twice in nbo parsing!')
                        if len(line.split()) > 8:  # for open shell systems
                            orbital_occupation[key] = [float(line.split()[-3]), float(line.split()[-1])]
                        else:  # For closed shell systems
                            orbital_occupation[key] = [float(line.split()[-2]), float(0)]

    if output_type == 'ORCA':
        finished, finalenergy, s_squared, s_squared_ideal, implicit_solvation_energy = output.wordgrab(
            ['****ORCA TERMINATED NORMALLY****', 'FINAL', '<S**2>', 'S*(S+1)', 'CPCM Dielectric    :'],
            [0, -1, -1, -1, 3], last_line=True)
        if finished == '****ORCA':
            finished = True

        timekey = output.wordgrab('TOTAL RUN TIME:', 'whole_line', last_line=True)[0]
        if type(timekey) == list:
            time = (float(timekey[3]) * 24 * 60 * 60
                    + float(timekey[5]) * 60 * 60
                    + float(timekey[7]) * 60
                    + float(timekey[9])
                    + float(timekey[11]) * 0.001)

        if finished:
            charge = output.wordgrab(['Total Charge'], [-1], last_line=True)[0]
            charge = int(round(charge, 0))  # Round to nearest integer value (it should always be very close)

        opt_energies = output.wordgrab('FINAL SINGLE POINT ENERGY', -1)[0]
        geo_opt_cycles, min_energy = len(opt_energies), min(opt_energies)

    return_dict = {}
    return_dict['name'] = name
    return_dict['charge'] = charge
    return_dict['finalenergy'] = try_float(finalenergy)
    return_dict['time'] = try_float(time)
    return_dict['s_squared'] = try_float(s_squared)
    return_dict['s_squared_ideal'] = try_float(s_squared_ideal)
    return_dict['finished'] = finished
    return_dict['min_energy'] = try_float(min_energy)
    return_dict['scf_error'] = scf_error
    return_dict['thermo_grad_error'] = thermo_grad_error
    return_dict['solvation_energy'] = implicit_solvation_energy
    return_dict['optimization_cycles'] = geo_opt_cycles
    return_dict['thermo_vib_energy'] = try_float(thermo_vib)
    return_dict['thermo_vib_free_energy'] = try_float(thermo_vib_f)
    return_dict['thermo_suspect'] = thermo_suspect
    return_dict['orbital_occupation'] = orbital_occupation
    return_dict['oscillating_scf_error'] = oscillating_scf_error
    return_dict['outfile_path'] = outfile_path
    return return_dict


def read_infile(outfile_path):
    # Takes the path to either the outfile or the infile of a job
    # Returns a dictionary of the job settings included in that infile

    root = outfile_path.rsplit('.', 1)[0]
    unique_job_name = os.path.split(root)[-1]
    inp = textfile(root + '.in')
    if '#ORCA' in inp.lines[0]:
        qm_code = 'orca'
    else:
        qm_code = 'terachem'

    if qm_code == 'terachem':
        # account for multiple keywords for dispersion
        disp0 = inp.wordgrab(['dispersion '], [1], last_line=True)
        disp1 = inp.wordgrab(['dftd '], [1], last_line=True)
        if disp0 != [None]:
            dispersion = disp0[0]
        else:
            dispersion = disp1[0]
        charge, spinmult, solvent, run_type, levelshifta, levelshiftb, method, hfx, basis, coordinates, guess = inp.wordgrab(
            ['charge ', 'spinmult ', 'epsilon ',
             'run ', 'levelshiftvala ',
             'levelshiftvalb ', 'method ',
             'HFX ', 'basis ', 'coordinates ', 'guess '],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            last_line=True)
        charge, spinmult = int(charge), int(spinmult)
        if guess:
            guess = True
        else:
            guess = False
        if method[0] == 'u':
            method = method[1:]

        convergence_thresholds = inp.wordgrab(
            ['min_converge_gmax ', 'min_converge_grms ', 'min_converge_dmax ', 'min_converge_drms ', 'min_converge_e ',
             'convthre '],
            [1] * 6, last_line=True)
        if not convergence_thresholds[0]:
            convergence_thresholds = None

        multibasis = inp.wordgrab(['$multibasis', '$end'], [0, 0], last_line=True, matching_index=True)
        if not multibasis[0]:
            multibasis = False
        else:
            multibasis = inp.lines[multibasis[0] + 1:multibasis[1]]

        constraints = inp.wordgrab(['$constraint_freeze', '$end'], [0, 0], last_line=True, matching_index=True)
        if not constraints[0]:
            constraints = False
        else:
            constraints = inp.lines[constraints[0] + 1:constraints[1]]

        if constraints and multibasis:
            raise Exception(
                'The current implementation of tools.read_infile() is known to behave poorly when an infile specifies both a multibasis and constraints')

    elif qm_code == 'orca':
        ligand_basis, run_type, method, parallel_environment, charge, spinmult, coordinates = inp.wordgrab(
            ['! MULLIKEN'] * 3 + [r'%pal'] + [r'xyzfile'] * 3,
            [2, 3, 4, 2, 1, 2, 3], last_line=True)

        charge, spinmult = int(charge), int(spinmult)
        if run_type == 'opt':
            run_type = 'minimize'

        levelshift, solvent, metal_basis = inp.wordgrab([r'%scf', r'%cpcm', r'%basis'], [0] * 3,
                                                        matching_index=True, last_line=True)

        if levelshift:
            levelshift = inp.lines[levelshift + 1]
            levelshift = levelshift.split()
            levelshift = levelshift[2]
        if solvent:
            solvent = inp.lines[solvent + 1]
            solvent = solvent.split()
            solvent = solvent[1]
        if metal_basis:
            metal_basis = inp.lines[metal_basis + 1]
            metal_basis = metal_basis.split()
            metal_basis = metal_basis[2]
            metal_basis = metal_basis[1:-1]

        levelshifta, levelshiftb = levelshift, levelshift
        if ligand_basis == '6-31G*' and metal_basis == 'LANL2DZ':
            basis = 'lacvps_ecp'
        else:
            raise Exception(
                'read_infile() is unable to parse this basis set/ecp combo: ' + ligand_basis + ' ' + metal_basis)

        # The following settings should not appear in a orca infile because they are not specified in the write_input() functionality for orca
        hfx, convergence_thresholds, multibasis, dispersion, guess, constraints = None, None, None, None, None, None

    return_dict = {}

    for prop, prop_name in zip([unique_job_name, charge, spinmult, solvent, run_type, levelshifta, levelshiftb, method, hfx,
                                basis, convergence_thresholds, multibasis, constraints, dispersion, coordinates, guess,
                                qm_code],
                               ['name', 'charge', 'spinmult', 'solvent', 'run_type', 'levelshifta', 'levelshiftb',
                                'method', 'hfx',
                                'basis', 'convergence_thresholds', 'multibasis', 'constraints', 'dispersion',
                                'coordinates', 'guess', 'qm_code']):
        return_dict[prop_name] = prop
    return return_dict


# Read the global and local configure files to determine the derivative jobs requested and the settings for job recovery
# The global configure file should be in the same directory where resub() is called
# The local configure file should be in the same directory as the .out file
def read_configure(home_directory, outfile_path):
    def load_configure_file(directory):
        def strip_new_line(string):
            if string[-1] == '\n':
                return string[:-1]
            else:
                return string

        if directory == 'in place':
            directory = os.getcwd()

        configure = os.path.join(directory, 'configure')
        if os.path.isfile(configure):
            with open(configure, 'r') as f:
                configure = f.readlines()
            configure = list(map(strip_new_line, configure))
            return configure
        else:
            return []
    home_configure = load_configure_file(home_directory)
    if outfile_path:
        local_configure = load_configure_file(os.path.split(outfile_path)[0])
    else:
        local_configure = []
    # Determine which derivative jobs are requested
    solvent, vertEA, vertIP, thermo, dissociation, hfx_resample, functionalsSP, mbe = False, False, False, False, False, False, False, False
    for line in home_configure + local_configure:
        if 'solvent' in line or 'Solvent' in line:
            solvent = [float(p) for p in line.split()[1:]]
        if 'vertEA' in line or 'VertEA' in line:
            vertEA = True
        if 'vertIP' in line or 'VertIP' in line:
            vertIP = True
        if 'functionalsSP' in line or 'FunctionalsSP' in line:
            functionalsSP = [str(p) for p in line.split()[1:]]
        if ('thermo' in line and 'thermo_grad_error' not in line) or ('Thermo' in line and 'Thermo_grad_error' not in line):
            thermo = True
        if 'dissociation' in line or 'Dissociation' in line:
            dissociation = True
        if 'hfx_resample' in line or 'HFX_resample' in line:
            hfx_resample = True
        if 'mbe' in line or "MBE" in line:
            mbe = True

    # Determine global settings for this run
    max_jobs, max_resub, levela, levelb, method, hfx, geo_check, sleep, job_recovery, dispersion = False, False, False, False, False, False, False, False, [], False
    ss_cutoff, hard_job_limit, use_molscontrol, general_sp = False, False, False, False
    run_psi4, psi4_config = False, {}
    dissociated_ligand_charges, dissociated_ligand_spinmults = {}, {}
    for configure in [home_configure, local_configure]:
        for line in configure:
            if 'max_jobs' in line.split(':'):
                max_jobs = int(line.split(':')[-1])
            if 'max_resub' in line.split(':'):
                max_resub = int(line.split(':')[-1])
            if 'levela' in line.split(':'):
                levela = float(line.split(':')[-1])
            if 'levelb' in line.split(':'):
                levelb = float(line.split(':')[-1])
            if 'method' in line.split(':'):
                method = line.split(':')[-1]
            if 'hfx' in line.split(':'):
                hfx = float(line.split(':')[-1])
            if 'geo_check' in line.split(':'):
                geo_check = line.split(':')[-1]
            if 'sleep' in line.split(':'):
                sleep = int(line.split(':')[-1])
            if 'job_recovery' in line.split(':'):
                job_recovery = line.split(':')[-1]
                # convert the string form of a python list to an actual list
                job_recovery = job_recovery[1:-1]
                job_recovery = job_recovery.split(',')
            if 'dispersion' in line.split(':'):
                dispersion = line.split(':')[-1]
            if 'ss_cutoff' in line.split(':'):
                ss_cutoff = float(line.split(':')[-1])
            if 'hard_job_limit' in line.split(':'):
                hard_job_limit = int(line.split(':')[-1])
            if 'dissociated_ligand_charge' in line.split(':'):
                dissociated_ligand_charges[line.split(':')[-1].split()[0]] = int(line.split(':')[-1].split()[1])
            if 'dissociated_ligand_spinmult' in line.split(':'):
                dissociated_ligand_spinmults[line.split(':')[-1].split()[0]] = int(line.split(':')[-1].split()[1])
            if "use_molscontrol" in line.split(':'):
                use_molscontrol = bool(int(line.split(":")[-1]))
            if "general_sp" in line.split(':'):
                print("general SP jobs activated.")
                localpath = line.split(":")[-1].replace(" ", "")
                if os.path.isfile(localpath):
                    with open(os.getcwd() + "/" + localpath, "r") as f:
                        try:
                            general_sp = json.load(f)
                        except json.JSONDecodeError:
                            raise ValueError("%s is not a valid json file." % localpath)
                else:
                    raise ValueError("%s does not exits." % localpath)
            if "run_psi4" in line.split(':'):
                print("Psi4 jobs activated.")
                run_psi4 = True
                localpath = line.split(":")[-1].replace(" ", "")
                if os.path.isfile(localpath):
                    with open(os.getcwd() + "/" + localpath, "r") as f:
                        try:
                            psi4_config = json.load(f)
                        except json.JSONDecodeError:
                            raise ValueError("%s is not a valid json file." % localpath)
                else:
                    raise ValueError("%s does not exits." % localpath)
    # If global settings not specified, choose defaults:
    if (not max_jobs) and isinstance(max_jobs, bool):
        max_jobs = 50
    if not max_resub:
        max_resub = 5
    if not levela:
        levela = 0.25
    if not levelb:
        levelb = 0.25
    if not method:
        method = 'b3lyp'
    if not hfx:
        hfx = 0.20
    if not sleep:
        sleep = 7200
    if not ss_cutoff:
        ss_cutoff = 1.0
    if (not hard_job_limit):
        hard_job_limit = 190

    return {'solvent': solvent, 'vertEA': vertEA, 'vertIP': vertIP, 'thermo': thermo, 'dissociation': dissociation,
            'hfx_resample': hfx_resample, 'max_jobs': max_jobs, 'max_resub': max_resub, 'levela': levela,
            'levelb': levelb, 'method': method, 'hfx': hfx, 'geo_check': geo_check, 'sleep': sleep,
            'job_recovery': job_recovery, 'dispersion': dispersion, 'functionalsSP': functionalsSP,
            'ss_cutoff': ss_cutoff, 'hard_job_limit': hard_job_limit,
            'dissociated_ligand_spinmults': dissociated_ligand_spinmults,
            'dissociated_ligand_charges': dissociated_ligand_charges,
            "use_molscontrol": use_molscontrol, "general_sp": general_sp,
            "run_psi4": run_psi4, "psi4_config": psi4_config, 'mbe': mbe}


def read_charges(PATH):
    # Takes the path to either the outfile or the charge_mull.xls and returns the charges
    PATH = convert_to_absolute_path(PATH)
    if len(PATH.rsplit('.', 1)) > 1:
        if PATH.rsplit('.', 1)[1] == 'out':
            PATH = os.path.join(os.path.split(PATH)[0], 'scr', 'charge_mull.xls')
    try:
        charge_mull = textfile(PATH)
        split_lines = [i.split() for i in charge_mull.lines]
        charges = [i[1] + ' ' + i[2] for i in split_lines]
        return charges
    except:
        return []


def read_mullpop(PATH):
    # Takes the path to either the outfile or the mullpop and returns the mullikan populations
    PATH = convert_to_absolute_path(PATH)
    if len(PATH.rsplit('.', 1)) > 1:
        if PATH.rsplit('.', 1)[1] == 'out':
            PATH = os.path.join(os.path.split(PATH)[0], 'scr', 'mullpop')

    mullpop = textfile(PATH)
    ### If multiple frames in mullpop, grab last frame
    total_lines = mullpop.wordgrab(['------------ ---------- ----------'], [1], matching_index=True)[0]
    if len(total_lines) > 1:
        mullpop.lines = mullpop.lines[total_lines[-2] + 2:]

    split_lines = [i.split() for i in mullpop.lines]
    if len(split_lines[2]) == 6:
        pops = [i[1] + ' ' + i[5] for i in split_lines[1:-2]]
    else:
        pops = [i[1] + ' ' + i[5] + ' ' + i[9] for i in split_lines[2:-2]]

    return pops


def write_input(input_dictionary=dict(), name=None, charge=None, spinmult=None,
                run_type='energy', method='b3lyp', solvent=False,
                guess=False, custom_line=None, levelshifta=0.25, levelshiftb=0.25,
                convergence_thresholds=None, basis='lacvps_ecp', hfx=None, constraints=None,
                multibasis=False, coordinates=False, dispersion=False, qm_code='terachem',
                parallel_environment=4, precision='dynamic', dftgrid=1, dynamicgrid='yes',
                machine='gibraltar'):
    # Writes a generic input file for terachem or ORCA
    # The neccessary parameters can be supplied as arguements or as a dictionary. If supplied as both, the dictionary takes priority

    infile = dict()
    # If the input_dictionary exists,parse it and set the parameters, overwritting other specifications
    for prop, prop_name in zip([charge, spinmult, solvent, run_type, levelshifta, levelshiftb, method, hfx,
                                basis, convergence_thresholds, multibasis, constraints, dispersion, coordinates,
                                guess, custom_line, qm_code, parallel_environment, name, precision, dftgrid,
                                dynamicgrid, machine],
                               ['charge', 'spinmult', 'solvent', 'run_type', 'levelshifta', 'levelshiftb', 'method',
                                'hfx',
                                'basis', 'convergence_thresholds', 'multibasis', 'constraints', 'dispersion',
                                'coordinates', 'guess', 'custom_line', 'qm_code', 'parallel_environment', 'name',
                                'precision', 'dftgrid', 'dynamicgrid', 'machine']):
        if prop_name in list(input_dictionary.keys()):
            infile[prop_name] = input_dictionary[prop_name]
        else:
            infile[prop_name] = prop

    if (not infile['charge'] and infile['charge'] != 0) or (not infile['spinmult'] and infile['spinmult'] != 0) or (
            not infile['name']) or (not infile['coordinates']):
        print(('Name: ' + infile['name']))
        print(('Charge: ' + str(infile['charge'])))
        print(('Spinmult: ' + str(infile['spinmult'])))
        raise Exception('Minimum parameters not specified for writing infile')
    if type(infile['charge']) != int or type(infile['spinmult']) != int:
        print(('Charge Type: ' + str(type(infile['charge']))))
        print(('Spinmult Type: ' + str(type(infile['spinmult']))))
        raise Exception('Spin and Charge should both be integers!')

    if infile['qm_code'] == 'terachem':
        write_terachem_input(infile)
    elif infile['qm_code'] == 'orca':
        write_orca_input(infile)
    else:
        raise Exception('QM code: ' + infile['qm_code'] + ' not recognized!')


def write_terachem_input(infile_dictionary):
    infile = infile_dictionary

    if infile['spinmult'] != 1:
        infile['method'] = 'u' + infile['method']

    text = ['levelshiftvalb ' + str(infile['levelshiftb']) + '\n',
            'levelshiftvala ' + str(infile['levelshifta']) + '\n',
            'run ' + infile['run_type'] + '\n',
            'scf diis+a\n',
            'coordinates ' + infile['coordinates'] + '\n',
            'levelshift yes\n',
            'spinmult ' + str(infile['spinmult']) + '\n',
            'scrdir ./scr\n',
            'basis ' + infile['basis'] + '\n',
            'timings yes\n',
            'charge ' + str(infile['charge']) + '\n',
            'method ' + str(infile['method']) + '\n',
            'precision ' + str(infile['precision']) + '\n',
            'dftgrid ' + str(infile['dftgrid']) + '\n',
            'dynamicgrid ' + str(infile['dynamicgrid']) + '\n',
            'new_minimizer yes\n',
            'ml_prop yes\n',
            'poptype mulliken\n',
            'bond_order_list yes\n',
            'end']

    if infile['custom_line']:
        text = text[:15] + [infile['custom_line'] + '\n'] + text[15:]
    if infile['guess']:
        if type(infile['guess']) == bool:
            if infile['spinmult'] == 1:
                infile['guess'] = 'c0'
            else:
                infile['guess'] = 'ca0 cb0'
        text = text[:-1] + ['guess ' + infile['guess'] + '\n',
                            'end']
    if infile['run_type'] != 'ts':
        text = text[:-1] + ['maxit 500\n',
                            'end']

    if type(infile['convergence_thresholds']) == list:
        if infile['convergence_thresholds'][0]:
            thresholds = [line if line.endswith('\n') else line + '\n' for line in infile['convergence_thresholds']]
            tight_thresholds = ("min_converge_gmax " + thresholds[0] + "min_converge_grms "
                                + thresholds[1] + "min_converge_dmax " + thresholds[2]
                                + "min_converge_drms " + thresholds[3] + "min_converge_e "
                                + thresholds[4] + "convthre " + thresholds[5])
            text = text[:-1] + ['\n', tight_thresholds, 'end']

    if infile['dispersion']:
        text = text[:-1] + ['dispersion ' + infile['dispersion'] + '\n', 'end']

    if infile['multibasis']:
        multibasis = [line if line.endswith('\n') else line + '\n' for line in infile['multibasis']]
        text = text[:-1] + ['\n', '$multibasis\n'] + multibasis + ['$end\n', 'end']

    if infile['constraints']:
        constraints = [line if line.endswith('\n') else line + '\n' for line in infile['constraints']]
        text = text[:-1] + ['\n', '$constraint_freeze\n'] + constraints + ['$end\n', 'end']

    if type(infile['hfx']) == int or type(infile['hfx']) == float:
        text = text[:-1] + ['\n',
                            'HFX ' + str(infile['hfx']) + '\n',
                            'end']

    if infile['solvent']:  # Adds solvent correction, if requested
        text = text[:-1] + ['\n',
                            'pcm cosmo\n',
                            'epsilon %.2f\n' % float(infile['solvent']),
                            'pcm_radii read\n',
                            'pcm_radii_file /home/harperd/pcm_radii\n',
                            'end']
    if infile['machine'] in ['gibraltar', 'bridges']:
        text = text[:-1] + ['nbo yes\n', 'gpus 1\n', 'end']
    elif infile['machine'] in ['comet']:
        text = text[:-1] + ['gpus 2\n', 'end']
    else:
        raise ValueError('Machine not known!')

    with open(infile['name'] + '.in', 'w') as fout:
        for lines in text:
            fout.write(lines)


def write_orca_input(infile_dictionary):
    infile = infile_dictionary

    # The orca input writting isn't as smart as the Terachem input writting
    # Ensure that the orca input isn't passed a keyword that it doesn't know how to handle yet
    if str(infile['levelshifta']) != str(infile['levelshiftb']):
        raise Exception('ORCA input does not support 2 different levelshift values for openshell systems! ' + str(
            infile['levelshifta']) + ' ' + str(infile['levelshiftb']))
    for element in ['constraints', 'dispersion', 'hfx', 'multibasis', 'convergence_thresholds', 'guess']:
        if element in infile.keys():
            if infile[element]:
                raise Exception('Keyword (' + element + ') not yet implemented for orca in the job manager')

    # ORCA requires explicit definition of the ECP and non-ecp basis
    if infile['basis'] == 'lacvps_ecp':
        ligand_basis = '6-31G*'
        metal_basis = 'LANL2DZ'
    else:
        raise Exception(infile['basis'] + 'not implemented in the job manager for use with orca!')

    # Determine the atoms which need to have an effective core potential added
    metals = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
              'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
    metal_basis_line = ''
    for metal in metals:
        metal_basis_line += ('  NewGTO ' + metal + ' "' + metal_basis + '" end\n')

    # Convert the keywords for run_type from terachem to orca
    if infile['run_type'] == 'minimize':
        infile['run_type'] = 'opt'

    # Note that max core is set to 2000MB, this is 2/3 of the amount allocated in the jobscript (on a per processor basis)
    # The SCF is known (according to the ORCA manual) to exceed allotted memory, so this provides some wiggle room
    if not infile['solvent']:
        first_line = r'! MULLIKEN ' + ligand_basis + ' ' + infile['run_type'] + ' ' + infile['method'] + ' printbasis\n'
    else:
        first_line = r'! MULLIKEN ' + ligand_basis + ' ' + infile['run_type'] + ' ' + infile[
            'method'] + ' CPCM printbasis\n'
    text = ['#ORCA input\n',
            first_line,
            r'%' + 'maxcore 2000\n',
            r'%' + 'pal nprocs ' + str(infile['parallel_environment']) + ' end\n',
            r'*' + 'xyzfile ' + str(infile['charge']) + ' ' + str(infile['spinmult']) + ' ' + infile[
                'name'] + '.xyz\n\n',
            r'%' + 'scf\n  Shift Shift ' + str(infile['levelshifta']) + ' ErrOff 0.1 end\nend\n\n',
            r'%' + 'basis\n' + metal_basis_line + 'end\n\n']

    if infile['solvent']:
        text += [r'%' + 'cpcm\n  epsilon ' + str(infile['solvent']) + '\nend\n\n']
    if infile['custom_line']:
        text = text + [infile['custom_line'] + '\n']

    with open(infile['name'] + '.in', 'w') as input_file:
        for lines in text:
            input_file.write(lines)


def write_jobscript(name, custom_line=None, time_limit='96:00:00', qm_code='terachem', parallel_environment=4,
                    machine='gibraltar', use_molscontrol=False, queues=['gpus', 'gpusnew']):
    # Writes a generic obscript
    # custom line allows the addition of extra lines just before the export statement

    if qm_code == 'terachem':
        write_terachem_jobscript(name, custom_line=custom_line, time_limit=time_limit,
                                 machine=machine, use_molscontrol=use_molscontrol,
                                 queues=queues)
    elif qm_code == 'orca':
        write_orca_jobscript(name, custom_line=custom_line, time_limit=time_limit,
                             parallel_environment=parallel_environment,
                             machine=machine, use_molscontrol=use_molscontrol)
    else:
        raise Exception('QM code: ' + qm_code + ' not recognized for jobscript writing!')


def write_terachem_jobscript(name, custom_line=None, time_limit='96:00:00', terachem_line=True,
                             machine='gibraltar', use_molscontrol=False, queues=['gpus', 'gpusnew']):
    # if use_molscontrol and machine != 'gibraltar':
    #     raise ValueError("molscontrol is only implemented on gibraltar for now.")

    if machine == 'gibraltar':
        if not use_molscontrol:
            text = ['#$ -S /bin/bash\n',
                    '#$ -N ' + name + '\n',
                    '#$ -cwd\n',
                    '#$ -R y\n',
                    '#$ -l h_rt=' + time_limit + '\n',
                    '#$ -l h_rss=8G\n',
                    '#$ -q '+'|'.join(queues)+'\n',
                    '#$ -l gpus=1\n',
                    '#$ -pe smp 1\n',
                    "# -fin " + "%s.in\n" % name,
                    "# -fin " + "%s.xyz\n" % name,
                    '# -fout scr/\n',
                    'source /etc/profile.d/modules.sh\n',
                    'module load terachem/tip\n',
                    'export OMP_NUM_THREADS=1\n'
                    ]
        else:
            text = ['#$ -S /bin/bash\n',
                    '#$ -N ' + name + '\n',
                    '#$ -cwd\n',
                    '#$ -R y\n',
                    '#$ -l h_rt=' + time_limit + '\n',
                    '#$ -l h_rss=8G\n',
                    '#$ -q '+'|'.join(queues)+'\n',
                    '#$ -l gpus=1\n',
                    '#$ -pe smp 1\n',
                    "# -fin " + "%s.in\n" % name,
                    "# -fin molscontrol_config.json\n",
                    "# -fin " + "%s.xyz\n" % name,
                    '# -fout scr/\n',
                    'source /etc/profile.d/modules.sh\n',
                    "source activate /home/crduan/.conda/envs/mols_keras/\n",
                    'module load terachem/tip\n',
                    'export OMP_NUM_THREADS=1\n'
                    ]
    elif machine == 'bridges':
        if int(time_limit.split(':')[0]) > 45:
            time_limit = '45:00:00'
        text = ['#!/bin/bash\n',
                '#SBATCH -J ' + name + '\n',
                '#SBATCH -N 1\n',
                '#SBATCH -p GPU-shared\n',
                '#SBATCH --ntasks-per-node 1\n'
                '#SBATCH -t ' + time_limit + '\n',
                '#SBATCH -C EGRESS\n',
                '#SBATCH --gres=gpu:k80:1\n\n',
                'set -x\n',
                'module load intel/17.4\n',
                'module load mpi/intel_mpi\n',
                'module load cuda/9.2\n',
                'export TeraChem=/home/ffangliu/production/build\n',
                'export PATH=$TeraChem/bin/:$PATH\n',
                'export NBOEXE=$TeraChem/nbo6/nbo6.i4.exe\n',
                'export LD_LIBRARY_PATH=$TeraChem/lib:$LD_LIBRARY_PATH\n'
                ]
    elif machine == 'comet':
        if int(time_limit.split(':')[0]) > 45:
            time_limit = '45:00:00'
        text = ['#!/bin/bash\n',
                '#SBATCH -J ' + name + '\n',
                '#SBATCH --nodes 1\n',
                '#SBATCH -p gpu-shared\n',
                '#SBATCH --mem=16G\n',
                '#SBATCH --ntasks-per-node=1\n'
                '#SBATCH -t ' + time_limit + '\n',
                '#SBATCH --export=ALL\n',
                '#SBATCH --gres=gpu:2\n\n',
                'export OMP_NUM_THREADS=2\n',
                'module purge\n',
                'export MODULEPATH="/share/apps/compute/modulefiles:${MODULEPATH}"\n',
                'module load gnu/7.2.0\n',
                'module load intel/2016.3.210\n',
                'module load  intelmpi/2016.3.210\n',
                'module load cuda/9.2\n',
                'export TeraChem=/oasis/projects/nsf/mit136/fangliu/src/production/build\n',
                'export PATH=$TeraChem/bin/:$PATH\n',
                'export NBOEXE=$TeraChem/nbo6/nbo6.i4.exe\n',
                'export LD_LIBRARY_PATH=$TeraChem/lib:$LD_LIBRARY_PATH\n',
                'export OpenMM=/oasis/projects/nsf/mit136/fangliu/opt/openmm\n',
                'export CPATH=$OpenMM/include:$CPATH\n',
                'export LD_LIBRARY_PATH=$OpenMM/lib:$LD_LIBRARY_PATH\n',
                'export LIBRARY_PATH=$OpenMM/lib:$LIBRARY_PATH\n',
                'export PATH=$OpenMM/bin:$PATH\n',
                'export OPENMM_INCLUDE_PATH=$OpenMM/include\n',
                'export OPENMM_LIB_PATH=$OpenMM/lib\n',
                'export OPENMM_PLUGIN_DIR=$OpenMM/lib/plugins\n'
                ]
    else:
        raise ValueError('Job manager does not know how to run Terachem on this machine!')
    if terachem_line and machine == 'gibraltar':
        if not use_molscontrol:
            text += ['terachem ' + name + '.in ' + '> $SGE_O_WORKDIR/' + name + '.out\n']
        else:
            text += ["cp %s.xyz initgeo.xyz" % name]
            text += ['terachem ' + name + '.in ' + '> $SGE_O_WORKDIR/' + name + '.out &\n']
            text += ["PID_KILL=$!\n"]
            text += ["molscontrol $PID_KILL &\n"]
            text += ["wait\n"]
            text += ["mv *.log $SGE_O_WORKDIR\n"]
            text += ["mv features.json $SGE_O_WORKDIR/dyanmic_features.json\n"]
    elif terachem_line and machine in ['bridges', 'comet']:
        text += ['terachem ' + name + '.in ' + '> ' + name + '.out\n']
    if custom_line:
        if type(custom_line) == list:
            text = text[:12] + custom_line + text[12:]
        else:
            text = text[:12] + [custom_line + '\n'] + text[12:]
    text += ['sleep 30']

    with open(name + '_jobscript', 'w') as jobscript:
        for i in text:
            jobscript.write(i)


def write_molscontrol_config():
    pass


def write_orca_jobscript(name, custom_line=None, time_limit='96:00:00', parallel_environment=4,
                         machine='gibraltar', use_molscontrol=False):
    # Write a generic orca jobscript

    if use_molscontrol:
        raise NotImplementedError('molscontrol is currently not supported for orca.')

    memory_allocation = str(
        int(parallel_environment) * 3)  # allocate memory based on 192 GB for 64 processors on the new cpu nodes

    if machine == 'gibraltar':
        text = ['#$ -S /bin/bash\n',
                '#$ -N ' + name + '\n',
                '#$ -cwd\n',
                '#$ -R y\n',
                '#$ -l h_rt=' + time_limit + '\n',
                '#$ -l h_rss=' + memory_allocation + 'G\n',
                '#$ -q cpus\n',
                '#$ -l cpus=1\n',
                '#$ -pe smp ' + str(parallel_environment) + '\n',
                '# -fin ' + name + '.in\n',
                '# -fin ' + name + '.xyz\n\n',
                'source /etc/profile.d/modules.sh\n',
                'module module load intel\n',
                'module module load orca\n',
                'export PATH=/home/harperd/software/openmpi/bin:$PATH\n',
                'export LD_LIBRARY_PATH=/home/harperd/software/openmpi/lib:$LD_LIBRARY_PATH\n\n',
                '/opt/orca/orca_4_1_2_linux_x86-64_openmpi313/orca ' + name + '.in  > $SGE_O_WORKDIR/' + name + '.out\n\n',
                'mkdir $SGE_O_WORKDIR/scr\n',
                'cp ' + name + '.trj $SGE_O_WORKDIR/scr/optim.xyz\n',
                'cp ' + name + '.gbw $SGE_O_WORKDIR/scr/\n',
                'cp ' + name + '.prop $SGE_O_WORKDIR/scr/\n',
                'cp ' + name + '.opt $SGE_O_WORKDIR/scr/\n',
                'cp ' + name + '_property.txt $SGE_O_WORKDIR/scr/\n']
    else:
        raise ValueError('Job manager does not know how to run ORCA on this machine!')
    if custom_line:
        if type(custom_line) == list:
            text = text[:12] + custom_line + text[12:]
        else:
            text = text[:12] + [custom_line + '\n'] + text[12:]

    with open(name + '_jobscript', 'w') as jobscript:
        for i in text:
            jobscript.write(i)


def get_scf_progress(outfile):
    flag = False
    flag_linecheck = False
    with open(outfile, 'r',errors='ignore') as fo:
        start = False
        for line in fo:
            ll = line.split()
            if "Start SCF Iterations" in line:
                start = True
                energy_this_scf = []
            if len(ll) == 11 and ll[0].isdigit() and start:
                energy_this_scf.append(float(ll[-2]))
            if "FINAL ENERGY:" in line and start:
                start = False
                flag = is_oscillate(energy_this_scf)
                if flag:
                    break
    with open(outfile, 'r',errors='ignore') as fo:
        for line in fo:
            if "WARNING: Final energy is higher than the lowest energy by" in line:
                e = float(line.split()[-1].strip("."))
                if e > 1:
                    flag_linecheck = True
    return (flag and flag_linecheck)


def is_oscillate(energy_this_scf):
    if len(energy_this_scf) > 1:
        if np.std(energy_this_scf) > 1:
            return True
    return False
