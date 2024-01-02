#!/usr/bin/env python
import os
import shutil
import glob
import numpy as np
import molSimplify.job_manager.tools as tools
import molSimplify.job_manager.moltools as moltools
from molSimplify.job_manager.classes import resub_history
import molSimplify.job_manager.manager_io as manager_io


def load_history(PATH):
    """Load a resub history object.

    Parameters
    ----------
        PATH : str
            The name of an output file or history object pickle file.

    Returns
    -------
        history : resub_history
            resub_history class instance for history object of job.

    """
    # takes the path to either an outfile or the resub_history pickle
    # returns the resub_history class object

    history = resub_history()
    history.read(PATH)
    return history


def abandon_job(PATH):
    """Abandons a job with a given path. This function is never and should never be called by the job manager.
    Only called manually for troublesome jobs.

    Parameters
    ----------
        PATH : str
            The name of an output file or history object pickle file.

    """
    # takes the path to either an outfile or the resub_history pickle
    # sets the jobs status to be abandoned

    # This function is never and should never be called by the job manager.
    # It is for manually use with particularly troublesome jobs
    history = load_history(PATH)
    history.abandon()
    history.save()


def save_scr(outfile_path, rewrite_inscr=True):
    """Archive the scr file so it isn't overwritten in future resubs.

    Parameters
    ----------
        outfile_path : str
            The name of an output file.
        rewrite_inscr : bool, optional
            Determines whether to copy this runs wfn and optimized geometry to the inscr directory. Default is True.

    Returns
    -------
        scrpaths : str
            Path to new scr directory.


    """
    root = os.path.split(outfile_path)[0]
    scr_path = os.path.join(root, 'scr')

    if os.path.isdir(scr_path):
        # extract the optimized geometry, if it exists
        optim = glob.glob(os.path.join(scr_path, 'optim.xyz'))
        if len(optim) > 0:
            tools.extract_optimized_geo(optim[0])

        if rewrite_inscr:
            # save the files necessary to resub the job to a folder called inscr
            save_paths = []
            save_paths.extend(glob.glob(os.path.join(scr_path, 'c0')))
            save_paths.extend(glob.glob(os.path.join(scr_path, 'ca0')))
            save_paths.extend(glob.glob(os.path.join(scr_path, 'cb0')))
            save_paths.extend(glob.glob(os.path.join(scr_path, 'optimized.xyz')))
            if os.path.isdir(os.path.join(root, 'inscr')):
                shutil.rmtree(os.path.join(root, 'inscr'))
            os.mkdir(os.path.join(root, 'inscr'))
            for path in save_paths:
                shutil.copy(path, os.path.join(root, 'inscr', os.path.split(path)[-1]))

        # archive the scr under a new name so that we can write a new one
        old_scrs = glob.glob(scr_path + '_*')
        old_scrs = [int(i[-1]) for i in old_scrs]

        if len(old_scrs) > 0:
            new_scr = str(max(old_scrs) + 1)
        else:
            new_scr = '0'
        # print("current_scr: ", scr_path)
        # print("backup_scr: ", scr_path + '_' + new_scr)
        shutil.move(scr_path, scr_path + '_' + new_scr)

        return os.path.join(os.path.split(outfile_path)[0], 'scr') + '_' + new_scr


def save_run(outfile_path, rewrite_inscr=True, save_scr_flag=True):
    """Save the outfile within the resub_history pickle object.

    Parameters
    ----------
        outfile_path : str
            The name of an output file.
        rewrite_inscr : bool, optional
            Determines whether to copy this runs wfn and optimized geometry to the inscr directory. Default is True.
        save_scr_flag : bool, optional
            Determine wheether to store scr. Default is True.

    """
    def write(list_of_lines, path):
        with open(path, 'w') as fil:
            for i in list_of_lines:
                fil.write(i)

    if save_scr_flag:
        scr_path = save_scr(outfile_path, rewrite_inscr=rewrite_inscr)
    else:
        scr_path = os.path.join(os.path.split(outfile_path)[0], 'scr')

    history = resub_history()
    history.read(outfile_path)

    with open(outfile_path, 'r',errors='ignore') as f:
        out_lines = f.readlines()
    history.outfiles.append(out_lines)

    folder = os.path.dirname(outfile_path)
    name = os.path.basename(folder)

    infile_path = os.path.join(folder,name + '.in')
    with open(infile_path, 'r') as f:
        in_lines = f.readlines()
    history.infiles.append(in_lines)

    jobscript_path = os.path.join(folder,name + '_jobscript')
    with open(jobscript_path, 'r') as f:
        job_lines = f.readlines()
    history.jobscripts.append(job_lines)

    xyz_path = os.path.join(folder,name + '.xyz')
    with open(xyz_path, 'r') as f:
        xyz_lines = f.readlines()
    history.xyzs.append(xyz_lines)

    history.save()

    if scr_path:
        # Additionally, write this information to textfile so it's earch to find
        home = os.getcwd()
        os.chdir(scr_path)
        write(out_lines, 'old_outfile')
        write(in_lines, 'old_infile')
        write(job_lines, 'old_job')
        write(xyz_lines, 'old_xyz')
        os.chdir(home)


def reset(outfile_path):
    """Returns the run to the state it was after the first run, before job recovery acted on it.

    Parameters
    ----------
        outfile_path : str
            The name of an output file.

    """
    # Returns the run to the state it was after the first run, before job recovery acted on it

    pickle_path = outfile_path.rsplit('.', 1)[0] + '.pickle'
    if os.path.isfile(pickle_path):

        print('Resetting run: ' + os.path.split(outfile_path)[-1].rsplit('.', 1)[0])
        old_path = os.path.join(os.path.split(outfile_path)[0], 'pre_reset')
        if not os.path.isdir(old_path):
            os.mkdir(old_path)

        # Find all the stdout and stderr files related to previous runs.
        queue_output = glob.glob(outfile_path.rsplit('.', 1)[0] + '.e*')
        queue_output.extend(glob.glob(outfile_path.rsplit('.', 1)[0] + '.pe*'))
        queue_output.extend(glob.glob(outfile_path.rsplit('.', 1)[0] + '.po*'))
        queue_output.extend(glob.glob(outfile_path.rsplit('.', 1)[0] + '.o*'))
        queue_output = [i for i in queue_output if i[-1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']]

        # remove all files from states after the specified state
        move = []
        identifier = 1
        while True:
            move.extend(glob.glob(os.path.join(os.path.split(outfile_path)[0], 'scr_' + str(identifier))))
            identifier += 1
            if len(glob.glob(os.path.join(os.path.split(outfile_path)[0], 'scr_' + str(identifier)))) == 0:
                break  # break when all scr_? files are found.

        # remove all files for derivative jobs spawned based on this job
        derivative_types = ['solvent', 'vertEA', 'vertIP', 'thermo', 'kp', 'rm', 'ultratight', 'HFXresampling',
                            'functional']
        possible = [i for i in glob.glob(os.path.join(os.path.split(outfile_path)[0], '*')) if os.path.isdir(i)]
        for folder in possible:
            if os.path.split(outfile_path)[1].rsplit('.', 1)[0] in folder:
                derivative = False
                for typ in derivative_types:
                    if typ in folder:
                        derivative = True
                if derivative:
                    shutil.rmtree(folder)

        # rename outfile and jobscript files

        folder = os.path.dirname(outfile_path)
        name = os.path.basename(folder)
        basepath = os.path.join(folder,name)


        shutil.move(outfile_path, basepath + '.old')  # rename old out so it isn't found in .out searches
        shutil.move(basepath + '_jobscript', basepath + '_oldjob')  # rename old jobscript so it isn't thought to be  job that hasn't started yet
        move.append(basepath + '.old')
        move.append(basepath + '.xyz')
        move.append(basepath + '.in')
        move.append(basepath + '_oldjob')
        if os.path.isdir(os.path.join(os.path.split(outfile_path)[0], 'inscr')):
            move.append(os.path.join(os.path.split(outfile_path)[0], 'inscr'))
        move.extend(queue_output)
        scr_path = os.path.join(os.path.split(outfile_path)[0], 'scr')
        move.append(scr_path)
        for path in move:
            # move the paths to their new location, Random numbers prevent clashes
            try:
                shutil.move(path,
                            os.path.join(old_path, str(np.random.randint(999999999)) + '_' + os.path.split(path)[-1]))
            except FileNotFoundError:
                print('No file found for: ' + path)

        # Rewrite the .xyz, .in, jobscript, and .out file to be the same as they were after the first run
        history = resub_history()
        history.read(pickle_path)
        outfile = history.outfiles[0]
        infile = history.infiles[0]
        jobscript = history.jobscripts[0]
        xyz = history.xyzs[0]
        with open(outfile_path, 'w') as writer:
            for i in outfile:
                writer.write(i)
        with open(basepath + '.in', 'w') as writer:
            for i in infile:
                writer.write(i)
        with open(basepath + '.xyz', 'w') as writer:
            for i in xyz:
                writer.write(i)
        with open(basepath+ '_jobscript', 'w') as writer:
            for i in jobscript:
                writer.write(i)

        shutil.move(scr_path + '_0', scr_path)
        shutil.move(pickle_path, os.path.join(old_path, str(np.random.randint(999999999)) + '_resub_history'))


def simple_resub(outfile_path):
    """Resubmits a job without changing parameters. Particularly useful for CUDA errors.

    Parameters
    ----------
        outfile_path : str
            The name of an output file.

    Returns
    -------
        Resub_flag : bool
            True if resubmitted.

    """
    # Resubmits a job without changing parameters. Particularly useful for CUDA errors.
    save_run(outfile_path, rewrite_inscr=False)
    history = resub_history()
    history.read(outfile_path)
    history.resub_number += 1
    history.notes.append('Resubbed for unknown error')
    history.save()

    root = outfile_path.rsplit('.', 1)[0]

    tools.qsub(root + '_jobscript')
    return True


def clean_resub(outfile_path):
    """Resubmits a job with default parameters, useful for undoing level shift or hfx alterations.

    Parameters
    ----------
        outfile_path : str
            The name of an output file.

    Returns
    -------
        Resub_flag : bool
            True if resubmitted.

    """
    # Resubmits a job with default parameters, useful for undoing level shift or hfx alterations
    save_run(outfile_path)
    history = resub_history()
    history.read(outfile_path)
    history.resub_number += 1
    history.status = 'Normal'
    history.notes.append('Needs clean resub')
    history.needs_resub = False
    history.save()

    machine = tools.get_machine()
    root = outfile_path.rsplit('.', 1)[0]
    name = os.path.split(root)[-1]
    directory = os.path.split(outfile_path)[0]
    infile_dict = manager_io.read_infile(outfile_path)

    home = os.getcwd()
    if len(directory) > 0:  # if the string is blank, then we're already in the correct directory
        os.chdir(directory)

    if os.path.isfile('inscr/optimized.xyz'):
        coordinates = 'inscr/optimized.xyz'  # Should trigger for optimization runs
    elif os.path.isfile(name + '.xyz'):
        coordinates = name + '.xyz'  # Should trigger for single point runs
    else:
        raise ValueError('No coordinates idenfied for clean in resubmission in directory ' + os.getcwd())

    configure_dict = manager_io.read_configure('in_place', outfile_path)

    infile_dict['coordinates'] = coordinates
    infile_dict['method'] = configure_dict['method']
    infile_dict['levelshifta'], infile_dict['levelshiftb'] = configure_dict['levela'], configure_dict['levelb']
    infile_dict['dispersion'] = configure_dict['dispersion']
    infile_dict['constraints'] = False
    infile_dict['machine'] = machine

    if infile_dict['spinmult'] == 1:
        infile_dict['guess'] = 'inscr/c0'
        manager_io.write_input(infile_dict)
    else:
        infile_dict['guess'] = 'inscr/ca0 inscr/cb0'
        manager_io.write_input(infile_dict)

    manager_io.write_jobscript(name, custom_line='# -fin inscr/', machine=machine)
    os.chdir(home)
    tools.qsub(root + '_jobscript')
    return True


def resub_spin(outfile_path):
    """Resubmits a spin contaminated job with blyp to help convergence to a non-spin contaminated solution.

    Parameters
    ----------
        outfile_path : str
            The name of an output file.

    Returns
    -------
        Resub_flag : bool
            True if resubmitted.

    """
    # resubmits a spin contaminated job with blyp to help convergence to a non-spin contaminated solution
    history = resub_history()
    history.read(outfile_path)
    resubbed_before = False
    if 'Spin contaminated, lowering HFX to aid convergence' in history.notes:
        resubbed_before = True
        history.status = os.path.split(outfile_path)[
                             -1] + ' has been submitted with lower HFX and still converges to a spin contaminated solution'
        history.save()
    if 'Needs clean resub' in history.notes:
        resubbed_before = True
        history.status = os.path.split(outfile_path)[
                             -1] + ' job recovery has failed - requesting resub_spin() after clean resubmission round'
        history.save()
    if 'HFXresampling' in outfile_path:
        resubbed_before = True
        history.status = os.path.split(outfile_path)[
                             -1] + ' is spin contaminated, but submitting with lower HFX does not make sense for HFX resampling jobs'
        history.save()

    if not resubbed_before:
        save_run(outfile_path, rewrite_inscr=False)
        history = resub_history()
        history.read(outfile_path)
        history.resub_number += 1
        history.status = 'HFX altered to assist convergence'
        history.needs_resub = True
        history.notes.append('Spin contaminated, lowering HFX to aid convergence')
        history.save()

        machine = tools.get_machine()
        root = outfile_path.rsplit('.', 1)[0]
        name = os.path.split(root)[-1]
        directory = os.path.split(outfile_path)[0]
        infile_dict = manager_io.read_infile(outfile_path)

        home = os.getcwd()
        if len(directory) > 0:  # if the string is blank, then we're already in the correct directory
            os.chdir(directory)

        infile_dict['method'] = 'blyp'
        infile_dict['machine'] = machine
        manager_io.write_input(infile_dict)

        manager_io.write_jobscript(name, machine=machine)
        os.chdir(home)
        tools.qsub(root + '_jobscript')
        return True

    else:
        return False


def resub_scf(outfile_path):
    """Resubmits a job that's having trouble converging the scf with different level shifts (1.0 and 0.1).

    Parameters
    ----------
        outfile_path : str
            The name of an output file.

    Returns
    -------
        Resub_flag : bool
            True if resubmitted.

    """
    # Resubmits a job that's having trouble converging the scf with different level shifts (1.0 and 0.1)
    history = resub_history()
    history.read(outfile_path)
    resubbed_before = False
    if 'SCF convergence error, level shifts adjusted to aid convergence' in history.notes:
        resubbed_before = True
        history.status = os.path.split(outfile_path)[
                             -1] + ' has been submitted with levels shifted and is still encountering an scf error'
        history.save()
    if 'Needs clean resub' in history.notes:
        resubbed_before = True
        history.status = os.path.split(outfile_path)[
                             -1] + ' job recovery has failed - requesting resub_scf() after clean resubmission round'
        history.save()

    if not resubbed_before:
        save_run(outfile_path, rewrite_inscr=False)
        history = resub_history()
        history.read(outfile_path)
        history.resub_number += 1
        history.status = 'Level shifts adjusted to assist convergence'
        history.needs_resub = True
        history.notes.append('SCF convergence error, level shifts adjusted to aid convergence')
        history.save()

        machine = tools.get_machine()
        root = outfile_path.rsplit('.', 1)[0]
        name = os.path.split(root)[-1]
        directory = os.path.split(outfile_path)[0]
        infile_dict = manager_io.read_infile(outfile_path)

        home = os.getcwd()
        if len(directory) > 0:  # if the string is blank, then we're already in the correct directory
            os.chdir(directory)
        infile_dict['levelshifta'], infile_dict['levelshiftb'] = 1.0, 0.1
        infile_dict['machine'] = machine
        manager_io.write_input(infile_dict)

        manager_io.write_jobscript(name, machine=machine)
        os.chdir(home)
        tools.qsub(root + '_jobscript')
        return True

    else:
        return False


def resub_oscillating_scf(outfile_path):
    """Resubmits a job that's having trouble converging the scf with different level shifts (1.0 and 0.1).

    Parameters
    ----------
        outfile_path : str
            The name of an output file.

    Returns
    -------
        Resub_flag : bool
            True if resubmitted.

    """

    # Resubmits a job that's having trouble converging the scf with different level shifts (1.0 and 0.1)
    history = resub_history()
    history.read(outfile_path)
    resubbed_before = False
    if 'SCF convergence error, precision and grid adjusted to aid convergence' in history.notes:
        resubbed_before = True
        history.status = os.path.split(outfile_path)[
                             -1] + ' has been submitted with higher precision and grid and is still encountering an scf error'
        history.save()
    if 'Needs clean resub' in history.notes:
        resubbed_before = True
        history.status = os.path.split(outfile_path)[
                             -1] + ' job recovery has failed - requesting resub_oscillating_scf() after clean resubmission round'
        history.save()

    if not resubbed_before:
        save_run(outfile_path, rewrite_inscr=False)
        history = resub_history()
        history.read(outfile_path)
        history.resub_number += 1
        history.status = 'precision and grid adjusted to assist convergence'
        history.notes.append('SCF convergence error, precision and grid adjusted to aid convergence')
        history.save()

        machine = tools.get_machine()
        root = outfile_path.rsplit('.', 1)[0]
        name = os.path.split(root)[-1]
        directory = os.path.split(outfile_path)[0]
        infile_dict = manager_io.read_infile(outfile_path)

        home = os.getcwd()
        if len(directory) > 0:  # if the string is blank, then we're already in the correct directory
            os.chdir(directory)
        infile_dict['precision'], infile_dict['dftgrid'], infile_dict['dynamicgrid'] = "double", 5, "no"
        infile_dict['machine'] = machine
        manager_io.write_input(infile_dict)

        manager_io.write_jobscript(name, machine=machine)
        os.chdir(home)
        tools.qsub(root + '_jobscript')
        return True
    else:
        return False


def resub_bad_geo(outfile_path, home_directory):
    """Resubmits a job that's converged to a bad geometry with additional contraints.

    Parameters
    ----------
        outfile_path : str
            The name of an output file.
        home_directory : str
            Path to the base directory of the run.

    Returns
    -------
        Resub_flag : bool
            True if resubmitted.

    """
    # Resubmits a job that's converged to a bad geometry with additional contraints
    history = resub_history()
    history.read(outfile_path)
    resubbed_before = False
    if 'Bad geometry detected, adding constraints and trying again' in history.notes:
        resubbed_before = True
        history.status = os.path.split(outfile_path)[
                             -1] + " has been submitted with additional constraints and still isn't a good geometry"
        history.save()
    if 'Needs clean resub' in history.notes:
        resubbed_before = True
        history.status = os.path.split(outfile_path)[
                             -1] + ' job recovery has failed - requesting resub_bad_geo after clean resubmission round'
        history.save()

    if not resubbed_before:
        save_run(outfile_path, rewrite_inscr=True)
        history = resub_history()
        history.read(outfile_path)
        history.resub_number += 1
        history.status = 'Constraints added to help convergence'
        history.needs_resub = True
        history.notes.append('Bad geometry detected, adding constraints and trying again')
        history.save()

        machine = tools.get_machine()
        root = outfile_path.rsplit('.', 1)[0]
        name = os.path.split(root)[-1]
        directory = os.path.split(outfile_path)[0]
        infile_dict = manager_io.read_infile(outfile_path)

        if infile_dict['constraints']:
            raise Exception(
                'resub.py does not currently support the use of external atom constraints. These will be overwritten by clean_resub() during job recovery')

        goal_geo = manager_io.read_configure(home_directory, outfile_path)['geo_check']
        if not goal_geo:
            raise Exception(
                'Goal geometry not specified, job ' + outfile_path + ' should not have been labelled bad geo!')
        else:
            metal_index, bonded_atom_indices = moltools.get_metal_and_bonded_atoms(outfile_path, goal_geo)
            # convert indexes from zero-indexed to one-indexed
            metal_index += 1
            bonded_atom_indices = [index + 1 for index in bonded_atom_indices]
            # Convert to TeraChem input syntax
            constraints = ['bond ' + str(metal_index) + '_' + str(index) + '\n' for index in bonded_atom_indices]

        home = os.getcwd()
        if len(directory) > 0:  # if the string is blank, then we're already in the correct directory
            os.chdir(directory)

        infile_dict['constraints'] = constraints
        infile_dict['machine'] = machine
        manager_io.write_input(infile_dict)

        manager_io.write_jobscript(name, machine=machine)
        os.chdir(home)
        tools.qsub(root + '_jobscript')
        return True

    else:
        return False


def resub_tighter(outfile_path):
    """Resubmits a thermo job with the gradient error problem. Finds the parent job and resubmits it with a tighter scf
    convergence criteria.

    Parameters
    ----------
        outfile_path : str
            The name of an output file.

    Returns
    -------
        Resub_flag : bool
            True if resubmitted.

    """
    # Takes the path to the outfile of a thermo job with the gradient error problem
    # Finds the parent job and resubmits it with a tighter scf convergence criteria

    name = os.path.split(outfile_path)[-1].rsplit('.', 1)[0]
    parent_name = name.rsplit('_', 1)[0]
    parent_directory = os.path.split(os.path.split(outfile_path)[0])[0]
    parent_path = os.path.join(parent_directory, parent_name + '.out')
    ultratight_path = os.path.join(parent_directory, parent_name + '_ultratight', parent_name + '_ultratight.out')

    scr_needs_to_be_saved = False
    if os.path.exists(ultratight_path):  # This ultratight resubmission has happend before, need to archive the results
        save_run(ultratight_path, rewrite_inscr=False, save_scr_flag=False)
        scr_needs_to_be_saved = True  # Need to save the scr AFTER prepping the new ultratight run. This helps keep compatibility with other functions

        history = resub_history()
        history.read(ultratight_path)
        history.resub_number += 1
        history.status = 'Running with tightened convergence thresholds'
        history.needs_resub = False
        history.notes.append('Further tightening convergence thresholds')
        history.save()

    jobscript = tools.prep_ultratight(parent_path)  # Prep tighter convergence run
    if scr_needs_to_be_saved:
        save_scr(ultratight_path, rewrite_inscr=False)
    tools.qsub(jobscript)  # Submit tighter convergence run

    # Set the original thermo run to wait for the ultratight run to finish
    history = resub_history()
    history.read(outfile_path)
    history.waiting = ultratight_path
    history.save()

    return True


def resub_thermo(outfile_path):
    """Similar to simple resub, but specific for addressing thermo gradient errors.
    hecks for the existance of an ultratight version of this run. If it exists,
    uses the most up to date version for the new thermo run

    Parameters
    ----------
        outfile_path : str
            The name of an output file.

    Returns
    -------
        Resub_flag : bool
            True if resubmitted.

    """
    # Similar to simple resub, but specific for addressing thermo gradient errors
    # Checks for the existance of an ultratight version of this run. If it exists, uses the most up to date version for the new thermo run

    save_run(outfile_path, rewrite_inscr=False)
    history = resub_history()
    history.read(outfile_path)
    history.resub_number += 1
    history.status = 'Normal'
    history.notes.append('Resubmitting thermo, possibly with a better initial geo')
    history.needs_resub = False
    history.save()

    name = os.path.split(outfile_path)[-1]
    name = name.rsplit('.', 1)[0]
    directory = os.path.split(outfile_path)[0]
    parent_name = name.rsplit('_', 1)[0]
    parent_directory = os.path.split(os.path.split(outfile_path)[0])[0]
    ultratight_dir = os.path.join(parent_directory, parent_name + '_ultratight')

    infile_dict = manager_io.read_infile(outfile_path)

    if os.path.exists(ultratight_dir):
        if os.path.exists(os.path.join(ultratight_dir, 'scr', 'optim.xyz')):
            tools.extract_optimized_geo(os.path.join(ultratight_dir, 'scr', 'optim.xyz'))
            shutil.copy(os.path.join(ultratight_dir, 'scr', 'optimized.xyz'), outfile_path.rsplit('.', 1)[0] + '.xyz')
        else:
            raise Exception('Unable to identify the ultratight geometry for run: ' + outfile_path)

        if infile_dict['spinmult'] == 1 and os.path.exists(os.path.join(ultratight_dir, 'scr', 'c0')):
            shutil.copy(os.path.join(ultratight_dir, 'scr', 'c0'), os.path.join(directory, 'c0'))
        elif infile_dict['spinmult'] != 1 and os.path.exists(
                os.path.join(ultratight_dir, 'scr', 'ca0')) and os.path.exists(
                os.path.join(ultratight_dir, 'scr', 'cb0')):
            shutil.copy(os.path.join(ultratight_dir, 'scr', 'ca0'), os.path.join(directory, 'ca0'))
            shutil.copy(os.path.join(ultratight_dir, 'scr', 'cb0'), os.path.join(directory, 'cb0'))
        else:
            raise Exception('Unable to find wavefunction files for ultratight geometry for run: ' + outfile_path)
    else:
        raise Exception(
            'An ultratight run does not exist for this thermo file. Consider calling simple_resub() or resub_tighter() instead of resub_thermo()')

    jobscript = outfile_path.rsplit('.', 1)[0] + '_jobscript'
    tools.qsub(jobscript)
    return True
