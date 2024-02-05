# @file dft_obs.py
#  Contains dft_observation class.
#
#  Written by JP Janet for HJK Group
#
#  Dpt of Chemical Engineering, MIT

from molSimplify.Informatics.lacRACAssemble import (
    generate_all_ligand_autocorrelations,
    generate_all_ligand_deltametrics,
    generate_full_complex_autocorrelations,
    generate_metal_autocorrelations,
    generate_metal_deltametrics,
    generate_all_ligand_misc)
from molSimplify.Classes.mol3D import mol3D

# DFT observations used to postprocess DFT results by measuring ligand properties


class dft_observation:
    def __init__(self, name, geopath):
        self.name = name
        self.descriptors = list()
        self.descriptor_names = list()
        self.mol = False
        self.health = False
        self.comments = list()
        self.geopath = geopath
        self.coord = 'undef'

    def sety(self, y_value):
        self.yvalue = y_value

    def obtain_mol3d(self):
        this_mol = mol3D()
        this_mol.readfromxyz(self.geopath)
        if this_mol.natoms > 0:
            self.mol = this_mol
            self.natoms = this_mol.natoms
            self.health = True
        else:
            self.comments.append('geo file appears empty')
            self.health = False

    def get_coord(self):
        self.coord = len(self.mol.getBondedAtomsSmart(self.mol.findMetal()[0]))

    def get_descriptor_vector(self, lig_only, simple, flag_name=False, loud=False):

        self.get_coord()
        if not lig_only and (self.coord == 6):
            results_dictionary = generate_all_ligand_misc(self.mol, loud)
            self.append_descriptors(
                results_dictionary['colnames'], results_dictionary['result_ax'], 'misc', 'ax')
            self.append_descriptors(
                results_dictionary['colnames'], results_dictionary['result_eq'], 'misc', 'eq')
        print(('after adding misc descriptors... ' +
               str(len(self.descriptor_names))))
        if self.coord == 6:  # oct only
            results_dictionary = generate_all_ligand_autocorrelations(
                self.mol, depth=3, loud=loud, flag_name=flag_name)
            self.append_descriptors(
                results_dictionary['colnames'], results_dictionary['result_ax_full'], 'f', 'ax')
            self.append_descriptors(
                results_dictionary['colnames'], results_dictionary['result_eq_full'], 'f', 'eq')
            print(('after adding full ax/eq descriptors... ' +
                   str(len(self.descriptor_names))))
            if not simple and not lig_only:
                self.append_descriptors(
                    results_dictionary['colnames'], results_dictionary['result_ax_con'], 'lc', 'ax')
                self.append_descriptors(
                    results_dictionary['colnames'], results_dictionary['result_eq_con'], 'lc', 'eq')
                results_dictionary = generate_all_ligand_deltametrics(
                    self.mol, depth=3, loud=True, flag_name=flag_name)
                self.append_descriptors(
                    results_dictionary['colnames'], results_dictionary['result_ax_con'], 'D_lc', 'ax')
                self.append_descriptors(
                    results_dictionary['colnames'], results_dictionary['result_eq_con'], 'D_lc', 'eq')

                print(('after adding lc ax/eq descriptors... ' +
                       str(len(self.descriptor_names))))
        if not lig_only:
            if not simple:
                results_dictionary = generate_metal_autocorrelations(
                    self.mol, depth=3, loud=loud)
                self.append_descriptors(
                    results_dictionary['colnames'], results_dictionary['results'], 'mc', 'all')
                results_dictionary = generate_metal_deltametrics(
                    self.mol, depth=3, loud=loud)
                self.append_descriptors(
                    results_dictionary['colnames'], results_dictionary['results'], 'D_mc', 'all')
            results_dictionary = generate_full_complex_autocorrelations(
                self.mol, depth=3, loud=loud)
            self.append_descriptors(
                results_dictionary['colnames'], results_dictionary['results'], 'f', 'all')
        print(('after adding full complex descriptors... ' +
               str(len(self.descriptor_names))))

    def append_descriptors(self, list_of_names, list_of_props, prefix, suffix):
        for names in list_of_names:
            if hasattr(names, '__iter__'):
                names = ["-".join([prefix, str(i), suffix]) for i in names]
                self.descriptor_names += names
            else:
                names = "-".join([prefix, str(names), suffix])
                self.descriptor_names.append(names)
        for values in list_of_props:
            if hasattr(values, '__iter__'):
                self.descriptors.extend(values)
            else:
                self.descriptors.append(values)


def write_descriptor_csv(list_of_runs):
    with open('descriptor_file.csv', 'w') as f:
        f.write('runs,')
        n_cols = len(list_of_runs[0].descriptor_names)
        for i, names in enumerate(list_of_runs[0].descriptor_names):
            if i < (n_cols-1):
                f.write(names+',')
            else:
                f.write(names+'\n')
        for runs in list_of_runs:
            try:
                f.write(runs.name)
                for properties in runs.descriptors:
                    f.write(','+str(properties))
                f.write('\n')
            except AttributeError:
                pass
