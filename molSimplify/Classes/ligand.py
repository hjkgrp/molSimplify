# @file ligand.py
#  Defines ligand class for postprocessing DFT results by measuring ligand properties
#
#  Written by JP Janet for HJK Group
#
#  Dpt of Chemical Engineering, MIT
import numpy as np
from itertools import combinations
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.globalvars import bondivdw, vdwrad
from molSimplify.utils.decorators import deprecated
from molSimplify.Scripts.nx_helpers import mol3D_to_networkx
import networkx as nx


# Ligand class for postprocessing DFT results by measuring ligand properties
class ligand:
    """Ligand class for postprocessing DFT results to measure ligand properties.

    Parameters
    ----------
        master_mol : mol3D
            mol3D class instance for ligand mol.
        index_list : list
            List for ligand connection atom indices.
        dent : int
            Denticity of the ligand
        read_lig : bool, optional
            str of file or (xyz/mol2 string) of saved ligand, by default False
    Raises
    ------
        ValueError
        If read_ligand type is unknown.
    """
    def __init__(self, master_mol, index_list, dent, read_lig=False):
        if isinstance(read_lig, str):
            thismol = mol3D()
            if 'TRIPOS' in read_lig:
                thismol.readfrommol2(read_lig, readstring=True)
            # Xyz filename
            elif read_lig[-4:] == '.xyz':
                thismol.readfromxyz(read_lig)
            # mol2 filename
            elif read_lig[-5:] == '.mol2':
                thismol.readfrommol2(read_lig)
            # checking for number at start of string -> indicates xyz string
            elif (len(read_lig.split('\n')) > 3) & (read_lig.split('\n')[0].replace(' ', '').isnumeric()):
                thismol.readfromstring(read_lig)
            # checking for similar file without header
            elif (len(read_lig.split('\n')[0].split()) == 4) and read_lig.split('\n')[0].split()[0]:
                thismol.readfromstring(read_lig)
            else:
                raise ValueError('Not Recognized Structure Type!')
            self.master_mol = thismol
            met = thismol.findMetal()[0]  # Pull out metal
            self.index_list = [x for x in range(thismol.natoms) if x != met]
            self.dent = len(thismol.getBondedAtomsSmart(met))
            self.ext_int_dict = {i: j for i, j in enumerate(self.index_list)}
            self.mol2string = thismol.writemol2('ligand', writestring=True)
            self.lig_mol_graph_det = thismol.get_mol_graph_det()
            self.percent_buried_volume = False
        else:
            self.master_mol = master_mol
            self.index_list = sorted(index_list)
            self.dent = dent
            self.ext_int_dict = dict()  # store
            self.mol2string = False
            self.lig_mol_graph_det = False
            self.percent_buried_volume = False

    def __repr__(self):
        if hasattr(self, 'mol'):
            return f'ligand({self.mol.make_formula(latex=False)})'
        else:
            # Default python repr:
            return object.__repr__(self)

    def obtain_mol3d(self):
        """Getting the mol3D of the ligand. Deprecated. Map between int and ext indcies.
        Obtain the ligand from the complex mol3D object.
        """
        this_mol = mol3D()
        this_ext_int_dict = dict()
        j = 0
        # the old routine where all atoms in the master_mol are gone through from 0 to natoms-1
        # for i in range(0, self.master_mol.natoms):
        #     if i in self.index_list:
        # the new rountine where the indices are taken out directly. This way the order of atoms is preserved
        for i in self.index_list:
            this_mol.addAtom(self.master_mol.getAtom(i))
            this_ext_int_dict.update({i: j})
            j += 1  # keep count of how many are added
        self.mol = this_mol
        if len(self.master_mol.graph):  # Save graph to ligand mol3D object
            delete_inds = [x for x in range(self.master_mol.natoms) if x not in self.index_list]
            self.mol.graph = np.delete(np.delete(self.master_mol.graph, delete_inds, 0), delete_inds, 1)
        if self.master_mol.bo_dict:
            save_bo_dict = self.master_mol.get_bo_dict_from_inds(self.index_list)
            self.mol.bo_dict = save_bo_dict
        if hasattr(self.master_mol, 'prop_tensor'):
            prop_tensor = np.zeros(shape=(len(self.index_list), len(self.master_mol.prop_name)))
            c = 0
            for i in self.index_list:
                prop_tensor[c, :] = self.master_mol.prop_tensor[i, :]
                c += 1
            self.mol.prop_tensor = prop_tensor
            self.mol.prop_name = self.master_mol.prop_name
        self.ext_int_dict = this_ext_int_dict

    def get_lig_mol2(self, transition_metals_only=True, inds=None, include_metal=True, bimetal=False):
        """Write out ligand mol2 string and molecular graph determinant.
        Include Metal flagged with Symbol "X" for placeholder status.
        Parameters
        ----------
            transition_metals_only : bool
                flag only transition metals with findMetal() function.
            inds : list
                indices of metals if passing in multimetal system

        Returns
        -------
            lig_mol_graph_det : str
                Molecular graph determinant.
            ligand_mol2_string : str
                Mol2 string for the ligand.
            catoms_indices : list
                List of catom (connection atom) indices - only returned if include_metal is set to False. Zero-indexed.

        """
        this_mol2 = mol3D()
        if inds:
            metal_ind = inds
        else:
            metal_ind = self.master_mol.findMetal(transition_metals_only=transition_metals_only)
        this_mol2_inds = self.index_list.copy()
        if include_metal:
            this_mol2_inds += metal_ind
        this_mol2_inds = sorted(this_mol2_inds)

        if (not bimetal):
            # Set up a binary vector indicating whether each atom is a connecting atom (1) or not (0)
            catoms_indices = self.master_mol.getBondedAtomsSmart(metal_ind)
            catom_selector = np.zeros(self.master_mol.natoms)
            catom_selector[catoms_indices] = 1

        # Add the metal with symbol = 'M'
        new_metal_inds = []
        for j, i in enumerate(this_mol2_inds):
            if i in metal_ind:
                atom = self.master_mol.getAtom(i)
                atom.mutate('X')
                this_mol2.addAtom(atom)
                new_metal_inds.append(j)
            else:
                this_mol2.addAtom(self.master_mol.getAtom(i))
        if len(self.master_mol.graph):  # Save graph to ligand mol3D object
            delete_inds = [x for x in range(self.master_mol.natoms) if x not in this_mol2_inds]
            this_mol2.graph = np.delete(np.delete(self.master_mol.graph, delete_inds, 0), delete_inds, 1)
            if (not bimetal):
                catom_selector = np.delete(catom_selector, delete_inds)
                catoms_indices = np.nonzero(catom_selector)[0]

        # #### Check for multiple metal centers. Save more coordinated one.
        if include_metal:
            if len(new_metal_inds) == 2:
                minds = new_metal_inds
                metal_cns = [len(this_mol2.getBondedAtomsSmart(x)) for x in minds]
                delind = minds[np.argmin(metal_cns)]
                this_mol2.deleteatom(minds[np.argmin(metal_cns)])
                del this_mol2_inds[delind]
        if self.master_mol.bo_dict:
            save_bo_dict = self.master_mol.get_bo_dict_from_inds(this_mol2_inds)
            this_mol2.bo_dict = save_bo_dict
        try:
            lig_mol_graph_det = this_mol2.get_mol_graph_det()
        except:
            lig_mol_graph_det = None
        lig_mol2_string = this_mol2.writemol2('ligand', writestring=True)
        self.mol2string = lig_mol2_string
        self.lig_mol_graph_det = lig_mol_graph_det

        if include_metal or bimetal:
            return lig_mol_graph_det, lig_mol2_string
        else:
            return lig_mol_graph_det, lig_mol2_string, catoms_indices

    def percent_buried_vol(self, radius=3.5, gridspec=0.1,
                           bondiscale=1.17, hydrogens=True):
        """Calculate the percent buried volume as described in https://doi.org/10.1039/B922984A,
        and https://chemistry-europe.onlinelibrary.wiley.com/doi/abs/10.1002/ejic.200801160.
        Bondi VDW radii are used where possible

        Parameters
        ----------
        radius : float, optional
            Radius of sphere around metal center consider in Angstroms, by default 3.5
            Paper recommends 3.5
        gridspec : float, optional
            xyz grid spacing in Angstroms, by default 0.1
            Paper recommends 0.05 - this is slower
        bondiscale : float, optional
            Scale of the bondi vdw radii for ligand atoms, by default 1.17
            Paper recommends 1.17
        hydrogens : bool, optional
            Include hydrogens in percent buried volume, by default True
            Paper recommends to not include hydrogens.

        Returns
        -------
        percent_buried_volume : float
            Percentage of the volume of a sphere around the metal that is buried under ligand atoms.
            Represents bulkiness of the ligand around the metal center.
        """
        if not self.mol2string:
            _, _ = self.get_lig_mol2()
        thismol = mol3D()
        thismol.readfrommol2(self.mol2string, readstring=True)
        # Accounting for hydrogens by default - otherwise deleting.
        if not hydrogens:
            thismol.deleteHs()
        natoms = thismol.natoms
        # Catch case where just hydrogen ligand
        if natoms == 1:
            percent_buried = 0.0
        else:
            met = thismol.findMetal()[0]
            coords = thismol.coordsvect() - thismol.coordsvect()[met]  # Center coordinates to the metal
            syms = [x for i, x in enumerate(thismol.symvect()) if i != met]
            radvect = []
            for sym in syms:
                if sym in bondivdw:  # Take in the bondivdw radii reccomeneded in paper
                    radvect.append(bondivdw[sym]*bondiscale)
                else:  # Else take in Newer definition of vdw radii
                    radvect.append(vdwrad[sym]*bondiscale)
            radiusvect = np.array(radvect)
            inds = np.array([x for x in range(natoms) if x != met])
            coords = coords[inds]  # Get rid of metal location from coords
            x_ = np.arange(-radius, radius, gridspec)
            y_ = np.arange(-radius, radius, gridspec)
            z_ = np.arange(-radius, radius, gridspec)
            grid = np.meshgrid(x_, y_, z_, indexing='ij')
            mgrid = list(map(np.ravel, grid))
            combined = np.vstack(mgrid).T  # Flatten meshgrid to nx3
            init_coords = np.array([[0, 0, 0]])  # We have set metal to (0,0,0)
            # Get distance of all gridpoints from (0,0,0) -> Filter out gridpoints further
            init_dists = np.linalg.norm(init_coords[:, None, :]-combined[None, :, :], axis=-1)
            # Get rid of gridpoints further away than radius and metal
            combined = combined[np.where(init_dists[0, :] <= radius)[0]]
            # Get rid of ligand atoms that won't interact at all
            met_ds = np.linalg.norm(init_coords[:, None, :]-coords[None, :, :], axis=-1)
            met_ds = met_ds - radius - radiusvect
            mfilter = np.where(met_ds[0] < 0)
            coords = coords[mfilter]
            radiusvect = radiusvect[mfilter]
            # Calc distance of all remaining atomic coords to all grid points
            ds = np.linalg.norm(coords[:, None, :]-combined[None, :, :], axis=-1)
            # Compare distances to radii of atoms, flag any less than than or equal to the radii as buried
            buried = len(np.where(np.any(np.less_equal(ds[:, :], radiusvect[:, None]), axis=0))[0])
            total = ds.shape[1]
            # Percent buried volume
            percent_buried = float(buried)/total*100
        self.percent_buried_volume = percent_buried
        return percent_buried


def ligand_breakdown(mol, BondedOct=False, silent=True, transition_metals_only=True):
    """Extract axial and equatorial components of a octahedral complex.

    Parameters
    ----------
        mol : mol3D
            mol3D class instance for the complex.
        BondedOct : bool, optional
            Enforce octahedral bonding. Default is False.
        silent : bool, optional
            Silence extra printout. Default is True.

    Returns
    -------
        liglist : list of list of int
            List of ligands. Length is the number of ligands. Each inner list contains the global indices of a ligand
        ligdents : list of int
            List of ligand denticities. Length is the number of ligands
        ligcons : list of list of int
            List of ligand connection indices (in mol). Length is the number of ligands

    """
    # this function takes an octahedral
    # complex and returns ligands
    metal_index = mol.findMetal(transition_metals_only=transition_metals_only)[0]
    # print(BondedOct)
    bondedatoms = mol.getBondedAtomsSmart(metal_index, oct=BondedOct)
    # print('!!!!!boundatoms', bondedatoms)
    # print('from get oct' + str(bondedatoms))
    # print('***\n')
    bonded_atom_symbols = [mol.getAtom(i).symbol() for i in bondedatoms]
    if not silent:
        print(('result of ligand ligand_breakdown', bonded_atom_symbols))
    liglist = []
    ligdents = []
    ligcons = []
    for atom in bondedatoms:
        if not silent:
            print(('this atom type is ' + mol.getAtom(atom).symbol()))
            print(('conection number ' + str(atom) + " of " + str(bondedatoms)))
        fragment = mol.findsubMol(atom, metal_index)
        this_cons = [x for x in fragment if (x in bondedatoms)]
        if not silent:
            print(('fragment', fragment))
            print(('this_cons', this_cons))
        unique = True
        for i, unique_ligands in enumerate(liglist):
            if sorted(fragment) == sorted(unique_ligands):
                unique = False
                matched = i
        if unique:
            liglist.append(sorted(fragment))
            ligdents.append(1)
            ligcons.append(this_cons)
        else:
            ligdents[matched] += 1
    return liglist, ligdents, ligcons


@deprecated('Use ligand_assign_consistent instead.')
def ligand_assign_original(mol, liglist, ligdents, ligcons, loud=False, name=False, eq_sym_match=False):
    """Assign axial and equatorial portions. Deprecated. Use ligand_assign_consistent. For octahedral geometries.

    Parameters
    ----------
        mol : mol3D
            mol3D class instance for the complex.
        liglist : list
            List of ligands
        ligdents : list
            List of ligand denticities
        ligcons : list
            List of ligand connection indices (in mol)
        loud : bool, optional
            Enable extra printout. Default is False.
        name : bool, optional
            Name ligands to write to XYZ. Default is False. Broken, do not use.
        eq_sym_match : bool, optional
            Default is False. Broken, do not use.


    Returns
    -------
        ax_ligand_list : list
            Ligands designated as axial ligands.
        eq_ligand_list : list
            Ligands designated as equatorial ligands.
        ax_natoms_list : list
            Number of atoms in axial ligands.
        eq_natoms_list : list
            Number of atoms in equatorial ligands.
        ax_con_int_list : list
            Connecting atoms indices of axial ligands.
        eq_con_int_list : list
            Connecting atoms indices of equatorial ligands.
        ax_con_list : list
            Connecting atoms of axial ligands.
        eq_con_list : list
            Connecting atoms of equatorial ligands.
        built_ligand_list : list
            List of ligand classes for all ligands.

    """
    valid = True
    # loud = False
    pentadentate = False
    hexadentate = False
    built_ligand_list = list()
    lig_natoms_list = list()
    unique_ligands = list()
    ligand_counts = list()
    all_ligand_counts = [0, 0, 0, 0, 0, 0]
    ligand_records = list()
    ax_con_int_list = list()
    eq_con_int_list = list()
    ax_natoms_list = list()
    eq_natoms_list = list()
    n_ligs = len(liglist)
    max_dent = max(ligdents)
    min_dent = min(ligdents)
    if loud:
        print('********************************************')
        print(("n_ligs = " + str(n_ligs)))
        print(("max d = " + str(max_dent)))
        print(("min_dent = " + str(min_dent)))
        print(("ligand list is" + str(liglist)))
        print(('denticities are  ' + str(ligdents)))
    if (max(ligdents) == 4) and (min(ligdents) != 1):
        valid = False
        print(('bad denticities: ' + str(ligdents)))
        print(('min denticities: ' + str(min(ligdents))))
    if max(ligdents) > 4:
        # ### Handling of pentadentate ligands goes here. #####
        if max(ligdents) == 5 and min(ligdents) == 1:
            pentadentate = True
        elif max(ligdents) == 6 and min(ligdents) == 6:
            hexadentate = True
        else:
            valid = False
            print(('bad denticities: ' + str(ligdents)))
            print(('max denticities: ' + str(min(ligdents))))
    if n_ligs > 3 and min(ligdents) > 1:
        valid = False
        print(('too many ligs ' + str((n_ligs))))
    eq_lig_list = list()
    ax_lig_list = list()
    ax_con_list = list()
    eq_con_list = list()
    for i, ligand_indices in enumerate(liglist):
        this_ligand = ligand(mol, ligand_indices, ligdents[i])
        this_ligand.obtain_mol3d()
        # lig_natoms_list.append(this_ligand.mol.natoms) ## old one with obtain_mol3d
        built_ligand_list.append(this_ligand)
        lig_natoms_list.append(len(this_ligand.index_list))
    for j, built_ligs in enumerate(built_ligand_list):
        # test if ligand is unique
        sl = [atom.symbol() for atom in built_ligs.master_mol.getAtomwithinds(built_ligs.index_list)]
        # _sl = [atom.symbol() for atom in built_ligs.mol.getAtoms()] ## old one with obtain_mol3d
        if loud:
            print(('checking lig ' + str(j) + ' : ' + str(sl)))
        unique = 1
        for i, other_sl in enumerate(unique_ligands):
            if sorted(sl) == sorted(other_sl):
                # duplicate
                unique = 0
                ligand_counts[i] += 1
        if unique == 1:
            unique_ligands.append(sl)
            ligand_counts.append(1)
            ligand_records.append(j)
    # loop to bin ligands:
    for j, built_ligs in enumerate(built_ligand_list):
        # test if ligand is unique
        sl = [atom.symbol() for atom in built_ligs.master_mol.getAtomwithinds(built_ligs.index_list)]
        for i, other_sl in enumerate(unique_ligands):
            if sorted(sl) == sorted(other_sl):
                # duplicate
                # print(i,ligand_counts[i])
                all_ligand_counts[j] = ligand_counts[i]

    if loud:
        print(('unique ligands' + str(unique_ligands)))
        print(('ligand counts' + str(ligand_counts)))
        print(('ligand records ' + str(ligand_records)))
        print((str(max(ligand_counts)) +
               ' is the max and min in  ' + str(min(ligand_counts))))
    n_unique_ligs = len(unique_ligands)
    if (n_ligs == 3) or (n_ligs == 4):  # most common case,
        # one/two equatorial and 2 axial mono
        # or three bidentate
        for i, ligs in enumerate(liglist):
            if ligdents[i] == 1 and min_dent == 1:  # anything with equatorial monos will
                # have higher than 4 n_ligs
                ax_lig_list.append(i)
                if loud:
                    print(('choosing ' + str(i) + ' as ax based on dent =1'))
                ax_con_list.append(ligcons[i])
            if (ligdents[i] >= 2) and (min_dent == 1):
                eq_lig_list.append(i)
                if loud:
                    print(('choosing lig ' + str(i) + ' as eq based on high dent'))
                eq_con_list.append(ligcons[i])
        if (n_ligs == 3) and (min_dent == max_dent):
            if n_unique_ligs == 1:
                # take any 2, they are all the same
                if loud:
                    print('triple bidentate case')
                ax_lig_list.append(0)
                eq_lig_list.append(1)
                eq_lig_list.append(2)
                ax_con_list.append(ligcons[0])
                eq_con_list.append(ligcons[1])
                eq_con_list.append(ligcons[2])
            elif min_dent == 2 and max_dent == 2 and n_ligs == 3 and not n_unique_ligs == 1:
                # this is a hetero/bidentate case
                for i, ligs in enumerate(liglist):
                    if all_ligand_counts[i] == 2:
                        eq_lig_list.append(i)
                        eq_con_list.append(ligcons[i])
                    elif all_ligand_counts[i] == 1:
                        ax_lig_list.append(i)
                        ax_con_list.append(ligcons[i])
    elif (n_ligs == 6):  # all mono  case,
        minz = 500
        maxz = -500
        if loud:
            print('monodentate case')
        allowed = list(range(0, 6))
        not_eq = list()
        for j, built_ligs in enumerate(built_ligand_list):
            this_z = sum([mol.getAtom(ii).coords()[2]
                          for ii in ligcons[j]]) / len(ligcons[j])
            if this_z < minz:
                minz = this_z
                bot_lig = j
                bot_con = ligcons[j]
            if loud:
                print(('updating bot axial to ' + str(bot_lig)))
            if this_z > maxz:
                maxz = this_z
                top_lig = j
                top_con = ligcons[j]
            if loud:
                print(('updating top axial to ' + str(top_lig)))
        not_eq.append(bot_lig)
        not_eq.append(top_lig)

        allowed = [x for x in allowed if ((x not in not_eq))]
        if len(allowed) != 4:
            print(('error in decomp of monodentate case!', allowed))
        eq_lig_list = allowed
        eq_con_list = [ligcons[i] for i in allowed]
        ax_lig_list = [top_lig, bot_lig]
        ax_con_list = [top_con, bot_con]
        if loud:
            print(('geometric eq_list ' + str(eq_lig_list)))
            print(('geometric ax_list ' + str(eq_lig_list)))
        if (max(ligand_counts) != 4) or (min(ligand_counts) != 2):
            if loud:
                print('not a 4-6 case')
            if (max(ligand_counts) == 6):
                if loud:
                    print('6-homoleptic, using geo values')
            # ax=ligand_records[ligand_counts.index(6)]
            # eq_lig=ligand_records[ligand_counts.index(6)]
            else:
                if loud:
                    print('monodentates not the same, using geo values ')
                    print(ligand_counts)
                    print(unique_ligands)
        elif n_unique_ligs == 2:
            if loud:
                print('this is a  4-6 case')
            allowed = list(range(0, 6))
            ax_lig_list = [i for i in allowed if (all_ligand_counts[i] == 2)]
            eq_lig_list = [i for i in allowed if (all_ligand_counts[i] == 4)]
            ax_con_list = [ligcons[i] for i in ax_lig_list]
            eq_con_list = [ligcons[i] for i in eq_lig_list]
    elif n_ligs == 2 and pentadentate:
        # ### Handling for pentadentate scaffolds ####
        if loud:
            print('pentadentate case')
        allowed = [0, 1]
        not_eq = list()
        for j, built_ligs in enumerate(built_ligand_list):
            if len(ligcons[j]) == 1:
                # ### This is the axial ligand ####
                # print(j, 'axial lig')
                top_lig = j
                top_con = ligcons[j]
                not_eq.append(top_lig)
            else:
                pentadentate_coord_list = np.array([mol.getAtom(
                    ii).coords() for ii in ligcons[j]])
                # #### Adjusting this so that by default, any 4 within the same plane will be assigned as eq. ###
                if loud:
                    print('pentadentate coord LIST!')
                    print(pentadentate_coord_list)
                point_combos = combinations([0, 1, 2, 3, 4], 4)
                error_list = []
                combo_list = []
                for i, combo in enumerate(point_combos):
                    combo_list.append(list(combo))
                    A = np.zeros((4, 3))
                    b = np.zeros(4)
                    for point_i, point_num in enumerate(combo):
                        coordlist = pentadentate_coord_list[point_num]
                        A[point_i, :] = [coordlist[0], coordlist[1], 1]
                        b[point_i] = coordlist[2]
                    # #### This code builds the best fit plane between 4 points,
                    # #### Then calculates the variance of the 4 points with respect to the plane
                    # #### The 4 that have the least variance are flagged as the eq plane.
                    try:
                        fit = np.linalg.pinv(A.T @ A) @ A.T @ b
                        errors = b - A @ fit
                        error_var = np.var(errors)
                        error_list.append(error_var)
                    except np.linalg.LinAlgError:
                        error_list.append(0)
                if loud:
                    print('combos below')
                    print(combo_list)
                    print('errors next, argmin combo selected')
                    print(error_list)
                not_ax_points = combo_list[np.argmin(error_list)]
                if len(set(not_ax_points)) != 4:
                    print('The equatorial plane is not being assigned correctly. Please check.')
                    # sardines
                else:
                    bot_idx = list(set(range(5)) - set(not_ax_points))[0]
                    if loud:
                        print(('This is bot_idx', bot_idx))
                    bot_lig = j
                    bot_con = [ligcons[j][bot_idx]]

        allowed = list(set(allowed) - set(not_eq))
        if loud:
            print(('this is the allowed list', allowed, not_eq))
        eq_lig_list = allowed
        eq_con_list = [
            list(set([ligcons[i] for i in allowed][0]) - set(top_con) - set(bot_con))]
        ax_lig_list = [top_lig, bot_lig]
        ax_con_list = [top_con, bot_con]
        if loud:
            print(('con lists', eq_con_list, ax_con_list))
        ###########################################################################################
        # In the above, the pentadentate ligand is classified as both axial and equatorial.       #
        # The lc atoms are decided by the z-position. Thus the pentadentate ligand has 4 eq-lc    #
        # and 1 ax-lc. Currently should be able to check this and set that up.                    #
        ###########################################################################################
    elif n_ligs == 1 and hexadentate:
        allowed = [0, 1]
        if loud:
            print('hexadentate case')
        not_eq = list()
        for j, built_ligs in enumerate(built_ligand_list):
            hexadentate_coord_list = np.array([mol.getAtom(
                ii).coords() for ii in ligcons[j]])
            # #### Adjusting this so that by default, any 4 within the same plane will be assigned as eq. ###
            if loud:
                print('hexadentate coord LIST!')
                print(hexadentate_coord_list)
            # point_combos = combinations([1,2,3,4,5,6],4)
            pair_combos = list(combinations([0, 1, 2, 3, 4, 5], 2))
            angle_list = []
            pair_list = []
            for i, pair in enumerate(pair_combos):
                pair_list.append(list(pair))
                p1 = np.squeeze(np.array(hexadentate_coord_list[list(pair)[0]]))
                p2 = np.squeeze(np.array(hexadentate_coord_list[list(pair)[1]]))
                m = np.array([mol.getAtom(mol.findMetal()[0]).coords()])
                v1u = np.squeeze(np.array((m - p1) / np.linalg.norm((m - p1))))
                v2u = np.squeeze(np.array((m - p2) / np.linalg.norm((m - p2))))
                # print('v1v2',v1u,v2u)
                angle = np.rad2deg(np.arccos(np.clip(np.dot(v1u, v2u), -1.0, 1.0)))
                if loud:
                    print(('pair of atoms, then angle', pair, angle))
                angle_list.append(angle)
            argsort_angle_list = np.squeeze(np.array(angle_list)).argsort()[-3:][::-1]
            point_combos = [pair_list[argsort_angle_list[0]] + pair_list[argsort_angle_list[1]],
                            pair_list[argsort_angle_list[1]] + pair_list[argsort_angle_list[2]],
                            pair_list[argsort_angle_list[2]] + pair_list[argsort_angle_list[0]]]
            error_list = []
            combo_list = []
            fitlist = []
            for i, combo in enumerate(point_combos):
                combo_list.append(combo)
                A = np.zeros((4, 3))
                b = np.zeros(4)
                for point_i, point_num in enumerate(combo):
                    coordlist = hexadentate_coord_list[point_num]
                    A[point_i, :] = [coordlist[0], coordlist[1], 1]
                    b[point_i] = coordlist[2]
                # #### This code builds the best fit plane between 4 points,
                # #### Then calculates the variance of the 4 points with respect to the plane
                # #### The 4 that have the least variance are flagged as the eq plane.
                fit = np.linalg.pinv(A.T @ A) @ A.T @ b
                fitlist.append(fit)
                errors = b - A @ fit
                error_var = np.var(errors)
                error_list.append(error_var)
            if loud:
                print('combos below')
                print(combo_list)
                print('errors next, argmin combo selected')
                print(error_list)
            best_fit_planes = np.squeeze(np.array(error_list)).argsort()[:3]
            perpdist = []
            perpcombo = []
            for fitnum, best_fit in enumerate(best_fit_planes):
                temp_fit = fitlist[best_fit]
                temp_combo = combo_list[best_fit]
                perpcombo.append(int(best_fit))
                temp_ax = set(range(0, 6)) - set(temp_combo)
                ax_dist = []
                for point_num in temp_ax:
                    coordlist = hexadentate_coord_list[point_num]
                    planez = [coordlist[0], coordlist[1], 1] * temp_fit
                    # planez = temp_fit[0] * coordlist[0] + temp_fit[1] * coordlist[1] + fit[2]
                    plane_coords = [coordlist[0], coordlist[1], planez]
                    adjusted_coords = [coordlist[0], coordlist[1], coordlist[2]]
                    squared_dist = np.sum((np.array(adjusted_coords) - np.array(plane_coords)) ** 2)
                    dist = np.squeeze(np.sqrt(squared_dist))
                    if loud:
                        print(('dist', dist))
                    ax_dist.append(dist)
                perpdist.append(np.mean(ax_dist))
            if loud:
                print(("Perpendicular distance is", perpdist, perpcombo, len(perpdist), len(best_fit_planes)))
            not_ax_points = combo_list[perpcombo[np.argmax(np.array(perpdist))]]
            if len(set(not_ax_points)) != 4:
                print('The equatorial plane is not being assigned correctly. Please check.')
                # sardines
            else:
                bot_idx = list(set(range(6)) - set(not_ax_points))[0]
                top_idx = list(set(range(6)) - set(not_ax_points))[1]
                if loud:
                    print(('This is bot_idx', bot_idx))
                bot_lig = j
                top_lig = j
                bot_con = [ligcons[j][bot_idx]]
                top_con = [ligcons[j][top_idx]]
        allowed = list(set(allowed) - set(not_eq))
        if loud:
            print(('this is the allowed list', allowed, not_eq))
        eq_lig_list = [top_lig]
        eq_con_list = eq_con_list = [
            list(set([ligcons[0][i] for i in not_ax_points]))]
        ax_lig_list = [top_lig, bot_lig]
        ax_con_list = [top_con, bot_con]
        if loud:
            print(('con lists', eq_con_list, ax_con_list))

    # ############## DONE WITH CLASSIFICATION ######
    # ax_lig=ligand_records[ligand_counts.index(2)]
    # eq_lig=ligand_records[ligand_counts.index(4)]
    ax_ligand_list = [built_ligand_list[i] for i in ax_lig_list]
    eq_ligand_list = [built_ligand_list[i] for i in eq_lig_list]
    if loud and valid:
        print(('lig_nat_list', lig_natoms_list))
        print(('eq_liq is ind ', eq_lig_list))
        print(('ax_liq is ind ', ax_lig_list))
        print(('ax built lig [0] ext ind :' +
               str(list(built_ligand_list[ax_lig_list[0]].ext_int_dict.keys()))))
        if len(ax_lig_list) > 1:
            print(('ax built lig [1] ext ind :' +
                   str(list(built_ligand_list[ax_lig_list[1]].ext_int_dict.keys()))))
        print(('eq built lig [0] ext ind: ' +
               str(list(built_ligand_list[eq_lig_list[0]].ext_int_dict.keys()))))
        print(('eq_con is ' + str((eq_con_list))))
        print(('ax_con is ' + str((ax_con_list))))
    for j, ax_con in enumerate(ax_con_list):
        current_ligand_index_list = built_ligand_list[ax_lig_list[j]].index_list
        ax_con_int_list.append([current_ligand_index_list.index(i) for i in ax_con])
        # ax_con_int_list.append(
        #     [built_ligand_list[ax_lig_list[j]].ext_int_dict[i] for i in ax_con])
        # # convert to interal index ## old one with obtain_mol3d
    for j, eq_con in enumerate(eq_con_list):
        current_ligand_index_list = built_ligand_list[eq_lig_list[j]].index_list
        eq_con_int_list.append([current_ligand_index_list.index(i) for i in eq_con])
        # eq_con_int_list.append(
        #     [built_ligand_list[eq_lig_list[j]].ext_int_dict[i] for i in eq_con])
        # # convert to interal index ## old one with obtain_mol3d
    if loud:
        print(('int eq ' + str(eq_con_int_list)))
        print(('ext eq ' + str(eq_con_list)))
        print('**********************************************')
    for ax_lig in ax_lig_list:
        ax_natoms_list.append(lig_natoms_list[ax_lig])
    for eq_lig in eq_lig_list:
        eq_natoms_list.append(lig_natoms_list[eq_lig])
    return (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list,
            ax_con_int_list, eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list)


# ## DISCLAIMER!!! Please be careful while modifying any part of 'ligand_assign_consistent'
# as that could affect everything else ###
def ligand_assign_consistent(mol, liglist, ligdents, ligcons, loud=False,
                             name=False, use_z=False, eq_sym_match=False):
    """This ligand assignment code handles octahedral complexes consistently. Assigns any octahedral complex.

    Parameters
    ----------
        mol : mol3D
            mol3D class instance for the complex.
        liglist : list
            List of ligands
        ligdents : list
            List of ligand denticities
        ligcons : list
            List of ligand connection indices (in mol)
        loud : bool, optional
            Enable extra printout. Default is False.
        name : bool, optional
            Name ligands to write to XYZ. Default is False.
        use_z : bool, optional
            Use z-dimension to determine axial ligands, as oppossed to connectivity.
        eq_sym_match : bool, optional
            Enforce eq plane to have connecting atoms with same symbol. Default is False.


    Returns
    -------
        ax_ligand_list : list
            Ligands designated as axial ligands.
        eq_ligand_list : list
            Ligands designated as equatorial ligands.
        ax_natoms_list : list
            Number of atoms in axial ligands.
        eq_natoms_list : list
            Number of atoms in equatorial ligands.
        ax_con_int_list : list
            Connecting atoms indices of axial ligands.
        eq_con_int_list : list
            Connecting atoms indices of equatorial ligands.
        ax_con_list : list
            Connecting atoms of axial ligands.
        eq_con_list : list
            Connecting atoms of equatorial ligands.
        built_ligand_list : list
            List of ligand classes for all ligands.

    """
    # ###### This ligand assignment code handles octahedral complexes consistently.
    # ###### It should be able to assign any octahedral complex
    angle_cutoff = 130  # Angle cutoff for linear
    valid = True
    hexadentate = False
    pentadentate = False
    metal_index = mol.findMetal()[0]  # Get metal index and coords
    m_coord = np.array(mol.getAtom(metal_index).coords())
    ligand_records = list()
    ax_con_int_list = list()  # Atom refs in for ligand mol3D objects in ax
    eq_con_int_list = list()  # Atom refs in for ligand mol3D objects in eq
    ax_natoms_list = list()
    eq_natoms_list = list()
    eq_lig_list = list()
    ax_lig_list = list()
    ax_con_list = list()
    eq_con_list = list()
    symbol_list = list()
    n_ligs = len(liglist)
    max_dent = max(ligdents)
    min_dent = min(ligdents)
    # ##### Utility functions for ligand MW and Angles

    def getMW(lig):  # Get total MW of ligand mol3d object
        mol = lig.master_mol
        lig_idx = lig.index_list
        mw = 0
        for i, atom in enumerate(mol.getAtoms()):
            if i in lig_idx:
                mw += atom.mass
        return mw
    # ## Below, take all combinations of two atoms, and measure their angles through the metal center

    def getAngle(coord_list, pair, m_coord):  # Get Angle of atom pair through metal center (stored at coord_list[0])
        # print("coord_list: ", coord_list)
        # print("pair: ", pair)
        try:
            p1 = np.squeeze(np.array(coord_list[pair[0]]))
            p2 = np.squeeze(np.array(coord_list[pair[1]]))
            m = m_coord
            v1u = np.squeeze(np.array((m - p1) / np.linalg.norm((m - p1))))
            v2u = np.squeeze(np.array((m - p2) / np.linalg.norm((m - p2))))
            angle = np.rad2deg(np.arccos(np.clip(np.dot(v1u, v2u), -1.0, 1.0)))
            # print("angle: ", angle)
        except IndexError:
            angle = 0
        return angle
    if loud:
        print('********************************************')
        print(("n_ligs = " + str(n_ligs)))
        print(("max d = " + str(max_dent)))
        print(("min_dent = " + str(min_dent)))
        print(("ligand list is" + str(liglist)))
        print(('denticities are  ' + str(ligdents)))
    # Flag Hexa/Pentadentate, check if denticities incorrect for Octahedral Complex
    if max(ligdents) > 4:  # Hexa/Pentadentate ligands flagging
        print(max(ligdents))
        if max(ligdents) == 5 and min(ligdents) == 1:
            pentadentate = True
        elif max(ligdents) == 6 and min(ligdents) == 6:
            hexadentate = True
        elif max(ligdents) == 5 and min(ligdents) == 5:
            print("here")
            hexadentate = True
        else:
            valid = False
            print(('bad denticities: ' + str(ligdents)))
            print(('max denticities: ' + str(min(ligdents))))
    elif n_ligs > 3 and min(ligdents) > 1:  # Catch errors in ligands
        valid = False
        print(('too many ligs ' + str((n_ligs))))
    # Build Ligands and get MWs of ligands
    built_ligand_list = list()
    lig_natoms_list = list()
    lig_mol_weights = list()
    for i, ligand_indices in enumerate(liglist):
        this_ligand = ligand(mol, ligand_indices, ligdents[i])
        this_ligand.obtain_mol3d()
        built_ligand_list.append(this_ligand)
        lig_natoms_list.append(len(this_ligand.index_list))
        lig_mol_weights.append(getMW(this_ligand))
    # ## Here, set a flat list of all connecting atoms (should be of length 6 for octahedral complexes)
    flat_ligcons = list()
    flat_lig_mol_weights = list()
    flat_lig_refs = list()
    for i, item in enumerate(ligcons):
        flat_ligcons += item  # Flat ligand list
        flat_lig_mol_weights += [lig_mol_weights[i]]*len(item)  # Flat lig mws
        flat_lig_refs += [i]*len(item)  # Referred back to individual ligands
    lig_con_weights = [atom.mass for atom in mol.getAtomwithinds(flat_ligcons)]  # Use for hexadentates
    # ## Obtain coordinates for the connecting atoms. Flat coord list ends up being used for comparisons.
    flat_coord_list = np.array([mol.getAtom(ii).coords() for ii in flat_ligcons])
    # Bin and sort ligands as Unique
    unique_ligands = list()
    lig_con_symbols_list = list()
    unique_ligcons = list()
    ligand_counts = list()
    for j, built_ligs in enumerate(built_ligand_list):
        # test if ligand is unique
        sl = sorted([atom.symbol() for atom in built_ligs.master_mol.getAtomwithinds(built_ligs.index_list)])
        # Added check for if ligand connecting atoms are also identical
        sl_ligcons = sorted([atom.symbol() for atom in mol.getAtomwithinds(ligcons[j])])
        symbol_list.append(sl)
        lig_con_symbols_list.append(sl_ligcons)
        if loud:
            print(('checking lig ' + str(j) + ' : ' + str(sl)))
        unique = 1  # Flag for detecting unique ligands
        for i, other_sl in enumerate(unique_ligands):
            if sl == other_sl and sl_ligcons == unique_ligcons[i]:
                # Duplicate
                unique = 0
                ligand_counts[i] += 1
        if unique == 1:
            unique_ligands.append(sl)
            unique_ligcons.append(sl_ligcons)
            ligand_counts.append(1)
            ligand_records.append(j)
    # Pair list contains all pairs of combinations of connecting atom points
    pair_combos = list(combinations([0, 1, 2, 3, 4, 5], 2))  # Pairs of coordinates of ligand-connecting atoms
    angle_list = list()
    pair_list = list()
    for i, pair in enumerate(pair_combos):
        pair_list.append(list(pair))
        angle = getAngle(flat_coord_list, pair_list[-1], m_coord)
        if loud:
            print(('pair of atoms, then angle', pair, angle))
        angle_list.append(angle)  # Save angle between connecting atom points thorugh metal.
    argsort_angle_list = np.squeeze(np.array(angle_list)).argsort()[-3:][::-1]
    # ## Get the 3 largest angles through the metal, which should define the x, y, and z planes of symmetry
    # ## Then define those planes of symmetry with the atom indices (point_combos)
    # ## Each of the point combos represents 4 points that should most approximately be in a plane
    # ## Defining the x,y, and z axes of the octahedral complex
    point_combos = [pair_list[argsort_angle_list[0]] + pair_list[argsort_angle_list[1]],
                    pair_list[argsort_angle_list[1]] + pair_list[argsort_angle_list[2]],
                    pair_list[argsort_angle_list[2]] + pair_list[argsort_angle_list[0]]]
    # ### Next, fit each of these planes with a best fit plane.
    # ### The plane that fits best will be the equatorial plane by default.
    # ### In some cases, the plane must be overruled (i.e seesaw tetradentates)
    # ### For special cases like the seesaw, there is consistent handling such that
    # ### there is consistent behavior for the ones within the same ligand class.
    error_list = list()  # Error of planes
    combo_list = list()
    fitlist = list()  # Fit of planes
    mw_plane_list = list()  # Total ligand MW in plane for monodentates/bi/tridentates
    mw_plane_lig_con_list = list()  # LC-atom MW in plane for hexadentates / non-planar tridentates
    for i, combo in enumerate(point_combos):
        combo_list.append(combo)
        if loud:
            print(('combo', combo))
        A = np.zeros((4, 3))
        b = np.zeros(4)
        mw_plane = 0
        mw_lig_cons = 0
        for point_i, point_num in enumerate(combo):
            coordlist = flat_coord_list[point_num]
            mw_plane += flat_lig_mol_weights[point_num]
            mw_lig_cons += lig_con_weights[point_num]
            A[point_i, :] = [coordlist[0], coordlist[1], 1]
            b[point_i] = coordlist[2]
        mw_plane_list.append(mw_plane)
        mw_plane_lig_con_list.append(mw_lig_cons)
        try:
            fit = np.linalg.pinv(A.T @ A) @ A.T @ b
            fitlist.append(fit)
            errors = b - A @ fit
            error_var = np.var(errors)
            error_list.append(error_var)
        except np.linalg.LinAlgError:
            error_list.append(0)  # perfect fit plane may suffer matrix singularity issues.
    # Print errors if loud
    if loud:
        print('combos below')
        print(combo_list)
        print('errors next, argmin combo selected')
        print(error_list)
    # ### Find the points that correspond to the best fit plane through 4 points.
    # ### Eq points are used later. Eq points has the number of the connection atoms
    # ### across from each other. It pulls from a list of 0, 1, 2, 3, 4, 5, and gets
    # ### which 4 are the atoms that are in the equatorial plane.
    eq_points = combo_list[np.argmin(np.array(error_list))]
    max_mw_idx = np.argmax(np.array(mw_plane_list))  # Saved for planar 3X3 dentate cases
    eq_points_max_mw = combo_list[max_mw_idx]
    eq_points_max_con_mw = combo_list[np.argmax(np.array(mw_plane_lig_con_list))]
    # If there is no difference in MW bewteen different planes, flag as symmetric compound
    if np.var(np.array(mw_plane_list)) == 0.0:
        symmetric = True
    else:
        symmetric = False
    # Print out state of affairs if loud
    if loud:
        print(('unique ligands' + str(unique_ligands)))
        print(('ligand counts' + str(ligand_counts)))
        print(('ligand records ' + str(ligand_records)))
        print((str(max(ligand_counts)) +
               ' is the max and min in  ' + str(min(ligand_counts))))
    n_unique_ligs = len(unique_ligands)  # Number of unique ligands
    if (n_ligs == 6):  # All monodentate
        allowed = list(range(0, 6))
        if n_unique_ligs == 1:  # Purely Homoleptic monodentate, by best fit plane
            if loud:
                print('homoleptic monodentate')
            eq_lig_list = eq_points  # Assign 4 lig_cons to equatorial plane, by best fit plane
            eq_con_list = [ligcons[j] for j in eq_lig_list]
            ax_lig_list = list(set(allowed)-set(eq_lig_list))  # Last 2 lig_cons to axial positions
            ax_con_list = [ligcons[j] for j in ax_lig_list]
        elif n_unique_ligs == 2:  # Mix of 2 monodentates
            if loud:
                print(('monodentate {}+{} ligands'.format(max(ligand_counts), min(ligand_counts))))
                print((ligand_counts, unique_ligands))
            eq_lig_list = list()
            if use_z:
                minz = 500
                maxz = -500
                if loud:
                    print('monodentate case')
                allowed = list(range(0, 6))
                not_eq = list()
                for j, built_ligs in enumerate(built_ligand_list):
                    this_z = sum([mol.getAtom(ii).coords()[2]
                                  for ii in ligcons[j]]) / len(ligcons[j])
                    if this_z < minz:
                        minz = this_z
                        bot_lig = j
                        bot_con = ligcons[j]
                    if loud:
                        print(('updating bot axial to ' + str(bot_lig)))
                    if this_z > maxz:
                        maxz = this_z
                        top_lig = j
                        top_con = ligcons[j]
                    if loud:
                        print(('updating top axial to ' + str(top_lig)))
                not_eq.append(bot_lig)
                not_eq.append(top_lig)
                allowed = [x for x in allowed if ((x not in not_eq))]
                if len(allowed) != 4:
                    print(('error in decomp of monodentate case!', allowed))
                eq_lig_list = allowed
                eq_con_list = [ligcons[i] for i in allowed]
                ax_lig_list = [top_lig, bot_lig]
                ax_con_list = [top_con, bot_con]
            else:
                if max(ligand_counts) == 5:  # 5+1 Monodentate
                    five_repeats = list()
                    for i, ligand_count in enumerate(ligand_counts):
                        temp_unique = unique_ligands[i]
                        temp_ligsym_unique = unique_ligcons[i]
                        for j, built_ligs in enumerate(built_ligand_list):
                            sym_list = sorted([
                                atom.symbol() for atom in
                                built_ligs.master_mol.getAtomwithinds(built_ligs.index_list)])
                            ligcon_inds = [x for x in built_ligs.index_list if x in flat_ligcons]
                            sl_ligcon = sorted([atom.symbol() for atom in built_ligs.master_mol.getAtomwithinds(ligcon_inds)])
                            if sym_list != temp_unique or sl_ligcon != temp_ligsym_unique:
                                continue
                            elif (ligand_count == 5):
                                # ### In the 5+1 monodentate case, 4 of the 5 are assigned to be equatorial,
                                five_repeats.append(j)
                            elif (ligand_count == 1):
                                ax_lig_list.append(j)
                                ax_con_list.append(ligcons[j])
                    # ## Calculate which of the 5 repeats has highest angle from 1 unique
                    # ## Set highest angle as axial, rest assigned as equatorial
                    coord_list = [flat_coord_list[ax_lig_list[0]]]+[flat_coord_list[x] for x in five_repeats]
                    pair_combos = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
                    angle_list = list()
                    for pair in pair_combos:
                        angle = getAngle(coord_list, pair, m_coord)
                        if loud:
                            print(('pair of atoms, angle', pair, angle))
                        angle_list.append(angle)
                    ax_lig = five_repeats[np.argmax(angle_list)]
                    ax_lig_list.append(ax_lig)
                    ax_con_list.append(ligcons[ax_lig])
                    eq_lig_list = list(set(five_repeats) - set(ax_lig_list))
                    eq_con_list = [ligcons[j] for j in eq_lig_list]
                elif max(ligand_counts) == 4:  # This can be either seesaw configuration or planar configuration, 4+2 cases
                    four_repeats = list()
                    for i, ligand_count in enumerate(ligand_counts):
                        temp_unique = unique_ligands[i]
                        temp_ligsym_unique = unique_ligcons[i]
                        for j, built_ligs in enumerate(built_ligand_list):
                            sym_list = sorted([
                                atom.symbol() for atom in
                                built_ligs.master_mol.getAtomwithinds(built_ligs.index_list)])
                            ligcon_inds = [x for x in built_ligs.index_list if x in flat_ligcons]
                            sl_ligcon = sorted([atom.symbol() for atom in built_ligs.master_mol.getAtomwithinds(ligcon_inds)])
                            if sym_list != temp_unique or sl_ligcon != temp_ligsym_unique:
                                continue
                            elif (ligand_count == 4) and len(four_repeats) < 4:
                                four_repeats.append(j)
                    if loud:
                        print(('this is four repeats', four_repeats))
                    four_repeats_cons = [ligcons[j] for j in four_repeats]
                    pair_combos = list(combinations([0, 1, 2, 3], 2))
                    angle_list = list()
                    pair_list = list()
                    coord_list = np.array([mol.getAtom(ii[0]).coords() for ii in four_repeats_cons])
                    for i, pair in enumerate(pair_combos):
                        pair_list.append(list(pair))
                        angle = getAngle(coord_list, list(pair), m_coord)
                        if loud:
                            print(('pair of atoms, then angle', pair, angle))
                        angle_list.append(angle)
                    # ### Seesaws will have only 1 ~180 degree angle, whereas planar ligands will have two.
                    # ### Thus, after measuring the angles for all tetradentate connecting atoms through the metal,
                    # ### looking at the angle of the first (not zeroeth) element tells us whether or not we have a seesaw.
                    test_angle = np.sort(np.array(angle_list))[::-1][1]
                    if test_angle < angle_cutoff:
                        seesaw = True
                        # ### In the seesaw, the two furthest apart are denoted as axial.
                        # ### The equatorial plane consists of 2 seesaw connecting atoms, and two monodentates.
                        axial_pair = [four_repeats_cons[val] for val in list(pair_list[np.argmax(np.array(angle_list))])]
                    else:
                        seesaw = False
                        # ### In the planar set, the two monodentates are axial, and tetradentate is equatorial.
                        axial_pair = list(set(allowed) - set(four_repeats))
                    if not seesaw:
                        eq_lig_list = four_repeats
                        eq_con_list = four_repeats_cons  # ADDED
                        ax_lig_list = axial_pair
                        ax_con_list = [ligcons[axial_pair[0]], ligcons[axial_pair[1]]]
                    else:  # 2 points in eq plane, seesaw case
                        # ### In the seesaw, the two furthest apart are denoted as axial.
                        # ### The equatorial plane consists of 2 seesaw connecting atoms, and two monodentates.
                        eq_plane_cons = list(set(flat_ligcons) - set([val[0] for val in axial_pair]))
                        eq_con_list = [ligcons[j] for j in range(len(ligcons)) if ligcons[j] in eq_plane_cons]
                        eq_lig_list = [i for i in range(len(ligcons)) if ligcons[i] in eq_plane_cons]
                        ax_lig_list = [i for i in range(len(ligcons)) if ligcons[i] not in eq_plane_cons]
                        ax_con_list = [ligcons[j] for j in range(len(ligcons)) if ligcons[j] not in eq_plane_cons]
                else:  # 3,3 monodentate case - just use the planes to define ax vs eq by maximum MW
                    if symmetric:  # If MWs of ligands match, but connecting atoms different, choose max MW con atom
                        eq_ligcons = list(set([flat_ligcons[j] for j in eq_points_max_con_mw]))
                    else:
                        eq_ligcons = list(set([flat_ligcons[j] for j in eq_points_max_mw]))
                    eq_lig_list = eq_points_max_mw
                    eq_con_list = [ligcons[j] for j in eq_lig_list]
                    ax_lig_list = list(set(allowed)-set(eq_points_max_mw))
                    ax_con_list = [ligcons[j] for j in ax_lig_list]
        elif n_unique_ligs == 3:  # Mix of 3 monodentates
            if loud:
                print(f'monodentate {max(ligand_counts)}+{min(ligand_counts)}+{6-max(ligand_counts)-min(ligand_counts)}')
            # ### Need to identify if in seesaw-style configuration or planar configuration
            if use_z:
                minz = 500
                maxz = -500
                if loud:
                    print('monodentate case')
                allowed = list(range(0, 6))
                not_eq = list()
                for j, built_ligs in enumerate(built_ligand_list):
                    this_z = sum([mol.getAtom(ii).coords()[2]
                                  for ii in ligcons[j]]) / len(ligcons[j])
                    if this_z < minz:
                        minz = this_z
                        bot_lig = j
                        bot_con = ligcons[j]
                    if loud:
                        print(('updating bot axial to ' + str(bot_lig)))
                    if this_z > maxz:
                        maxz = this_z
                        top_lig = j
                        top_con = ligcons[j]
                    if loud:
                        print(('updating top axial to ' + str(top_lig)))
                not_eq.append(bot_lig)
                not_eq.append(top_lig)
                allowed = [x for x in allowed if ((x not in not_eq))]
                if len(allowed) != 4:
                    print(('error in decomp of monodentate case!', allowed))
                eq_lig_list = allowed
                eq_con_list = [ligcons[i] for i in allowed]
                ax_lig_list = [top_lig, bot_lig]
                ax_con_list = [top_con, bot_con]
            else:
                if max(ligand_counts) == 4:  # Seesaw vs. Planar
                    four_repeats = list()
                    for i, ligand_count in enumerate(ligand_counts):
                        temp_unique = unique_ligands[i]
                        temp_ligsym_unique = unique_ligcons[i]
                        for j, built_ligs in enumerate(built_ligand_list):
                            sym_list = sorted([
                                atom.symbol() for atom in
                                built_ligs.master_mol.getAtomwithinds(built_ligs.index_list)])
                            ligcon_inds = [x for x in built_ligs.index_list if x in flat_ligcons]
                            sl_ligcon = sorted([atom.symbol() for atom in built_ligs.master_mol.getAtomwithinds(ligcon_inds)])
                            if (sym_list == temp_unique) and (sl_ligcon == temp_ligsym_unique) and \
                                    (ligand_count == 4) and len(four_repeats) < 4:
                                four_repeats.append(j)
                    if loud:
                        print(('this is four repeats', four_repeats))
                    four_repeats_cons = [ligcons[j] for j in four_repeats]
                    pair_combos = list(combinations([0, 1, 2, 3], 2))
                    angle_list = []
                    pair_list = []
                    coord_list = np.array([mol.getAtom(ii[0]).coords() for ii in four_repeats_cons])
                    for i, pair in enumerate(pair_combos):
                        pair_list.append(list(pair))
                        angle = getAngle(coord_list, pair_list[-1], m_coord)
                        if loud:
                            print(('pair of atoms, then angle', pair, angle))
                        angle_list.append(angle)
                    # ### Seesaws will have only 1 ~180 degree angle, whereas planar ligands will have two.
                    # ### Thus, after measuring the angles for all tetradentate connecting atoms through the metal,
                    # ### looking at the angle of the first (not zeroeth) element tells us whether or not we have a seesaw.
                    test_angle = np.sort(np.array(angle_list))[::-1][1]
                    if test_angle < angle_cutoff:
                        seesaw = True
                        # ### In the seesaw, the two furthest apart are denoted as axial.
                        # ### The equatorial plane consists of 2 seesaw connecting atoms, and two monodentates.
                        axial_pair = [four_repeats_cons[val] for val in list(pair_list[np.argmax(np.array(angle_list))])]
                    else:
                        seesaw = False
                        # ### In the planar set, the two monodentates are axial, and tetradentate is equatorial.
                        axial_pair = list(set(allowed) - set(four_repeats))
                    if not seesaw:  # Planar
                        eq_lig_list = four_repeats
                        eq_con_list = four_repeats_cons  # ADDED
                        ax_lig_list = axial_pair
                        ax_con_list = [ligcons[axial_pair[0]], ligcons[axial_pair[1]]]
                    else:  # 2 points in eq plane, seesaw case
                        # ### In the seesaw, the two furthest apart are denoted as axial.
                        # ### The equatorial plane consists of 2 seesaw connecting atoms, and two monodentates.
                        eq_plane_cons = list(set(flat_ligcons) - set([val[0] for val in axial_pair]))
                        eq_con_list = [ligcons[j] for j in range(len(ligcons)) if ligcons[j] in eq_plane_cons]
                        eq_lig_list = [i for i in range(len(ligcons)) if ligcons[i] in eq_plane_cons]
                        ax_lig_list = [i for i in range(len(ligcons)) if ligcons[i] not in eq_plane_cons]
                        ax_con_list = [ligcons[j] for j in range(len(ligcons)) if ligcons[j] not in eq_plane_cons]
                elif max(ligand_counts) == 3:  # 3+2+1 - Planar 3 or not
                    three_repeats = list()
                    two_repeats = list()
                    for i, ligand_count in enumerate(ligand_counts):
                        temp_unique = unique_ligands[i]
                        temp_ligsym_unique = unique_ligcons[i]
                        for j, built_ligs in enumerate(built_ligand_list):
                            sym_list = sorted([
                                atom.symbol() for atom in
                                built_ligs.master_mol.getAtomwithinds(built_ligs.index_list)])
                            ligcon_inds = [x for x in built_ligs.index_list if x in flat_ligcons]
                            sl_ligcon = sorted([atom.symbol() for atom in built_ligs.master_mol.getAtomwithinds(ligcon_inds)])
                            if sym_list != temp_unique or sl_ligcon != temp_ligsym_unique:
                                continue
                            elif (ligand_count == 3) and len(three_repeats) < 3:
                                three_repeats.append(j)
                            elif (ligand_count == 2) and len(two_repeats) < 2:
                                two_repeats.append(j)
                    if loud:
                        print(('this is three repeats', three_repeats))
                    three_repeats_cons = [ligcons[j] for j in three_repeats]
                    two_repeats_cons = [ligcons[j] for j in two_repeats]
                    pair_combos = list(combinations([0, 1, 2], 2))
                    angle_list = []
                    pair_list = []
                    coord_list = np.array([mol.getAtom(ii[0]).coords() for ii in three_repeats_cons])
                    for i, pair in enumerate(pair_combos):  # Test angle between 3 identical
                        pair_list.append(list(pair))
                        angle = getAngle(coord_list, pair_list[-1], m_coord)
                        if loud:
                            print(('pair of atoms, then angle', pair, angle))
                        angle_list.append(angle)
                    test_angle = np.sort(np.array(angle_list))[::-1][0]
                    if test_angle < angle_cutoff:  # If less than cutoff, non-planar
                        planar = False
                        coord_list_two = np.array([mol.getAtom(ii[0]).coords() for ii in two_repeats_cons])
                        coord_list_three = np.array([mol.getAtom(ii[0]).coords() for ii in three_repeats_cons])
                        angle_list = list()
                        # unused
                        # m = np.array([mol.getAtom(mol.findMetal()[0]).coords()])
                        pair_combos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
                        for k, pair in enumerate(pair_combos):
                            p1 = np.squeeze(np.array(coord_list_two[pair[0]]))
                            p2 = np.squeeze(np.array(coord_list_three[pair[1]]))
                            v1u = np.squeeze(np.array((m_coord - p1) / np.linalg.norm((m_coord - p1))))
                            v2u = np.squeeze(np.array((m_coord - p2) / np.linalg.norm((m_coord - p2))))
                            angle = np.rad2deg(np.arccos(np.clip(np.dot(v1u, v2u), -1.0, 1.0)))
                            angle_list.append(angle)
                        # ### If not planar, considering the plane with the bidentate to be equatorial.
                        # ### Along with 2 opposite Tridentate connecting atoms
                        # ### Consistent with tridentate cases.
                        top_2_angles = np.squeeze(np.array(angle_list)).argsort()[-2:][::-1]
                        first_pair = pair_combos[top_2_angles[0]]
                        second_pair = pair_combos[top_2_angles[1]]
                        conlist = list()
                        conlist += two_repeats_cons[first_pair[0]]+three_repeats_cons[first_pair[1]]
                        conlist += two_repeats_cons[second_pair[0]]+three_repeats_cons[second_pair[1]]
                        if loud:
                            print(('eq points reassigned', conlist))
                        eq_points_defined = [j for j in allowed if ligcons[j][0] in conlist]
                    else:
                        planar = True
                        mono_dentate_idx_set = list(set(allowed)-set(three_repeats))
                        coord_list = np.array([mol.getAtom(ii[0]).coords() for ii in three_repeats_cons])
                        m = np.array([mol.getAtom(mol.findMetal()[0]).coords()])
                        mono_con_list = []
                        for mono_dentate_idx in mono_dentate_idx_set:
                            current_con = ligcons[mono_dentate_idx]
                            mono_con_list.append(current_con[0])
                            p1 = np.array(mol.getAtom(current_con[0]).coords())
                            angle_list = []
                            for k, tri_con in enumerate(three_repeats_cons):
                                p2 = np.squeeze(np.array(coord_list[k]))
                                v1u = np.squeeze(np.array((m - p1) / np.linalg.norm((m - p1))))
                                v2u = np.squeeze(np.array((m - p2) / np.linalg.norm((m - p2))))
                                angle = np.rad2deg(np.arccos(np.clip(np.dot(v1u, v2u), -1.0, 1.0)))
                                angle_list.append(angle)
                            test_angle = np.sort(np.array(angle_list))[::-1][0]
                            # ### If the tridentate is planar, only one of the bidentate connecting atoms
                            # ### will be across the tridentate. Thus the equatorial plane will be defined
                            # ### as the planar tridentate plus one connection atom for the bidentate
                            # ### The monodentate and the second bidentate connecting atoms are denoted axial.
                            if test_angle > angle_cutoff:
                                monodentate_eq_con = current_con
                                monodentate_eq_idx = mono_dentate_idx
                    if planar:
                        eq_lig_list = three_repeats+[monodentate_eq_idx]
                        eq_con_list = three_repeats_cons+[monodentate_eq_con]  # ADDED
                        ax_con_list = [[val] for val in mono_con_list if val not in monodentate_eq_con]
                        ax_lig_list = [i for i in range(len(ligcons)) if ligcons[i] not in eq_con_list]
                    else:
                        # ## Eq points set based on 2 + 2 of (3) Eq and mono+ 1(3) Ax
                        # ## Consistent with 3+2+1 below by denticity
                        eq_lig_list = eq_points_defined
                        eq_con_list = [ligcons[j] for j in eq_lig_list]
                        ax_lig_list = [val for i, val in enumerate(allowed) if val not in eq_points_defined]
                        ax_con_list = [ligcons[j] for j in ax_lig_list]
                else:  # Max mw determines eq plane (2+2+2 different monodentates)
                    eq_lig_list = eq_points_max_mw
                    eq_con_list = [ligcons[j] for j in eq_lig_list]
                    ax_lig_list = list(set(allowed)-set(eq_points_max_mw))
                    ax_con_list = [ligcons[j] for j in ax_lig_list]
        else:  # with more than 4 ligands, the ax/eq breaks down. Eq plane defined by max mw plane.
            if loud:
                print('monodentate more than 3 unique')
            eq_lig_list = eq_points_max_mw
            eq_con_list = [ligcons[j] for j in eq_lig_list]
            ax_lig_list = list(set(allowed)-set(eq_points_max_mw))
            ax_con_list = [ligcons[j] for j in ax_lig_list]
        # ###### Find and put trans species in eq plane in order opposite
        # ###### from each other [0,2],[1,3] in indices for monodentates
        eq_con_coords = [flat_coord_list[x] for x in eq_lig_list]
        combos = [(0, 1), (0, 2), (0, 3)]
        angle_list_eq = list()
        for pair in combos:
            angle = getAngle(eq_con_coords, pair, m_coord)
            if loud:
                print(('pair of atoms, angle', pair, angle))
            angle_list_eq.append(angle)
        opposite_lig = combos[np.argmax(angle_list_eq)][1]
        others = list(set([1, 2, 3])-set([opposite_lig]))
        eq_order = [0, others[0], opposite_lig, others[1]]
        # Reorder equatorial plane
        eq_con_list = [eq_con_list[x] for x in eq_order]
        eq_lig_list = [eq_lig_list[x] for x in eq_order]
    elif (n_ligs == 5):  # 2+1+1+1+1
        # unused
        # allowed = list(range(0, 5))
        if loud:
            print('bidentate 2+1+1+1+1 case')
        bidentate_ligand_idx = np.argmax(ligdents)
        bidentate_cons = ligcons[bidentate_ligand_idx]
        mono_dentate_idx_set = list(set(range(len(ligdents)))-set([bidentate_ligand_idx]))
        # unused
        # monodentate_cons = [ligcons[val] for val in mono_dentate_idx_set]
        coord_list = np.array([mol.getAtom(ii).coords() for ii in bidentate_cons])
        mono_con_list = list()
        monodentate_eq_cons = list()
        monodentate_eq_idxs = list()
        # 2-dentate in eq plane and opposite monodentates selected
        bidentate_axial = False
        for mono_dentate_idx in mono_dentate_idx_set:
            current_con = ligcons[mono_dentate_idx]
            mono_con_list.append(current_con[0])
            p1 = np.array(mol.getAtom(current_con[0]).coords())
            for k, bi_con in enumerate(bidentate_cons):
                p2 = np.squeeze(np.array(coord_list[k]))
                v1u = np.squeeze(np.array((m_coord - p1) / np.linalg.norm((m_coord - p1))))
                v2u = np.squeeze(np.array((m_coord - p2) / np.linalg.norm((m_coord - p2))))
                angle = np.rad2deg(np.arccos(np.clip(np.dot(v1u, v2u), -1.0, 1.0)))
                if angle > angle_cutoff:
                    monodentate_eq_cons.append(current_con)
                    monodentate_eq_idxs.append(mono_dentate_idx)
        if len(monodentate_eq_cons) < 2:
            bidentate_axial = True
        if not bidentate_axial:  # Normal
            eq_lig_list = [bidentate_ligand_idx, monodentate_eq_idxs[0], monodentate_eq_idxs[1]]
            eq_con_list = [bidentate_cons, monodentate_eq_cons[0], monodentate_eq_cons[1]]  # ADDED
            ax_lig_list = [val for val in mono_dentate_idx_set if val not in monodentate_eq_idxs]
            ax_con_list = [[val] for val in mono_con_list if val not in [x[0] for x in monodentate_eq_cons]]
        else:
            ax_lig_list = [bidentate_ligand_idx, bidentate_ligand_idx]
            ax_con_list = [[val] for val in bidentate_cons]
            eq_lig_list = [val for val in mono_dentate_idx_set]
            eq_con_list = [[val] for val in mono_con_list]
    elif (n_ligs == 4):  # 2+2+1+1 and 3+1+1+1 cases
        allowed = list(range(0, 4))
        if max(ligdents) == 3:
            if loud:
                print('3+1+1+1 case')
            tridentate_ligand_idx = np.argmax(ligdents)
            tridentate_cons = ligcons[tridentate_ligand_idx]
            mono_dentate_idx_set = list(set(range(len(ligdents)))-set([tridentate_ligand_idx]))
            # unused
            # monodentate_cons = [ligcons[val] for val in mono_dentate_idx_set]
            pair_combos = list(combinations([0, 1, 2], 2))
            angle_list = []
            pair_list = []
            coord_list = np.array([mol.getAtom(ii).coords() for ii in tridentate_cons])
            for i, pair in enumerate(pair_combos):
                pair_list.append(list(pair))
                angle = getAngle(coord_list, pair_list[-1], m_coord)
                if loud:
                    print(('pair of atoms, then angle', pair, angle))
                angle_list.append(angle)
            # ### The tridentate can either be planar or not. If planar, it will have at least one
            # ### angle that is close to 180 between connecting atoms, through the metal.
            test_angle = np.sort(np.array(angle_list))[::-1][0]
            planar = False
            if test_angle > angle_cutoff:
                planar = True
                coord_list = np.array([mol.getAtom(ii).coords() for ii in tridentate_cons])
                m = np.array([mol.getAtom(mol.findMetal()[0]).coords()])
                mono_con_list = []
                for mono_dentate_idx in mono_dentate_idx_set:
                    current_con = ligcons[mono_dentate_idx]
                    mono_con_list.append(current_con[0])
                    p1 = np.array(mol.getAtom(current_con[0]).coords())
                    angle_list = []
                    for k, tri_con in enumerate(tridentate_cons):
                        p2 = np.squeeze(np.array(coord_list[k]))
                        v1u = np.squeeze(np.array((m - p1) / np.linalg.norm((m - p1))))
                        v2u = np.squeeze(np.array((m - p2) / np.linalg.norm((m - p2))))
                        angle = np.rad2deg(np.arccos(np.clip(np.dot(v1u, v2u), -1.0, 1.0)))
                        angle_list.append(angle)
                    test_angle = np.sort(np.array(angle_list))[::-1][0]
                    # ### If the tridentate is planar, only one of the bidentate connecting atoms
                    # ### will be across the tridentate. Thus the equatorial plane will be defined
                    # ### as the planar tridentate plus one connection atom for the bidentate
                    # ### The monodentate and the second bidentate connecting atoms are denoted axial.
                    if test_angle > angle_cutoff:
                        monodentate_eq_con = current_con
                        monodentate_eq_idx = mono_dentate_idx
            if planar:
                eq_lig_list = [tridentate_ligand_idx, monodentate_eq_idx]
                eq_con_list = [tridentate_cons, monodentate_eq_con]  # ADDED
                ax_lig_list = [val for val in mono_dentate_idx_set if val not in [monodentate_eq_idx]]
                ax_con_list = [[val] for val in mono_con_list if val not in monodentate_eq_con]
            else:  # WORKs
                # ## any equatorial plane will have 2 of the tridentate con atoms so take the eq plane with max mw
                tri_eq_ligcons = list(set(tridentate_cons).intersection(
                                      set([flat_ligcons[x] for x in eq_points_max_mw])))
                mono_eq_ligcons = list(set([flat_ligcons[x] for x in eq_points_max_mw])-set(tri_eq_ligcons))
                eq_lig_list = [tridentate_ligand_idx, flat_lig_refs[flat_ligcons.index(mono_eq_ligcons[0])],
                               flat_lig_refs[flat_ligcons.index(mono_eq_ligcons[1])]]
                eq_con_list = [tri_eq_ligcons, [mono_eq_ligcons[0]], [mono_eq_ligcons[1]]]
                ax_lig_list = [tridentate_ligand_idx,
                               list(set(mono_dentate_idx_set).difference(set([x for x in eq_lig_list[1:]])))[0]]
                tri_ax_ligcon = list(set(tridentate_cons).difference(
                                      set(tri_eq_ligcons)))
                ax_con_list = [tri_ax_ligcon,
                               list(set(flat_ligcons).difference(
                                   set(mono_eq_ligcons+tri_ax_ligcon+tri_eq_ligcons)))]
        elif max(ligdents) == 2:
            # ### Need to handle case with both equatorial and triple-bidentate style.
            if loud:
                print('2+2+1+1 case')
            allowed = list(range(0, 4))
            bidentate_ligand_idx1 = np.squeeze(np.array(ligdents)).argsort()[-2:][::-1][0]
            bidentate_ligand_idx2 = np.squeeze(np.array(ligdents)).argsort()[-2:][::-1][1]
            bidentate_cons1 = ligcons[bidentate_ligand_idx1]
            bidentate_cons2 = ligcons[bidentate_ligand_idx2]
            pair_combos = list(combinations([0, 1, 2, 3], 2))
            angle_list = []
            pair_list = []
            coord_list = np.array([mol.getAtom(ii).coords() for ii in bidentate_cons1]
                                  + [mol.getAtom(ii).coords() for ii in bidentate_cons2])
            for i, pair in enumerate(pair_combos):
                pair_list.append(list(pair))
                angle = getAngle(coord_list, pair_list[-1], m_coord)
                if loud:
                    print(('pair of atoms, then angle', pair, angle))
                angle_list.append(angle)
            # ### Seesaws will have only 1 ~180 degree angle, whereas planar ligands will have two.
            # ### Thus, after measuring the angles for all tetradentate connecting atoms through the metal,
            # ### looking at the angle of the first (not zeroeth) element tells us whether or not we have a seesaw.
            test_angle = np.sort(np.array(angle_list))[::-1][1]
            if loud:
                print(('ANGLE LIST', angle_list, 'sorted', np.sort(np.array(angle_list))[::-1]))
            if test_angle < angle_cutoff:
                seesaw = True
                temp_cons = bidentate_cons1+bidentate_cons2
                # Axial pair in bidentates identified
                axial_pair = [[temp_cons[val]] for val in list(pair_list[np.argmax(np.array(angle_list))])]
            else:
                seesaw = False
            if not seesaw:  # Planar
                eq_lig_list = [bidentate_ligand_idx1, bidentate_ligand_idx2]
                eq_con_list = [bidentate_cons1, bidentate_cons2]
                ax_lig_list = list(set(allowed)-set([bidentate_ligand_idx1, bidentate_ligand_idx2]))
                ax_con_list = [ligcons[j] for j in ax_lig_list]
            else:  # 2 points in eq plane, seesaw case
                ax_con_list = axial_pair
                flat_ax_con_list = [item for sublist in ax_con_list for item in sublist]
                ax_lig_list = [j for j, val in enumerate(ligcons) if flat_ax_con_list[0] in val] + \
                              [j for j, val in enumerate(ligcons) if flat_ax_con_list[1] in val]
                eq_ligcons = set(flat_ligcons) - set(flat_ax_con_list)
                eq_con_bidentate_list = [
                    list(set(ligcons[0]).intersection(eq_ligcons)),
                    list(set(ligcons[1]).intersection(eq_ligcons)),
                    list(set(ligcons[2]).intersection(eq_ligcons)),
                    list(set(ligcons[3]).intersection(eq_ligcons))]
                eq_con_bidentate_list = [val for val in eq_con_bidentate_list
                                         if len(val) > 0]
                eq_con_list = eq_con_bidentate_list
                flat_eq_con_list = [item for sublist in eq_con_list for item in sublist]
                eq_lig_list = [
                    j for j, val in enumerate(ligcons) if
                    len(set(val).intersection(set(flat_eq_con_list))) > 0]
    elif (n_ligs == 3):  # 2+2+2 or 4+1+1, can be seesaw/planar or 3+2+1, seesaw/planar
        if max(ligdents) == 4:  # 4+1+1
            if loud:
                print('4+1+1 case')
            allowed = list(range(0, 3))
            tetradentate_ligand_idx = np.argmax(ligdents)
            tetradentate_cons = ligcons[tetradentate_ligand_idx]
            pair_combos = list(combinations([0, 1, 2, 3], 2))
            angle_list = []
            pair_list = []
            coord_list = np.array([mol.getAtom(ii).coords() for ii in tetradentate_cons])
            for i, pair in enumerate(pair_combos):
                pair_list.append(list(pair))
                angle = getAngle(coord_list, pair_list[-1], m_coord)
                if loud:
                    print(('pair of atoms, then angle', pair, angle))
                angle_list.append(angle)
            # ### Seesaws will have only 1 ~180 degree angle, whereas planar ligands will have two.
            # ### Thus, after measuring the angles for all tetradentate connecting atoms through the metal,
            # ### looking at the angle of the first (not zeroeth) element tells us whether or not we have a seesaw.
            test_angle = np.sort(np.array(angle_list))[::-1][1]
            if test_angle < angle_cutoff:
                seesaw = True
                # ### In the seesaw, the two furthest apart are denoted as axial.
                # ### The equatorial plane consists of 2 seesaw connecting atoms, and two monodentates.
                axial_pair = [tetradentate_cons[val] for val in list(pair_list[np.argmax(np.array(angle_list))])]
            else:
                seesaw = False
                # ### In the planar set, the two monodentates are axial, and tetradentate is equatorial.
                axial_pair = list(set(allowed) - set([tetradentate_ligand_idx]))
            if not seesaw:  # Planar
                eq_lig_list = [tetradentate_ligand_idx]
                eq_con_list = [tetradentate_cons]  # ADDED
                ax_lig_list = axial_pair
                ax_con_list = [ligcons[axial_pair[0]], ligcons[axial_pair[1]]]
            else:  # 2 points in eq plane, seesaw case
                eq_plane = list(set(flat_ligcons) - set(axial_pair))
                # ### Find the two connecting atoms of the seesaw that lie in the equatorial plane
                tet_eq = list(set(eq_plane).intersection(set(tetradentate_cons)))
                eq_con_list = [ligcons[j] if len(ligcons[j]) == 1 else tet_eq for j in range(len(ligcons))]
                eq_lig_list = allowed
                ax_lig_list = [tetradentate_ligand_idx, tetradentate_ligand_idx]
                ax_con_list = [[axial_pair[0]], [axial_pair[1]]]
        elif max(ligdents) == 3:  # 3+2+1
            if loud:
                print('3+2+1 case')
            allowed = list(range(0, 3))
            tridentate_ligand_idx = np.argmax(ligdents)
            monodentate_ligand_idx = np.argmin(ligdents)
            bidentate_ligand_idx = list(set(allowed) - set([tridentate_ligand_idx]) - set([monodentate_ligand_idx]))[0]
            tridentate_cons = ligcons[tridentate_ligand_idx]
            bidentate_cons = ligcons[bidentate_ligand_idx]
            monodentate_cons = ligcons[monodentate_ligand_idx]
            pair_combos = list(combinations([0, 1, 2], 2))
            angle_list = []
            pair_list = []
            coord_list = np.array([mol.getAtom(ii).coords() for ii in tridentate_cons])
            for i, pair in enumerate(pair_combos):
                pair_list.append(list(pair))
                angle = getAngle(coord_list, pair_list[-1], m_coord)
                if loud:
                    print(('pair of atoms, then angle', pair, angle))
                angle_list.append(angle)
            # ### The tridentate can either be planar or not. If planar, it will have at least one
            # ### angle that is close to 180 between connecting atoms, through the metal.
            test_angle = np.sort(np.array(angle_list))[::-1][0]
            bidentate_axial = False
            if test_angle < angle_cutoff:
                planar = False
                coord_list = np.array([mol.getAtom(ii).coords() for ii in tridentate_cons])
                p1 = np.array(mol.getAtom(monodentate_cons[0]).coords())
                angle_list = []
                for k, tri_con in enumerate(tridentate_cons):
                    p2 = np.squeeze(np.array(coord_list[k]))
                    v1u = np.squeeze(np.array((m_coord - p1) / np.linalg.norm((m_coord - p1))))
                    v2u = np.squeeze(np.array((m_coord - p2) / np.linalg.norm((m_coord - p2))))
                    angle = np.rad2deg(np.arccos(np.clip(np.dot(v1u, v2u), -1.0, 1.0)))
                    angle_list.append(angle)
                # ### If not planar, one connection atom (for the monodentate ligand) will be opposite
                # ### the connections for the tridentate ligands. Assigning this atom as the opposite monodentate.
                # ### If not planar, considering the plane with the bidentate to be equatorial.
                opposite_monodentate = [tridentate_cons[np.argmax(np.array(angle_list))]]
                tridentate_cons.remove(tridentate_cons[np.argmax(np.array(angle_list))])
            else:  # Planar
                planar = True
                coord_list = np.array([mol.getAtom(ii).coords() for ii in tridentate_cons])
                # unused:
                # m = np.array([mol.getAtom(mol.findMetal()[0]).coords()])
                bidentate_eq_con = []
                for bi_con in bidentate_cons:
                    p1 = np.array(mol.getAtom(bi_con).coords())
                    angle_list = []
                    for k, tri_con in enumerate(tridentate_cons):
                        p2 = np.squeeze(np.array(coord_list[k]))
                        v1u = np.squeeze(np.array((m_coord - p1) / np.linalg.norm((m_coord - p1))))
                        v2u = np.squeeze(np.array((m_coord - p2) / np.linalg.norm((m_coord - p2))))
                        angle = np.rad2deg(np.arccos(np.clip(np.dot(v1u, v2u), -1.0, 1.0)))
                        angle_list.append(angle)
                    test_angle = np.sort(np.array(angle_list))[::-1][0]
                    # ### If the tridentate is planar, only one of the bidentate connecting atoms
                    # ### will be across the tridentate. Thus the equatorial plane will be defined
                    # ### as the planar tridentate plus one connection atom for the bidentate
                    # ### The monodentate and the second bidentate connecting atoms are denoted axial.
                    if test_angle > angle_cutoff:
                        bidentate_eq_con = [bi_con]
                        bidentate_ax_con = [list(set(bidentate_cons) - set([bi_con]))[0]]
                if len(bidentate_eq_con) < 1:
                    bidentate_axial = True
            if planar and not bidentate_axial:
                eq_lig_list = [tridentate_ligand_idx, bidentate_ligand_idx]
                eq_con_list = [tridentate_cons, bidentate_eq_con]  # ADDED
                ax_lig_list = [monodentate_ligand_idx, bidentate_ligand_idx]
                ax_con_list = [monodentate_cons, bidentate_ax_con]
            elif bidentate_axial:
                eq_lig_list = [tridentate_ligand_idx, monodentate_ligand_idx]
                eq_con_list = [tridentate_cons, monodentate_cons]
                ax_lig_list = [bidentate_ligand_idx, bidentate_ligand_idx]
                ax_con_list = [[val] for val in bidentate_cons]
            else:
                eq_lig_list = [tridentate_ligand_idx, bidentate_ligand_idx]
                eq_con_list = [tridentate_cons, bidentate_cons]  # ADDED
                ax_lig_list = [monodentate_ligand_idx, tridentate_ligand_idx]
                ax_con_list = [monodentate_cons, opposite_monodentate]
        elif (max(ligdents) == 2) and (min(ligdents) == 2):  # 2+2+2
            if loud:
                print('triple bidentate case')
            allowed = list(range(0, 3))
            bidentate_cons_1 = ligcons[0]
            bidentate_cons_2 = ligcons[1]
            bidentate_cons_3 = ligcons[2]
            # ### Using previously defined eq plane with max_mw to define equatorial plane
            eq_ligcons = set([flat_ligcons[j] for j in eq_points_max_mw])
            eq_con_bidentate_list = [list(set(bidentate_cons_1).intersection(eq_ligcons)),
                                     list(set(bidentate_cons_2).intersection(eq_ligcons)),
                                     list(set(bidentate_cons_3).intersection(eq_ligcons))]
            # ### all 3 ligands classified as equatorial because at least one atom connects to the equatorial plane for all
            # ### axial only has two, where the axial connection is defined
            if any([len(x) < 1 for x in eq_con_bidentate_list]):
                eq_con_list = [x for x in eq_con_bidentate_list if len(x) > 0]
                eq_lig_list = [i for i, x in enumerate(eq_con_bidentate_list)
                               if len(x) > 0]
                ax_lig = list(set(allowed)-set(eq_lig_list))[0]
                ax_lig_list = [ax_lig, ax_lig]
                ax_con_list = [[val] for val in ligcons[ax_lig]]
            else:
                eq_con_list = [eq_con_bidentate_list[0], eq_con_bidentate_list[1], eq_con_bidentate_list[2]]
                eq_lig_list = allowed
                ax_con_list = [list(set(ligcons[i]) - set(eq_con_bidentate_list[i])) for i in allowed if
                               (len(eq_con_bidentate_list[i]) == 1)]
                ax_lig_list = [i for i in allowed if (len(eq_con_bidentate_list[i]) == 1)]
    elif (n_ligs == 2 and not pentadentate):  # 4+2 or 3+3
        if ((max(ligdents) == 4) and (min(ligdents) == 2)):  # 4+2
            if loud:
                print('4+2 dentate case')
            # ### 4+2 cases are handled in the exact same way as 4+1+1 seesaws. See above for explanation.
            # ### The seesaw tetradentate ligand has two axial and two equatorial connecting atoms.
            allowed = list(range(0, 2))
            tetradentate_ligand_idx = np.argmax(ligdents)
            tetradentate_cons = ligcons[tetradentate_ligand_idx]
            bidentate_ligand_idx = np.argmin(ligdents)
            bidentate_cons = ligcons[bidentate_ligand_idx]
            pair_combos = list(combinations([0, 1, 2, 3], 2))
            angle_list = []
            pair_list = []
            coord_list = np.array([mol.getAtom(ii).coords() for ii in tetradentate_cons])
            for i, pair in enumerate(pair_combos):
                pair_list.append(list(pair))
                angle = getAngle(coord_list, pair_list[-1], m_coord)
                if loud:
                    print(('pair of atoms, then angle', pair, angle))
                angle_list.append(angle)
            bidentate_axial = False
            if len(np.where(np.array(angle_list) > angle_cutoff)[0]) > 1:
                bidentate_axial = True
            if not bidentate_axial:
                axial_pair = [tetradentate_cons[val] for val in list(pair_list[np.argmax(np.array(angle_list))])]
                eq_plane = list(set(flat_ligcons) - set(axial_pair))
                tet_eq = list(set(eq_plane).intersection(set(tetradentate_cons)))
                eq_con_list = [ligcons[j] if len(ligcons[j]) == 2 else tet_eq for j in range(len(ligcons))]
                eq_lig_list = allowed
                ax_lig_list = [tetradentate_ligand_idx, tetradentate_ligand_idx]
                ax_con_list = [[axial_pair[0]], [axial_pair[1]]]
            else:  # Bidentate is Axial
                ax_lig_list = [bidentate_ligand_idx, bidentate_ligand_idx]
                ax_con_list = [[val] for val in bidentate_cons]
                eq_lig_list = [tetradentate_ligand_idx]
                eq_con_list = [tetradentate_cons]
        if ((max(ligdents) == 3) and (min(ligdents) == 3)):  # 3+3
            if loud:
                print('3+3 dentate case')
            # unused
            # allowed = list(range(0, 2))
            tridentate_cons_1 = ligcons[0]
            tridentate_cons_2 = ligcons[1]
            pair_combos = list(combinations([0, 1, 2], 2))
            angle_list = list()
            pair_list = list()
            coord_list = np.array([mol.getAtom(ii).coords() for ii in tridentate_cons_1])
            for i, pair in enumerate(pair_combos):
                pair_list.append(list(pair))
                angle = getAngle(coord_list, pair_list[-1], m_coord)
                if loud:
                    print(('pair of atoms, then angle', pair, angle))
                angle_list.append(angle)
            test_angle = np.sort(np.array(angle_list))[::-1][0]
            # ### Like the above 3+2+1 case, the tridentate must be classified into planar or not.
            # ### If not planar, both ligands are both axial and equatorial. (by con Mol Weight)
            # ### If planar, the ligand that falls on the eq_points (as found above), is only equatorial,
            # ### but the tridentate ligand with one equatorial position and two axial is classified as both.
            if test_angle < angle_cutoff:
                planar = False
            else:
                planar = True
            if not planar:  # Seesaw
                eq_ligcons = set([flat_ligcons[j] for j in eq_points_max_con_mw])
                eq_con_list = [list(set(tridentate_cons_1).intersection(eq_ligcons)),
                               list(set(tridentate_cons_2).intersection(eq_ligcons))]
                ax_con_list = [list(set(tridentate_cons_1) - set(eq_con_list[0])),
                               list(set(tridentate_cons_2) - set(eq_con_list[1]))]
                eq_lig_list = [0, 1]
                ax_lig_list = [0, 1]
            else:  # Planar
                eq_ligcons = set([flat_ligcons[j] for j in eq_points_max_mw])
                eq_lig1 = list(set(tridentate_cons_1).intersection(eq_ligcons))
                eq_lig2 = list(set(tridentate_cons_2).intersection(eq_ligcons))
                if len(eq_lig1) == 2 or len(eq_lig2) == 2:  # Catch cases where 2 + 2 planar
                    new_combo_idx = list(set([x for x in range(3)])-set([max_mw_idx]))[0]
                    eq_points_max_mw = combo_list[new_combo_idx]
                    eq_ligcons = set([flat_ligcons[j] for j in eq_points_max_mw])
                    eq_lig1 = list(set(tridentate_cons_1).intersection(eq_ligcons))
                    eq_lig2 = list(set(tridentate_cons_2).intersection(eq_ligcons))
                eq_con_list = [eq_lig1, eq_lig2]
                eq_lig_list = [0, 1]
                if len(eq_lig1) == 1:
                    ax_con_list = [list(set(tridentate_cons_1) - set(eq_lig1))]
                    ax_lig_list = [0]
                else:
                    ax_con_list = [list(set(tridentate_cons_2) - set(eq_lig2))]
                    ax_lig_list = [1]
    elif (n_ligs == 2 and pentadentate):  # 5+1
        # ### Handling for pentadentate scaffolds ####
        if loud:
            print('pentadentate case')
        # unused
        # allowed = [0, 1]
        not_eq = list()
        if len(ligcons[0]) == 1:
            # ### This is the axial ligand ####
            # print(j, 'axial lig')
            top_lig = 0
            top_con = ligcons[0]
            not_eq.append(top_lig)
            pent_lig = 1
        else:
            top_lig = 1
            top_con = ligcons[1]
            not_eq.append(top_lig)
            pent_lig = 0
        pentadentate_coord_list = np.array([mol.getAtom(
                    ii).coords() for ii in ligcons[pent_lig]])
        # #### Adjusting this so that by default, any 4 within the same plane will be assigned as eq. ###
        m = np.array([mol.getAtom(mol.findMetal()[0]).coords()])
        p1 = np.array(mol.getAtom(top_con[0]).coords())
        angle_list = []
        for coord in pentadentate_coord_list:
            p2 = coord
            v1u = np.squeeze(np.array((m - p1) / np.linalg.norm((m - p1))))
            v2u = np.squeeze(np.array((m - p2) / np.linalg.norm((m - p2))))
            angle = np.rad2deg(np.arccos(np.clip(np.dot(v1u, v2u), -1.0, 1.0)))
            angle_list.append(angle)
        bot_idx = np.argmax(np.array(angle_list))
        bot_con = [ligcons[pent_lig][bot_idx]]
        bot_lig = pent_lig
        # unused:
        # not_ax_points = combo_list[np.argmin(error_list)]
        if loud:
            print(('This is bot_idx', bot_idx))
            print((flat_ligcons, top_con, bot_con))
        eq_lig_list = [bot_lig]
        eq_con_list = [list(set(flat_ligcons) - set(top_con) - set(bot_con))]
        ax_lig_list = [top_lig, bot_lig]
        ax_con_list = [top_con, bot_con]
        if loud:
            print(('con lists', eq_con_list, ax_con_list))
        ###########################################################################################
        # In the above, the pentadentate ligand is classified as both axial and equatorial.       #
        # The lc atoms are decided by the z-position. Thus the pentadentate ligand has 4 eq-lc    #
        # and 1 ax-lc. Currently should be able to check this and set that up.                    #
        ###########################################################################################
    elif (n_ligs == 1 and hexadentate):  # 6
        if loud:
            print('hexadentate case')
        not_ax_points = eq_points_max_con_mw  # Define by maximum mw plane for hexadentates
        bot_idx = list(set(range(6)) - set(not_ax_points))[0]
        top_idx = list(set(range(6)) - set(not_ax_points))[1]
        if lig_con_weights[top_idx] > lig_con_weights[bot_idx]:  # Move heavier atom to bottom
            tmp_idx = top_idx
            top_idx = bot_idx
            bot_idx = tmp_idx
        if loud:
            print(('This is bot_idx', bot_idx))
        bot_con = [ligcons[0][bot_idx]]
        top_con = [ligcons[0][top_idx]]
        eq_lig_list = [0]
        eq_con_list = [list(set([ligcons[0][i] for i in not_ax_points]))]
        ax_lig_list = [0, 0]
        ax_con_list = [top_con, bot_con]
        if loud:
            print(('con lists', eq_con_list, ax_con_list))
    if eq_sym_match:  # Enforce eq plane to have connecting atoms with same symbol
        flat_eq_con_list = [item for sublist in eq_con_list for item in sublist]
        flat_eq_con_syms = set([mol.getAtom(item).symbol() for item in flat_eq_con_list])
        if len(flat_eq_con_syms) != 1:  # If more than 1 different type of symbol in eq plane!
            if loud:
                print('Correcting for eq plane with identical chemical symbols for con atoms.')
            pair_combos = list(combinations([0, 1, 2, 3, 4, 5], 2))
            pair_list = list()
            for pair in pair_combos:
                pair_list.append(list(pair))
            point_combos = [pair_list[argsort_angle_list[0]] + pair_list[argsort_angle_list[1]],
                            pair_list[argsort_angle_list[1]] + pair_list[argsort_angle_list[2]],
                            pair_list[argsort_angle_list[2]] + pair_list[argsort_angle_list[0]]]
            symbols_combos = list()
            for combo in point_combos:
                tmp_sym_combos = set()
                for point_num in combo:
                    tmp_sym_combos.add(mol.getAtom(flat_ligcons[point_num]).symbol())
                symbols_combos.append(len(tmp_sym_combos))  # Save number of distinct con_atoms
            # Get plane with fewest distinct types of connecting atoms
            eq_plane_min_atom_types = point_combos[np.argmin(symbols_combos)]
            eq_ligcons = [flat_ligcons[i] for i in eq_plane_min_atom_types]
            ax_ligcons = list(set(flat_ligcons)-set(eq_ligcons))
            eq_lig_list = list()
            eq_con_list = list()
            for lig_con in eq_ligcons:
                lig_ref = flat_lig_refs[flat_ligcons.index(lig_con)]
                if lig_ref in eq_lig_list:
                    eq_con_list[eq_lig_list.index(lig_ref)].append(lig_con)
                else:
                    eq_con_list.append([lig_con])
                    eq_lig_list.append(lig_ref)
            ax_lig_list = list()
            ax_con_list = list()
            for lig_con in ax_ligcons:
                lig_ref = flat_lig_refs[flat_ligcons.index(lig_con)]
                if lig_ref in ax_lig_list:
                    ax_con_list[ax_lig_list.index(lig_ref)].append(lig_con)
                else:
                    ax_con_list.append([lig_con])
                    ax_lig_list.append(lig_ref)
        # ###### Code in case further atom-wise bond-length constraints wanted ######
        # else: # All eq same symbol. Ensure axial are flagged as two different bond lengths.
        #     con_bond_lengths = [np.round(mol.getDistToMetal(x,metal_index),6) for x in flat_ligcons]
        #     bl_set = list(set(con_bond_lengths)) # Check if there are 2 different bond lengths
        #     if len(bl_set) == 2 and (con_bond_lengths.count(bl_set[0])==2
        #                              or con_bond_lengths.count(bl_set[1])==2):
        #         con_bls = np.array(con_bond_lengths)
        #         type1_idxs = np.where(con_bls==bl_set[0])[0]
        #         type2_idxs = np.where(con_bls==bl_set[1])[0]
        #         if len(type1_idxs) > len(type2_idxs):
        #             eq_ligcons = [flat_ligcons[i] for i in type1_idxs]
        #             ax_ligcons = list(set(flat_ligcons)-set(eq_ligcons))
        #         else:
        #             eq_ligcons = [flat_ligcons[i] for i in type2_idxs]
        #             ax_ligcons = list(set(flat_ligcons)-set(eq_ligcons))
        #         eq_lig_list = list()
        #         eq_con_list = list()
        #         for lig_con in eq_ligcons:
        #             lig_ref = flat_lig_refs[flat_ligcons.index(lig_con)]
        #             if lig_ref in eq_lig_list:
        #                 eq_con_list[eq_lig_list.index(lig_ref)].append(lig_con)
        #             else:
        #                 eq_con_list.append([lig_con])
        #                 eq_lig_list.append(lig_ref)
        #         ax_lig_list = list()
        #         ax_con_list = list()
        #         for lig_con in ax_ligcons:
        #             lig_ref = flat_lig_refs[flat_ligcons.index(lig_con)]
        #             if lig_ref in ax_lig_list:
        #                 ax_con_list[ax_lig_list.index(lig_ref)].append(lig_con)
        #             else:
        #                 ax_con_list.append([lig_con])
        #                 ax_lig_list.append(lig_ref)
    # Build ligand list for ax/eq positions, compile all information
    ax_ligand_list = [built_ligand_list[i] for i in ax_lig_list]
    eq_ligand_list = [built_ligand_list[i] for i in eq_lig_list]
    if loud and valid:
        print(('lig_nat_list', lig_natoms_list))
        print(('eq_liq is ind ', eq_lig_list))
        print(('ax_liq is ind ', ax_lig_list))
        print(('ax built lig [0] ext ind :' +
               str(list(built_ligand_list[ax_lig_list[0]].ext_int_dict.keys()))))
        if len(ax_lig_list) > 1:
            print(('ax built lig [1] ext ind :' +
                   str(list(built_ligand_list[ax_lig_list[1]].ext_int_dict.keys()))))
        print(('eq built lig [0] ext ind: ' +
               str(list(built_ligand_list[eq_lig_list[0]].ext_int_dict.keys()))))
        print(('eq_con is ' + str((eq_con_list))))
        print(('ax_con is ' + str((ax_con_list))))

    for j, ax_con in enumerate(ax_con_list):
        current_ligand_index_list = built_ligand_list[ax_lig_list[j]].index_list
        ax_con_int_list.append([current_ligand_index_list.index(i) for i in ax_con])
    for j, eq_con in enumerate(eq_con_list):
        current_ligand_index_list = built_ligand_list[eq_lig_list[j]].index_list
        eq_con_int_list.append([current_ligand_index_list.index(i) for i in eq_con])
    if loud:
        print(('int eq ' + str(eq_con_int_list)))
        print(('ext eq ' + str(eq_con_list)))
        print('**********************************************')
    for ax_lig in ax_lig_list:
        ax_natoms_list.append(lig_natoms_list[ax_lig])
    for eq_lig in eq_lig_list:
        eq_natoms_list.append(lig_natoms_list[eq_lig])
    return (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list,
            ax_con_int_list, eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list)


def ligand_assign_alleq(mol, liglist, ligdents, ligcons):
    ax_ligand_list, ax_con_int_list = [], []
    eq_ligand_list, eq_con_int_list = [], []
    for i, ligand_indices in enumerate(liglist):
        this_ligand = ligand(mol, ligand_indices, ligdents[i])
        this_ligand.obtain_mol3d()
        eq_con = ligcons[i]
        eq_ligand_list.append(this_ligand)
        current_ligand_index_list = this_ligand.index_list
        eq_con_int_list.append([current_ligand_index_list.index(i) for i in eq_con])
    return ax_ligand_list, eq_ligand_list, ax_con_int_list, eq_con_int_list


def get_lig_symmetry(mol, loud=False, htol=3):
    """Handles ligand symmetry assignment.

    Parameters
    ----------
        mol : mol3D
            mol3D class instance.
        loud : bool
            Flag for extra printout. Default is False.
        htol : int, optional
            Tolerance for hydrogens in matching ligands. Default is 3.

    Returns
    -------
        outstring : str
            ligand symmetry plane

    """
    liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=True)
    ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, \
        ax_con_int_list, eq_con_int_list, ax_con_list, \
        eq_con_list, built_ligand_list = ligand_assign_consistent(mol, liglist, ligdents, ligcons, loud=loud)
    max_dent = max(ligdents)
    min_dent = min(ligdents)
    flat_ligcons = [item for sublist in ligcons for item in sublist]
    # ## Below, take all combinations of two atoms, and measure their angles through the metal center

    def compare_ligs(ligs):
        unique_ligands = list()
        unique_ligcons = list()
        unique_hs = list()
        for j, built_ligs in enumerate(ligs):
            # test if ligand is unique without hydrogens added?
            sl = sorted([atom.symbol() for atom in built_ligs.master_mol.getAtomwithinds(built_ligs.index_list)])
            hs = len([item for item in sl if item == 'H'])
            sl = [item for item in sl if item != 'H']
            if len(sl) < 1:
                sl = ['H']
            ligcon_inds = [x for x in built_ligs.index_list if x in flat_ligcons]
            sl_ligcon = sorted([atom.symbol() for atom in built_ligs.master_mol.getAtomwithinds(ligcon_inds)])
            unique = 1  # Flag for detecting unique ligands
            for i, other_sl in enumerate(unique_ligands):
                if sl == other_sl and sl_ligcon == unique_ligcons[i] and np.isclose(hs, unique_hs[i], atol=float(htol)):
                    # Duplicate
                    unique = 0
            if unique == 1:
                unique_ligands.append(sl)
                unique_ligcons.append(sl_ligcon)
                unique_hs.append(hs)
        if len(unique_ligands) < len(ligs):
            copy_in_list = True
        else:
            copy_in_list = False
        return copy_in_list
    # Build Ligands and get MWs of ligands
    built_ligand_list = list()
    for i, ligand_indices in enumerate(liglist):
        this_ligand = ligand(mol, ligand_indices, ligdents[i])
        this_ligand.obtain_mol3d()
        built_ligand_list.append(this_ligand)
    # ## Obtain coordinates for the connecting atoms. Flat coord list ends up being used for comparisons.
    # Bin and sort ligands as Unique
    unique_ligands = list()
    unique_ligcons = list()
    unique_counts = list()
    unique_hs = list()
    for j, built_ligs in enumerate(built_ligand_list):
        # test if ligand is unique without hydrogens added
        sl = sorted([atom.symbol() for atom in built_ligs.master_mol.getAtomwithinds(built_ligs.index_list)])
        hs = len([item for item in sl if item == 'H'])
        sl = [item for item in sl if item != 'H']
        if len(sl) < 1:
            sl = ['H']
        ligcon_inds = [x for x in built_ligs.index_list if x in flat_ligcons]
        sl_ligcon = sorted([atom.symbol() for atom in built_ligs.master_mol.getAtomwithinds(ligcon_inds)])
        unique = 1  # Flag for detecting unique ligands
        for i, other_sl in enumerate(unique_ligands):
            if sl == other_sl and sl_ligcon == unique_ligcons[i] and np.isclose(hs, unique_hs[i], atol=float(htol)):
                # Duplicate
                unique = 0
                unique_counts[i] += 1
        if unique == 1:
            unique_ligands.append(sl)
            unique_ligcons.append(sl_ligcon)
            unique_counts.append(1)
            unique_hs.append(hs)
    n_unique_ligs = len(unique_ligands)  # Number of unique ligands
    max_eq_count = max([len(x) for x in eq_con_list])  # Maximum cons of same lig in eq plane
    if max_dent == 6:
        outstring = '6'
    elif max_dent == 5 and min_dent == 1:
        outstring = '51'
    elif max_dent == 4 and min_dent == 2:
        if liglist[np.argmin(ligdents)] == ax_ligand_list[0].index_list:
            outstring = '4planar2'
        else:
            outstring = '42'
    elif max_dent == 4 and min_dent == 1:
        # 411c, 411t, 4|11|c, 4|11|t
        if n_unique_ligs == 2:
            # 4|11|c, 4|11|t
            if max_eq_count == 4:
                outstring = '4|11|t'
            elif max_eq_count == 2:
                outstring = '4|11|c'
            else:
                outstring = 'Error411'
        elif n_unique_ligs == 3:
            # 411c, 411t
            if max_eq_count == 4:
                outstring = '411t'
            elif max_eq_count == 2:
                outstring = '411c'
            else:
                outstring = 'Error411'
        else:
            outstring = 'Error411'
    elif max_dent == 3 and min_dent == 3:
        # 33f, 33m, |33|f, |33|m
        if n_unique_ligs == 2:
            if max_eq_count == 2:
                outstring = '33f'
            elif max_eq_count == 3:
                outstring = '33m'
            else:
                outstring = 'Error33'
        else:
            if max_eq_count == 2:
                outstring = '|33|f'
            elif max_eq_count == 3:
                outstring = '|33|m'
            else:
                outstring = 'Error33'
    elif max_dent == 3 and 2 in ligdents:
        # 3f21, 3m21
        if max_eq_count == 2:
            outstring = '3f21'
        elif max_eq_count == 3:
            if ax_ligand_list[1].index_list == ax_ligand_list[0].index_list:
                outstring = '3m2t1'
            else:
                outstring = '3m21'
        else:
            outstring = 'Error321'
    elif max_dent == 3 and 2 not in ligdents:
        # 3f111, 3m111, 3f|11|1, 3m|11|t1, 3m|11|c1,
        # 3m|111|, 3f|111|
        if n_unique_ligs == 4:
            if max_eq_count == 2:
                outstring = '3f111'
            else:
                outstring = '3m111'
        elif n_unique_ligs == 3:
            if max_eq_count == 2:
                outstring = '3f|11|1'
            else:
                if compare_ligs(ax_ligand_list):
                    outstring = '3m|11|t1'
                else:
                    outstring = '3m|11|c1'
        elif n_unique_ligs == 2:
            if max_eq_count == 2:
                outstring = '3f|111|'
            else:
                outstring = '3m|111|'
        else:
            outstring = 'Error3111'
    elif max_dent == 2 and min_dent == 2:
        # 222, 2|22|, |222|
        if ax_ligand_list[1].index_list == ax_ligand_list[0].index_list:
            outstring = '22planar2'
        elif n_unique_ligs == 3:
            outstring = '222'
        elif n_unique_ligs == 2:
            outstring = '|22|2'
        elif n_unique_ligs == 1:
            outstring = '|222|'
        else:
            outstring = 'Error222'
    elif max_dent == 2 and len(ligdents) == 4:
        # 2211c, 2211t, |22|11c, |22|11t, 22|11|c, 22|11|t
        # |22||11|c, |22||11|t
        if ax_ligand_list[1].index_list == ax_ligand_list[0].index_list:
            outstring = '2t211'
        elif max_eq_count == 2:  # Must be planar bidentates, trans monodentates
            if compare_ligs(eq_ligand_list):  # Bidentates identical
                if compare_ligs(ax_ligand_list):  # Monodentates identical
                    outstring = '|22||11|t'
                else:
                    outstring = '|22|11t'
            else:  # Bidentates different
                if compare_ligs(ax_ligand_list):  # Monodentates identical
                    outstring = '22|11|t'
                else:
                    outstring = '2211t'
        elif max_eq_count == 1:  # Must seesaw bidentates, cis-monodentates
            if compare_ligs(ax_ligand_list):  # Bidentates identical
                eq_ligand_list.remove(ax_ligand_list[0])  # remove one bidentate
                if compare_ligs(eq_ligand_list):  # Monodentates identical
                    outstring = '|22||11|c'
                else:
                    outstring = '|22|11c'
            else:  # Bidentates different
                if compare_ligs(eq_ligand_list):  # Monodentates identical
                    outstring = '22|11|c'
                else:
                    outstring = '2211c'
        else:
            outstring = 'Error2211'
    elif max_dent == 2 and len(ligdents) == 5:
        # 21111, 2|11|t11, 2|11|c11t, 2|11|c11c , 2|111|m1, 2|111|f1
        # 2|1111|, 2|11|c|11|t, 2|11|c|11|c
        if liglist[np.argmax(ligdents)] == ax_ligand_list[0].index_list:
            outstring = '2t1111'
        elif n_unique_ligs == 5:
            outstring = '21111'
        elif n_unique_ligs == 4:
            if compare_ligs(ax_ligand_list):  # trans
                outstring = '2|11|t11'
            else:
                if compare_ligs(eq_ligand_list):  # trans different monodentates
                    outstring = '2|11|c11t'
                else:
                    outstring = '2|11|c11c'
        elif n_unique_ligs == 3:
            if max(unique_counts) == 3:
                if compare_ligs(ax_ligand_list):  # mer
                    outstring = '2|111|m1'
                else:
                    outstring = '2|111|f1'
            elif max(unique_counts) == 2:
                if compare_ligs(ax_ligand_list):  # trans
                    outstring = '2|11|c|11|t'
                else:
                    outstring = '2|11|c|11|c'
            else:
                outstring = 'Error21111'
        elif n_unique_ligs == 2:
            outstring = '2|1111|'
        else:
            outstring = 'Error21111'
    elif max_dent == 1:
        trans_pairs = [ax_ligand_list, [eq_ligand_list[0], eq_ligand_list[2]],
                       [eq_ligand_list[1], eq_ligand_list[3]]]
        if n_unique_ligs == 6:
            outstring = '111111'
        elif n_unique_ligs == 5:
            # Compare all trans positions. If not trans set as cis
            trans_count = 0
            for pair in trans_pairs:
                if compare_ligs(pair):
                    trans_count += 1
            if trans_count == 1:
                outstring = '|11|t1111'
            elif trans_count == 0:
                outstring = '|11|c1111'
            else:
                outstring = 'Error|11|1111'
        elif n_unique_ligs == 4:
            # |11|c|11|t11, |11|c|11|c11t, |11|c|11|c11c, |11|t|11|t11,
            # |111|f/m111
            if max(unique_counts) == 3:  # fac/mer
                mer = False
                for pair in trans_pairs:
                    if compare_ligs(pair):
                        mer = True
                if mer:
                    outstring = '|111|m111'
                else:
                    outstring = '|111|f111'
            elif max(unique_counts) == 2:
                trans_count = 0
                for pair in trans_pairs:
                    if compare_ligs(pair):
                        trans_count += 1
                if trans_count == 2:
                    outstring = '|11|t|11|t11'
                elif trans_count == 1:
                    outstring = '|11|c|11|t11'
                elif trans_count == 0:
                    planes = [eq_ligand_list, [eq_ligand_list[0]]+[eq_ligand_list[2]]+ax_ligand_list,
                              [eq_ligand_list[1]]+[eq_ligand_list[3]]+ax_ligand_list]
                    plane_counts = 0
                    for plane in planes:
                        if compare_ligs(plane):
                            plane_counts += 1
                    if plane_counts == 1:
                        outstring = '|11|c|11|c11t'
                    elif plane_counts == 2:
                        outstring = '|11|c|11|c11c'
                    else:
                        outstring = 'Error|11|c|11|c11'
                else:
                    outstring = 'Error|11||11|11'
            else:
                outstring = 'Error|11||11|11'
        elif n_unique_ligs == 3:
            # |11|c|11|c|11|t, |11|t|11|t|11|t, |11|c|11|c|11|c
            # |1111|11c, |1111|11t
            # |111|f|11|1, |111|m|11|t1, |111|m|11|c1
            if max(unique_counts) == 4:
                trans_count = 0
                for pair in trans_pairs:
                    if compare_ligs(pair):
                        trans_count += 1
                if trans_count == 2:
                    outstring = '|1111|11t'
                elif trans_count == 1:
                    outstring = '|1111|11c'
                else:
                    outstring = 'Error|1111|11'
            elif max(unique_counts) == 3:
                trans_count = 0
                for pair in trans_pairs:
                    if compare_ligs(pair):
                        trans_count += 1
                if trans_count == 2:
                    outstring = '|111|m|11|t1'
                elif trans_count == 1:
                    outstring = '|111|m|11|c1'
                elif trans_count == 0:
                    outstring = '|111|f|11|1'
                else:
                    outstring = 'Error|111||11|1'
            elif max(unique_counts) == 2:
                trans_count = 0
                for pair in trans_pairs:
                    if compare_ligs(pair):
                        trans_count += 1
                if trans_count == 3:
                    outstring = '|11|t|11|t|11|t'
                elif trans_count == 1:
                    outstring = '|11|c|11|c|11|t'
                elif trans_count == 0:
                    outstring = '|11|c|11|c|11|c'
                else:
                    outstring = 'Error|11||11|11|'
            else:
                outstring = 'Error_3U_mono'
        elif n_unique_ligs == 2:
            # |11111|1
            # |111|f|111|,|111|m|111|
            # |1111||11|t, |1111||11|c
            if max(unique_counts) == 5:
                outstring = '|11111|1'
            elif max(unique_counts) == 4:
                trans_count = 0
                for pair in trans_pairs:
                    if compare_ligs(pair):
                        trans_count += 1
                if trans_count == 3:
                    outstring = '|1111||11|t'
                elif trans_count == 1:
                    outstring = '|1111||11|c'
                else:
                    outstring = 'Error|1111||11|'
            elif max(unique_counts) == 3:
                trans_count = 0
                for pair in trans_pairs:
                    if compare_ligs(pair):
                        trans_count += 1
                if trans_count == 2:
                    outstring = '|111|m|111|'
                elif trans_count == 0:
                    outstring = '|111|f|111|'
                else:
                    outstring = 'Error|111||111|'
            else:
                outstring = 'Error_2U_mono'
        elif n_unique_ligs == 1:
            outstring = '|111111|'
        else:
            outstring = 'Error_Monodentate'
    else:
        outstring = 'Error_None_Fit'
    return outstring
