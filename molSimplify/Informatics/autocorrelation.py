from molSimplify.Classes.globalvars import globalvars
from molSimplify.Classes.ligand import (
    ligand_assign_original,
    ligand_breakdown,
    )
from molSimplify.Scripts.geometry import distance
import numpy as np

# ########## UNIT CONVERSION
HF_to_Kcal_mol = 627.503

globs = globalvars()


def append_descriptors(descriptor_names, descriptors, list_of_names, list_of_props, prefix, suffix):
    """
    Utility to build standardly formatted RACs.

    Parameters
    ----------
        descriptor_names : list
            Descriptors names list to be appended to.
        descriptors : list
            Descriptors list to be appended to.
        list_of_names : list
            Names to be added.
        list_of_props : list
            Types of RACs.
        prefix : str
            Prefix to be added to names.
        suffix : str
            Suffix to be added to names.

    Returns
    -------
        descriptor_names : list
            Compiled list of descriptor names.
        descriptors : list
            Compiled list of descriptor values.

    """
    try:
        basestring
    except NameError:
        basestring = str

    for names in list_of_names:
        if not isinstance(names, basestring):
            names = ["-".join([prefix, str(i), suffix]) for i in names]
            descriptor_names += names
        else:
            names = "-".join([prefix, str(names), suffix])
            descriptor_names.append(names)
    for values in list_of_props:
        if not isinstance(names, basestring):
            descriptors.extend(values)
        else:
            descriptors.append(values)
    return descriptor_names, descriptors


def append_descriptor_derivatives(descriptor_derivative_names, descriptor_derivatives,
                                  mat_of_names, dmat, prefix, suffix):
    """
    Utility to build standardly formatted RACs derivatives.

    Parameters
    ----------
        descriptor_derivative_names : list
            RAC names, will be a matrix, will be appended to.
        descriptor_derivatives : list
            RAC, will be appended to.
        mat_of_names : list
            Names, will be added.
        dmat : list
            Mat of RAC derivatives.
        prefix : str
            RAC prefix.
        suffix : str
            RAC suffix.

    Returns
    -------
        descriptor_derivative_names : list
            Compiled list (matrix) of descriptor derivative names.
        descriptor_derivatives : list
            Derivatives of RACs w.r.t atomic props (matrix).

    """
    for names in mat_of_names:
        jnames = ["-".join([prefix, str(i), suffix]) for i in names]
        descriptor_derivative_names.append(jnames)
    if descriptor_derivatives is None:
        descriptor_derivatives = dmat
    else:
        descriptor_derivatives = np.row_stack([descriptor_derivatives, dmat])
    return descriptor_derivative_names, descriptor_derivatives


def autocorrelation(mol, prop_vec, orig, d, oct=True, use_dist=False, size_normalize=False):
    """
    Calculate and return the products autocorrelation.

    Parameters
    ----------
        mol : mol3D
            mol3D object to calculate autocorrelation over.
        prop_vec : np.array
            Property of atoms in mol in order of index.
        orig : int
            Zero-indexed starting atom.
        d : int
            Number of hops to travel.
        oct : bool, optional
            Flag is octahedral complex, by default True.
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False.
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.

    Returns
    -------
        result_vector : np.array
            assembled products autocorrelations

    """
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    if not use_dist:
        result_vector[hopped] = prop_vec[orig] * prop_vec[orig]
    else:
        result_vector[hopped] = 0.5 * abs(prop_vec[orig]) ** 2.4 / mol.natoms
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        for inds in new_active_set:
            if not use_dist:
                result_vector[hopped] += prop_vec[orig] * prop_vec[inds]
            else:
                this_dist = mol.getDistToMetal(orig, inds)
                result_vector[hopped] += prop_vec[orig] * prop_vec[inds] / this_dist
            historical_set.update(active_set)
        active_set = new_active_set
        if size_normalize:
            result_vector[hopped] /= mol.natoms
    return (result_vector)


def autocorrelation_derivative(mol, prop_vec, orig, d, oct=True):
    """
    Returns derivative vector of products autocorrelations.

    Parameters
    ----------
        mol : mol3D
            mol3D object to calculate derivatives over.
        prop_vec : np.array
            Property of atoms in mol in order of index.
        orig : int
            Zero-indexed starting atom.
        d : int
            Number of hops to travel.
        oct : bool, optional
            Flag is octahedral complex, by default True.

    Returns
    -------
        derivative_mat : np.array
            RAC derivatives matrix.

    """
    derivative_mat = np.zeros((d + 1, len(prop_vec)))
    # Loop for each atom.
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    for derivate_ind in range(0, len(prop_vec)):
        if derivate_ind == orig:
            derivative_mat[hopped, derivate_ind] = 2 * prop_vec[orig]
        else:
            derivative_mat[hopped, derivate_ind] = 0
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # Prepare all atoms attached to this connection.
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        for inds in new_active_set:
            for derivate_ind in range(0, len(prop_vec)):
                if derivate_ind == orig:
                    derivative_mat[hopped, derivate_ind] += prop_vec[inds]
                elif derivate_ind == inds:
                    derivative_mat[hopped, derivate_ind] += prop_vec[orig]
            historical_set.update(active_set)
        active_set = new_active_set
    return (derivative_mat)


def deltametric(mol, prop_vec, orig, d, oct=True, use_dist=False, size_normalize=False):
    """
    Returns the deltametric autocorrelation.

    Parameters
    ----------
        mol : mol3D
            mol3D object to calculate deltametric autocorrelation over.
        prop_vec : np.array
            Property of atoms in mol in order of index.
        orig : int
            Zero-indexed starting atom.
        d : int
            Number of hops to travel.
        oct : bool, optional
            Flag is octahedral complex, by default True.
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False.
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.

    Returns
    -------
        results_vector : np.array
            Deltametric autocorrelations.

    """
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # Prepare all atoms attached to this connection.
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        for inds in new_active_set:
            if not use_dist:
                result_vector[hopped] += prop_vec[orig] - prop_vec[inds]
            else:
                this_dist = mol.getDistToMetal(orig, inds)
                result_vector[hopped] += (prop_vec[orig] - prop_vec[inds]) / (this_dist + 1e-6)
            historical_set.update(active_set)
        active_set = new_active_set
        if size_normalize:
            result_vector[hopped] /= mol.natoms
    return (result_vector)


def deltametric_derivative(mol, prop_vec, orig, d, oct=True):
    """
    Returns the deltametric autocorrelation derivative vector.

    Parameters
    ----------
        mol : mol3D
            mol3D object to calculate deltametric autocorrelation derivative over.
        prop_vec : np.array
            Property of atoms in mol in order of index.
        orig : int
            Zero-indexed starting atom.
        d : int
            Number of hops to travel.
        oct : bool, optional
            Flag is octahedral complex, by default True.

    Returns
    -------
        derivative_mat : np.array
            Deltametric autocorrelation derivatives matrix.

    """
    derivative_mat = np.zeros((d + 1, len(prop_vec)))
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    # The zero-depth element is always zero.
    for derivate_ind in range(0, len(prop_vec)):
        derivative_mat[hopped, derivate_ind] = 0.0
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # Prepare all atoms attached to this connection.
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        for inds in new_active_set:
            for derivate_ind in range(0, len(prop_vec)):
                if derivate_ind == orig:
                    derivative_mat[hopped, derivate_ind] += 1
                elif derivate_ind == inds:
                    derivative_mat[hopped, derivate_ind] += -1
        historical_set.update(active_set)
        active_set = new_active_set
    return (derivative_mat)


def construct_property_vector(mol, prop, oct=True, modifier=False, MRdiag_dict={}, transition_metals_only=True):
    """
    Assigns the value of property for atom i (zero index) in mol to position i in returned vector.

    Parameters
    ----------
        mol : mol3D
            Molecule to generate property vector for.
        prop : str
            Property to generate vector for - Acceptable prop values: ['electronegativity',
            'nuclear_charge', 'ident', 'topology', 'ox_nuclear_charge', 'size', 'vdwrad',
            'group_number', 'polarizability', 'bondvalence', 'num_bonds',
            'bondvalence_devi', 'bodavrg', 'bodstd', 'charge']
        oct : bool, optional
            Flag if octahedral complex, by default True.
        modifier : bool, optional
            If passed - dict, used to modify prop vector (e.g., for adding
            ONLY used with  ox_nuclear_charge    ox or charge)
            {"Fe":2, "Co": 3} etc, by default False.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.
        transition_metals_only : bool, optional
            Flag if only transition metals counted as metals, by default True.

    Returns
    -------
        w : np.array
            Property vector for mol by atom.

    """
    allowed_strings = [
    'electronegativity', 'nuclear_charge', 'ident', 'topology',
    'ox_nuclear_charge', 'size', 'vdwrad', 'group_number', 'polarizability',
    'bondvalence', 'num_bonds', 'bondvalence_devi', 'bodavrg', 'bodstd', 'charge',
    ]
    if len(MRdiag_dict):
        for k in list(MRdiag_dict):
            allowed_strings += [k]
    prop_dict = dict()
    w = np.zeros(mol.natoms)
    done = False
    if prop not in allowed_strings:
        print(('error, property  ' + str(prop) + ' is not a vaild choice'))
        print((' options are  ' + str(allowed_strings)))
        return False
    if prop == 'electronegativity':
        prop_dict = globs.endict()
    elif prop == 'size':
        at_keys = list(globs.amass().keys())
        for keys in at_keys:
            values = globs.amass()[keys][2]
            prop_dict.update({keys: values})
    elif prop == 'nuclear_charge':
        at_keys = list(globs.amass().keys())
        for keys in at_keys:
            values = globs.amass()[keys][1]
            prop_dict.update({keys: values})
    elif prop == 'group_number':  # Uses number of valence electrons
        at_keys = list(globs.amass().keys())
        for keys in at_keys:
            values = globs.amass()[keys][3]
            prop_dict.update({keys: values})
    elif prop == 'ox_nuclear_charge':
        if not modifier:
            print('Error, must give modifier with ox_nuclear_charge')
            return False
        else:
            at_keys = list(globs.amass().keys())
            for keys in at_keys:
                values = globs.amass()[keys][1]
                if keys in list(modifier.keys()):
                    values -= float(modifier[keys])  # assumes oxidation state provided (i.e. Fe(IV))
                prop_dict.update({keys: values})
    elif prop == 'polarizability':
        prop_dict = globs.polarizability()
        for i, atoms in enumerate(mol.getAtoms()):
            atom_type = atoms.symbol()
            w[i] = prop_dict[atom_type]
    elif prop == 'ident':
        at_keys = list(globs.amass().keys())
        for keys in at_keys:
            prop_dict.update({keys: 1})
    elif prop == 'topology':
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = len(mol.getBondedAtomsSmart(i, oct=oct))
        done = True
    elif prop == 'vdwrad':
        prop_dict = globs.vdwrad()
        for i, atoms in enumerate(mol.getAtoms()):
            atom_type = atoms.symbol()
            if atom_type in globs.metalslist():
                w[i] = globs.amass()[atoms.symbol()][2]
            else:
                w[i] = prop_dict[atoms.symbol()]
        done = True
    elif prop == 'bondvalence':
        assert len(mol.getAtoms()) == len(mol.bv_dict)
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = mol.bv_dict[i]
        done = True
    elif prop == 'num_bonds':
        for i, atom in enumerate(mol.getAtoms()):
            if not atom.ismetal(transition_metals_only=transition_metals_only):
                w[i] = globs.bondsdict()[atom.symbol()]
            else:
                w[i] = len(mol.getBondedAtomsSmart(i, oct=oct))
        done = True
    elif prop == 'bondvalence_devi':
        assert len(mol.getAtoms()) == len(mol.bvd_dict)
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = mol.bvd_dict[i]
        done = True
    elif prop == 'bodavrg':
        assert len(mol.getAtoms()) == len(mol.bodavrg_dict)
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = mol.bodavrg_dict[i]
        done = True
    elif prop == 'bodstd':
        assert len(mol.getAtoms()) == len(mol.bodstd_dict)
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = mol.bodstd_dict[i]
        done = True
    elif prop == 'charge':
        assert len(mol.getAtoms()) == len(mol.charge_dict)
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = mol.charge_dict[i]
        done = True
    elif prop in ['A25PBE', 'B1', 'rND_PBE', 'IND_PBE', 'nLUMO_MP2', 'T1',
                  'largest_amp', 'TAE', 'C0^2', 'nLUMO_CAS', '%CorrE_orca']:
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = MRdiag_dict[prop][mol.getAtom(i).symbol()]
        done = True
    if not done:
        for i, atoms in enumerate(mol.getAtoms()):
            w[i] = prop_dict[atoms.symbol()]
    return (w)


def full_autocorrelation(mol, prop, d, oct=True, modifier=False, use_dist=False, size_normalize=False, MRdiag_dict={}, transition_metals_only=True):
    """
    Calculate full scope product autocorrelations (i.e., start at every atom up to depth d).

    Parameters
    ----------
        mol : mol3D
            Molecule to calculate full scope RAC over.
        prop : str
            Property to evaluate.
        d : int
            Depth of full scope autocorrelation.
        oct : bool, optional
            Is octahedral flag, by default True.
        modifier : bool, optional
            Use ox modifier, by default False.
        use_dist : bool, optional
            Weigh autocorrelation by distance of atoms from each other, by default False.
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.
        transition_metals_only : bool, optional
            Flag if only transition metals counted as metals, by default True.

    Returns
    -------
        autocorrelation_vector : np.array
            Full scope product autocorrelation values.

    """
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier, MRdiag_dict=MRdiag_dict, transition_metals_only=transition_metals_only)
    index_set = list(range(0, mol.natoms))
    autocorrelation_vector = np.zeros(d + 1)
    for centers in index_set:
        autocorrelation_vector += autocorrelation(mol, w, centers, d, oct=oct, use_dist=use_dist,
                                                  size_normalize=size_normalize)
    return (autocorrelation_vector)


def full_autocorrelation_derivative(mol, prop, d, oct=True, modifier=False):
    """
    Calculate full scope product autocorrelations derivatives
    (i.e., start at every atom up to depth d).

    Parameters
    ----------
        mol : mol3D
            Molecule to calculate full scope RAC over.
        prop : str
            Property to evaluate.
        d : int
            Depth of scope to evaluate.
        oct : bool, optional
            Is octahedral flag, by default True.
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False.
        modifier : bool, optional
            Use ox modifier, by default False.

    Returns
    -------
        autocorrelation_derivative_mat : np.array
            Full scope autocorrelation derivative matrix.

    """
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
    index_set = list(range(0, mol.natoms))
    autocorrelation_derivative_mat = np.zeros((d + 1, mol.natoms))
    for centers in index_set:
        autocorrelation_derivative_mat += autocorrelation_derivative(mol, w, centers, d, oct=oct)
    return (autocorrelation_derivative_mat)


def generate_full_complex_autocorrelations(mol,
                                           depth=4, oct=True,
                                           flag_name=False, modifier=False,
                                           use_dist=False, size_normalize=False,
                                           Gval=False, NumB=False, polarizability=False,
                                           MRdiag_dict={},
                                           transition_metals_only=True):
    """
    Utility to manage full complex autocorrelation generation and labeling.
    Works on any molecule, not just TM complexes.

    Parameters
    ----------
        mol : mol3D
            Molecule used for full scope.
        depth : int, optional
            Depth of autocorrelations to evaluate, by default 4.
        oct : bool, optional
            Is an octahedral complex, by default True.
        flag_name : bool, optional
            Prepend "f_all" to results to track full complex, by default False.
        modifier : bool, optional
            Use ox_modifier on metal charge, by default False.
        use_dist : bool, optional
            Weigh autocorrelations by interatomic distances, by default False.
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        Gval : bool, optional
            Use G value as RAC, by default False.
        NumB : bool, optional
            Use number of bonds as RAC, by default False.
        polarizability : bool, optional
            Use polarizability (alpha) as RAC, by default False.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.
        transition_metals_only : bool, optional
            Flag if only transition metals counted as metals, by default True.

    Returns
    -------
        results_dictionary : dict
            Formatted dictionary with {'colnames': colnames, 'results': result}.
            For key colnames, value is list of lists.
            For key results, value is list of np.array.

    """
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    if polarizability:
        allowed_strings += ["polarizability"]
        labels_strings += ["alpha"]
    if len(MRdiag_dict):
        allowed_strings, labels_strings = [], []
        for k in list(MRdiag_dict):
            allowed_strings += [k]
            labels_strings += [k]
    for ii, properties in enumerate(allowed_strings):
        metal_ac = full_autocorrelation(mol, properties, depth,
                                        oct=oct, modifier=modifier,
                                        use_dist=use_dist,
                                        size_normalize=size_normalize,
                                        MRdiag_dict=MRdiag_dict,
                                        transition_metals_only=transition_metals_only)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_f_all': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_full_complex_autocorrelation_derivatives(mol, depth=4, oct=True, flag_name=False,
                                                      modifier=False, Gval=False, NumB=False):
    """
    Utility to manage full complex autocorrelation derivative generation and labeling.

    Parameters
    ----------
        mol : mol3D
            Molecule used for full scope.
        depth : int, optional
            Depth of autocorrelations to evaluate, by default 4.
        oct : bool, optional
            Is an octahedral complex, by default True.
        flag_name : bool, optional
            Prepend "f_all" to results to track full complex, by default False.
        modifier : bool, optional
            Use ox_modifier on metal charge, by default False.
        Gval : bool, optional
            Use G value as RAC, by default False.
        NumB : bool, optional
            Use number of bonds as RAC, by default False.

    Returns
    -------
        results_dictionary : dict
            Formatted dictionary with {'colnames': colnames, 'results': result}.

    """
    result = None
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        f_ac_der = full_autocorrelation_derivative(mol, properties, depth, oct=oct, modifier=modifier)
        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])
        # colnames.append(this_colnames)
        if result is None:
            result = f_ac_der
        else:
            result = np.row_stack([result, f_ac_der])
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_f_all': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def atom_only_autocorrelation(mol, prop, d, atomIdx, oct=True, use_dist=False, size_normalize=False, MRdiag_dict={}):
    """
    Calculate product autocorrelation vectors from a given atom or list of atoms
    (e.g., up to depth 4 from the connecting atoms).

    Parameters
    ----------
        mol : mol3D
            Molecule to calculate atom-only autocorrelations from.
        prop : str
            Property to evaluate.
        d : int
            Depth to calculate derivatives over.
        atomIdx : int or list
            Atoms from which the autocorrelation vector should be centered.
            List of indices or a single index.
        oct : bool, optional
            Use octahedral flag, by default True.
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False.
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.

    Returns
    -------
        autocorrelation_vector : np.array
            List of atom-only autocorrelations.

    """
    w = construct_property_vector(mol, prop, oct, MRdiag_dict=MRdiag_dict)
    autocorrelation_vector = np.zeros(d + 1)
    if hasattr(atomIdx, "__len__"): # Indicative of a list of indices.
        for elements in atomIdx:
            autocorrelation_vector += autocorrelation(mol, w, elements, d, oct=oct, use_dist=use_dist,
                                                      size_normalize=size_normalize)
        autocorrelation_vector = np.divide(autocorrelation_vector, len(atomIdx)) # Averaging.
    else: # Single index.
        autocorrelation_vector += autocorrelation(mol, w, atomIdx, d, oct=oct, use_dist=use_dist,
                                                  size_normalize=size_normalize)
    return (autocorrelation_vector)


def atom_only_autocorrelation_derivative(mol, prop, d, atomIdx, oct=True):
    """
    Calculate product autocorrelation derivative vectors from a given atom or list of atoms
    (e.g., up to depth 4 from the connecting atoms).

    Parameters
    ----------
        mol : mol3D
            Molecule to calculate atom-only autocorrelation derivatives from.
        prop : str
            Property to evaluate.
        d : int
            Depth to calculate derivatives over.
        atomIdx : int or list
            Atoms from which the autocorrelation vector should be centered.
            List of indices or a single index.
        oct : bool, optional
            Use octahedral flag, by default True.

    Returns
    -------
        autocorrelation_derivative_mat : np.array
            List of atom-only autocorrelation derivatives.

    """
    w = construct_property_vector(mol, prop, oct)
    autocorrelation_derivative_mat = np.zeros((d + 1, mol.natoms))
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            autocorrelation_derivative_mat += autocorrelation_derivative(mol, w, elements, d, oct=oct)
        autocorrelation_derivative_mat = np.divide(autocorrelation_derivative_mat, len(atomIdx))
    else:
        autocorrelation_derivative_mat += autocorrelation_derivative(mol, w, atomIdx, d, oct=oct)
    return (autocorrelation_derivative_mat)


def metal_only_autocorrelation(
    mol, prop, d, oct=True,
    func=autocorrelation, modifier=False,
    use_dist=False, size_normalize=False,
    MRdiag_dict={}, transition_metals_only=True
    ):
    """
    Calculate the metal_only product autocorrelations
    (e.g., metal-centered atom-only RACs).

    Averaged over all metals.

    Parameters
    ----------
        mol : mol3D
            Molecule with metal to calculate MC product RACs for.
        prop : str
            Property to evaluate.
        d : int
            Depth of autocorrelation.
        oct : bool, optional
            Use octahedral geometry evaluations, by default True.
        func : function, optional
            Which function to evaluate mc-racs by, by default autocorrelation.
        modifier : bool, optional
            Use ox_modifier, by default False.
            If passed - dict, used to modify prop vector (e.g., for adding
            ONLY used with  ox_nuclear_charge    ox or charge)
            {"Fe":2, "Co": 3} etc, by default False.
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False.
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.
        transition_metals_only : bool, optional
            Flag if only transition metals counted as metals, by default True.

    Returns
    -------
        autocorrelation_vector: np.array
            MC atom-only RACs vector.

    """
    autocorrelation_vector = np.zeros(d + 1)
    metal_idxs = mol.findMetal(transition_metals_only=transition_metals_only)
    if len(metal_idxs) == 0:
        raise Exception('No metal found in mol object.')
    n_met = len(metal_idxs)

    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier, MRdiag_dict=MRdiag_dict, transition_metals_only=transition_metals_only)
    for metal_ind in metal_idxs:
        autocorrelation_vector += func(mol, w, metal_ind, d, oct=oct, use_dist=use_dist, size_normalize=size_normalize)
    autocorrelation_vector = np.divide(autocorrelation_vector, n_met)
    return (autocorrelation_vector)


def metal_only_autocorrelation_derivative(
    mol, prop, d, oct=True,
    func=autocorrelation_derivative, modifier=False,
    transition_metals_only=True):
    """
    Calculate the metal_only product autocorrelation derivatives
    (e.g., metal-centered atom-only RAC derivatives).

    Parameters
    ----------
        mol : mol3D
            Molecule with metal to calculate MC product RAC derivatives for.
        prop : str
            Property to evaluate.
        d : int
            Depth of autocorrelation.
        oct : bool, optional
            Use octahedral geometry evaluations, by default True.
        func : function, optional
            Which function to evaluate mc-racs by, by default autocorrelation_derivative.
        modifier : bool, optional
            Use ox_modifier, by default False.
        transition_metals_only : bool, optional
            Flag if only transition metals counted as metals, by default True.

    Returns
    -------
        autocorrelation_vector: np.array
            MC atom-only RAC derivatives vector (matrix).

    """
    autocorrelation_vector_derivative = np.zeros(d + 1)
    metal_idxs = mol.findMetal(transition_metals_only=transition_metals_only)
    if len(metal_idxs) == 0:
        raise Exception('No metal found in mol object.')
    n_met = len(metal_idxs)

    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier, transition_metals_only=transition_metals_only)
    for metal_ind in metal_idxs:
        autocorrelation_vector_derivative += func(mol, w, metal_ind, d, oct=oct)
    autocorrelation_vector_derivative = np.divide(autocorrelation_vector_derivative, n_met)
    return (autocorrelation_vector_derivative)


def atom_only_deltametric(mol, prop, d, atomIdx, oct=True, modifier=False,
                          use_dist=False, size_normalize=False, MRdiag_dict={}):
    """
    Calculate deltametric autocorrelation vectors from a given atom or list of atoms
    (e.g., up to depth 4 from the connecting atoms).

    Parameters
    ----------
        mol : mol3D
            Molecule to calculate atom-only autocorrelations from.
        prop : str
            Property to evaluate.
        d : int
            Depth to calculate derivatives over.
        atomIdx : int or list
            Atoms from which the autocorrelation vector should be centered.
            List of indices or a single index.
        oct : bool, optional
            Use octahedral flag, by default True.
        modifier : bool, optional
            Use ox modifier, by default False.
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False.
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.

    Returns
    -------
        deltametric_vector : np.array
            List of atom-only deltametric autocorrelations.

    """
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier, MRdiag_dict=MRdiag_dict)
    deltametric_vector = np.zeros(d + 1)
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            deltametric_vector += deltametric(mol, w, elements, d, oct=oct, use_dist=use_dist, size_normalize=size_normalize)
        deltametric_vector = np.divide(deltametric_vector, len(atomIdx))
    else:
        deltametric_vector += deltametric(mol, w, atomIdx, d, oct=oct, use_dist=use_dist, size_normalize=size_normalize)
    return (deltametric_vector)


def atom_only_deltametric_derivative(mol, prop, d, atomIdx, oct=True, modifier=False):
    """
    Calculate deltametric autocorrelation derivative vectors
    from a given atom or list of atoms
    (e.g., up to depth 4 from the connecting atoms).

    Parameters
    ----------
        mol : mol3D
            Molecule to calculate atom-only deltametric autocorrelation derivatives from.
        prop : str
            Property to evaluate.
        d : int
            Depth to calculate derivatives over.
        atomIdx : int or list
            Atoms from which the autocorrelation vector should be centered.
            List of indices or a single index.
        oct : bool, optional
            Use octahedral flag, by default True.
        modifier : bool, optional
            Use ox_modifier, by default False.

    Returns
    -------
        deltametric_derivative_mat : np.array
            Matrix of atom-only deltametric autocorrelation derivatives.

    """
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
    deltametric_derivative_mat = np.zeros((d + 1, mol.natoms))
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            deltametric_derivative_mat += deltametric_derivative(mol, w, elements, d, oct=oct)
        deltametric_derivative_mat = np.divide(deltametric_derivative_mat, len(atomIdx))
    else:
        deltametric_derivative_mat += deltametric_derivative(mol, w, atomIdx, d, oct=oct)
    return (deltametric_derivative_mat)


def metal_only_deltametric(
    mol, prop, d, oct=True,
    func=deltametric, modifier=False,
    use_dist=False, size_normalize=False,
    MRdiag_dict={}, transition_metals_only=True):
    """
    Gets the metal atom-only deltametric RAC.

    Parameters
    ----------
        mol : mol3D
            Molecule with metal to calculate MC deltametric RACs.
        prop : str
            Property to evaluate.
        d : int
            Depth of deltametric.
        oct : bool, optional
            Use octahedral geometry evaluations, by default True.
        func : function, optional
            Which function to evaluate mc-racs by, by default deltametric.
        modifier : bool, optional
            Use ox_modifier, by default False.
            If passed - dict, used to modify prop vector (e.g., for adding
            ONLY used with  ox_nuclear_charge    ox or charge)
            {"Fe":2, "Co": 3} etc, by default False.
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False.
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.
        transition_metals_only : bool, optional
            Flag if only transition metals counted as metals, by default True.

    Returns
    -------
        deltametric_vector : np.array
            Metal-centered deltametric RAC vector.

    """
    deltametric_vector = np.zeros(d + 1)
    metal_idxs = mol.findMetal(transition_metals_only=transition_metals_only)
    if len(metal_idxs) == 0:
        raise Exception('No metal found in mol object.')
    n_met = len(metal_idxs)

    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier, MRdiag_dict=MRdiag_dict, transition_metals_only=transition_metals_only)
    for metal_ind in metal_idxs:
        deltametric_vector += func(mol, w, metal_ind, d, oct=oct, use_dist=use_dist, size_normalize=size_normalize)
    deltametric_vector = np.divide(deltametric_vector, n_met)
    return (deltametric_vector)


def metal_only_deltametric_derivative(
    mol, prop, d, oct=True,
    func=deltametric_derivative, modifier=False,
    transition_metals_only=True):
    """
    Gets the metal atom-only deltametric derivatives.

    Parameters
    ----------
        mol : mol3D
            Molecule with metal to calculate MC deltametric RAC derivatives for.
        prop : str
            Property to evaluate.
        d : int
            Depth of deltametric.
        oct : bool, optional
            Use octahedral geometry evaluations, by default True.
        func : function, optional
            Which function to evaluate mc-racs by, by default deltametric_derivative.
        modifier : bool, optional
            Use ox_modifier, by default False.
        transition_metals_only : bool, optional
            Flag if only transition metals counted as metals, by default True.

    Returns
    -------
        deltametric_vector_derivative : np.array
            Metal-centered deltametric derivatives vector (matrix).

    """
    deltametric_vector_derivative = np.zeros(d + 1)
    metal_idxs = mol.findMetal(transition_metals_only=transition_metals_only)
    if len(metal_idxs) == 0:
        raise Exception('No metal found in mol object.')
    n_met = len(metal_idxs)

    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier, transition_metals_only=transition_metals_only)
    for metal_ind in metal_idxs:
        deltametric_vector_derivative += func(mol, w, metal_ind, d, oct=oct)
    deltametric_vector_derivative = np.divide(deltametric_vector_derivative, n_met)
    return (deltametric_vector_derivative)


def generate_metal_autocorrelations(mol, depth=4, oct=True, flag_name=False,
                                    modifier=False, Gval=False, NumB=False, polarizability=False,
                                    use_dist=False, size_normalize=False, MRdiag_dict={},
                                    transition_metals_only=True):
    """
    Utility for generating all metal-centered product autocorrelations for a complex.

    Parameters
    ----------
        mol : mol3D
            Molecule to get mc-RACs for.
        depth : int, optional
            Depth of RACs to calculate, by default 4.
        oct : bool, optional
            Use octahedral criteria for structure evaluation, by default True.
            If complex is octahedral, will use better bond checks.
        flag_name : bool, optional
            Shift RAC names slightly, by default False.
        modifier : bool, optional
            Use ox_modifier for metal, by default False.
        Gval : bool, optional
            Use G value as descriptor property, by default False.
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False.
        polarizability : bool, optional
            Use polarizability (alpha) as descriptor property, by default False.
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False.
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.
        transition_metals_only : bool, optional
            Flag if only transition metals counted as metals, by default True.

    Returns
    -------
        results_dictionary: dict
            Dictionary of all geo-based MC-RAC product descriptors -
            {'colnames': colnames, 'results': result}.
            For key colnames, value is list of lists.
            For key results, value is list of np.array.

    """
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    if polarizability:
        allowed_strings += ['polarizability']
        labels_strings += ['alpha']
    if len(MRdiag_dict):
        allowed_strings, labels_strings = [], []
        for k in list(MRdiag_dict):
            allowed_strings += [k]
            labels_strings += [k]
    for ii, properties in enumerate(allowed_strings):
        metal_ac = metal_only_autocorrelation(mol, properties, depth, oct=oct,
                                              modifier=modifier, use_dist=use_dist,
                                              size_normalize=size_normalize,
                                              MRdiag_dict=MRdiag_dict,
                                              transition_metals_only=transition_metals_only)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_ac': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_autocorrelation_derivatives(mol, depth=4, oct=True, flag_name=False,
                                               modifier=False, NumB=False, Gval=False):
    """
    Utility for generating all metal-centered product autocorrelation derivatives for a complex.

    Parameters
    ----------
        mol : mol3D
            Molecule to get mc-RAC derivatives for.
        depth : int, optional
            Depth of RACs to calculate, by default 4.
        oct : bool, optional
            Use octahedral criteria for structure evaluation, by default True.
            If complex is octahedral, will use better bond checks.
        flag_name : bool, optional
            Shift RAC names slightly, by default False.
        modifier : bool, optional
            Use ox_modifier for metal, by default False.
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False.
        Gval : bool, optional
            Use G value as descriptor property, by default False.

    Returns
    -------
        results_dictionary : dict
            Dictionary of all geo-based MC-RAC product descriptor derivatives
            {'colnames': colnames, 'results': result}.

    """
    result = None
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        metal_ac_der = metal_only_autocorrelation_derivative(mol, properties, depth,
                                                             oct=oct, modifier=modifier)
        for i in range(0, depth + 1):
            colnames.append([f'd{labels_strings[ii]}-{i}/d{labels_strings[ii]}{j}' for j in
                             range(0, mol.natoms)])

        if result is None:
            result = metal_ac_der
        else:
            result = np.row_stack([result, metal_ac_der])
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_ac': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_deltametrics(mol, depth=4, oct=True, flag_name=False,
                                modifier=False, Gval=False, NumB=False, polarizability=False,
                                use_dist=False, size_normalize=False, MRdiag_dict={},
                                transition_metals_only=True):
    """
    Utility for generating all metal-centered deltametric autocorrelations for a complex.

    Parameters
    ----------
        mol : mol3D
            Molecule to get D_mc-RACs for.
        depth : int, optional
            Depth of RACs to calculate, by default 4.
        oct : bool, optional
            Use octahedral criteria for structure evaluation, by default True.
            If complex is octahedral, will use better bond checks.
        flag_name : bool, optional
            Shift RAC names slightly, by default False.
        modifier : bool, optional
            Use ox_modifier for metal, by default False.
        Gval : bool, optional
            Use G value as descriptor property, by default False.
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False.
        polarizability : bool, optional
            Use polarizability (alpha) as descriptor property, by default False.
        use_dist : bool, optional
            Weigh autocorrelation by physical distance of atom from original, by default False.
        size_normalize : bool, optional
            Whether or not to normalize by the number of atoms.
        MRdiag_dict : dict, optional
            Keys are ligand identifiers, values are MR diagnostics like E_corr.
        transition_metals_only : bool, optional
            Flag if only transition metals counted as metals, by default True.

    Returns
    -------
        results_dictionary: dict
            Dictionary of all geo-based MC-RAC deltametric descriptors -
            {'colnames': colnames, 'results': result}.
            For key colnames, value is list of lists.
            For key results, value is list of np.array.

    """
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    if polarizability:
        allowed_strings += ['polarizability']
        labels_strings += ['alpha']
    if len(MRdiag_dict):
        allowed_strings, labels_strings = [], []
        for k in list(MRdiag_dict):
            allowed_strings += [k]
            labels_strings += [k]
    for ii, properties in enumerate(allowed_strings):
        metal_ac = metal_only_deltametric(mol, properties, depth, oct=oct,
                                          modifier=modifier,
                                          use_dist=use_dist, size_normalize=size_normalize,
                                          MRdiag_dict=MRdiag_dict,
                                          transition_metals_only=transition_metals_only)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_del': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_deltametric_derivatives(mol, depth=4, oct=True, flag_name=False,
                                           modifier=False, NumB=False, Gval=False):
    """
    Utility for generating all metal-centered deltametric autocorrelation derivatives
    for a complex.

    Parameters
    ----------
        mol : mol3D
            Molecule to get D_mc-RAC derivatives for.
        depth : int, optional
            Depth of RACs to calculate, by default 4.
        oct : bool, optional
            Use octahedral criteria for structure evaluation, by default True.
            If complex is octahedral, will use better bond checks.
        flag_name : bool, optional
            Shift RAC names slightly, by default False.
        modifier : bool, optional
            Use ox_modifier for metal, by default False.
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False.
        Gval : bool, optional
            Use G value as descriptor property, by default False.

    Returns
    -------
        results_dictionary: dict
            Dictionary of all geo-based MC-RAC deltametric descriptor derivatives -
            {'colnames': colnames, 'results': result}.

    """
    result = None
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        metal_ac_der = metal_only_deltametric_derivative(mol, properties, depth, oct=oct,
                                                         modifier=modifier)
        for i in range(0, depth + 1):
            colnames.append([f'd{labels_strings[ii]}-{i}/d{labels_strings[ii]}{j}' for j in
                             range(0, mol.natoms)])
        if result is None:
            result = metal_ac_der
        else:
            result = np.row_stack([result, metal_ac_der])
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_del': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_atomonly_autocorrelations(mol, atomIdx, depth=4, oct=True, Gval=False, NumB=False, polarizability=False):
    """
    This function gets autocorrelations for a molecule starting in one single atom only.

    Parameters
    ----------
        mol : mol3D
            mol3D molecule to analyze.
        atomIdx : int or list of int
            Index or list of indices of atoms to start autocorrelation from.
        depth : int, optional
            Depth of autocorrelations.
        oct : bool, optional
            Use octahedral criteria for structure evaluation, by default True.
            If complex is octahedral, will use better bond checks.
        Gval : bool, optional
            Use G value as RAC, by default False.
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False.
        polarizability : bool, optional
            Use polarizability (alpha) as RAC, by default False.

    Returns
    -------
        results_dictionary : dict
            Dictionary of atom only RAC names and values.
            For key colnames, value is list of lists.
            For key results, value is list of np.array.

    """
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    if polarizability:
        allowed_strings += ['polarizability']
        labels_strings += ['alpha']
    for ii, properties in enumerate(allowed_strings):
        atom_only_ac = atom_only_autocorrelation(mol, properties, depth, atomIdx, oct=oct)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(atom_only_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_atomonly_autocorrelation_derivatives(mol, atomIdx, depth=4, oct=True, NumB=False, Gval=False):
    # # This function gets the d/dx for autocorrelations for a molecule starting
    # # in one single atom only.
    # Inputs:
    #       mol - mol3D class
    #       atomIdx - int, index of atom3D class
    result = None
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        atom_only_ac = atom_only_autocorrelation_derivative(mol, properties, depth, atomIdx, oct=oct)
        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])
        if result is None:
            result = atom_only_ac
        else:
            result = np.row_stack([result, atom_only_ac])
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_atomonly_deltametrics(mol, atomIdx, depth=4, oct=True, Gval=False, NumB=False, polarizability=False):
    """
    This function gets deltametrics for a molecule starting in one single atom only.

    Parameters
    ----------
        mol : mol3D
            mol3D molecule to analyze.
        atomIdx : int
            index of atom3D class.
        depth : int, optional
            Depth of deltametrics.
        oct : bool, optional
            Use octahedral criteria for structure evaluation, by default True.
            If complex is octahedral, will use better bond checks.
        Gval : bool, optional
            Use G value as RAC, by default False.
        NumB : bool, optional
            Use number of bonds as descriptor property, by default False.
        polarizability : bool, optional
            Use polarizability (alpha) as RAC, by default False.

    Returns
    -------
        results_dictionary : dict
            Dictionary of atom only deltametric names and values.
            For key colnames, value is list of lists.
            For key results, value is list of np.array.

    """
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    if polarizability:
        allowed_strings += ["polarizability"]
        labels_strings += ["alpha"]
    for ii, properties in enumerate(allowed_strings):
        atom_only_ac = atom_only_deltametric(mol, properties, depth, atomIdx, oct=oct)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(atom_only_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_atomonly_deltametric_derivatives(mol, atomIdx, depth=4, oct=True, NumB=False, Gval=False):
    # # This function gets deltametrics for a molecule starting
    # # in one single atom only.
    # Inputs:
    #       mol - mol3D class
    #       atomIdx - int, index of atom3D class
    result = None
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        atom_only_ac_der = atom_only_deltametric_derivative(mol, properties, depth, atomIdx, oct=oct)
        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])
        if result is None:
            result = atom_only_ac_der
        else:
            result = np.row_stack([result, atom_only_ac_der])
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def get_metal_index(mol, transition_metals_only=True):
    """
    Utility for getting metal index of molecule, and printing warning
    if more than one metal index found.
    Parameters
    ----------
        mol : mol3D
            Molecule to get metal index for.
        transition_metals_only : bool, optional
            Flag if only transition metals counted as metals, by default True.
    Returns
    -------
        f_metal_idx: int
            Index of the first metal atom.
    """
    metal_idx = mol.findMetal(transition_metals_only=transition_metals_only)
    if len(metal_idx) > 1:
        print('Warning: More than one metal in mol object. Choosing the first one.')
    f_metal_idx = metal_idx[0]
    return f_metal_idx


def ratiometric(mol, prop_vec_num, prop_vec_den, orig, d, oct=True):
    """This function returns the ratiometrics for one atom.

    Parameters
    ----------
        mol : mol3D class
        prop_vec : vector, property of atoms in mol in order of index
        orig : int, zero-indexed starting atom
        d : int, number of hops to travel
        oct : bool, if complex is octahedral, will use better bond checks

    Returns
    -------
        result_vector : vector of prop_vec_num / prop_vec_den

    """
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    result_vector[hopped] = prop_vec_num[orig] / prop_vec_den[orig]
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # Prepare all atoms attached to this connection.
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        for inds in new_active_set:
            result_vector[hopped] += prop_vec_num[orig] / prop_vec_den[inds]
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def summetric(mol, prop_vec, orig, d, oct=True):
    """This function returns the summetrics for one atom.

    Parameters
    ----------
        mol : mol3D class
        prop_vec : vector, property of atoms in mol in order of index
        orig : int, zero-indexed starting atom
        d : int, number of hops to travel
        oct : bool, if complex is octahedral, will use better bond checks

    Returns
    -------
        result_vector : vector of prop_vec_num / prop_vec_den

    """
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    result_vector[hopped] = prop_vec[orig] + prop_vec[orig]
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # Prepare all atoms attached to this connection.
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        for inds in new_active_set:
            result_vector[hopped] += prop_vec[orig] + prop_vec[inds]
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def autocorrelation_catoms(mol, prop_vec, orig, d, oct=True, catoms=None):
    # Calculate the autocorrelation for the orig to certain connecting atoms.
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    result_vector[hopped] = prop_vec[orig] * prop_vec[orig]
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # Prepare all atoms attached to this connection.
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            if this_atom == orig and (catoms is not None):
                this_atoms_neighbors = catoms
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        for inds in new_active_set:
            result_vector[hopped] += prop_vec[orig] * prop_vec[inds]
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def deltametric_catoms(mol, prop_vec, orig, d, oct=True, catoms=None):
    # Calculate the deltametrics for the orig to certain connecting atoms.
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # Prepare all atoms attached to this connection.
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            if this_atom == orig and (catoms is not None):
                this_atoms_neighbors = catoms
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        for inds in new_active_set:
            result_vector[hopped] += prop_vec[orig] - prop_vec[inds]
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def multiatom_only_autocorrelation(mol, prop, d, oct=True,
                                   func=autocorrelation, modifier=False,
                                   additional_elements=False):
    autocorrelation_vector = np.zeros(d + 1)
    metal_list = mol.findMetal()
    if additional_elements:
        for element in additional_elements:
            metal_list += mol.findAtomsbySymbol(element)
    n_met = len(metal_list)
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
    for metal_ind in metal_list:
        autocorrelation_vector += func(mol, w, metal_ind, d, oct=oct)
    autocorrelation_vector = np.divide(autocorrelation_vector, n_met)
    return (autocorrelation_vector)


def atom_only_ratiometric(mol, prop_num, prop_den, d, atomIdx, oct=True):
    # atomIdx must b either a list of indices
    # or a single index.
    w_num = construct_property_vector(mol, prop_num, oct)
    w_den = construct_property_vector(mol, prop_den, oct)
    autocorrelation_vector = np.zeros(d + 1)
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            autocorrelation_vector += ratiometric(mol, w_num, w_den, elements, d, oct=oct)
        autocorrelation_vector = np.divide(autocorrelation_vector, len(atomIdx))
    else:
        autocorrelation_vector += ratiometric(mol, w_num, w_den, atomIdx, d, oct=oct)
    return (autocorrelation_vector)


def atom_only_summetric(mol, prop, d, atomIdx, oct=True):
    # atomIdx must b either a list of indices
    # or a single index.
    w = construct_property_vector(mol, prop, oct)
    autocorrelation_vector = np.zeros(d + 1)
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            autocorrelation_vector += summetric(mol, w, elements, d, oct=oct)
        autocorrelation_vector = np.divide(autocorrelation_vector, len(atomIdx))
    else:
        autocorrelation_vector += summetric(mol, w, atomIdx, d, oct=oct)
    return (autocorrelation_vector)


def multiatom_only_deltametric(mol, prop, d, oct=True,
                               func=deltametric, modifier=False,
                               additional_elements=False):
    deltametric_vector = np.zeros(d + 1)
    metal_list = mol.findMetal()
    if additional_elements:
        for element in additional_elements:
            metal_list += mol.findAtomsbySymbol(element)
    n_met = len(metal_list)
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
    for metal_ind in mol.findMetal():
        deltametric_vector += func(mol, w, metal_ind, d, oct=oct)
    deltametric_vector = np.divide(deltametric_vector, n_met)
    return (deltametric_vector)


def metal_only_layer_density(mol, prop, d, oct=True):
    try:
        metal_ind = get_metal_index(mol)
        print(('metal_index is: %d' % metal_ind))
        w = construct_property_vector(mol, prop, oct=oct)
        density_vector = layer_density_in_3D(mol, w, metal_ind, d, oct=oct)
    except IndexError:
        print('Error, no metal found in mol object!')
        return False
    return density_vector


def layer_density_in_3D(mol, prop_vec, orig, d, oct=True):
    # # This function returns the density (prop^3/(d+1)^3)
    # # for one atom.
    # Inputs:
    #    mol - mol3D class
    #    prop_vec - vector, property of atoms in mol in order of index
    #    orig -  int, zero-indexed starting atom
    #    d - int, number of hops to travel
    #    oct - bool, if complex is octahedral, will use better bond checks
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    result_vector[hopped] = prop_vec[orig] ** 3 / (hopped + 1) ** 3
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # Prepare all atoms attached to this connection.
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        for inds in new_active_set:
            result_vector[hopped] += prop_vec[inds] ** 3 / (hopped + 1) ** 3
            historical_set.update(active_set)
        active_set = new_active_set
    return result_vector


def find_ligand_autocorrelations_oct(mol, prop, loud, depth, name=False,
                                     oct=True, custom_ligand_dict=False):
    # # This function takes a
    # # symmetric (axial == axial,
    # # equatorial == equatorial)
    # # octahedral complex
    # # and returns autocorrelations for
    # # the axial and equatorial ligands.
    # # custom_ligand_dict allows the user to skip the breakdown
    # # in cases where 3D geo is not correct/formed
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=oct)
        (ax_ligand_list, eq_ligand_list, ax_natoms_list,
         eq_natoms_list, ax_con_int_list, eq_con_int_list,
         ax_con_list, eq_con_list, built_ligand_list) = ligand_assign_original(
            mol, liglist, ligdents, ligcons, loud, name=False)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    # Count ligands.
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)
    # Get full ligand AC.
    ax_ligand_ac_full = []
    eq_ligand_ac_full = []
    for i in range(0, n_ax):
        if not list(ax_ligand_ac_full):
            ax_ligand_ac_full = full_autocorrelation(ax_ligand_list[i].mol, prop, depth)
        else:
            ax_ligand_ac_full += full_autocorrelation(ax_ligand_list[i].mol, prop, depth)
    ax_ligand_ac_full = np.divide(ax_ligand_ac_full, n_ax)
    for i in range(0, n_eq):
        if not list(eq_ligand_ac_full):
            eq_ligand_ac_full = full_autocorrelation(eq_ligand_list[i].mol, prop, depth)
        else:
            eq_ligand_ac_full += full_autocorrelation(eq_ligand_list[i].mol, prop, depth)
    eq_ligand_ac_full = np.divide(eq_ligand_ac_full, n_eq)

    # Get partial ligand AC.
    ax_ligand_ac_con = []
    eq_ligand_ac_con = []

    for i in range(0, n_ax):
        if not list(ax_ligand_ac_con):
            ax_ligand_ac_con = atom_only_autocorrelation(ax_ligand_list[i].mol, prop, depth, ax_con_int_list[i])
        else:
            ax_ligand_ac_con += atom_only_autocorrelation(ax_ligand_list[i].mol, prop, depth, ax_con_int_list[i])
    ax_ligand_ac_con = np.divide(ax_ligand_ac_con, n_ax)
    for i in range(0, n_eq):
        if not list(eq_ligand_ac_con):
            eq_ligand_ac_con = atom_only_autocorrelation(eq_ligand_list[i].mol, prop, depth, eq_con_int_list[i])
        else:
            eq_ligand_ac_con += atom_only_autocorrelation(eq_ligand_list[i].mol, prop, depth, eq_con_int_list[i])
    eq_ligand_ac_con = np.divide(eq_ligand_ac_con, n_eq)

    return ax_ligand_ac_full, eq_ligand_ac_full, ax_ligand_ac_con, eq_ligand_ac_con


def find_ligand_autocorrelation_derivatives_oct(mol, prop, loud, depth, name=False,
                                                oct=True, custom_ligand_dict=False):
    # # This function takes a
    # # symmetric (axial == axial,
    # # equatorial == equatorial)
    # # octahedral complex
    # # and returns autocorrelations for
    # # the axial and equatorial ligands.
    # # custom_ligand_dict allows the user to skip the breakdown
    # # in cases where 3D geo is not correct/formed
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=oct)
        (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list,
         eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list) = ligand_assign_original(
            mol, liglist, ligdents, ligcons, loud, name=False)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    # Count ligands.
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)
    # Get full ligand AC.
    ax_ligand_ac_full_derivative = None
    eq_ligand_eq_full_derivative = None

    # Allocate the full Jacobian matrix.
    ax_full_j = np.zeros([depth + 1, mol.natoms])
    eq_full_j = np.zeros([depth + 1, mol.natoms])
    ax_con_j = np.zeros([depth + 1, mol.natoms])
    eq_con_j = np.zeros([depth + 1, mol.natoms])

    # Full ligand ACs
    for i in range(0, n_ax):  # For each ax ligand
        ax_ligand_ac_full_derivative = full_autocorrelation_derivative(ax_ligand_list[i].mol, prop, depth)
        # Now we need to map back to full positions.
        for ii, row in enumerate(ax_ligand_ac_full_derivative):
            for original_ids in list(ax_ligand_list[i].ext_int_dict.keys()):
                ax_full_j[ii, original_ids] += np.divide(row[ax_ligand_list[i].ext_int_dict[original_ids]], n_ax)

    for i in range(0, n_eq):  # For each eq ligand
        # Now we need to map back to full positions.
        eq_ligand_eq_full_derivative = full_autocorrelation_derivative(eq_ligand_list[i].mol, prop, depth)
        for ii, row in enumerate(eq_ligand_eq_full_derivative):
            for original_ids in list(eq_ligand_list[i].ext_int_dict.keys()):
                eq_full_j[ii, original_ids] += np.divide(row[eq_ligand_list[i].ext_int_dict[original_ids]], n_eq)

    # Ligand connection ACs
    for i in range(0, n_ax):
        ax_ligand_ac_con_derivative = atom_only_autocorrelation_derivative(ax_ligand_list[i].mol, prop, depth,
                                                                           ax_con_int_list[i])
        # Now we need to map back to full positions.
        for ii, row in enumerate(ax_ligand_ac_con_derivative):
            for original_ids in list(ax_ligand_list[i].ext_int_dict.keys()):
                ax_con_j[ii, original_ids] += np.divide(row[ax_ligand_list[i].ext_int_dict[original_ids]], n_ax)

    for i in range(0, n_eq):
        eq_ligand_ac_con_derivative = atom_only_autocorrelation_derivative(eq_ligand_list[i].mol, prop, depth,
                                                                           eq_con_int_list[i])
        # Now we need to map back to full positions.
        for ii, row in enumerate(eq_ligand_ac_con_derivative):
            for original_ids in list(eq_ligand_list[i].ext_int_dict.keys()):
                eq_con_j[ii, original_ids] += np.divide(row[eq_ligand_list[i].ext_int_dict[original_ids]], n_eq)

    return ax_full_j, eq_full_j, ax_con_j, eq_con_j


def find_ligand_autocorrs_and_deltametrics_oct_dimers(mol, prop, depth, name=False,
                                                      oct=True, custom_ligand_dict=False):
    # # This function takes a
    # # symmetric (axial == axial,
    # # equatorial == equatorial)
    # # octahedral complex
    # # and returns autocorrelations for
    # # the axial and equatorial ligands.
    # # custom_ligand_dict allows the user to skip the breakdown
    # # in cases where 3D geo is not correct/formed
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    if not custom_ligand_dict:
        raise ValueError('No custom ligand dict provided!')
    else:
        ax1_ligand_list = custom_ligand_dict["ax1_ligand_list"]
        ax2_ligand_list = custom_ligand_dict["ax2_ligand_list"]
        ax3_ligand_list = custom_ligand_dict["ax3_ligand_list"]
        ax1_con_int_list = custom_ligand_dict["ax1_con_int_list"]
        ax2_con_int_list = custom_ligand_dict["ax2_con_int_list"]
        ax3_con_int_list = custom_ligand_dict["ax3_con_int_list"]
        axligs = [ax1_ligand_list, ax2_ligand_list, ax3_ligand_list]
        axcons = [ax1_con_int_list, ax2_con_int_list, ax3_con_int_list]
        n_axs = [len(i) for i in axligs]

    # Get full ligand AC.
    ax_ligand_ac_fulls = [False, False, False]

    for axnum in range(3):
        ax_ligand_ac_full = list()
        for i in range(0, n_axs[axnum]):
            if not list(ax_ligand_ac_full):
                ax_ligand_ac_full = full_autocorrelation(axligs[axnum][i].mol, prop, depth)
            else:
                ax_ligand_ac_full += full_autocorrelation(axligs[axnum][i].mol, prop, depth)
        ax_ligand_ac_full = np.divide(ax_ligand_ac_full, n_axs[axnum])
        ax_ligand_ac_fulls[axnum] = ax_ligand_ac_full

    # Get partial ligand AC.
    ax_ligand_ac_cons = [False, False, False]

    for axnum in range(3):
        ax_ligand_ac_con = list()
        for i in range(0, n_axs[axnum]):
            if not list(ax_ligand_ac_con):
                ax_ligand_ac_con = atom_only_autocorrelation(axligs[axnum][i].mol, prop, depth, axcons[axnum][i])
            else:
                ax_ligand_ac_con += atom_only_autocorrelation(axligs[axnum][i].mol, prop, depth, axcons[axnum][i])
        ax_ligand_ac_con = np.divide(ax_ligand_ac_con, n_axs[axnum])
        ax_ligand_ac_cons[axnum] = ax_ligand_ac_con

    # Get deltametrics.
    ax_delta_cons = [False, False, False]

    for axnum in range(3):
        ax_delta_con = list()
        for i in range(0, n_axs[axnum]):
            if not list(ax_delta_con):
                ax_delta_con = atom_only_deltametric(axligs[axnum][i].mol, prop, depth, axcons[axnum][i])
            else:
                ax_delta_con += atom_only_deltametric(axligs[axnum][i].mol, prop, depth, axcons[axnum][i])
        ax_delta_con = np.divide(ax_delta_con, n_axs[axnum])
        ax_delta_cons[axnum] = ax_delta_con

    return ax_ligand_ac_fulls + ax_ligand_ac_cons + ax_delta_cons


def find_ligand_deltametrics_oct(mol, prop, loud, depth, name=False, oct=True, custom_ligand_dict=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    # # This function takes a
    # # octahedral complex
    # # and returns deltametrics for
    # # the axial and equatorial ligands.
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=oct)
        (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list,
         eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list) = ligand_assign_original(
            mol, liglist, ligdents, ligcons, loud, name=False)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    # Count ligands.
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)

    # Get partial ligand AC.
    ax_ligand_ac_con = []
    eq_ligand_ac_con = []

    for i in range(0, n_ax):
        if not list(ax_ligand_ac_con):
            ax_ligand_ac_con = atom_only_deltametric(ax_ligand_list[i].mol, prop, depth, ax_con_int_list[i])
        else:
            ax_ligand_ac_con += atom_only_deltametric(ax_ligand_list[i].mol, prop, depth, ax_con_int_list[i])
    ax_ligand_ac_con = np.divide(ax_ligand_ac_con, n_ax)
    for i in range(0, n_eq):
        if not list(eq_ligand_ac_con):
            eq_ligand_ac_con = atom_only_deltametric(eq_ligand_list[i].mol, prop, depth, eq_con_int_list[i])
        else:
            eq_ligand_ac_con += atom_only_deltametric(eq_ligand_list[i].mol, prop, depth, eq_con_int_list[i])
    eq_ligand_ac_con = np.divide(eq_ligand_ac_con, n_eq)

    return ax_ligand_ac_con, eq_ligand_ac_con


def find_ligand_deltametric_derivatives_oct(mol, prop, loud, depth, name=False, oct=True, custom_ligand_dict=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    # # This function takes a
    # # octahedral complex
    # # and returns deltametrics for
    # # the axial and equatorial ligands.
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=oct)
        (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list,
         eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list) = ligand_assign_original(
            mol, liglist, ligdents, ligcons, loud, name=False)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]

    # Count ligands.
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)

    # Allocate the full Jacobian matrix.
    ax_con_j = np.zeros([depth + 1, mol.natoms])
    eq_con_j = np.zeros([depth + 1, mol.natoms])

    for i in range(0, n_ax):
        ax_ligand_ac_con_derivative = atom_only_deltametric_derivative(ax_ligand_list[i].mol, prop, depth,
                                                                       ax_con_int_list[i])
        # Now we need to map back to full positions.
        for ii, row in enumerate(ax_ligand_ac_con_derivative):
            for original_ids in list(ax_ligand_list[i].ext_int_dict.keys()):
                ax_con_j[ii, original_ids] += np.divide(row[ax_ligand_list[i].ext_int_dict[original_ids]], n_ax)

    for i in range(0, n_eq):
        eq_ligand_ac_con_derivative = atom_only_deltametric_derivative(eq_ligand_list[i].mol, prop, depth,
                                                                       eq_con_int_list[i])
        for ii, row in enumerate(eq_ligand_ac_con_derivative):
            for original_ids in list(eq_ligand_list[i].ext_int_dict.keys()):
                eq_con_j[ii, original_ids] += np.divide(row[eq_ligand_list[i].ext_int_dict[original_ids]], n_eq)

    return ax_con_j, eq_con_j


def generate_all_ligand_autocorrelations(mol, loud, depth=4, name=False, flag_name=False,
                                         custom_ligand_dict=False, NumB=False, Gval=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    result_ax_full = list()
    result_eq_full = list()
    result_ax_con = list()
    result_eq_con = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        (ax_ligand_ac_full,
         eq_ligand_ac_full,
         ax_ligand_ac_con,
         eq_ligand_ac_con) = find_ligand_autocorrelations_oct(
             mol,
             properties,
             loud=loud,
             depth=depth,
             name=name,
             oct=True,
             custom_ligand_dict=custom_ligand_dict)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result_ax_full.append(ax_ligand_ac_full)
        result_eq_full.append(eq_ligand_ac_full)
        result_ax_con.append(ax_ligand_ac_con)
        result_eq_con.append(eq_ligand_ac_con)
    if flag_name:
        results_dictionary = {'colnames': colnames,
                              'result_ax_full_ac': result_ax_full,
                              'result_eq_full_ac': result_eq_full,
                              'result_ax_con_ac': result_ax_con,
                              'result_eq_con_ac': result_eq_con}
    else:
        results_dictionary = {'colnames': colnames,
                              'result_ax_full': result_ax_full,
                              'result_eq_full': result_eq_full,
                              'result_ax_con': result_ax_con,
                              'result_eq_con': result_eq_con}
    return results_dictionary


def generate_all_ligand_autocorrelation_derivatives(mol, loud, depth=4, name=False, flag_name=False,
                                                    custom_ligand_dict=False, NumB=False, Gval=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    result_ax_full = None
    result_eq_full = None
    result_ax_con = None
    result_eq_con = None

    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        ax_ligand_ac_full, eq_ligand_ac_full, ax_ligand_ac_con, eq_ligand_ac_con = find_ligand_autocorrelation_derivatives_oct(
            mol,
            properties,
            loud=loud,
            depth=depth,
            name=name,
            oct=True,
            custom_ligand_dict=custom_ligand_dict)
        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])
        if result_ax_full is None:
            result_ax_full = ax_ligand_ac_full
        else:
            result_ax_full = np.row_stack([result_ax_full, ax_ligand_ac_full])

        if result_eq_full is None:
            result_eq_full = eq_ligand_ac_full
        else:
            result_eq_full = np.row_stack([result_eq_full, eq_ligand_ac_full])

        if result_ax_con is None:
            result_ax_con = ax_ligand_ac_con
        else:
            result_ax_con = np.row_stack([result_ax_con, ax_ligand_ac_con])

        if result_eq_con is None:
            result_eq_con = eq_ligand_ac_con
        else:
            result_eq_con = np.row_stack([result_eq_con, eq_ligand_ac_con])

    if flag_name:
        results_dictionary = {'colnames': colnames, 'result_ax_full_ac': result_ax_full,
                              'result_eq_full_ac': result_eq_full,
                              'result_ax_con_ac': result_ax_con, 'result_eq_con_ac': result_eq_con}
    else:
        results_dictionary = {'colnames': colnames, 'result_ax_full': result_ax_full, 'result_eq_full': result_eq_full,
                              'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}
    return results_dictionary


def generate_all_ligand_autocorrs_and_deltametrics_dimers(mol, loud, depth=4, name=False, flag_name=False,
                                                          custom_ligand_dict=False, NumB=False, Gval=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    result_ax1_full = list()
    result_ax2_full = list()
    result_ax3_full = list()
    result_ax1_con = list()
    result_ax2_con = list()
    result_ax3_con = list()
    result_delta_ax1_con = list()
    result_delta_ax2_con = list()
    result_delta_ax3_con = list()

    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        # lig_autocorrs is a list of length 6 (ax{i}_ligand_ac_fulls, ax{i}_ligand_ac_cons).
        lig_autocorrs = find_ligand_autocorrs_and_deltametrics_oct_dimers(mol,
                                                                          properties,
                                                                          depth=depth,
                                                                          name=name,
                                                                          oct=True,
                                                                          custom_ligand_dict=custom_ligand_dict)
        this_colnames = []
        assert all([len(i) > 0 for i in lig_autocorrs]), 'Some ligand autocorrelations are empty! %s' % lig_autocorrs
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result_ax1_full.append(lig_autocorrs[0])
        result_ax2_full.append(lig_autocorrs[1])
        result_ax3_full.append(lig_autocorrs[2])
        result_ax1_con.append(lig_autocorrs[3])
        result_ax2_con.append(lig_autocorrs[4])
        result_ax3_con.append(lig_autocorrs[5])
        result_delta_ax1_con.append(lig_autocorrs[6])
        result_delta_ax2_con.append(lig_autocorrs[7])
        result_delta_ax3_con.append(lig_autocorrs[8])

    results_dictionary = {'colnames': colnames,
                          'result_ax1_full': result_ax1_full,
                          'result_ax2_full': result_ax2_full,
                          'result_ax3_full': result_ax3_full,
                          'result_ax1_con': result_ax1_con,
                          'result_ax2_con': result_ax2_con,
                          'result_ax3_con': result_ax3_con,
                          'result_delta_ax1_con': result_delta_ax1_con,
                          'result_delta_ax2_con': result_delta_ax2_con,
                          'result_delta_ax3_con': result_delta_ax3_con}
    return results_dictionary


def generate_all_ligand_deltametrics(mol, loud, depth=4, name=False, flag_name=False,
                                     custom_ligand_dict=False, NumB=False, Gval=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]

    result_ax_con = list()
    result_eq_con = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        ax_ligand_ac_con, eq_ligand_ac_con = find_ligand_deltametrics_oct(mol, properties, loud, depth, name, oct=True,
                                                                          custom_ligand_dict=custom_ligand_dict)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result_ax_con.append(ax_ligand_ac_con)
        result_eq_con.append(eq_ligand_ac_con)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'result_ax_con_del': result_ax_con,
                              'result_eq_con_del': result_eq_con}
    else:
        results_dictionary = {'colnames': colnames, 'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}
    return results_dictionary


def generate_all_ligand_deltametric_derivatives(mol, loud, depth=4, name=False, flag_name=False,
                                                custom_ligand_dict=False, NumB=False, Gval=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]

    result_ax_con = None
    result_eq_con = None
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        ax_ligand_ac_con, eq_ligand_ac_con = find_ligand_deltametric_derivatives_oct(mol, properties, loud, depth, name,
                                                                                     oct=True,
                                                                                     custom_ligand_dict=custom_ligand_dict)

        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])
        if result_ax_con is None:
            result_ax_con = ax_ligand_ac_con
        else:
            result_ax_con = np.row_stack([result_ax_con, ax_ligand_ac_con])
        if result_eq_con is None:
            result_eq_con = eq_ligand_ac_con
        else:
            result_eq_con = np.row_stack([result_eq_con, eq_ligand_ac_con])
    if flag_name:
        results_dictionary = {'colnames': colnames, 'result_ax_con_del': result_ax_con,
                              'result_eq_con_del': result_eq_con}
    else:
        results_dictionary = {'colnames': colnames, 'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}
    return results_dictionary


def generate_multiatom_autocorrelations(mol, depth=4, oct=True, flag_name=False, additional_elements=False):
    # oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    for ii, properties in enumerate(allowed_strings):
        metal_ac = multiatom_only_autocorrelation(mol, properties, depth, oct=oct, additional_elements=additional_elements)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_ac': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_eff_autocorrelations(oxmodifier, mol, depth=4, oct=True, flag_name=False, transition_metals_only=True):
    # # oxmodifier - dict, used to modify prop vector (e.g. for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    metal_ox_ac = metal_only_autocorrelation(mol, 'group_number', depth, oct=oct, modifier=oxmodifier, transition_metals_only=transition_metals_only)
    this_colnames = []
    for i in range(0, depth + 1):
        this_colnames.append('Gval' + '-' + str(i))
    colnames.append(this_colnames)
    result.append(metal_ox_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_eff_deltametrics(oxmodifier, mol, depth=4, oct=True, flag_name=False, transition_metals_only=True):
    # # oxmodifier - dict, used to modify prop vector (e.g. for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    metal_ox_ac = metal_only_deltametric(mol, 'group_number', depth, oct=oct, modifier=oxmodifier, transition_metals_only=transition_metals_only)
    this_colnames = []
    for i in range(0, depth + 1):
        this_colnames.append('Gval' + '-' + str(i))
    colnames.append(this_colnames)
    result.append(metal_ox_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_multiatom_deltametrics(mol, depth=4, oct=True, flag_name=False, additional_elements=False):
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    for ii, properties in enumerate(allowed_strings):
        metal_ac = multiatom_only_deltametric(mol, properties, depth, oct=oct, additional_elements=additional_elements)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_del': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_full_complex_coulomb_autocorrelations(mol,
                                                   depth=3, oct=True,
                                                   flag_name=False, modifier=False,
                                                   use_dist=False, transition_metals_only=True):
    result = list()
    colnames = []
    allowed_strings = ['ident', 'topology', 'group_number', "num_bonds"]
    labels_strings = ['I', 'T', 'Gval', "NumB"]
    for ii, properties in enumerate(allowed_strings):
        metal_ac = full_autocorrelation(mol, properties, depth,
                                        oct=oct, modifier=modifier,
                                        use_dist=use_dist,
                                        transition_metals_only=transition_metals_only)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_f_all': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


#### Possibly Needed - ox_ utilities ###
def generate_metal_ox_autocorrelations(oxmodifier, mol, depth=4,
                                       oct=True, flag_name=False,
                                       use_dist=False, size_normalize=False):
    # # oxmodifier - dict, used to modify prop vector (e.g., for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #    oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    metal_ox_ac = metal_only_autocorrelation(mol, 'ox_nuclear_charge', depth, oct=oct,
                                             modifier=oxmodifier,
                                             use_dist=use_dist, size_normalize=size_normalize)
    this_colnames = []
    for i in range(0, depth + 1):
        this_colnames.append('O' + '-' + str(i))
    colnames.append(this_colnames)
    result.append(metal_ox_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_autocorrelation_derivatives(oxmodifier, mol, depth=4, oct=True, flag_name=False):
    # # oxmodifier - dict, used to modify prop vector (e.g., for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #    oct - bool, if complex is octahedral, will use better bond checks
    result = None
    colnames = []
    metal_ox_ac = metal_only_autocorrelation_derivative(mol, 'ox_nuclear_charge', depth, oct=oct,
                                                        modifier=oxmodifier)
    for i in range(0, depth + 1):
        colnames.append(['d' + 'O' + '-' + str(i) + '/d' + 'O' + str(j) for j in range(0, mol.natoms)])
    result = metal_ox_ac
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_deltametrics(oxmodifier, mol, depth=4, oct=True,
                                   flag_name=False, use_dist=False, size_normalize=False):
    # # oxmodifier - dict, used to modify prop vector (e.g., for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #    oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    metal_ox_ac = metal_only_deltametric(mol, 'ox_nuclear_charge', depth, oct=oct,
                                         modifier=oxmodifier, use_dist=use_dist,
                                         size_normalize=size_normalize)
    this_colnames = []
    for i in range(0, depth + 1):
        this_colnames.append('O' + '-' + str(i))
    colnames.append(this_colnames)
    result.append(metal_ox_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_deltametric_derivatives(oxmodifier, mol, depth=4, oct=True,
                                              flag_name=False):
    # # oxmodifier - dict, used to modify prop vector (e.g., for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #    oct - bool, if complex is octahedral, will use better bond checks
    colnames = []
    metal_ox_ac = metal_only_deltametric_derivative(mol, 'ox_nuclear_charge',
                                                    depth, oct=oct, modifier=oxmodifier)
    for i in range(0, depth + 1):
        colnames.append(['d' + 'O' + '-' + str(i) + '/d' + 'O' + str(j) for j in range(0, mol.natoms)])

    result = metal_ox_ac
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary
### End of ox utilities ###
