#  @file helpers.py
#  Contains some useful helper functions for classes.
#
#  Written by HJK Group
#
#  Dpt of Chemical Engineering, MIT

from molSimplify.Classes.monomer3D import monomer3D
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.atom3D import atom3D


def read_atom(line):
    """ Reads a line of a pdb into an atom dictionary.

    Parameters
    ----------
        line : str
            a line of a pdb file

    Returns
    -------
        atom_dict : dictionary
            dictionary containing attributes of atom when reading the pdb
    """

    labels = ['Type', 'SerialNum', 'Name', 'AltLoc', 'ResName', 'ChainID',
              'ResSeq', 'ICode', 'X', 'Y', 'Z', 'Occupancy', 'TempFactor',
              'Element', 'Charge']
    data = [line[0:6].strip(), int(line[6:11]), line[12:16].strip(),
            line[16:17].strip(), line[17:20].strip(), line[21:22].strip(),
            int(line[22:26]), line[26:27].strip(), float(line[30:38]),
            float(line[38:46]), float(line[46:54]), float(line[54:60]),
            float(line[60:66]), line[70:78].strip(), line[78:80].strip()]
    atom_dict = dict(zip(labels, data))
    atom_dict['Element'] = atom_dict['Element'][0] + atom_dict['Element'][1:].lower()
    atom_dict["Line"] = line
    return atom_dict


def makeMol(a_dict, mols, conf, chains, prev_a_dict, bonds, aa=True):
    """ Creates an monomer3D from a_dicts and adds it to the appropriate places.

    Parameters
    ----------
        a_dict : dictionary
            created from the read_atom method
        mols : dictionary
            keyed by chain and location, valued by monomer3Ds or mol3Ds
        conf : list
            contains amino acid conformations
        chains : dictionary
            contains amino acid chains
        prev_a_dict : dictionary
            the a_dict from the previous run
        bonds : dictionary
            contains bonding info
        aa : boolean
            True if mols consists of monomer3Ds, False if consists of mol3Ds

    Returns
    -------
        mols, conf, chains, prev_a_dict as updated
    """
    loc = a_dict['AltLoc']
    ploc = prev_a_dict["AltLoc"]
    hack = False
    hack2 = False
    m = 0
    key = (a_dict['ChainID'], a_dict['ResSeq'])
    if key not in mols.keys():
        if aa:
            m = monomer3D(a_dict['ResName'], a_dict['ChainID'], a_dict['ResSeq'],
                     a_dict['Occupancy'], loc)
        else:
            m = mol3D(a_dict['ResName'], loc)
        mols[key] = [m]
    elif loc != '' and ploc != '':
        li = ord(ploc)
        if loc > chr(li) and len(mols[key]) <= li-64:
            if aa:
                m = monomer3D(a_dict['ResName'], a_dict['ChainID'],
                         a_dict['ResSeq'], a_dict['Occupancy'], loc)
                m.temp_list = mols[key][-1].temp_list
            else:
                m = mol3D(a_dict['ResName'], loc)
                m.temp_list = mols[key][-1].temp_list
            if mols[key][-1].loc != loc:
                mols[key].append(m)
        else:
            weirdo = True
            for mol in mols[key]:
                if mol.loc == loc:
                    weirdo = False
            if weirdo:
                if aa:
                    m = monomer3D(a_dict['ResName'], a_dict['ChainID'],
                             a_dict['ResSeq'], a_dict['Occupancy'], loc)
                    m.temp_list = mols[key][-1].temp_list
                else:
                    m = mol3D(a_dict['ResName'], loc)
                    m.temp_list = mols[key][-1].temp_list
                if mols[key][-1].loc != loc:
                    mols[key].append(m)
    elif loc == '':
        # atom must be added to multiple conformations
        hack = True
    elif key in mols.keys() and ploc == '' and len(mols[key]) == 1:
        hack2 = True
        mols[key][0].setLoc(loc)
        mols[key][0].temp_list = mols[key][0].atoms.copy()
    prev_a_dict = a_dict
    if a_dict['ChainID'] not in chains.keys():
        chains[a_dict['ChainID']] = []  # initialize key of chain dictionary
    if m != 0 and key not in conf and loc != '' and float(a_dict['Occupancy']) < 1:
        conf.append(key)
    if m != 0 and m not in chains[a_dict['ChainID']] and key not in conf:
        chains[a_dict['ChainID']].append(m)
    if loc == '' or ploc == '':
        m = mols[key][0]
    else:
        for i in mols[key]:
            if i.loc == loc:
                m = i
    atom = atom3D(Sym=a_dict['Element'], xyz=[a_dict['X'], a_dict['Y'],
                                              a_dict['Z']],
                  Tfactor=a_dict['TempFactor'], occup=a_dict['Occupancy'],
                  greek=a_dict['Name'], line=a_dict['Line'])
    if not hack2:
        for a in m.temp_list:
            if type(a) == tuple:
                m.addAtom(a[1], a[0])
            else:
                m.addAtom(a)
        m.temp_list = []
    if hack:
        for i in mols[key]:
            i.addAtom(atom, a_dict['SerialNum'])
    else:
        m.addAtom(atom, a_dict['SerialNum'])  # terminal Os may be missing
        if aa:
            if atom.greek == "C":
                m.c.append(atom)
            if atom.greek == "N":
                m.n.append(atom)
    if key in conf and chains[a_dict['ChainID']] != []:
        chains[a_dict['ChainID']] = []
    if aa:
        m.setBonds()
        bonds.update(m.bonds)
        if m.prev is None and (a_dict['ChainID'], a_dict['ResSeq'] - 1) in mols.keys():
            m.setPrev(mols[(a_dict['ChainID'], a_dict['ResSeq'] - 1)][0])
        prev_mol = m.prev
        if prev_mol is not None:
            if prev_mol.next is None:
                prev_mol.setNext(m)
            for n in m.n:
                for c in prev_mol.c:
                    bonds[n].add(c)
                    bonds[c].add(n)
    return atom, mols, conf, chains, prev_a_dict, bonds
