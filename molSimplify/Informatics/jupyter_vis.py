"""
Py3Dmol install: (works in both Python2/3 conda environments)
conda install -c rmg py3dmol
Some Documentation: https://pypi.org/project/py3Dmol/
3DMol.js backend: http://3dmol.csb.pitt.edu/index.html
"""
import math as m
import numpy as np

import py3Dmol

from molSimplify.Classes.mol3D import mol3D


def type_convert(structures):
    """Handle multiple types of structures passed. List of xyz, mol2 files,
    or list of xyz, mol2strings.

    Parameters
    ----------
    structures : list
        Structures you want visualized: can either be a list or individual:
        mol2 strings, mol2 files, xyz strings, xyz files, or mol3D objects
    """
    outlist = []
    if not isinstance(structures, str):
        # Convert other array-like arguments to a list.
        structures = list(structures)
    if isinstance(structures, list):
        for i, x in enumerate(structures):
            if isinstance(x, mol3D):
                outlist.append(x)
                continue
            mol = mol3D()
            # mol2string
            if 'TRIPOS' in x:
                mol.readfrommol2(x, readstring=True)
            # Xyz filename
            elif x[-4:] == '.xyz':
                mol.readfromxyz(x)
            # mol2 filename
            elif x[-5:] == '.mol2':
                mol.readfrommol2(x)
            # checking for number at start of string -> indicates xyz string
            elif (len(x.split('\n')) > 3) & (x.split('\n')[0].replace(' ', '').isnumeric()):
                mol.readfromstring(x)
            # checking for similar file without header
            elif (len(x.split('\n')[0].split()) == 4) and x.split('\n')[0].split()[0]:
                mol.readfromstring(x)
            else:
                raise ValueError('Not Recognized Structure Type for index: ' + str(i))
            outlist.append(mol)
    elif isinstance(structures, str):
        x = structures
        mol = mol3D()
        # mol2string
        if 'TRIPOS' in x:
            mol.readfrommol2(x, readstring=True)
        # Xyz filename
        elif x[-4:] == '.xyz':
            mol.readfromxyz(x)
        # mol2 filename
        elif x[-5:] == '.mol2':
            mol.readfrommol2(x)
        # checking for number at start of string -> indicates xyz string
        elif (len(x.split('\n')) > 3) & (x.split('\n')[0].replace(' ', '').isnumeric()):
            mol.readfromstring(x)
        # checking for similar file without header
        elif (len(x.split('\n')[0].split()) == 4) and x.split('\n')[0].split()[0]:
            mol.readfromstring(x)
        else:
            raise ValueError('Not Recognized Structure Type Passed')
        outlist.append(mol)
    elif isinstance(structures, mol3D):
        outlist = structures
    else:
        raise ValueError('Not Recognized Structure Type Passed')
    return outlist


def view_structures(structures, w=400, h=400, columns=2,
                    representation='stick', labelsize=18,
                    labels=False, readstring=True):
    """
    py3Dmol view atoms object(s)
    xyz_names = xyz files that will be rendered in a tiled format in jupyter (list,str)
    w = width of frame (or subframes) in pixels (int)
    h = height of frame (or subframes) in pixels (int)
    cols = number of columns in subframe (int)
    representation = how the molecule will be viewed (str)
    labelsize = size of the data label (in Points) (int)
    labels = turn labels on/off (bool)
    """
    mol3Ds = type_convert(structures)
    if len(mol3Ds) == 1:
        view_ats = py3Dmol.view(width=w, height=h)
        mol = mol3Ds[0]
        label = False
        if isinstance(labels, str):
            label = labels
        elif isinstance(labels, list):
            label = labels[0]
        elif isinstance(labels, bool):
            if labels:
                label = mol.make_formula(latex=False)

        metal_atom_index = mol.findMetal()  # will be empty list if no metals
        coords = mol.coords()
        if metal_atom_index:  # Take advantage of empty list
            label_posits = mol.getAtomCoords(metal_atom_index[0])
        else:
            label_posits = mol.centersym()  # Put it at the geometric center of the molecule.
        view_ats.addModel(coords, 'xyz')  # Add the molecule
        view_ats.setStyle({representation: {'colorscheme': 'Jmol'}})
        if label:
            view_ats.addLabel(
                "{}".format(label),
                {'position': {'x': '{}'.format(label_posits[0]),
                 'y': '{}'.format(label_posits[1]), 'z': '{}'.format(label_posits[2])},
                 'backgroundColor': "'black'", 'backgroundOpacity': '0.3',
                 'fontOpacity': '1', 'fontSize': '{}'.format(labelsize),
                 'fontColor': "white", 'inFront': 'true'})
        view_ats.zoomTo()
        view_ats.show()
    elif len(mol3Ds) < 50:
        rows = int(m.ceil(float(len(mol3Ds))/columns))
        w = w*columns
        h = h*rows
        # Initialize Layout
        view_ats = py3Dmol.view(width=w, height=h, linked=False, viewergrid=(rows, columns))
        # Check for labels and populate
        if isinstance(labels, bool):
            if labels:
                label = [x.make_formula(latex=False) for x in mol3Ds]
            else:
                label = []
        elif isinstance(labels, list) or isinstance(labels, np.ndarray):
            if (len(labels) != len(mol3Ds)):
                print('Wrong amount of labels passed, defaulting to chemical formulas.')
                label = [x.make_formula(latex=False) for x in mol3Ds]
            else:  # Force them all to be strings.
                label = [str(x) for x in labels]
        else:
            raise ValueError('What sort of labels are wanting? Not recognized.')
        x, y = 0, 0  # Subframe position
        for i, item in enumerate(mol3Ds):
            mol = item
            coords = mol.coords()
            metal_atom_index = mol.findMetal()
            if metal_atom_index:
                label_posits = mol.getAtomCoords(metal_atom_index[0])
            else:
                label_posits = mol.centersym()
            view_ats.addModel(coords, 'xyz', viewer=(x, y))
            view_ats.setStyle({representation: {'colorscheme': 'Jmol'}}, viewer=(x, y))
            if len(label) > 0:
                view_ats.addLabel(
                    "{}".format(label[i]),
                    {'position': {'x': '{}'.format(label_posits[0]),
                     'y': '{}'.format(label_posits[1]), 'z': '{}'.format(label_posits[2])},
                     'backgroundColor': "'black'", 'backgroundOpacity': '0.5',
                     'fontOpacity': '1', 'fontSize': '{}'.format(labelsize),
                     'fontColor': "white", 'inFront': 'true'}, viewer=(x, y))
            view_ats.zoomTo(viewer=(x, y))
            if y+1 < columns:  # Fill in columns
                y += 1
            else:
                x += 1
                y = 0
        view_ats.show()
    else:
        raise ValueError('Warning. Passing this many structures WILL cause your kernel to crash.')
