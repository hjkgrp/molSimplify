# @file globalvars.py
# @file globalvars.py
#  Contains useful constants used throughout the code.
#
#  Written by Kulik Group
#
#  Department of Chemical Engineering, MIT

import os
import inspect
import glob
import platform
import sys
import subprocess
from molSimplify.utils.metaclasses import Singleton

# Dictionary containing atomic mass, atomic number, covalent radius, num valence electrons
# Data from http://www.webelements.com/ (last accessed May 13th 2015)
amassdict = {'X': (1.0, 0, 0.77, 0), 'H': (1.0079, 1, 0.37, 1),
             'D': (2.0141, 1, 0.37, 1), 'He': (4.002602, 2, 0.46, 2),
             'Li': (6.94, 3, 1.33, 1), 'Be': (9.0121831, 4, 1.02, 2), 'B': (10.83, 5, 0.85, 3),
             'C': (12.0107, 6, 0.77, 4), 'N': (14.0067, 7, 0.75, 5), 'O': (15.9994, 8, 0.73, 6),
             'F': (18.9984, 9, 0.71, 7), 'Ne': (20.1797, 10, 0.67, 8), 'Na': (22.99, 11, 1.55, 1),
             'Mg': (24.30, 12, 1.39, 2), 'Al': (26.98, 13, 1.26, 3), 'Si': (28.08, 14, 1.16, 4),
             'P': (30.9738, 15, 1.06, 5), 'S': (32.065, 16, 1.02, 6), 'Cl': (35.453, 17, 0.99, 7),
             'Ar': (39.948, 18, 0.96, 8), 'K': (39.10, 19, 1.96, 1), 'Ca': (40.08, 20, 1.71, 2),
             'Sc': (44.96, 21, 1.7, 3), 'Ti': (47.867, 22, 1.36, 4), 'V': (50.94, 23, 1.22, 5),
             'Cr': (51.9961, 24, 1.27, 6), 'Mn': (54.938, 25, 1.39, 7), 'Fe': (55.84526, 26, 1.25, 8),
             'Co': (58.9332, 27, 1.26, 9), 'Ni': (58.4934, 28, 1.21, 10), 'Cu': (63.546, 29, 1.38, 11),
             'Zn': (65.39, 30, 1.31, 12), 'Ga': (69.72, 31, 1.24, 3), 'Ge': (72.63, 32, 1.21, 4),
             'As': (74.92, 33, 1.21, 5), 'Se': (78.96, 34, 1.16, 6), 'Br': (79.904, 35, 1.14, 7),
             'Kr': (83.798, 36, 1.17, 8), 'Rb': (85.47, 37, 2.10, 1), 'Sr': (87.62, 38, 1.85, 2),
             'Y': (88.91, 39, 1.63, 3), 'Zr': (91.22, 40, 1.54, 4), 'Nb': (92.91, 41, 1.47, 5),
             'Mo': (95.96, 42, 1.38, 6), 'Tc': (98.9, 43, 1.56, 7), 'Ru': (101.1, 44, 1.25, 8),
             'Rh': (102.9, 45, 1.25, 9), 'Pd': (106.4, 46, 1.20, 10), 'Ag': (107.9, 47, 1.28, 11),
             'Cd': (112.4, 48, 1.48, 12), 'In': (111.818, 49, 1.42, 3), 'Sn': (118.710, 50, 1.40, 4),
             'Sb': (121.760, 51, 1.40, 5), 'Te': (127.60, 52, 1.99, 6), 'I': (126.90447, 53, 1.40, 7),
             'Xe': (131.293, 54, 1.31, 8), 'Cs': (132.9055, 55, 2.32, 1), 'Ba': (137.327, 56, 1.96, 2),
             'La': (138.9, 57, 1.69, 3), 'Ce': (140.116, 58, 1.63, 4), 'Pr': (140.90766, 59, 1.76, 5),
             'Nd': (144.242, 60, 1.74, 6), 'Pm': (145, 61, 1.73, 7), 'Sm': (150.36, 62, 1.72, 8),
             'Eu': (151.964, 63, 1.68, 9), 'Gd': (157.25, 64, 1.69, 10), 'Tb': (158.92535, 65, 1.68, 11),
             'Dy': (162.500, 66, 1.67, 12), 'Ho': (164.93033, 67, 1.66, 13), 'Er': (167.259, 68, 1.65, 14),
             'Tm': (168.93422, 69, 1.64, 15), 'Yb': (173.045, 70, 1.70, 16), 'Lu': (174.9668, 71, 1.62, 3),
             'Hf': (178.5, 72, 1.50, 8), 'Ta': (180.9, 73, 1.38, 5), 'W': (183.8, 74, 1.46, 6),
             'Re': (186.2, 75, 1.59, 7), 'Os': (190.2, 76, 1.28, 8), 'Ir': (192.2, 77, 1.37, 9),
             'Pt': (195.1, 78, 1.23, 10), 'Au': (197.0, 79, 1.24, 11), 'Hg': (200.6, 80, 1.49, 2),
             'Tl': (204.38, 81, 1.44, 3), 'Pb': (207.2, 82, 1.44, 4), 'Bi': (208.9804, 83, 1.51, 5),
             'Po': (208.98, 84, 1.90, 6), 'At': (209.99, 85, 2.00, 7), 'Rn': (222.6, 86, 142, 4),
             'Fr': (223.02, 87, 3.48, 8), 'Ra': (226.03, 88, 2.01, 2), 'Ac': (277, 89, 1.86, 3),
             'Th': (232.0377, 90, 1.75, 4), 'Pa': (231.04, 91, 2.00, 5), 'U': (238.02891, 92, 1.70, 6),
             'Np': (237.05, 93, 1.90, 7), 'Pu': (244.06, 94, 1.75, 8), 'Am': (243.06, 95, 1.80, 9),
             'Cm': (247.07, 96, 1.69, 10), 'Bk': (247.07, 97, 1.68, 11), 'Cf': (251.08, 98, 1.68, 12)}

# Pa and onward should be checked


# van der Waals radii for elements
# Data from DOI: 10.1039/C3DT50599E, Dalton Trans., 2013, 42, 8617-8636
vdwrad = {'H': 1.2, 'He': 1.43, 'Li': 2.12, 'Be': 1.98, 'B': 1.91,
          'C': 1.77, 'N': 1.66, 'O': 1.50, 'F': 1.46, 'Ne': 1.58, 'Na': 2.50,
          'Mg': 2.51, 'Al': 2.25, 'Si': 2.19, 'P': 1.90, 'S': 1.89,
          'Cl': 1.82, 'Ar': 1.83, 'K': 2.73, 'Ca': 2.62, 'Sc': 2.58,
          'Ti': 2.46, 'V': 2.42, 'Cr': 2.45, 'Mn': 2.45, 'Fe': 2.44,
          'Co': 2.40, 'Ni': 2.40, 'Cu': 2.38, 'Zn': 2.39, 'Ga': 2.32,
          'Ge': 2.29, 'As': 1.88, 'Se': 1.82, 'Br': 1.86, 'Kr': 2.25,
          'Rb': 3.21, 'Sr': 2.84, 'Y': 2.75, 'Zr': 2.52, 'Nb': 2.56,
          'Mo': 2.45, 'Tc': 2.44, 'Ru': 2.46, 'Rh': 2.44, 'Pd': 2.15,
          'Ag': 2.53, 'Cd': 2.49, 'In': 2.43, 'Sn': 2.42, 'Sb': 2.47,
          'Te': 1.99, 'I': 2.04, 'Xe': 2.06, 'Cs': 3.48, 'Ba': 3.03,
          'La': 2.98, 'Ce': 2.88, 'Pr': 2.92, 'Nd': 2.95, 'Sm': 2.90,
          'Eu': 2.87, 'Gd': 2.83, 'Tb': 2.79, 'Dy': 2.87, 'Ho': 2.81,
          'Er': 2.83, 'Tm': 2.79, 'Yb': 2.80, 'Lu': 2.74, 'Hf': 2.63,
          'Ta': 2.53, 'W': 2.57, 'Re': 2.49, 'Os': 2.48, 'Ir': 2.41,
          'Pt': 2.29, 'Au': 2.32, 'Hg': 2.45, 'Tl': 2.47, 'Pb': 2.60,
          'Bi': 2.54, 'Ac': 2.8, 'Th': 2.93, 'Pa': 2.88, 'U': 2.71,
          'Np': 2.82, 'Pu': 2.81, 'Am': 2.83, 'Cm': 3.05, 'Bk': 3.4,
          'Cf': 3.05, 'Es': 2.7}

# Bondi van der Waals radii for elements
# From: doi:10.1021/j100881a503
# Accessed: http://www.knowledgedoor.com/2/elements_handbook/bondi_van_der_waals_radius.html 4/8/2021
bondivdw = {'Ar': 1.88, 'As': 1.85, 'Br': 1.85, 'Cd': 1.62, 'C': 1.70,
            'Cl': 1.75, 'Cu': 1.4, 'F': 1.47, 'Ga': 1.87, 'Au': 1.66,
            'He': 1.40, 'H': 1.20, 'In': 1.93, 'I': 1.98, 'Kr': 2.02,
            'Pb': 2.02, 'Li': 1.82, 'Mg': 1.73, 'Hg': 1.70, 'Ne': 1.54,
            'Ni': 1.63, 'N': 1.55, 'O': 1.52, 'Pd': 1.63, 'P': 1.80, 'Pt': 1.7,
            'K': 2.75, 'Se': 1.90, 'Si': 2.10, 'Ag': 1.72, 'Na': 2.27, 'S': 1.80,
            'Te': 2.06, 'Tl': 1.96, 'Sn': 1.96, 'U': 1.86, 'Xe': 2.16, 'Zn': 1.39}

# Period definitions for all element symbols
# Data from https://en.wikipedia.org/wiki/Group_(periodic_table) (last accessed Sept. 12th 2019)

period_1 = ['H', 'He']

period_2 = ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']

period_3 = ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']

period_4 = ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
            'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr']

period_5 = ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
            'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sn', ' Te', 'I', 'Xe']

period_6 = ['Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
            'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re',
            'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']

period_7 = ['Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
            'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
            'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

periods_dict = {'period_1': period_1, 'period_2': period_2, 'period_3': period_3,
                'period_4': period_4, 'period_5': period_5, 'period_6': period_6,
                'period_7': period_7}

# Group definitions for all element symbols
# Data from https://en.wikipedia.org/wiki/Group_(periodic_table) (last accessed Sept. 12th 2019)

hydrogen = ['H']  # Note H not typically included in group 1

group_1 = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr']

group_2 = ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra']

group_3 = ['Sc', 'Y']  # Some IUPAC Initiatives to call either 'La' and 'Ac' grp 3 or 'Lu' and 'Lr

group_4 = ['Ti', 'Zr', 'Hf', 'Rf']

group_5 = ['V', 'Nb', 'Ta', 'Db']

group_6 = ['Cr', 'Mo', 'W', 'Sg']

group_7 = ['Mn', 'Tc', 'Re', 'Bh']

group_8 = ['Fe', 'Ru', 'Os', 'Hs']

group_9 = ['Co', 'Rh', 'Ir', 'Mt']

group_10 = ['Ni', 'Pd', 'Pt', 'Ds']

group_11 = ['Cu', 'Ag', 'Au', 'Rg']

group_12 = ['Zn', 'Cd', 'Hg', 'Cn']

group_13 = ['B', 'Al', 'Ga', 'In', 'Tl', 'Nh']

group_14 = ['C', 'Si', 'Ge', 'Sn', 'Pb', 'Fl']

group_15 = ['N', 'P', 'As', 'Sb', 'Bi', 'Mc']

group_16 = ['O', 'S', 'Se', 'Te', 'Po', 'Lv']

group_17 = ['F', 'Cl', 'Br', 'I', 'At', 'Ts']

group_18 = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og']

lanthanides = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
               'Ho', 'Er', 'Tm', 'Yb', 'Lu']

actinides = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
             'Es', 'Fm', 'Md', 'No', 'Lr']

groups_dict = {
    'group_1': group_1, 'group_2': group_2, 'group_3': group_3,
    'group_4': group_4, 'group_5': group_5, 'group_6': group_6,
    'group_7': group_7, 'group_8': group_8, 'group_9': group_9,
    'group_10': group_10, 'group_11': group_11, 'group_12': group_12,
    'group_13': group_13, 'group_14': group_14, 'group_15': group_15,
    'group_16': group_16, 'group_17': group_17, 'group_18': group_18,
    'lanthanides': lanthanides, 'actinides': actinides, 'hydrogen': hydrogen}

# Metals (includes alkali, alkaline earth, and transition metals)
alkali_and_alkaline_earth = [
    'Li', 'li', 'LI', 'lithium', 'Be', 'be', 'BE', 'beryllium',
    'Na', 'na', 'NA', 'sodium', 'Mg', 'mg', 'MG', 'magnesium',
    'K', 'k', 'potassium', 'Ca', 'ca', 'CA', 'calcium',
    'Rb', 'rb', 'RB', 'rubidium', 'Sr', 'sr', 'SR', 'strontium',
    'Cs', 'cs', 'CS', 'cesium', 'Ba', 'ba', 'BA', 'barium',
    'Fr', 'fr', 'FR', 'francium', 'Ra', 'ra', 'RA', 'radium']

heavy_metals_and_metalloids = [
    'Al', 'al', 'AL', 'aluminum', 'aluminium',
    'Ga', 'ga', 'GA', 'gallium', 'Ge', 'ge', 'GE', 'germanium',
    'As', 'as', 'AS', 'arsenic', 'In', 'in', 'IN', 'indium',
    'Sn', 'sn', 'SN', 'tin', 'Sb', 'sb', 'SB', 'antimony',
    'Te', 'te', 'TE', 'tellurium', 'Tl', 'tl', 'TL', 'thallium',
    'Pb', 'pb', 'PB', 'lead', 'Bi', 'bi', 'BI', 'bismuth',
    'Po', 'po', 'PO', 'polonium', 'At', 'at', 'AT', 'astatine',
    'La', 'la', 'LA', 'lanthanum',
    'Ce', 'ce', 'CE', 'cerium', 'Pr', 'pr', 'PR', 'praseodymium',
    'Nd', 'nd', 'ND', 'neodymium', 'Pm', 'pm', 'PM', 'promethium',
    'Sm', 'sm', 'SM', 'samarium', 'Eu', 'eu', 'EU', 'europium',
    'Gd', 'gd', 'GD', 'gadolinium', 'Tb', 'tb', 'TB', 'terbium',
    'Dy', 'dy', 'DY', 'dysprosium', 'Ho', 'ho', 'HO', 'holmium',
    'Er', 'er', 'ER', 'erbium', 'Tm', 'tm', 'TM', 'thulium',
    'Yb', 'yb', 'YB', 'ytterbium', 'Lu', 'lu', 'LU', 'lutetium',
    'Ac', 'ac', 'AC', 'actinium', 'Th', 'th', 'TH', 'thorium',
    'Pa', 'pa', 'PA', 'proactinium', 'U', 'u', 'uranium',
    'Np', 'np', 'NP', 'neptunium', 'Pu', 'pu', 'PU', 'plutonium',
    'Am', 'am', 'AM', 'americium', 'Cu', 'cu', 'CU', 'curium',
    'Bk', 'bk', 'BK', 'berkelium', 'Cf', 'cf', 'CF', 'californium',
    'Es', 'es', 'ES', 'einsteinium', 'Fm', 'fm', 'FM', 'fermium',
    'Md', 'md', 'MD', 'mendelevium', 'No', 'no', 'NO', 'nobelium',
    'Lr', 'lr', 'LR', 'lawrencium']

# ## The metals list below contains only TMs. See metalslist function for logic.
metalslist = [
    'Sc', 'sc', 'SC', 'scandium', 'Ti', 'ti', 'TI', 'titanium',
    'V', 'v', 'vanadium', 'Cr', 'cr', 'CR', 'chromium',
    'Mn', 'mn', 'MN', 'manganese', 'Fe', 'fe', 'FE', 'iron',
    'Co', 'co', 'CO', 'cobalt', 'Ni', 'ni', 'NI', 'nickel',
    'Cu', 'cu', 'CU', 'copper', 'Zn', 'zn', 'ZN', 'zinc',
    'Y', 'y', 'yttrium', 'Zr', 'zr', 'ZR', 'zirconium',
    'Nb', 'nb', 'NB', 'niobium', 'Mo', 'mo', 'MO', 'molybdenum',
    'Tc', 'tc', 'TC', 'technetium', 'Ru', 'ru', 'RU', 'ruthenium',
    'Rh', 'rh', 'RH', 'rhodium', 'Pd', 'pd', 'PD', 'palladium',
    'Ag', 'ag', 'AG', 'silver', 'Cd', 'cd', 'CD', 'cadmium',
    'Hf', 'hf', 'HF', 'hafnium', 'Ta', 'ta', 'TA', 'tantalum',
    'W', 'w', 'tungsten', 'Re', 're', 'RE', 'rhenium',
    'Os', 'os', 'OS', 'osmium', 'Ir', 'ir', 'IR', 'iridium',
    'Pt', 'pt', 'PT', 'platinum', 'Au', 'au', 'AU', 'gold',
    'Hg', 'hg', 'HG', 'mercury', 'X',
]

metals_conv = {
    'scandium': 'Sc', 'titanium': 'Ti', 'vanadium': 'V', 'chromium': 'Cr',
    'manganese': 'Mn', 'iron': 'Fe', 'cobalt': 'Co', 'nickel': 'Ni',
    'copper': 'Cu', 'zinc': 'Zn', 'yttrium': 'Y', 'zirconium': 'Zr',
    'niobium': 'Nb', 'molybdenum': 'Mo', 'technetium': 'Tc',
    'ruthenium': 'Ru', 'rhodium': 'Rh', 'palladium': 'Pd', 'silver': 'Ag',
    'cadmium': 'Cd', 'lanthanum': 'La', 'hafnium': 'Hf', 'tantalum': 'Ta',
    'tungsten': 'W', 'rhenium': 'Re', 'osmium': 'Os', 'iridium': 'Ir',
    'platinum': 'Pt', 'gold': 'Au', 'mercury': 'Hg'}

# d-electron counts of transition metals
mtlsdlist = {'sc': 1, 'ti': 2, 'v': 3, 'cr': 4, 'mn': 5, 'fe': 6, 'co': 7, 'ni': 8, 'cu': 9, 'zn': 10,
             'y': 1, 'zr': 2, 'nb': 3, 'mo': 4, 'tc': 5, 'ru': 6, 'rh': 7, 'pd': 8, 'ag': 9, 'cd': 10,
             'hf': 2, 'ta': 3, 'w': 4, 're': 5, 'os': 6, 'ir': 7, 'pt': 8, 'au': 9, 'hg': 10}

# Default spins for each d-electron count (make this metal/oxidation state specific)
defaultspins = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5',
                5: '6', 6: '5', 7: '4', 8: '3', 9: '2', 10: '1'}

# Elements sorted by atomic number
elementsbynum = ['H', 'He',
                 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                 'K', 'Ca',
                 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                 'Xe',
                 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']

# Electronegativity (Pauling) by atom symbol
endict = {"H": 2.20, "He": 4.16,
          "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
          "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
          "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
          "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81,
          "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Rb": 0.82, "Sr": 0.95, "Y": 1.22,
          "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Tc": 2.10, "Ru": 2.20, "Rh": 2.28,
          "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05, "I": 2.66,
          "Cs": 0.79, "Ba": 0.89, "Hf": 1.30, "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20,
          "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02,
          "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Sm": 1.17,
          "Gd": 1.20, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Lu": 1.27,
          "Fr": 0.7, "Ra": 0.9, "Ac": 1.1, "Th": 1.3, "Pa": 1.5, "U": 1.38, "Np": 1.36, "Pu": 1.28,
          "Am": 1.3, "Cm": 1.3, "Bk": 1.3, "Cf": 1.3, "Es": 1.3, "Fm": 1.3, "Md": 1.3, "No": 1.3,
          "Yb": 1.1, "Eu": 1.2, "Tb": 1.1, "Te": 2.10}

# Polarizability (alpha) by atom symbol
# From https://www.tandfonline.com/doi/full/10.1080/00268976.2018.1535143
# Last accessed 4/28/20

poldict = {"H": 4.50711, "He": 1.38375,
           "Li": 164.1125, "Be": 37.74, "B": 20.5, "C": 11.3, "N": 7.4,
           "O": 5.3, "F": 3.74, "Ne": 2.66, "Na": 162.7, "Mg": 71.2, "Al": 57.8, "Si": 37.3, "P": 25,
           "S": 19.4, "Cl": 14.6, "Ar": 11.083, "K": 289.7, "Ca": 160.8, "Sc": 97, "Ti": 100,
           "V": 87, "Cr": 83, "Mn": 68, "Fe": 62, "Co": 55, "Ni": 49, "Cu": 46.5, "Zn": 38.67,
           "Ga": 50, "Ge": 40, "As": 30, "Se": 28.9, "Br": 21, "Kr": 16.78, "Rb": 319.8, "Sr": 197.2,
           "Y": 162, "Zr": 112, "Nb": 98, "Mo": 87, "Tc": 79, "Ru": 72, "Rh": 66, "Pd": 26.14,
           "Ag": 55, "Cd": 46, "In": 65, "Sn": 53, "Sb": 43, "Te": 38, "I": 32.9, "Xe": 27.32,
           "Cs": 400.9, "Ba": 272, "La": 215, "Ce": 205, "Pr": 216, "Nd": 208, "Pm": 200, "Sm": 192,
           "Eu": 184, "Gd": 158, "Tb": 170, "Dy": 163, "Ho": 156, "Er": 150, "Tm": 144,
           "Yb": 139, "Lu": 137, "Hf": 103, "Ta": 74, "W": 68, "Re": 62, "Os": 57, "Ir": 54,
           "Pt": 48, "Au": 36, "Hg": 33.91, "Tl": 50, "Pb": 47, "Bi": 48, "Po": 44, "At": 42,
           "Rn": 35, "Fr": 317.8, "Ra": 246, "Ac": 203, "Pa": 154, "U": 129, "Np": 151, "Pu": 132,
           "Am": 131, "Cm": 144, "Bk": 125, "Cf": 122, "Es": 118, "Fm": 113, "Md": 109, "No": 110,
           "Lr": 320, "Rf": 112, "Db": 42, "Sg": 40, "Bh": 38, "Hs": 36, "Mt": 34, "Ds": 32,
           "Rg": 32, "Cn": 28, "Nh": 29, "Fl": 31, "Mc": 71, "Ts": 76, "Og": 58}


# Roman numerals
romans = {'I': '1', 'II': '2', 'III': '3', 'IV': '4',
          'V': '5', 'VI': '6', 'VII': '7', 'VIII': '8'}

# bondsdict
bondsdict = {"H": 1, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 3, "O": 2, "F": 1,
             "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 3, "S": 2, "Cl": 1,
             "As": 3, "Se": 2, "Br": 1, "I": 1, "He": 2}

# triple bonds dictionry: Defined as 0.5*(double bond dist + triple bond dist)
# bond lengths are from http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
tribonddict = {("C", "C"): 1.27, ("C", "N"): 1.235, ("C", "O"): 1.165, ("N", "N"): 1.175,
               ("N", "C"): 1.235, ("O", "C"): 1.165}

# Amino acids
amino_acids = {"ALA": "ALA", "A": "ALA",
               "ARG": "ARG", "R": "ARG", "ARN": "ARG",
               "ASN": "ASN", "N": "ASN",
               "ASP": "ASP", "D": "ASP", "ASH": "ASP",
               "CYS": "CYS", "C": "CYS", "CYS2": "CYS", "CYM": "CYS",
               "CYX": "CYS", "GLU": "GLU", "E": "GLU", "GLH": "GLU",
               "GLN": "GLN", "Q": "GLN", 'AGLN': 'GLN', 'BGLN': 'GLN',
               "GLY": "GLY", "G": "GLY",
               "HIS": "HIS", 'H': "HIS", "HID": "HIS", "HIE": "HIS",
               "HIP": "HIS", "HSE": "HIS", "HSD": "HIS", "HSP": "HIS",
               "ILE": "ILE", "I": "ILE", "LEU": "LEU", "L": "LEU",
               "LYS": "LYS", "K": "LYS", "LYN": "LYS",
               "MET": "MET", "M": "MET", "AMET": "MET", "BMET": "MET",
               "PHE": "PHE", "F": "PHE", "PRO": "PRO", "P": "PRO",
               "SER": "SER", "S": "SER", "ASER": "SER", "BSER": "SER",
               "THR": "THR", 'T': "THR", "TRP": "TRP", "W": "TRP",
               "TYR": "TYR", "Y": 'TYR',
               "VAL": "VAL", 'V': "VAL", 'AVAL': 'VAL', 'BVAL': 'VAL'
               }

# Common heteromolecules
het_mols = ['TAU', 'AKG', 'FE2', 'HOH', 'NAG', 'BMA', 'MAN', 'ACE', 'ACY', 'FE',
            'HEM', 'BLE', 'MG', 'BGC', 'WAT', 'CA', 'CL1']

# ---Geo_Check_Metrics------
dict_oct_check_loose = {"mono": {'num_coord_metal': 6,
                                 'rmsd_max': 0.4, 'atom_dist_max': 0.6,
                                 'oct_angle_devi_max': 16, 'max_del_sig_angle': 27,
                                 'dist_del_eq': 0.45, 'dist_del_all': 1.25,
                                 'devi_linear_avrg': 35, 'devi_linear_max': 40},
                        "multi": {'num_coord_metal': 6,
                                  'rmsd_max': 4, 'atom_dist_max': 0.6,
                                  'oct_angle_devi_max': 20, 'max_del_sig_angle': 35,
                                  'dist_del_eq': 0.45, 'dist_del_all': 1.25,
                                  'devi_linear_avrg': 35, 'devi_linear_max': 40}
                        }

dict_oct_check_st = {"mono": {'num_coord_metal': 6,
                              'rmsd_max': 0.3, 'atom_dist_max': 0.45,
                              'oct_angle_devi_max': 12, 'max_del_sig_angle': 22.5,
                              'dist_del_eq': 0.35, 'dist_del_all': 1.0,
                              'devi_linear_avrg': 20, 'devi_linear_max': 28},
                     "multi": {'num_coord_metal': 6,
                               'rmsd_max': 3, 'atom_dist_max': 0.45,
                               'oct_angle_devi_max': 20, 'max_del_sig_angle': 36,
                               'dist_del_eq': 0.35, 'dist_del_all': 1.0,
                               'devi_linear_avrg': 20, 'devi_linear_max': 28}
                     }

dict_oneempty_check_st = {"mono": {'num_coord_metal': 5,
                                   'rmsd_max': 0.4, 'atom_dist_max': 0.7,
                                   'oct_angle_devi_max': 15, 'max_del_sig_angle': 18,
                                   'dist_del_eq': 0.5, 'dist_del_all': 1,
                                   'devi_linear_avrg': 10, 'devi_linear_max': 20},
                          "multi": {'num_coord_metal': 5,
                                    'rmsd_max': 0.4, 'atom_dist_max': 0.7,
                                    'oct_angle_devi_max': 15, 'max_del_sig_angle': 18,
                                    'dist_del_eq': 0.5, 'dist_del_all': 1,
                                    'devi_linear_avrg': 10, 'devi_linear_max': 20}
                          }

dict_oneempty_check_loose = {"mono": {'num_coord_metal': 5,
                                      'rmsd_max': 0.6, 'atom_dist_max': 0.9,
                                      'oct_angle_devi_max': 20, 'max_del_sig_angle': 27,
                                      'dist_del_eq': 0.6, 'dist_del_all': 1.2,
                                      'devi_linear_avrg': 15, 'devi_linear_max': 28},
                             "multi": {'num_coord_metal': 5,
                                       'rmsd_max': 0.6, 'atom_dist_max': 0.9,
                                       'oct_angle_devi_max': 20, 'max_del_sig_angle': 27,
                                       'dist_del_eq': 0.6, 'dist_del_all': 1.2,
                                       'devi_linear_avrg': 15, 'devi_linear_max': 28}
                             }

dict_tetra_check_loose = {"mono": {'num_coord_metal': 4, 'rmsd_max': 0.4,
                                   'oct_angle_devi_max': 16, 'max_del_sig_angle': 27,
                                   'dist_del_all': 1.25,
                                   'devi_linear_avrg': 35, 'devi_linear_max': 40},
                          "multi": {'num_coord_metal': 4, 'rmsd_max': 0.4,
                                    'oct_angle_devi_max': 16, 'max_del_sig_angle': 27,
                                    'dist_del_all': 1.25,
                                    'devi_linear_avrg': 35, 'devi_linear_max': 40}
                          }

dict_tetra_check_st = {"mono": {'num_coord_metal': 4, 'rmsd_max': 0.3,
                                'oct_angle_devi_max': 12, 'max_del_sig_angle': 22.5,
                                'dist_del_all': 1,
                                'devi_linear_avrg': 20, 'devi_linear_max': 28},
                       "multi": {'num_coord_metal': 4, 'rmsd_max': 0.3,
                                 'oct_angle_devi_max': 12, 'max_del_sig_angle': 22.5,
                                 'dist_del_all': 1,
                                 'devi_linear_avrg': 20, 'devi_linear_max': 28}
                       }

dict_staus = {'good': 1, 'bad': 0}

oct_angle_ref = [[90, 90, 90, 90, 180] for x in range(6)]
tetra_angle_ref = [[109.47, 109.47, 109.47] for x in range(4)]
oneempty_angle_ref = [[90, 90, 90, 90], [180, 90, 90, 90], [180, 90, 90, 90],
                      [180, 90, 90, 90], [180, 90, 90, 90]]
geo_check_dictionary = {"dict_oct_check_loose": dict_oct_check_loose,
                        "dict_oct_check_st": dict_oct_check_st,
                        "dict_oneempty_check_st": dict_oneempty_check_st,
                        "dict_oneempty_check_loose": dict_oneempty_check_loose,
                        "dict_staus": dict_staus,
                        "oct_angle_ref": oct_angle_ref,
                        "oneempty_angle_ref": oneempty_angle_ref}
all_geometries = {
    3: ["trigonal planar", "T shape", "trigonal pyramidal"],
    4: ["tetrahedral", "square planar", "seesaw"],
    5: ["trigonal bipyramidal", "square pyramidal", "pentagonal planar"],
    6: ["octahedral", "pentagonal pyramidal", "trigonal prismatic"],
    7: ["pentagonal bipyramidal"],
}
all_angle_refs = {
    "trigonal planar": [[120, 120] for x in range(3)],
    "T shape": [[90, 90], [90, 180], [90, 180]],
    "tetrahedral": [[109.47, 109.47, 109.47] for x in range(4)],
    "square planar": [[90, 90, 180] for x in range(4)],
    "seesaw": [[90, 90, 180] for x in range(2)] + [[90, 90, 120] for x in range(2)],
    "trigonal pyramidal": [[109.5, 109.5] for x in range(3)],
    "trigonal bipyramidal": [[90, 90, 90, 180] for x in range(2)] + [[120, 120, 90, 90] for x in range(3)],
    "square pyramidal": [[90, 90, 90, 90]] + [[180, 90, 90, 90] for x in range(4)],
    "pentagonal planar": [[36, 36, 72, 72] for x in range(5)],
    "octahedral": [[90, 90, 90, 90, 180] for x in range(6)],
    "pentagonal pyramidal": [[90, 90, 90, 90, 90]] + [[36, 36, 72, 72, 90] for x in range(5)],
    "trigonal prismatic": [[75.3, 86.5, 86.5, 133.3, 133.3] for x in range(6)],
    "pentagonal bipyramidal": [[90, 90, 90, 90, 90, 180] for x in range(2)] + [[72, 72, 144, 144, 90, 90] for x in range(5)]
}


# Module for running bash commands
#  @param cmd String containing command to be run
#  @return bash output string
def mybash(cmd):
    """Function to run a bash command.

    Parameters
    ----------
        cmd : str
            String containing command to be run

    Returns
    -------
        output : str
            bash output string

    """
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()
    lines = stdout.decode('utf-8').splitlines()
    lines = [line+"\n" for line in lines if line !=""]
    return ''.join(lines)


# Defines global variables used throughout the code
class globalvars(metaclass=Singleton):
    """Globalvars class. Defines global variables used throughout the code, including periodic table.
    """
    def __init__(self):
        # Program name
        self.PROGRAM = 'molSimplify'
        s = '\nmolSimplify v1.3.3x\nFreely distributed under the GNU GPL license.\n'
        s += 'Copyright 2017 Kulik Lab @ MIT\n'
        s += 'Developed by: Efthymios Ioannidis (timis@mit.edu)\n'
        s += 'Contributions by:\n\tHeather J. Kulik (hjkulik@mit.edu)\n'
        s += '\t Terry Gani (terryg@mit.edu)\n'
        s += '\t JP Janet (jpjanet@mit.edu)\n'
        s += 'E. I. Ioannidis, T. Z. H. Gani, H. J. Kulik. J. Comput. Chem. 2016, 37, 2106-2117.\n'
        s += 'J.P. Janet, Q. Zhao, E.I. Ioannidis, H.J. Kulik. Mol. Simul. 2017,43(5-6), 327-345.\n'
        s += 'J.P. Janet, T. Z. H. Gani, A. H. Steeves, E. I. Ioannidis, H. J. Kulik. Ind. Eng. Chem. Res. 2017, 56(17), 4898-4910.\n'
        # About message
        self.about = s
        # ##### GET INFORMATION ######
        runfromcmd = False
        try:
            # ## check if running through commandline ###
            if sys.stdin.isatty():
                # running through command line
                runfromcmd = True
            else:
                runfromcmd = False
        except AttributeError:  # if sys.stdin does not have an isatty method
            runfromcmd = True
        # ## get running os ###
        self.linux, self.osx = False, False
        if platform.system().lower() in 'linux':
            self.linux = True
        elif platform.system().lower() in 'darwin':
            self.osx = True
        # get cwd
        # script filename (usually with path)
        _ = inspect.getfile(inspect.currentframe())
        cdir2 = os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe())))  # script directory
        cdir = cdir2.rsplit('/', 1)[0]
        # unused:
        # cdir2 = cdir
        homedir = os.path.expanduser("~")
        # create default molSimplify for mac
        if self.osx and not glob.glob(homedir + '/.' + self.PROGRAM) and not runfromcmd:
            txt = 'INSTALLDIR=/Applications/' + self.PROGRAM + '.app/Contents/Resources\n'
            with open(homedir + '/.' + self.PROGRAM, 'w') as fout:
                fout.write(txt)
        self.chemdbdir = ''
        self.multiwfn = ''
        self.custom_path = False
        # ##### check for ~/.molSimplify ######
        if glob.glob(homedir + '/.' + self.PROGRAM):
            with open(homedir + '/.' + self.PROGRAM, 'r') as fin:
                s = [_f for _f in fin.read().splitlines() if _f]
            d = dict()
            for ss in s:
                sp = [_f for _f in ss.split('=') if _f]
                try:
                    d[sp[0]] = sp[1]
                except IndexError:
                    pass
            if 'CHEMDBDIR' in list(d.keys()):
                self.chemdbdir = d['CHEMDBDIR']
            if 'MULTIWFN' in list(d.keys()):
                self.multiwfn = "'" + d['MULTIWFN'] + "'"
            if 'CUSTOM_DATA_PATH' in list(d.keys()):
                self.custom_path = d['CUSTOM_DATA_PATH']
        else:
            self.installdir = cdir
            with open(homedir + '/.' + self.PROGRAM, 'w') as fout:
                fout.write('CHEMDBDIR=\n')

        # Home directory
        self.homedir = homedir
        # Number of smiles ligands
        self.nosmiles = 0
        # Jobs directory
        self.rundir = homedir + '/Runs/'
        # Number of generated structures
        self.generated = 0
        # Additional debug output
        self.debug = False
        # SMARTS patterns for forced hydrogen removal
        self.remHsmarts = ["O=CN", "O=CO", "n", "N=CN", "nN"]
        # Default geometries for each coordination number if none specified
        self.defaultgeometry = {8: ('sqap', 'square_antiprismatic'), 7: ('pbp', 'pentagonal_bipyramidal'),
                                6: ('oct', 'octahedral'), 5: ('tbp', 'trigonal bipyramidal'), 4: ('thd', 'tetrahedral'),
                                3: ('trigonal planar', 'tpl'), 2: ('linear', 'li'), 1: ('one', 'one')}
        # Default oxidation states for elements
        self.defaultoxstate = {
            'au': 'I', 'gold': 'I', 'scandium': 'III', 'sc': 'III', 'ti': 'IV', 'titanium': 'IV'}
        # bent "linear" angle in degrees, e.g., in Fe(III)-superoxo or a bent nitrosyl
        self.linearbentang = 45

    # Returns atomic mass dictionary
    #  @param self The object pointer
    #  @return Atomic mass dictionary
    def amass(self):
        """Get the atomic mass dictionary.

        Returns
        -------
            amassdict : dict
                Dictionary containing atomic masses.
        """
        return amassdict

    def polarizability(self):
        """Get the polarizability dictionary.

        Returns
        -------
            poldict : dict
                Dictionary containing polarizabilities.
        """
        return poldict

    def tribonddict(self):
        """Get the triple bond dictionary.

        Returns
        -------
            tribonddict : dict
                Dictionary containing triple bond lengths.
        """
        return tribonddict

    def bondsdict(self):
        """Get the bond dictionary.

        Returns
        -------
            bondsdict : dict
                Dictionary containing bond lengths.
        """
        return bondsdict

    def elementsbynum(self):
        """Returns list of elements by number

        Returns
        -------
            elementsbynum : list
                List of elements by number
        """
        return elementsbynum

    def endict(self):
        """Returns electronegativity dictionary.

        Returns
        -------
            endict : list
                Electronegativity dictionary
        """
        return endict

    def vdwrad(self):
        """Returns VDW dictionary.

        Returns
        -------
            vdwrad : list
                Dictionary of VDW radii.
        """
        return vdwrad

    def metalslist(self, transition_metals_only=True):
        """Get the metals list.

        Returns
        -------
            metalslist : list
                List of available metals.
        """
        if not transition_metals_only:
            return metalslist + alkali_and_alkaline_earth + heavy_metals_and_metalloids
        else:
            return metalslist

    def groups(self):
        """Returns dict of elements by groups.

        Returns
        -------
            groups_dict : dict
                Groups dictionary.
        """
        return groups_dict

    def periods(self):
        """Returns dict of elements by periods.

        Returns
        -------
            periods_dict : dict
                Periods dictionary.
        """
        return periods_dict

    def geo_check_dictionary(self):
        """Returns list of geo check objects dictionary.

        Returns
        -------
            geo_check_dictionary : dict
                Geo check measurement dictionary.
        """
        return geo_check_dictionary

    def get_all_geometries(self):
        """Get available geometries.

        Returns
        -------
            all_geometries : list
                All available geometries.
        """
        return all_geometries

    def get_all_angle_refs(self):
        """Get references angle dict.

        Returns
        -------
            all_angle_refs : dict
                Reference angles for various geometries.
        """
        return all_angle_refs

    def add_custom_path(self, path):
        """Record custom path in ~/.molSimplify file

        Parameters
        ----------
            path : str
                Path to custom data ~/.molSimplify file.
        """
        homedir = os.path.expanduser("~")
        with open(homedir + '/.' + self.PROGRAM, 'a') as f:
            f.write('CUSTOM_DATA_PATH=' + str(path) + "\n")

    def bbcombs_mononuc(self):
        """Get backbone combinations dictionary

        Returns
        -------
            bbcombs_mononuc : dict
                Backbone combination dictionary for different geometries.
        """
        bbcombs_mononuc = dict()
        bbcombs_mononuc['one'] = [[1]]
        bbcombs_mononuc['li'] = [[1], [2]]
        bbcombs_mononuc['oct'] = [[1, 2, 3, 4, 5, 6],  # 6-dentate
                                  [1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [
                                      1, 2, 3, 5, 6], [1, 2, 4, 5, 6],  # 5-dentate
                                  [1, 3, 4, 5, 6], [2, 3, 4, 5, 6],  # 5-dentate
                                  [1, 2, 3, 4], [2, 5, 4, 6], [
                                      1, 5, 3, 6],  # 4-dentate
                                  [1, 2, 3], [1, 4, 2], [1, 4, 3], [1, 5, 3], [
                                      1, 6, 3], [2, 3, 4],  # 3-dentate
                                  [2, 5, 4], [2, 6, 4], [5, 4, 6], [5, 1, 6], [
                                      5, 2, 6], [5, 3, 6],  # 3-dentate
                                  [1, 2], [1, 4], [1, 5], [1, 6], [
                                      2, 3], [2, 5],  # 2-dentate
                                  [2, 6], [3, 5], [3, 6], [4, 5], [
                                      4, 6], [3, 4],  # 2-dentate
                                  [1], [2], [3], [4], [5], [6]]  # 1-dentate
        bbcombs_mononuc['pbp'] = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 6],  # 6/5-dentate
                                  [1, 2, 3, 5],  # 4-dentate
                                  [1, 2, 3], [1, 2, 4], [2, 1, 5], [3, 1, 6], [
                                      5, 6, 3], [2, 6, 5],  # 3-dentate
                                  [1, 2], [2, 3], [3, 4], [4, 5], [1, 7], [
                                      2, 6], [5, 7], [3, 6],  # 2-dentate
                                  [1], [2], [3], [4], [5], [6], [7]]  # 1-dentate
        bbcombs_mononuc['spy'] = [[1, 2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2],
                                  [1, 2], [1, 4], [2, 3], [3, 4], [4, 5], [
                                      2, 5], [3, 5], [1, 5], [1], [2], [3], [4],
                                  [5]]
        bbcombs_mononuc['sqp'] = [[1, 4, 2, 3], [1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2], [1, 2], [1, 4], [2, 3],
                                  [3, 4],
                                  [1], [2], [3], [4]]
        bbcombs_mononuc['tbp'] = [[1, 2, 3, 4, 5], [1, 3, 4, 5], [3, 2, 4], [4, 5, 3], [5, 1, 3], [4, 5], [5, 3],
                                  [3, 4],
                                  [1, 4], [1, 5], [1, 3], [2, 4], [2, 5], [2, 3], [1], [2], [3], [4], [5]]
        bbcombs_mononuc['thd'] = [[1, 2, 3, 4], [3, 2, 4], [2, 4, 1], [4, 1, 3], [2, 4], [4, 3], [3, 2], [1, 3], [1, 4],
                                  [2, 4], [1], [2], [3], [4]]
        bbcombs_mononuc['tpl'] = [[1, 2, 3], [
            1, 2], [2, 3], [1, 3], [1], [2], [3]]
        bbcombs_mononuc['tpr'] = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5], [1, 2, 5, 4], [5, 2, 3, 6], [1, 4, 6, 3],
                                  [1, 2, 3], [3, 6, 5],
                                  [2, 3], [2, 5], [5, 6], [6, 4], [4, 1], [1], [2], [3], [4], [5], [6]]
        return bbcombs_mononuc

    def testTF(self):
        """Tests to see whether keras and tensorflow are available.

        Returns
        -------
            tf_flag : bool
                True if tensorflow and keras are available.
        """
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            from tensorflow.keras.models import Model  # noqa: F401
            from tensorflow.keras.models import Sequential  # noqa: F401
            return True
        except ImportError:
            return False

    def testmatplotlib(self):
        """Tests to see if matplotlib is available

        Returns
        -------
            mpl_flag : bool
                True if matplotlib is available
        """
        try:
            # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import matplotlib.pyplot as plt  # noqa: F401
            import matplotlib.ticker as ticker  # noqa: F401
            return True
        except ImportError:
            return False

    def getAllAAs(self):
        """ Gets all amino acids

        Returns
        -------
            amino_acids : dictionary
                Dictionary of standard amino acids
        """
        return amino_acids
