import pytest
import argparse
import os
from openbabel import openbabel as ob
import numpy as np
from molSimplify.Scripts.inparse import *
from molSimplify.Scripts.generator import *
from molSimplify.Classes.globalvars import *
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Scripts.geometry import distance
from molSimplify.Classes.atom3D import atom3D
from pkg_resources import resource_filename, Requirement

infile = resource_filename(Requirement.parse(
    "molSimplify"), "tests/inputs/example_1_noff.in")
args = ['main.py', '-i', infile]
startgen(args, False, False)
infile = resource_filename(Requirement.parse(
    "molSimplify"), "tests/inputs/example_1.in")
args = ['main.py', '-i', infile]
startgen(args, False, False)

