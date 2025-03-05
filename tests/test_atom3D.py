import math
import pytest
from molSimplify.Classes.atom3D import atom3D


@ pytest.mark.parametrize(
	"sym, coords",
	[
	("Al", [0.8,3.9,3.9]),
	("Ir", [-1,2.4,7.7]),
	])
def test_repr(sym, coords):
	my_atom = atom3D(Sym=sym, xyz=coords)
	assert str(my_atom) == f"atom3D(Sym={sym}, xyz={coords})"


@pytest.mark.parametrize(
	"sym",
	[
	"Rb",
	"X",
	"Pb",
	"Ge",
	"Co",
	"Pt",
	"Ba",
	])
def test_symbol(sym):
	my_atom = atom3D(Sym=sym)
	assert sym == my_atom.symbol()


@pytest.mark.parametrize(
	"coords",
	[
	[1,0,1],
	[0,5,8],
	[-3,-2,-1],
	])
def test_coords(coords):
	my_atom = atom3D(xyz=coords)
	assert coords == my_atom.coords()


def test_defaults():
	my_atom = atom3D()
	assert "C" == my_atom.symbol()
	assert [0, 0, 0] == my_atom.coords()


@pytest.mark.parametrize(
	"coords1, coords2, correct_answer",
	[
	([0,0,0], [3,0,4], 5),
	([1,-1,-10], [40,2,3], 41.2189277),
	([5,4,5], [5,4,5], 0),
	])
def test_distance(coords1, coords2, correct_answer):
	my_atom1, my_atom2 = atom3D(xyz=coords1), atom3D(xyz=coords2)
	assert math.isclose(correct_answer, my_atom1.distance(my_atom2))
	assert math.isclose(correct_answer, my_atom2.distance(my_atom1))


@pytest.mark.parametrize(
	"coords1, coords2, correct_answer",
	[
	([0,0,0], [3,0,4], [-3,0,-4]),
	([1,-1,-10], [40,2,3], [-39,-3,-13]),
	([5,4,5], [5,4,5], [0,0,0]),
	])
def test_distancev(coords1, coords2, correct_answer):
	my_atom1, my_atom2 = atom3D(xyz=coords1), atom3D(xyz=coords2)
	assert correct_answer == my_atom1.distancev(my_atom2)

	negative_correct_answer = [-1*i for i in correct_answer]
	assert negative_correct_answer == my_atom2.distancev(my_atom1)


@pytest.mark.parametrize(
	"coords",
	[
	[1,1,1],
	[-5,5,10],
	[3,9,0],
	])
def test_setcoords(coords):
	my_atom = atom3D()
	my_atom.setcoords(coords)
	assert coords == my_atom.coords()


@pytest.mark.parametrize(
	"sym, mass, atno, rad",
	[
	("B", 10.83, 5, 0.85),
	("Tc", 98.9, 43, 1.56),
	("Cs", 132.9055, 55, 2.32),
	])
def test_mutate(sym, mass, atno, rad):
	my_atom = atom3D()
	my_atom.mutate(newType=sym)
	assert sym == my_atom.symbol()
	assert mass == my_atom.mass
	assert atno == my_atom.atno
	assert rad == my_atom.rad


@pytest.mark.parametrize(
    "sym, transition_metals_only, correct_answer",
    [
    ("Na", False, True),
    ("Na", True, False),
    ("Ca", False, True),
    ("Ca", True, False),
    ("Zr", False, True),
    ("Zr", True, True),
    ("Os", False, True),
    ("Os", True, True),
    ("Zn", False, True),
    ("Zn", True, True),
    ("C", False, False),
    ("C", True, False),
    ("Sb", False, True),
    ("Sb", True, False),
    ("Br", False, False),
    ("Br", True, False),
    ("Og", False, False),
    ("Og", True, False),
    ("Np", False, True),
    ("Np", True, False),
    ("Er", False, True),
    ("Er", True, False),
    ])
def test_ismetal(sym, transition_metals_only, correct_answer):
	my_atom = atom3D(Sym=sym)
	metal_flag = my_atom.ismetal(transition_metals_only=transition_metals_only)
	assert correct_answer == metal_flag


@pytest.mark.parametrize(
	"xyz, dxyz, new_xyz",
	[
	([1,2,3], [4,5,6], [5,7,9]),
	([-8,4,0], [0,1,1], [-8,5,1]),
	([17,100,-3.5], [5.2,-1,3], [22.2,99,-0.5]),
	])
def test_translate(xyz, dxyz, new_xyz):
	my_atom = atom3D(xyz=xyz)
	my_atom.translate(dxyz)
	assert new_xyz == my_atom.coords()
