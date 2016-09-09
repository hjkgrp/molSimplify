# Written by JP Janet for HJK Group
# Dpt of Chemical Engineering, MIT
import os, sys
import glob, re, math, random, string, numpy, pybel
from math import pi
from scipy.spatial import Delaunay, ConvexHull
#import networkx as nx
from geometry import *
from Classes.atom3D import *
from Classes.mol3D import*
from Classes.globalvars import globalvars
from operator import add

###############################
def cell_ffopt(ff,mol,frozenats):
    ### FORCE FIELD OPTIMIZATION ##
    # INPUT
    #   - ff: force field to use, available MMFF94, UFF< Ghemical, GAFF
    #   - mol: mol3D to be ff optimized
    #   - connected: indices of connection atoms to metal
    #   - constopt: flag for constrained optimization
    # OUTPUT
    #   - mol: force field optimized mol3D
    metals = range(21,31)+range(39,49)+range(72,81)
    ### check requested force field
    ffav = 'mmff94, uff, ghemical, gaff, mmff94s' # force fields
    if ff.lower() not in ffav:
        print 'Requested force field not available. Defaulting to MMFF94'
        ff = 'mmff94'
    ### convert mol3D to OBmol via xyz file, because AFTER/END option have coordinates
    backup_mol = mol3D()
    backup_mol.copymol3D(mol)
 #   print('bck ' + str(backup_mol.getAtom(0).coords()))
 #   print('mol_ibf ' + str(mol.getAtom(0).coords()))

    mol.writexyz('tmp.xyz')
    mol.OBmol = mol.getOBmol('tmp.xyz','xyzf')
    os.remove('tmp.xyz')
    ### initialize constraints
    constr = pybel.ob.OBFFConstraints()
    ### openbabel indexing starts at 1 ### !!!
    # convert metals to carbons for FF
    indmtls = []
    mtlsnums = []
    for iiat,atom in enumerate(mol.OBmol.atoms):
        if atom.atomicnum in metals:
            indmtls.append(iiat)
            mtlsnums.append(atom.atomicnum)
            atom.OBAtom.SetAtomicNum(19)
    for cat in frozenats:
        constr.AddAtomConstraint(cat+1) # indexing babel
    ### set up forcefield
    forcefield =pybel.ob.OBForceField.FindForceField(ff)
    obmol = mol.OBmol.OBMol
    forcefield.Setup(obmol,constr)
    ## force field optimize structure
    forcefield.ConjugateGradients(2500)
    forcefield.GetCoordinates(obmol)
    mol.OBmol = pybel.Molecule(obmol)

    # reset atomic number to metal
    for i,iiat in enumerate(indmtls):
        mol.OBmol.atoms[iiat].OBAtom.SetAtomicNum(mtlsnums[i])
    mol.convert2mol3D()

    en = forcefield.Energy()
 #   print(str(mol.OBmol.atoms[1].OBAtom.GetVector().GetZ()))
#    print(str(forcefield.Validate()))
   # print('mol_af ' + str(mol.getAtom(0).coords()))

  #  print('ff delta = ' + str(backup_mol.rmsd(mol)))
    del forcefield, constr, obmol
    return mol,en
################################
def import_from_cif(fst):
    #INPUT:
    # fst:  filename of cif file
    #OUTPUT:
    # unit_cell:  mol3D class of a single unit cell
    # cell_vector: list of lists of floats, each 
    #           corresponds to one of the defining cell
    #           vectors 
    cell_vector = list()
    unit_cell = mol3D()
    A = 0
    B = 0
    C = 0
    alpha =0
    beta = 0
    emsg =list()
    exit_status = 0
    gamma = 0
    obConversion = pybel.ob.OBConversion()
    obConversion.SetInAndOutFormats("cif", "xyz")
    mol = pybel.ob.OBMol()
    try:
        obConversion.ReadFile(mol, fst)
        fillUC = pybel.ob.OBOp.FindType("fillUC")
        fillUC = pybel.ob.OBOp.FindType("fillUC")
        fillUC.Do(mol, "strict")
        unit_cell.OBmol =pybel.Molecule(mol)
        unit_cell.convert2mol3D()
    except:
        emsg.append("Error in reading of cif file by pybel")
        exit_status = 1
    with open(fst) as f:
        lines = f.readlines()
        for line in lines:
            linesplit = line.split()
            if len(linesplit) != 0:
                if linesplit[0] == "_cell_length_a":
                    A = float((re.sub(r'\([^)]*\)','', ''.join(c for c in linesplit[1]))))
                if linesplit[0] == "_cell_length_b":
                   B = float((re.sub(r'\([^)]*\)','', ''.join(c for c in linesplit[1]))))
                if linesplit[0] == "_cell_length_c":
                    C = float((re.sub(r'\([^)]*\)','', ''.join(c for c in linesplit[1]))))

                if linesplit[0] == "_cell_angle_alpha":
                    alpha =float( ''.join(c for c in linesplit[1] if c not in '()').rstrip('.'))
                if linesplit[0] == "_cell_angle_beta":
                    beta =float( ''.join(c for c in linesplit[1] if c not in '()').rstrip('.'))
                if linesplit[0] == "_cell_angle_gamma":
                    gamma = float(''.join(c for c in linesplit[1] if c not in '()').rstrip('.'))
    # create cell vectors
    try:
        cell_vector.append([A,0,0])
        cell_vector.append([B*numpy.cos((gamma*pi)/180),B*numpy.sin((gamma*pi)/180),0])
        cx = C*numpy.cos((beta*pi)/180)
        cy = C*(numpy.cos((alpha*pi)/180)-numpy.cos((beta*pi)/180)*numpy.cos((gamma*pi/180)))/numpy.sin((gamma*pi)/180)
        cz = sqrt(C*C - cx*cx - cy*cy)
        cell_vector.append([cx,cy,cz])
    except:
        emsg = emsg.append('Error in creating unit cell from cif informtation')
        exit_status = 2
    for i,rows in enumerate(cell_vector):
        print(rows)
        for j,elements in enumerate(rows):
            if elements <= 1e-8:
                cell_vector[i][j] = 0
    if exit_status != 0:
        return emsg
    else:
        return unit_cell,cell_vector
#################################
def shave_surface_layer(super_cell):
    shaved_cell = mol3D()
    shaved_cell.copymol3D(super_cell)
    TOL = 1e-1
    extents = find_extents(super_cell)
    zmax = extents[2]
    del_list = list()
    for i,atoms in enumerate(super_cell.getAtoms()):
        coords = atoms.coords()
        if abs(coords[2] - zmax) < TOL:
            del_list.append(i)
    shaved_cell.deleteatoms(del_list)
    return shaved_cell
###############################
def shave_under_layer(super_cell):
    shaved_cell = mol3D()
    shaved_cell.copymol3D(super_cell)
    TOL = 1e-1
    zmin = 1000;
    for i,atoms in enumerate(super_cell.getAtoms()):
        coords = atoms.coords()
        if (coords[2] < zmin):
                zmin = coords[2]
    del_list = list()
    for i,atoms in enumerate(super_cell.getAtoms()):
        coords = atoms.coords()
        if abs(coords[2] - zmin) < TOL:
            del_list.append(i)
    shaved_cell.deleteatoms(del_list)
    return shaved_cell
###############################
def zero_z(super_cell):
        zeroed_cell = mol3D()
        zeroed_cell.copymol3D(super_cell)
        TOL = 1e-1
        zmin = 1000;
        for i,atoms in enumerate(super_cell.getAtoms()):
                coords = atoms.coords()
                if (coords[2] < zmin):
                        zmin = coords[2]
        zeroed_cell.translate([0,0,-1*zmin])
        return zeroed_cell

###############################
def point_in_box(point,box):
    outcome = False
    fx=(box[0][0] <= point[0])*(point[0] < box[0][1])
    fy=(box[1][0] <= point[1])*(point[1] < box[1][1])
    fz=(box[2][0] <= point[2])*(point[2] < box[2][1])
    if fz  and fy and fx:
        outcome = True
    return outcome
##############################
def points_below_plane(point,w):
    outcome = False
    zplane = w[2] + w[1]*point[1] + w[0]*point[0]
    if (point[2] <= zplane):
        outcome =  True
    return outcome
#################################
def cut_cell_to_index(unit_cell,cell_vector,miller_index):
    ## determine the plane:
        cut_cell = mol3D()
        cut_cell.copymol3D(unit_cell)
        h,k,l = miller_index
  #      print(str(h) + ' ' + str(k) +  ' ' +  str(l))
        disc,p,q = xgcd(k,l)
 #       print(str(p) + ' ' + str(q))
        cell_vector = numpy.array(cell_vector)
        k1= numpy.dot(p*(k*cell_vector[0]-h*cell_vector[1]) + q*(l*cell_vector[0] - h*cell_vector[2]),l*cell_vector[1] - k*cell_vector[2])
        k2= numpy.dot(l*(k*cell_vector[0]-h*cell_vector[1]) -k*(l*cell_vector[0] - h*cell_vector[2]),l*cell_vector[1] - k*cell_vector[2])
        tol = 1e-3
        if abs(k2)> tol:
            c = -1*int(round(k1/k2))
            p,q = p+c*l,q - c*k
        v1 = p*numpy.array(k*cell_vector[0]-h*cell_vector[1]) + q*numpy.array(l*cell_vector[0] + h*cell_vector[2])
        v2 = numpy.array(l*cell_vector[1]-k*cell_vector[2])
        disc,a,b = xgcd(p*k + q*l,h)
        v3 = numpy.array(b*cell_vector[0] + a*p*cell_vector[1]  + a*q*cell_vector[2])
        zint = miller_index[2]*cell_vector[2][2]
        yint = miller_index[1]*cell_vector[1][1]
        xint = miller_index[0]*cell_vector[0][0]
        w = [0,0,0]
        w[2] = zint
        w[1] = -w[2]/yint
        w[0] = -w[2]/xint
#        print('w = ' + str(w))
#        print(miller_index)
        plane_normal = normalize_vector(numpy.cross(vecdiff([xint,0,0],[0,0,zint]),vecdiff([0,yint,0],[0,0,zint])))
        angle = vecangle(plane_normal,[0,0,1])
        u =  numpy.cross(plane_normal,[0,0,1])
        return v1,v2,v3,angle,u
##################################
def center_of_sym(list_of_points):
    n = len(list_of_points)
#    print('lop = ' + str(list_of_points))
    csym = [0,0,0];
    csym = [float(sum(x)/n) for x in zip(*list_of_points)]
    return csym
##################################
def xgcd(b, n):
    # calculate x,y such that b*x + n*y = gcd(b,n)
    # by extended Euclidean algorithm 
    x0, x1, y0, y1 = 1, 0, 0, 1
    while n != 0:
        q, b, n = b // n, n, b % n
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return  b, x0, y0
#
##################################

def distance_zw(r1,r2):
    dx = r1[0] - r2[0]
    dy = r1[1] - r2[1]
    dz =150*( r1[2] - r2[2])
    d = sqrt(dx**2+dy**2+dz**2)
    return d
##################################
def mdistance(r1,r2):
    dx = r1[0] - r2[0]
    dy = r1[1] - r2[1]
    dz = r1[2] - r2[2]
    d = sqrt(numpy.power(dx,2) + numpy.power(dy,2) + numpy.power(dz,2))
    return d

###################################
def normalize_vector(v):
    length = distance(v,[0,0,0])
    if length:
        nv = [float(i)/length for i in v]
    else:
        nv = [0,0,0]
    return nv
###################################
def distance_2d_torus(R1,R2,dim):
    ### distance between points in Euclidean torus 
    dx =abs( R1[0] - R2[0])
    dy = abs(R1[1] - R2[1] )
    dz =abs(( R1[2] - R2[2]))
    d1 = sqrt(  numpy.power(dim[0] - dx,2)
              + numpy.power(dim[1] - dy,2)
              + numpy.power(dz,2))
    d2 = sqrt(  numpy.power(dim[0] - dx,2)
                     + numpy.power(dy,2)
                     + numpy.power(dz,2))
    d3 = sqrt(  numpy.power(dx,2)
              + numpy.power(dim[1] - dy,2)
              + numpy.power(dz,2))
    d = min(d1,d2,d3)
    return d
################################################################
def periodic_2d_distance(R1,R2,cell_vector):
    ### distance between points in Euclidean torus 
    ## STILL UNDER CONSTRUCTION, WIP WIP WIP ***
    dx =abs( R1[0] - R2[0])
    dy = abs(R1[1] - R2[1] )
    dz =abs(( R1[2] - R2[2]))
    for v1shifts in [-1,0,1]:
        for v1shifts in [-1,0,1]:
            for yshifts in [-1,0,0]:
                pass
    d1 = sqrt(  numpy.power(dim[0] - dx,2)
              + numpy.power(dim[1] - dy,2)
              + numpy.power(dz,2))
    d2 = sqrt(  numpy.power(dim[0] - dx,2)
                     + numpy.power(dy,2)
                     + numpy.power(dz,2))
    d3 = sqrt(  numpy.power(dx,2)
              + numpy.power(dim[1] - dy,2)
              + numpy.power(dz,2))
    d = min(d1,d2,d3)
    return d
################################################################
def periodic_mindist(mol,surf,dim):
    ### calculates minimum distance between atoms in 2 molecules ###
    # INPUT
    #   - mol: mol3D class,  molecule
    #   - surf: mol3D class, the surface
    #   - dim: list of float, replication 
    # OUTPUT
    #   - mind: minimum distance between atoms of the 2 mol objects
    mind = 1000
    for atom1 in mol.getAtoms():
        for atom0 in surf.getAtoms():
            if (distance_2d_torus(atom1.coords(),atom0.coords(),dim) < mind):
                mind = distance(atom1.coords(),atom0.coords())
    return mind
################################################################
def periodic_selfdist(mol,dim):
    ### calculates minimum distance between atoms in 2 molecules ##
    # INPUT
    #   - mol: mol3D class,  molecule
    #   - dim: list of floats, replication 
    # OUTPUT
    #   - mind: minimum distance between atoms of the 2 mol and periodic
    #             images
    mind = 1000
    for ii,atom1 in enumerate(mol.getAtoms()):
        for jj,atom0 in enumerate(mol.getAtoms()):
            if (distance_2d_torus(atom1.coords(),atom0.coords(),dim) < mind) and (ii !=jj):
                mind = distance(atom1.coords(),atom0.coords())
    return mind


##################################
def closest_torus_point(mol,dim):
    min_dist = 1000
    for atom1 in mol.getAtoms():
        R1 = atom1.coords()
        for atom2 in mol.getAtoms():
            R2 = atom2.coords()
            d = distance_2d_torus(R1,R2,dim)
            if (d<min_dist):
                min_dist = d
    return min_dist
##################################
def concave_hull(points,alpha):
    ## points should be tuples
    de = Delaunay(points)
    for i in de.simplices:
        tmp = []
        j = [points[c] for c in i]
        print(i)
        print(j)
    print(de)

points= [[1,1],[1,0],[0,1],[0,0]]

###################################
def unit_to_super(unit_cell,cell_vector,duplication_vector):
    # INPUT
    #   - unit_cell: mol3D class that contains the unit cell
    #   - cell_vector: list of float contains the cell vectors a,b,c
    #   - duplication_vector: list of int the number of duplications in each dim
    # OUTPUT
    #   - super_cell: mol3D class that contains the super cell
    super_cell  = mol3D()
    print(cell_vector)
    acell = duplication_vector[0]
    bcell = duplication_vector[1]
    ccell = duplication_vector[2]
    a = cell_vector[0]
    b = cell_vector[1]
    c = cell_vector[2]
    for atoms in unit_cell.getAtoms():
#        print('.......................................')
#        print(atoms.symbol())
#        print(atoms.coords())
        for i in range(0,acell):
            for j in range(0,bcell):
                for k in range(0,ccell):
#                    print(str(i) + str(j) + str(k))
                    dx = 0 + i*a[0] + j*b[0] + k*c[0]
                    dy = 0 + i*a[1] + j*b[1] + k*c[1]
                    dz = 0 + i*a[2] + j*b[2] + k*c[2]
                    trans_vect =(dx,dy,dz)
                    new_atom =atom3D(atoms.symbol(),atoms.coords())
                    new_atom.translate(trans_vect)
                    super_cell.addAtom(new_atom)
    return super_cell
##############################
def find_all_surface_atoms(super_cell,tol=1e-2,type_of_atom = False):
    # Get all atoms on the tope surface - NB, this will
    # not handle complex (2 or more) atom-type surfaces
    # if the atoms are 'layered', e.g. TiO2 - Ti under O2, 
    # no Ti will be found
    # INPUT: 
    #   - super_cell: mol3D class that contains the super cell
    #   - tol: float, max distance from extent plane to look
    #   - type_of_atom: optional, string, gets atoms of the given type on the face plane
    #                   if left out, will not care about types of atoms
    # OUPUT
    #   - avail_sites_list: list of int, indices of atoms on the surface
    #
    extents = find_extents(super_cell)
    target_height = extents[2]
    avail_sites_list = list()
    if type_of_atom:
        possible_atom_inds = super_cell.findAtomsbySymbol(type_of_atom)
    else:
        possible_atom_inds =range(0,super_cell.natoms)
    for indices in possible_atom_inds:
        z_dist = abs(super_cell.getAtom(indices).coords()[2] - target_height)
        if (z_dist <= tol):
            avail_sites_list.append(indices)
    return avail_sites_list
##############################
def find_extents(super_cell):
    # INPUT
    #   - super_cell: mol3D class that contains the super cell
    # OUPUT
    #   - extents: list of max coords of atoms on the surface
    xmax = 0
    zmax = 0
    ymax = 0
    for atoms in super_cell.getAtoms():
        coords = atoms.coords()
        x_ext = coords[0]# + atoms.rad
        y_ext = coords[1]# + atoms.rad
        z_ext = coords[2]# + atoms.rad
        xmax = max(xmax,x_ext)
        ymax = max(ymax,y_ext)
        zmax = max(zmax,z_ext)
    extents = [xmax,ymax,zmax]
    return extents
#####################################
def find_extents_cv(super_cell_vector):
    # INPUT
    #   - super_cell_vector: matrix of the three vectors that define the super cell
    # OUPUT
    #   - extents: list of max coords of the super cell
    xmax = 0
    zmax = 0
    ymax = 0
    for columns in super_cell_vector:
        xmax = max(xmax,abs(columns[0]))
        ymax = max(ymax,abs(columns[1]))
        zmax = max(zmax,abs(columns[2]))
    xmax = numpy.linalg.norm(super_cell_vector[0])
    ymax = numpy.linalg.norm(super_cell_vector[1])
    zmax = numpy.linalg.norm(super_cell_vector[2])

    extents = [xmax,ymax,zmax]
    return extents
#############################
def multialign_objective_function(payload,surface_coord_list,cand_list,bind_dist):
    # INPUT
    #   - payload: mol3D, the structure to add
    #   - surface_coord_list: list of list of 3 float, coordinates of the 
    #                        slab target points
    #   - cand_list: list of int, indices of the attachment points in the 
    #                payload
    #   - bind_dist: float, target alignment distance
    # OUPUT
    #   - cost: float, sum of squared error, the difference between
    #           the actual distance and the target
    cost = 0
    for indices in enumerate(cand_list):
       v1=(surface_coord_list[indices[0]])
       v2=payload.getAtom(int(indices[1])).coords()
       cost +=numpy.power((mdistance(v1,v2)) - bind_dist,2)
    return cost
#############################
def tracked_merge(payload,super_cell):
    # INPUT
    #   - super_cell: mol3D, the slab (and previously added adsorbates)
    #   - payload: mol3D, the structure to add
    # OUPUT
    #   - merged_cell: mol3D, merged combintation of payload and cell
    #   - payload_index: list of int, indices of payload atoms in cell
    #   - slab_index: list of int, indices of slab atoms in the cell
    payload_index = [i for i in xrange(0,payload.natoms)]
    slab_index = [i + payload.natoms for i in xrange(0,super_cell.natoms)]
    merged_cell = mol3D()
    merged_cell.copymol3D(payload)
    merged_cell.combine(super_cell)
    return merged_cell,payload_index,slab_index
#############################
def force_field_relax_with_slab(super_cell,payload,cand_list,its):
    # INPUT
    #   - super_cell: mol3D, the slab (and previously added adsorbates)
    #   - payload: mol3D, the structure to add
    #   - can_ind:  list of int, indices of taget attachement points in molecule
    # OUPUT
    #   - new_payload: mol3D, payload relaxed by force field with slab fixed
    new_payload = mol3D()
    new_payload.copymol3D(payload)
    cell_copy  = mol3D()
    surface_sites_list = find_all_surface_atoms(super_cell,tol=1e-2)
    for sites in surface_sites_list:
        cell_copy.addAtom(super_cell.getAtom(sites))
    merged_payload,payload_ind,slab_index = tracked_merge(new_payload,cell_copy)
    merged_payload.writexyz('modi.xyz')
    full_fixed_atoms_list = cand_list + slab_index # freeze the slab componentsp
    distorted_payload = mol3D()
    distorted_payload.copymol3D(merged_payload)
    print('in ff, distorded coords' + str(distorted_payload.getAtom(0).coords()));
    distorted_payload,enl = cell_ffopt('uff',merged_payload,full_fixed_atoms_list)
    print('after ff, distorded coords' + str(distorted_payload.getAtom(0).coords()));
    print(full_fixed_atoms_list)
#    distorted_payload.writexyz(str(its)+'modr.xyz')
    distorted_payload.deleteatoms(slab_index)
    return distorted_payload
#############################
def surface_center(super_cell):
    # INPUT
    #   - super_cell: mol3D, the slab (and previously added adsorbates)
    #   - payload: mol3D, the structure to add
    #   - can_ind:  list of int, indices of taget attachement points in molecule
    # OUPUT
    #   - new_payload: mol3D, payload relaxed by force field with slab fixed
    cell_copy  = mol3D()
    surface_sites_list = find_all_surface_atoms(super_cell,tol=1e-2)
    for sites in surface_sites_list:
        cell_copy.addAtom(super_cell.getAtom(sites))
    centroid = cell_copy.centersym()

    return centroid


##############################
def choose_nearest_neighbour(target_site,avail_sites_dict,occupied_sites_dict,super_cell,super_cell_vector):
    # INPUT
    #   - avail_sites_dict: dict with {index:[coords] } of {int,list of float}, free sites
    #   - occupied_sites_dict: dict with {index:[coords] } of {int,list of float}, occupied sites
    #   - target_site: list of doubles, coords that the new site should be close to 

    #   - weight: float in [0,1], how strongly the interace-absorbate distance is weighted
    #   - method: 'linear' a  linear combination of distance from centroid and neighbour
    #                             distance is used
    #                     'log' a logarithmic weighting is used - strong re
    # OUPUT
    #   - nn_site: index of nearest neighbour  site, a key for avail_sites_dict
    extents = find_extents_cv(super_cell_vector)
 #   print('extents = ' + str(extents))
    weight = 0.1 #favours adjaceny to point over distance from other occupied sites
    # get the nearest site to target
    score = 100000 # weighted assessment, lower is better
    avail_sites_list = avail_sites_dict.keys()
    occupied_sites_list = occupied_sites_dict.keys()
#    print('oc sites list  = '+str(occupied_sites_list))
    if (len(avail_sites_list) > 1 ): # more than 1 option, pick closest to target site 
        for indices in avail_sites_list:
                distance_to_target =  distance(target_site,avail_sites_dict[indices]) # NOT the torus distance - must be two cells in one unit
                distance_to_nearest_occupied =1000 
                for neighbours in occupied_sites_list:
                    # get distance to nearest neighbour
                    distance_to_nearest_occupied = min(distance_2d_torus(avail_sites_dict[indices],occupied_sites_dict[neighbours],extents),distance_to_nearest_occupied)
                    this_score =(1 - weight)*distance_to_target -  weight*distance_to_nearest_occupied
                if this_score < score:
                    score = this_score
  #                  print('New lowest score at score =' + str(score) + '\n')
                    nn_site = indices
    elif (len(avail_sites_list) == 1):
        nn_site = avail_sites_list[0]
    else:
        emsg = ('error: no free site is possible')
        print(emsg)
    return nn_site
#####################################
def choose_best_site(avail_sites_dict,occupied_sites_dict,centroid,super_cell,super_cell_vector,weight = 0.5, method = 'linear'):

    # INPUT
    #   - avail_sites_dict: dict with {index:[coords] } of {int,list of float}, free sites
    #   - occupied_sites_dict: dict with {index:[coords] } of {int,list of float}, occupied sites
    #   - weight: float in [0,1], how strongly the interace-absorbate distance is weighted
    #   - method: 'linear' a  linear combination of distance from centroid and neighbour
    #                             distance is used
    #                     'log' a logarithmic weighting is used - strong re
    # OUPUT
    #   - target_site: index of target site, a key for avail_sites_dict
    extents = find_extents_cv(super_cell_vector)
    centroid= surface_center(super_cell) 

    print('extents = ' + str(extents))
    print('centroid is at '+ str(centroid))
    skipalign = 0
    score = 100000 # weighted assessment, lower is better
    avail_sites_list = avail_sites_dict.keys()
    random.shuffle(avail_sites_list)
    occupied_sites_list = occupied_sites_dict.keys()
    print('oc sites list  = '+str(occupied_sites_list))
    print('ac sites list  = '+str(avail_sites_list))
    if (len(avail_sites_list) > 1 ): # more than 1 option, pick closest to center of plane 
        for indices in avail_sites_list:
#                distance_to_center =  distance_2d_torus(centroid,avail_sites_dict[indices],extents)
                distance_to_center =  mdistance(centroid,avail_sites_dict[indices])

                distance_to_nearest_occupied =1000 
                for neighbours in occupied_sites_list:
                    # get distance to nearest neighbour
                    #print('Neighbour:' + str(occupied_sites_dict[neighbours]) + ' point is at ' + str(avail_sites_dict[indices])  
                    distance_to_nearest_occupied = min(distance_2d_torus(avail_sites_dict[indices],occupied_sites_dict[neighbours],extents),distance_to_nearest_occupied)
#                print('dist to nearest = '   + str(distance_to_nearest_occupied) + ' at ' + str(avail_sites_dict[indices]) )
                if (method == 'linear'):
                    this_score =(1 - weight)*distance_to_center -  weight*distance_to_nearest_occupied
                elif (method == 'log'):
                    this_score = (1 - weight)*abs(numpy.log(distance_to_center)) - weight*abs(numpy.log(distance_to_nearest_occupied))
                if this_score < score:
                    score = this_score
                    target_site = indices
                    print('target site is ' + str(indices) + ' at ' +str(super_cell.getAtom(indices).coords()))  
    elif (len(avail_sites_list) == 1):
        target_site = avail_sites_list[0]
    else:
        emsg = ('error: no free site is possible')
        print(emsg)
    return target_site
#####################################
def align_payload_to_multi_site(payload,surface_coord_list,cand_list,bind_dist):
    # INPUT
    #   - payload: mol3D class that contains the molecule to place
    #   - align_coord: list of lists of float, positions on surface
    #   - cand_mask: list of int, indices of atoms in payload that will be aligned
    #               can also contain a string, mask, list of indicies
    # OUPUT
    #   - newpay_load: mol3D class with the atom in payload(cand_in) directly above
    #                  align_coord. )Does NOT change height

    # Get all atoms on the top surface - NB, this will not handle complex surfaces, split calls by atom type
    #print('align symbol is ' + payload.getAtom(cand_ind).symbol())
    new_payload = mol3D()
    new_payload.copymol3D(payload)
    #print(cand_list)
    payload_coord = center_of_sym([new_payload.getAtom(i).coords() for i in cand_list ])
    surface_coord = center_of_sym(surface_coord_list)

    vec1 =  vecdiff(surface_coord,payload.centersym())
    vec2 =  vecdiff(payload_coord,new_payload.centersym())
    print('\n vec1 is '+ str(vec1))
    print('vec2 is '+ str(vec2) + '\n')

    rotate_angle = vecangle(vec1,vec2)
    theta,u = rotation_params(payload_coord,new_payload.centersym(),surface_coord)
    print(theta)
    print('angle is '+ str(rotate_angle))
    print('normal is ' + str(u))
    new_payload = rotate_around_axis(new_payload,new_payload.centersym(),u,rotate_angle)
    cost =  multialign_objective_function(new_payload,surface_coord_list,cand_list,bind_dist)
    final_payload = new_payload

    # need to determine the collinearity of the points are co-plannar
    collinear_flag = False
    coplanar_flag = False
    if len(cand_list) == 2:
        collinear_flag = True
    if len(cand_list) == 3:
        coplanar_flag = True
        collinear_flag = checkcolinear(new_payload.getAtom(cand_list[0]).coords(),new_payload.getAtom(cand_list[1]).coords(),new_payload.getAtom(cand_list[2]).coords())
    elif len(cand_list) == 4:
        pass
        #coplanar_flag = checkplanar(new_payload.getAtom(cand_list[0]),new_payload.getAtom(cand_list[1]),new_payload.getAtom(cand_list[2]),new_payload.getAtom(cand_list[3]).coords())
    if collinear_flag: # there is a single line defining the axis - align this with 
        line_slope = vecdiff(new_payload.getAtom(cand_list[0]).coords(),new_payload.getAtom(cand_list[1]).coords())
        print('collinear case : line ' + str(line_slope))
        new_u = numpy.cross(line_slope,vecdiff(surface_coord,payload_coord))
        print('new u is ' + str(new_u))
    elif coplanar_flag:
        dvec1 =  vecdiff(new_payload.getAtom(cand_list[0]).coords(),new_payload.getAtom(cand_list[1]).coords())
        dvec2 =  vecdiff(new_payload.getAtom(cand_list[0]).coords(),new_payload.getAtom(cand_list[2]).coords())
        plane_vector = numpy.cross(dvec1,dvec2)
        print('coplanar case : normal ' + str(plane_vector))
        new_u = numpy.cross(plane_vector,vecdiff(surface_coord,payload_coord))
        print('new u is ' + str(new_u))

    if collinear_flag or coplanar_flag:
        print('starting rotation')
        for rotate_angle in range(-100,100):
            this_payload = mol3D()
            this_payload.copymol3D(final_payload)
            this_payload = rotate_around_axis(this_payload,this_payload.centersym(),new_u,float(rotate_angle)/10) #fine grained check
            this_cost = multialign_objective_function(this_payload,surface_coord_list,cand_list,bind_dist)
            if (this_cost < (cost)):
                #print('current cost = ' + str(this_cost) + ', the max is ' + str(cost))
                print('accepting rotate at theta  = ' + str(rotate_angle))
                cost = this_cost
                final_payload = this_payload
    return final_payload

##################################
def combine_multi_aligned_payload_with_cell(super_cell,super_cell_vector,payload,cand_list,surface_coord_list,bind_dist,duplicate = False,control_angle = False):
     #   This function does final lowering, rotate and merge of previously aligned molecule with surface
     #   Precede all calls to this funciton with allign_payload_to_Site to avoid strange behaviour
     # INPUT 
     #   - super_cell: mol3D class that contains the super cell
     #   - payload: mol3D class that contains that target molecule
     #   - payload_ind: int, index of atom in payload that will bind to the surface 
     #   - align_coord: list of float, coordinates of the target surface site
     #   - bind_dist: float, binding distance in A   
     #   - duplicate: logical, create a negative-z reflection as well?
     # OUPUT
     #   - combined_cel: mol3D class, loaded cell
    combined_cell = mol3D()
    combined_cell.copymol3D(super_cell)
    new_payload = mol3D()
    new_payload.copymol3D(payload)
    trial_cell = mol3D()
    trial_cell.copymol3D(combined_cell)

    ######## DEBUG ONLY #####
    backup_payload = mol3D()
    backup_payload.copymol3D(payload)
    ##########################

    extents = find_extents_cv(super_cell_vector)
    # the generalized distance descriptors
    payload_coord = center_of_sym([new_payload.getAtom(i).coords() for i in cand_list ])
    surface_coord = center_of_sym(surface_coord_list)

    vec =  vecdiff(surface_coord,payload_coord)
    cost =  multialign_objective_function(new_payload,surface_coord_list,cand_list,bind_dist)
    final_payload = new_payload
    if not control_angle:
        print('starting rotation')
        for rotate_angle in range(0,360):
            this_payload = mol3D()
            this_payload.copymol3D(new_payload)
            this_payload = rotate_around_axis(this_payload,this_payload.centersym(),vec,rotate_angle)
            this_cost = multialign_objective_function(this_payload,surface_coord_list,cand_list,bind_dist)
            if (this_cost < (cost)):
               cost = this_cost
               final_payload = this_payload

    print('cost after rotation =' + str(cost))

    # lower into positon
    # step size:
    factor = 0.20
    deltaZ =factor*(distance(payload_coord,surface_coord)-bind_dist)
    this_step_accepted = True
    num_bad_steps= 0 
    break_flag = False
    maxits = 150
    its = 0
    while (not break_flag) and (its < maxits):
        its +=1
        if (not this_step_accepted) and (num_bad_steps <= 4):
            factor = 0.1*factor
        elif (not this_step_accepted) and (num_bad_steps > 4):
            break_flag = True
        payload_coord = center_of_sym([final_payload.getAtom(i).coords() for i in cand_list ])
        # this will be lowered slowly, then rotate to optimize at each height
        trans_vec = [factor*deltaZ*element for element  in normalize_vector(vec)]
        this_payload = mol3D()
        this_payload.copymol3D(final_payload)
        this_payload.translate(trans_vec)
        this_cost = multialign_objective_function(this_payload,surface_coord_list,cand_list,bind_dist)
        this_dist = min(periodic_mindist(this_payload,combined_cell,extents),this_payload.mindist(combined_cell))
        this_coord = center_of_sym([this_payload.getAtom(i).coords() for i in cand_list ])

        this_deltaZ =(distance(this_coord,surface_coord)-bind_dist)

        print('cost  = ' + str(this_cost) +'/' +str(cost)+ '  i = ' + str(its) + '  dz =  ' + str(deltaZ) + ' dist  '+ str(this_dist) + ' b step  = '+ str(num_bad_steps) + ' nxt dz = ' + str(this_deltaZ))
        if (this_cost < (cost)) and (this_dist > 0.75) and (deltaZ > 1e-2):
            print('accepting down shift at i  = ' + str(its))
            cost = this_cost
            del final_payload
            final_payload = mol3D()
            if ( this_payload.mindist(combined_cell) < 1.5):
                print('ff on at iteration '  + str(its))
                distorted_payload = mol3D()
                distorted_payload.copymol3D(this_payload)
                print('Warning, a force-field relaxation is in progress. For large molecules (Natoms > 50), this may take a few minutes. Please be patient.')
                distorted_payload = force_field_relax_with_slab(super_cell,this_payload,cand_list,its)
                print(this_payload.getAtom(0).symbol() + ' at ' + str(this_payload.getAtom(cand_list[0]).coords()) + 'target at ' + str(surface_coord_list[0]))
                print(distorted_payload.getAtom(0).symbol() + ' at ' + str(distorted_payload.getAtom(cand_list[0]).coords()) + 'target at ' + str(surface_coord_list[0]))
                final_payload.copymol3D(distorted_payload)
                print(final_payload.getAtom(0).symbol() + ' at ' + str(final_payload.getAtom(cand_list[0]).coords()) + 'target at ' + str(surface_coord_list[0]))
            else:
                final_payload.copymol3D(this_payload)
            this_step_accepted = True
            factor = min(1.25*factor,0.8)
            deltaZ = this_deltaZ
            num_bad_steps = 0
        else:
            this_step_accepted = False
            num_bad_steps += 1 
    print('\n\n exit after '  + str(its) + ' iterations')
    print('target distance  = ' + str(bind_dist) + ', average deviation =  ' + str(sqrt(cost)/len(cand_list)))
    distances_list = []
    for indices in enumerate(cand_list):
        v1=(surface_coord_list[indices[0]])
        v2=final_payload.getAtom(int(indices[1])).coords()
        distances_list.append((distance(v1,v2)))
    print(' Target distance was  ' + str(bind_dist)+', achieved ' + str(distances_list))
    min_dist = final_payload.mindist(combined_cell)
   # now, rotate to maximize spacing, based on mask length
    rotate_on = False
    if len(cand_list) == 1:
      rotate_on = True
    elif len(cand_list) ==2:
        vec = vecdiff(final_payload.getAtom(cand_list[0]).coords(),final_payload.getAtom(cand_list[1]).coords())
        rotate_on = True
   #     print('2-points')
    if control_angle:
        rotate_on = False
    if rotate_on:
        trial_cell.combine(final_payload)
        trial_cell.writexyz('before_rot.xyz')
        print('starting strain rotation')
        for rotate_angle in range(0,360):
             this_payload = mol3D()
             this_payload.copymol3D(final_payload)
             payload_coord = center_of_sym([this_payload.getAtom(i).coords() for i in cand_list ])
             this_payload = rotate_around_axis(this_payload,payload_coord,vec,rotate_angle)
             this_dist = min(periodic_mindist(this_payload,combined_cell,extents),periodic_selfdist(this_payload,extents),this.payload.mindist(combined_cell))
             if (this_dist > (min_dist + 1e-3)):
                 print('current dist = ' + str(this_dist) + ', the max is ' + str(min_dist))
                 print('accepting rotate at theta  = ' + str(rotate_angle))
                 min_dist = this_dist
                 final_payload = this_payload
    if len(cand_list) > 1:
    # now, distort molecule based on FF to optimize bond length
        print('\n \n begining distortion ')
        nsteps = 20 
        dfactor =float(1)/nsteps
        trans_vec_list = list()
        distances_list = list()
        # fetch all of the remaining 
        for indices in enumerate(cand_list):
           v1=(surface_coord_list[indices[0]])
           v2=final_payload.getAtom(int(indices[1])).coords()
           trans_vec_list.append(normalize_vector(vecdiff(v2,v1)))
           distances_list.append(distance(v1,v2) - bind_dist)
        ens =[]
        cutoff = 5.0 # kcal/mol
        distorted_payload = mol3D()
        distorted_payload.copymol3D(final_payload)
        for ii in range(0,nsteps+1):
            for indices in enumerate(cand_list):
                this_translation = [(-1)*dfactor*distances_list[indices[0]] for i in trans_vec_list[indices[0]]]
                distorted_payload.getAtom(int(indices[1])).translate(this_translation)
            distorted_payload,enl = cell_ffopt('mmff94',distorted_payload,cand_list)
            ens.append(enl)
            this_cost = multialign_objective_function(distorted_payload,surface_coord_list,cand_list,bind_dist)
            this_dist =min(periodic_mindist(distorted_payload,combined_cell,extents),periodic_selfdist(distorted_payload,extents))
            del distances_list
            distances_list = list()
            for indices in enumerate(cand_list):
               v1=(surface_coord_list[indices[0]])
               v2=distorted_payload.getAtom(int(indices[1])).coords()
               distances_list.append((distance(v1,v2) - bind_dist))
            #print(str((abs(ens[-1] - ens[0]) < 5.0)) + str((this_cost < cost)) + str((this_dist >= (min_dist - 0.1))))

            if (abs(ens[-1] - ens[0]) < 5.0) and (this_cost < cost) and (this_dist >= (min_dist - 0.1)):
                 final_payload = distorted_payload
                 cost = this_cost
                 min_dist = this_dist
                 print('accepting distort')

    distances_list = []
    for indices in enumerate(cand_list):
        v1=(surface_coord_list[indices[0]])
        v2=final_payload.getAtom(int(indices[1])).coords()
        distances_list.append((distance(v1,v2)))
    print('\n\n Target distance was  ' + str(bind_dist)+', achieved ' + str(distances_list))


    if duplicate:
        second_payload = mol3D()
        second_payload.copymol3D(final_payload)
        xyline = [second_payload.centersym()[0],second_payload.centersym()[1],0]
        point = [xyline[0],xyline[1],0*(second_payload.centersym()[2])/2]
        rotate_angle = 180
        second_payload = rotate_around_axis(second_payload,point,xyline,rotate_angle)
        second_payload.translate([0,0,surface_coord[2]])
        final_payload.combine(second_payload)
    min_intercell_d = closest_torus_point(final_payload,extents)
    print('minimum inter-cell adsorbate atom distance is ' + str(min_intercell_d))
    combined_cell.combine(final_payload)
    return combined_cell
###################################
def molecule_placement_supervisor(super_cell,super_cell_vector,target_molecule,method,target_atom_type,align_dist,surface_atom_type = False,control_angle = False,align_ind = False, align_axis = False,
                                  duplicate = False, number_of_placements = 1, coverage = False,weighting_method = 'linear',weight = 0.5,masklength = 1):
    ######### parse input
    if ((number_of_placements != 1) or coverage) and ((method != 'alignpair') or (control_angle)):
        if not (method == 'alignpair'):
            print('Multiple placement NOT supported for method ' + method)
        if control_angle:
            print('Cannot support multiple placements and controlled align')
        print(' Setting single placement only')
        number_of_placements = 1
        coverage = False
    if  ((control_angle) and not (align_axis)) or ((align_axis)  and not (control_angle)):
        print('Cannot control angle and not provide axis or vice-versa')
        print('control angle is ' +str(control_angle))
        print('align_axis  is ' + str(align_axis))
        control_angle =  False
        align_axis = False
    if control_angle and not align_ind:
        print('align_ind not found, even though control_angle is on. Disabling controlled rotation')
        control_angle = false
    if (method == 'alignpair') and not surface_atom_type:
       print('Must provide surface binding atom type to use alignpair')
       print(' using centered placemented instead')
       method = 'center'
    if (method == 'alignpair'): # get all vaccancies 
        avail_sites_list =  find_all_surface_atoms(super_cell,tol=1,type_of_atom = surface_atom_type)
        avail_sites_dict = dict()
        for indices in avail_sites_list:
            avail_sites_dict[indices] = super_cell.getAtom(indices).coords()
        occupied_sites_dict = dict()
        # calculate max number of sites that need to be filled 
        max_sites =int(numpy.floor(float(len(avail_sites_dict.keys()))/masklength))
    if coverage:
        number_of_placements = int(numpy.ceil(max_sites*coverage))
        print('Coverage requested = ' + str(coverage))

    ######## prepare and allocate
    loaded_cell = mol3D()
    loaded_cell.copymol3D(super_cell)
    debug_cell = mol3D()
    debug_cell.copymol3D(loaded_cell)
    ####### begin actual work

    for placements in range(0,number_of_placements):
        sites_list = list() # list to hod all of the target sites on the surface
        if (method == 'center'):
            align_coord = centered_align_coord(super_cell_vector)
            sites_list.append(align_coord)
        elif (method == 'staggered'):
            align_coord = staggered2_align_coord(super_cell)
            sites_list.append(align_coord)

        elif (method == 'alignpair'):
            best_site = choose_best_site(avail_sites_dict,occupied_sites_dict,centered_align_coord(super_cell_vector),super_cell,super_cell_vector,weight,weighting_method)
            align_coord = super_cell.getAtom(best_site).coords()
            occupied_sites_dict[best_site] = avail_sites_dict.pop(best_site) # this transfers the site to occupied
            sites_list.append(align_coord)
            if masklength != 1: # this is if we need multiple sites
                for iterates in range(1,masklength):
                    print('in loop for ' + str(iterates))
                    nn_site = choose_nearest_neighbour(align_coord,avail_sites_dict,occupied_sites_dict,super_cell,super_cell_vector)
                    align_coord = super_cell.getAtom(nn_site).coords()
                    sites_list.append(align_coord)
                    occupied_sites_dict[nn_site] = avail_sites_dict.pop(nn_site) # this transfers the site to occupied
        #        print(occupied_sites_dict.keys())
        #        print(sites_list)
                align_coord = center_of_sym(sites_list)

        else:
            emsg = 'unkown method of molecule placement ' + method
            print(emsg)
            return emsg
        print('Targert for align is ' + str(align_coord))
        ########## actual placement
        payload = mol3D()
        payload.copymol3D(target_molecule)
        payload_rad = payload.molsize()

        trans_vec = vecdiff(align_coord,payload.centersym())
        payload.translate(trans_vec)
        extents = find_extents_cv(super_cell_vector)
        payload.translate([0,0,extents[2]+1.15*(payload_rad + align_dist)]) # place far above


        ###############################
        temp_pay = mol3D()
        temp_pay.copymol3D(payload)
        debug_cell.combine(temp_pay)
        debug_cell.writexyz('db1.xyz')

        ######### find matching atom in payload
        # need to determine if the target is an element or a mask
        globs = globalvars()
        elements = globs.elementsbynum()
        if target_atom_type in elements:
            # find all matches in target
            payload_targets = payload.findAtomsbySymbol(target_atom_type)
            if (len(payload_targets) > 1 ): # more than 1 option, pick closest to center of plane 
                maxd = 1000
                for indices in payload_targets:
                    dist = distance(payload.getAtom(indices).coords(),[extents[0]/2,extents[1]/2,extents[2]])
                    if (dist<maxd):
                        cand_ind = indices
                        maxd = dist
            elif (len(payload_targets) == 1):
                    cand_ind = payload_targets[0]
            else:
                    emsg = ('Error: no align of type' + target_atom_type+ ' is possible. Not found in target. Using atom 0 align')
                    cand_ind = 0
                    print(emsg)
            print('cand _ind = ' + str(cand_ind))
            cand_list = [cand_ind]
        else:
            print('target molecule mask on ' + str(target_atom_type))
            cand_ind = target_atom_type 
            cand_list = [(int(i)-1) for i in cand_ind]
            print('candidate list is ' +str(cand_list))
        ######## rotate for optimal approach
        payload = align_payload_to_multi_site(payload,sites_list,cand_list,align_dist) # align
        if control_angle:
            print('begining controlled rotation, targeting angle ' + str(control_angle) + ' to  line ' + str(align_axis))
            print('aligning '+ payload.getAtom(cand_ind).symbol() + ' with ' + payload.getAtom(align_ind).symbol())
            payload = axes_angle_align(payload,cand_ind,align_ind,align_axis,control_angle)

        print('payload cysm '+ str(payload.centersym()))
        #######################################
        temp_pay2 = mol3D()
        temp_pay2.copymol3D(payload)
        temp_pay2.translate([0,0,-5])
        debug_cell.combine(temp_pay2)
        debug_cell.writexyz('db2.xyz')

        ####### lower payload to distance, rotate to avoid conflicr
        loaded_cell = combine_multi_aligned_payload_with_cell(loaded_cell,super_cell_vector,payload,cand_list,sites_list,align_dist,duplicate,control_angle)

        ########################
        temp_pay3 = mol3D()
        temp_pay3.copymol3D(payload)
        debug_cell.combine(temp_pay3)
        debug_cell.writexyz('db3.xyz')
        print('number of atoms = ' + str(loaded_cell.natoms))
        print("\n")
    ###### run tests
    overlap_flag = loaded_cell.sanitycheck(0)
    min_dist = loaded_cell.mindistmol()
    if (number_of_placements > 1): 
        print('preparing ' + str(number_of_placements) + ' placements ')
        effectvie_coverage =  float(number_of_placements)/float(max_sites)
        print('giving effectvie coverage of ' + str(effectvie_coverage) + '\n')
    print('Is there overalp? ' + str(overlap_flag))
    min_intercell_d = closest_torus_point(payload,extents)
    print('minimum inter-cell adsorbate atom distance is ' + str(min_intercell_d))

    return  loaded_cell

###################################
def centered_align_coord(super_cell_vector):
    extents = find_extents_cv(super_cell_vector)
    centroid= [extents[0]/2,extents[1]/2,extents[2]]
    print('Centroid is at ' + str(centroid))
    align_coord = centroid
    return align_coord
###################################
def staggered2_align_coord(super_cell):
    max_dist = 1000
    avail_sites_list =  find_all_surface_atoms(super_cell,tol=1e-2,type_of_atom = False)
    close_list = list()
    extents = find_extents(super_cell)
    centroid= [extents[0]/2,extents[1]/2,extents[2]]
    for indices in avail_sites_list:
        this_dist = distance(centroid,super_cell.getAtom(indices).coords())
        if (this_dist < (max_dist - 1e-3)):
            max_dist = this_dist
            if (len(close_list) >1):
                 print('subseq')
                 close_list[1] = close_list[0] #save old index
                 close_list[0] = super_cell.getAtom(indices)
            elif (len(close_list) == 1):
                print('second atom found')
                close_list.append(super_cell.getAtom(indices))
                temp = close_list[0]
                close_list[0] = close_list[1]
                close_list[1] = temp
            elif (len(close_list) == 0):
                    print('first atom found')
                    close_list.append(super_cell.getAtom(indices))
    align_coord = [sum(x)/2 for x in zip(close_list[0].coords(),close_list[1].coords())]
    return align_coord    #### end of stagger
###################################
def axes_angle_align(payload,cand_ind,align_ind,align_target,angle):
     #   This function doe rotates a given payload molecule such that the X-Y projection of
     #   the cord joining the two atoms in cand_ind and  align_ind is aligned with the vector given in align_target
     # INPUT 
     #   - payload: mol3D class that contains that target molecule
     #   - cand_ind: int, index of atom in payload that is used as reference
     #   - align_ind: int, index of atom in payload that define the cord to align
     #   - align_target: list of 3 float, vector on the cell surface to align. Normally z=0    
     # OUPUT
     #   - new_payload: mol3D class, rotation of payload
    new_payload = mol3D()
    new_payload.copymol3D(payload)
    align_chord =vecdiff(new_payload.getAtom(cand_ind).coords(),new_payload.getAtom(align_ind).coords())
    align_chord[2] = 0 # project into X-Y
    normal_vect = numpy.cross(align_chord,align_target)
    rotate_angle = vecangle(align_chord,align_target) + angle
    print('my angle is ' + str(rotate_angle) + ' nv is ' +str(normal_vect))
    new_payload = rotate_around_axis(new_payload,new_payload.getAtom(cand_ind).coords(),normal_vect,rotate_angle)
    return new_payload
##########################################

def slab_module_supervisor(args,rootdir):
    ###################################
    ###################################
    ############# INPUT ###############
    ######### Default values #########
    ### Invocation
    slab_gen = False 
    place_on_slab = False 
    ### Required Input: slab generation
    #unit_cell = False
    #cell_vector = False
    ## OR
    cif_path = False
    duplication_vector =False 
    ## OR
    slab_size = False
    # optional_input
    miller_index  = False
    ## Required Input: placement
    #target_molecule =  False 
    align_distance_method = False
    # options are "physisorption","chemisorption","custom"
    align_dist = False # use in conjunction with "custom" above
    ## Optionial Input: placement
    align_method = 'center'
    # other options: 'center','staggered', 'alignpair'
    # for alignpair only:
    surface_atom_type = False 
    object_align = False
    num_surface_atoms = 1
    num_placements = 1 
    coverage = False
    multi_placement_centering = 0.95
    # for surface rotation:
    control_angle = False
    angle_control_partner = False
    angle_surface_axis = [1,1]

    # duplication
    duplicate = False

    ###### Now attempt input ####
    import_success = True
    emsg = list()
    multi_placement_centering_overide = False
    miller_flag  = False
    if (args.slab_gen): #0
        slab_gen = True
    if (args.unit_cell): #1 
        unit_cell = mol3D()
        # test if the unit cell is a .xyz file
        try:
            ext = os.path.splitext(args.unit_cell)[1]
            if (ext == '.xyz'):
                unit_cell.readfromxyz(args.unit_cell)
            elif (ext == '.mol'):
                unit_cell.OBmol = unit_cell.getOBmol(args.unit_cell)
                unit_cell.convert2mol3D()
        except:
            emsg.append('Unable to import unit cell at  ' + str(args.unit_cell))
            import_success == False
    if (args.cell_vector): #2
        cell_vector = args.cell_vector
    if (args.cif_path): #3
        cif_path = args.cif_path
    if (args.duplication_vector): #4
        duplication_vector = args.duplication_vector
    if (args.slab_size): #5
        slab_size = args.slab_size
    if (args.miller_index): #6
        miller_index = args.miller_index
        miller_flag = True
     # ## parse slab options
    if (args.place_on_slab): #0
        place_on_slab = True
    if (args.target_molecule): #1 
        target_molecule = mol3D()
        # test if the unit cell is a .xyz file
        ext = os.path.splitext(args.target_molecule)[1]
        try:
            ext = os.path.splitext(args.target_molecule)[1]
            if (ext == '.xyz'):
               target_molecule.readfromxyz(args.target_molecule)
            elif (ext == '.mol'):
               target_molecule.OBmol = unit_cell.getOBmol(args.target_molecule)
               target_molecule.convert2mol3D()
        except:
            emsg.append('Unable to import target at  ' + str(args.target_molecule))
            import_success = False
    if (args.align_distance_method): #2
        align_distance_method = args.align_distance_method
    if (args.align_dist): #3    
        align_dist = args.align_dist
    if (args.align_method): #4
        align_method = args.align_method
    if (args.object_align): #5
       object_align = args.object_align
    if (args.surface_atom_type):#6
       surface_atom_type = args.surface_atom_type
    if (args.num_surface_atoms): #7
       num_surface_atoms = args.num_surface_atoms
    if (args.num_placements): #8
       num_placements = args.num_placements
    if (args.coverage):#9
       coverage = args.coverage
    if (args.multi_placement_centering):#10
       multi_placement_centering = args.multi_placement_centering
       multi_placement_centering_overide= True
    if (args.control_angle):#11
       control_angle = args.control_angle
    if (args.angle_control_partner): #12
       angle_control_partner = args.angle_control_partner
    if (args.angle_surface_axis): #13
       angle_surface_axis = args.angle_surface_axis
       print('ang_surf_axis  '  +str(angle_surface_axis))
    if (args.duplicate):#14
       duplicate = True

    ### check inputs
    if not import_success:
        print(emsg)
        return emsg
    if num_placements >1 and not multi_placement_centering_overide:
        multi_placement_centering = 1 # reccomended for multiple placments

    if not slab_gen and not place_on_slab:
        emsg.append('Slab builder module not enabled, placement mode not enabled - no action taken ')
        print(emsg)
        return emsg
    if place_on_slab and not target_molecule:
        emsg.append('Placement requested, but no object given. Skipping')
        print(emsg)
    if place_on_slab and not align_dist and (align_distance_method =="custom"):
        emsg.append('No placement distance given, defaulting to covalent radii')
        print(emsg)
    if place_on_slab and align_dist and not align_distance_method:
        print("using custom align distance of " + str(align_dist))
        align_distance_method = "custom"

    ## resolve align distance
    if align_distance_method == "chemisorption":
        globs = globalvars()
        if align_method == "alignpair":
            if surface_atom_type in globs.elementsbynum():
                surf_rad = globs.amass()[surface_atom_type][2]
            if object_align in globs.elementsbynum():
                obj_rad = globs.amass()[object_align][2]
            else:
                obj_rad = globs.amass()[globs.elementsbynum()[object_align[0]]]
        align_dist = obj_rad + surf_rad
        print('Chemisorption align distance set to  ' + str(align_dist))


    ## Main calls
    if slab_gen:
        passivate = True
        if cif_path:
            try:
                unit_cell,cell_vector = import_from_cif(cif_path)
            except:
                emsg.append('unable to import cif at ' + str(cif_path))
                return emsg
        if miller_flag:
                v1,v2,v3,angle,u = cut_cell_to_index(unit_cell,cell_vector,miller_index)
                cell_vector = [v1,v2,v3]  # change basis of cell to reflect cut, will rotate after gen
                print('cell vector is now ')
                print(cell_vector)
        if slab_size:
            max_dims = find_extents_cv(cell_vector)
            print('max dims are' + str(max_dims))
            duplication_vector = [int(numpy.ceil(slab_size[i]/max_dims[i])) for i in [0,1,2]]
        print('\n cell vector is '  + str(cell_vector))
        print('\n')
        print('duplication vector is  '+  str(duplication_vector))
        print('\n')
        acell = duplication_vector[0]
        bcell = duplication_vector[1]
        ccell = duplication_vector[2]
        if miller_flag:
                duplication_vector[2] += 4 #enusre enough layers to get to height
        super_cell = unit_to_super(unit_cell,cell_vector,duplication_vector)
        if miller_flag:
                super_cell = rotate_around_axis(super_cell,[0,0,0],u,angle)
                old_cv =  [PointRotateAxis(u,[0,0,0],list(i),angle) for i in cell_vector]
                duplication_vector[2] -= 4
        super_cell_vector = [[i*duplication_vector[0] for i in cell_vector[0]],
                         [i*duplication_vector[1] for i in cell_vector[1]],
                         [i*duplication_vector[2] for i in cell_vector[2]]]
        if miller_flag:
            old_cell_vector = [[i*duplication_vector[0] for i in old_cv[0]],
                             [i*duplication_vector[1] for i in old_cv[1]],
                             [i*duplication_vector[2] for i in old_cv[2]]]

        super_cell_dim = find_extents(super_cell)
        if miller_flag:
                super_cell= shave_under_layer(super_cell)
                super_cell= shave_under_layer(super_cell)
                super_cell = zero_z(super_cell) 
                if not slab_size:
                        extents = find_extents_cv(super_cell_vector)
                        target_size = extents[2]
                        while super_cell_dim[2] > 1.1*slab_size[2]:
                                print('slab is too thick, shaving...')
                                super_cell = shave_surface_layer(super_cell)
                                super_cell_dim = find_extents(super_cell)
        if slab_size:
            while super_cell_dim[2] > 1.1*slab_size[2]:
                print('slab is too thick, shaving...')
                super_cell = shave_surface_layer(super_cell)
                super_cell_dim = find_extents(super_cell)
        if passivate:
            pass
        if not os.path.exists(rootdir + 'slab'):
                os.makedirs(rootdir + 'slab')
        super_cell.writexyz(rootdir + 'slab/super' +''.join( [str(i) for i in duplication_vector])+'.xyz')
        print ('\n Created a ' + str(acell)+'x'+str(bcell)+'x' + str(ccell)+' supercell.\n')
        points = [[1,1],[2,1],[0,1],[0,0],[0.5,0.5],[0,2]]
        concave_hull(points,0.1)
    elif not slab_gen: #placement only, skip slabbing!
        super_cell = unit_cell
        super_cell_vector = cell_vector

    if place_on_slab:
        if control_angle:
            print('control angle on')
            print(angle_surface_axis)
            angle_surface_axis.append(0)
            print(angle_surface_axis)
        loaded_cell =  molecule_placement_supervisor(super_cell,super_cell_vector,target_molecule,
                                                 align_method,object_align,align_dist,surface_atom_type,
                                                 control_angle = control_angle, align_ind = angle_control_partner, align_axis = angle_surface_axis,
                                                 duplicate = duplicate, number_of_placements = num_placements, coverage = coverage,
                                                 weighting_method = 'linear' ,weight = 0, masklength = num_surface_atoms)
        super_duper_cell = unit_to_super(loaded_cell,old_cell_vector,[2,2,1])
       
        if not os.path.exists(rootdir + 'loaded_slab'):
                os.makedirs(rootdir + 'loaded_slab')
        loaded_cell.writexyz(rootdir + 'loaded_slab/loaded.xyz')
        super_duper_cell.writexyz(rootdir + 'loaded_slab/SD.xyz')



