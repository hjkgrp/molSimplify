# Written by Tim Ioannidis for HJK Group
# Dpt of Chemical Engineering, MIT

##########################################################
############# This script performs a lot of ##############
######### different operations and manipulates ###########
################### atoms in 3D space ####################
##########################################################
import sys
import copy
from numpy import arccos, cross, dot, pi
from numpy import sin, cos, mat, array, arctan2
from numpy.linalg import det, svd
from math import pi ,sin, cos, sqrt
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.atom3D import atom3D


######################
### Euclidean norm ###
######################
def norm(u):
    # INPUT
    #   - u: n-element list
    # OUTPUT
    #   - d: Euclidean norm
    d = 0.0
    for u0 in u:
        d += (u0*u0)
    d = sqrt(d)
    return d
    
#################
### normalize ###
#################
def normalize(u):
    # INPUT
    #   - u: n-element list
    # OUTPUT
    #   - un: normalized vector
    d = norm(u)
    un = []
    if d > 1.0e-13:
        un.append(u/d)
    return un

#########################################
### Euclidean distance between points ###
#########################################
def distance(R1,R2):
    # INPUT
    #   - R1: 3-element list representing point 1
    #   - R2: 3-element list representing point 2
    # OUTPUT
    #   - d: Euclidean distance
    dx = R1[0] - R2[0] 
    dy = R1[1] - R2[1] 
    dz = R1[2] - R2[2] 
    d = sqrt(dx**2+dy**2+dz**2)
    return d

##########################################
### calculates difference vector r1-r2 ###
##########################################
def vecdiff(r1,r2):
    # INPUT
    #   - r1: list representing vector r1
    #   - r2: list representing vector r2
    #   - Rp: reference point through axis
    # OUTPUT
    #   - dr: list with element-wise difference r1-r2
    dr = [a-b for a,b in zip(r1,r2)]
    return dr
    
##########################################
### calculates midpoint between r1-r2  ###
##########################################
def midpt(r1,r2):
    # INPUT
    #   - r1: list representing vector r1
    #   - r2: list representing vector r2
    #   - Rp: reference point through axis
    # OUTPUT
    #   - dr: list with element-wise difference r1-r2
    m = [0.5*(a+b) for a,b in zip(r1,r2)]
    return m   
    
#########################################
### Checks if 3 points are collinear ###
#########################################
def checkcolinear(R1,R2,R3):
    # INPUT
    #   - R1: 3-element list representing point 1
    #   - R2: 3-element list representing point 2
    #   - R3: 3-element list representing point 3
    dr1 = vecdiff(R2,R1)
    dr2 = vecdiff(R1,R3)
    dd = cross(array(dr1),array(dr2))
    if norm(dd) < 1.e-01:
        return True
    else:
        return False
        
#####################################
### Checks if 4 points are planar ###
#####################################
def checkplanar(R1,R2,R3,R4):
    # INPUT
    #   - R1: 3-element list representing point 1
    #   - R2: 3-element list representing point 2
    #   - R3: 3-element list representing point 3
    #   - R4: 3-element list representing point 4
    r31 = vecdiff(R3,R1)
    r21 = vecdiff(R2,R1)
    r43 = vecdiff(R4,R3)
    cr0 = cross(array(r21),array(r43))
    dd = dot(r31,cr0)
    if abs(dd) < 1.e-1:
        return True
    else:
        return False

###############################################
### calculates angle between vectors r1, r2 ###
###############################################
def vecangle(r1,r2):
    # INPUT
    #   - r1: list representing vector r1
    #   - r2: list representing vector r2
    # OUTPUT
    #   - theta: angle between vectors in degrees
    # angle between r10 and r21
    if(norm(r2)*norm(r1) > 1e-16):
        theta = 180*arccos(dot(r2,r1)/(norm(r2)*norm(r1)))/pi
    else:
        theta = 0.0
    return theta

################################################
########## gets point in line on a  ############
############ predefined distance ###############
################################################
def getPointu(Rr, dist, u):
    # INPUT
    #   - Rr: coordinates of reference point
    #   - dist: final distance
    #   - u: direction
    # OUTPUT
    #   - P: final point
    # get float bond length
    bl = float(dist)
    # get unit vector through line r = r0 + t*u
    t = bl/norm(u) # get t as t=bl/norm(r1-r0)
    # get point
    P = [0,0,0]
    P[0] = t*u[0]+Rr[0]
    P[1] = t*u[1]+Rr[1]
    P[2] = t*u[2]+Rr[2]
    return P

##################################################
### gets perpendicular vector to plane r10,r21 ###
##### and the angle between the two vectors  #####
##################################################
def rotation_params(r0,r1,r2):
    # INPUT
    #   - r0: 3-d point 0
    #   - r1: 3-d point 1
    #   - r2: 3-d point 2
    # OUTPUT
    #   - theta: angle between r10=r1-r0 and r21=r2-r1
    #   - u: pependicular vector to plane r10 x r21
    r10 = [a-b for a,b in zip(r1,r0)]
    r21 = [a-b for a,b in zip(r2,r1)]
    # angle between r10 and r21
    if(norm(r21)*norm(r10) > 1e-16):
        theta = 180*arccos(dot(r21,r10)/(norm(r21)*norm(r10)))/pi
    else:
        theta = 0.0
    # get normal vector to plane r0 r1 r2
    u = cross(r21,r10)
    # check for collinear case
    if norm(u) < 1e-16:
        # pick random perpendicular vector
        if (abs(r21[0]) > 1e-16):
            u = [(-r21[1]-r21[2])/r21[0],1,1]
        elif (abs(r21[1]) > 1e-16):
            u = [1,(-r21[0]-r21[2])/r21[1],1]
        elif (abs(r21[2]) > 1e-16):
            u = [1,1,(-r21[0]-r21[1])/r21[2]]
    return theta,u

###################################################
######### aligns mol0 with respect to mol1 ########
########### to minimize the RMSD value ############
########### using the Kabsch algorithm ############
## http://en.wikipedia.org/wiki/Kabsch_algorithm ##
###################################################
def kabsch(mol0,mol1):
    # INPUT
    #   - mol0: molecule to be aligned
    #   - mol1: reference molecule
    # OUTPUT
    #   - mol0: aligned molecule
    # get coordinates and matrices P,Q
    P, Q = [],[]
    for atom0,atom1 in zip(mol0.GetAtoms(),mol1.GetAtoms()):
        P.append(atom0.coords())
        Q.append(atom1,coords())
    # Computation of the covariance matrix
    C = dot(transpose(P), Q)
    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = svd(C)
    d = (det(V) * det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:,-1] = -V[:,-1]
    # Create Rotation matrix U
    U = dot(V, W)
    # Rotate P
    P = dot(P, U)
    # write back coordinates
    for i,atom in enumerate(mol0.GetAtoms()):
        atom.setcoords(P[i])
    return mol0
    
#################################################
####### contains roto-translation matrix ########
####### for reflection of r through plane #######
########## with normal u and point rp ###########
#################################################
def ReflectPlane(u,r,Rp):
    # INPUT
    #   - u: normal vector to plane [ux,uy,uz]
    #   - Rp: reference point [x,y,z]
    #   - r: point to be reflected [x,y,z]
    # OUTPUT
    #   - rn: reflected point [x,y,z]
    un = norm(u)
    if (un > 1e-16):
        u[0] = u[0]/un
        u[1] = u[1]/un
        u[2] = u[2]/un
    # construct augmented vector rr = [r;1]
    d = -u[0]*Rp[0]-u[1]*Rp[1]-u[2]*Rp[2]
    # reflection matrix
    R=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    rn = [0,0,0]
    R[0][0] = 1-2*u[0]*u[0]
    R[0][1] = -2*u[0]*u[1] 
    R[0][2] = -2*u[0]*u[2] 
    R[0][3] = -2*u[0]*d
    R[1][0] = -2*u[1]*u[0] 
    R[1][1] = 1-2*u[1]*u[1] 
    R[1][2] = -2*u[1]*u[2] 
    R[1][3] = -2*u[1]*d
    R[2][0] = -2*u[2]*u[0]
    R[2][1] = -2*u[1]*u[2]
    R[2][2] = 1-2*u[2]*u[2] 
    R[2][3] = -2*u[2]*d
    R[3][3] = 1 
    # get new point
    rn[0] = R[0][0]*r[0]+R[0][1]*r[1]+R[0][2]*r[2] + R[0][3] 
    rn[1] = R[1][0]*r[0]+R[1][1]*r[1]+R[1][2]*r[2] + R[1][3] 
    rn[2] = R[2][0]*r[0]+R[2][1]*r[1]+R[2][2]*r[2] + R[2][3] 
    return rn

#################################################
####### contains roto-translation matrix ########
######### for rotation of r about axis u ########
######## through point rp by angle theta ########
#################################################
def PointRotateAxis(u,rp,r,theta):
    # INPUT
    #   - u: direction vector of axis [ux,uy,uz]
    #   - rp: reference point [x,y,z]
    #   - r: point to be reflected [x,y,z]
    #   - theta: angle of rotation in degrees
    # OUTPUT
    #   - rn: reflected point [x,y,z]
    # construct augmented vector rr = [r;1]
    rr = r
    rr.append(1)
    # rotation matrix about arbitrary line through rp
    R=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    rn = [0,0,0]
    R[0][0] = cos(theta)+u[0]**2*(1-cos(theta))
    R[0][1] = u[0]*u[1]*(1-cos(theta))-u[2]*sin(theta)
    R[0][2] = u[0]*u[2]*(1-cos(theta))+u[1]*sin(theta)
    R[0][3] = (rp[0]*(u[1]**2+u[2]**2)-u[0]*(rp[1]*u[1]+rp[2]*u[2]))*(1-cos(theta))
    R[0][3] += (rp[1]*u[2]-rp[2]*u[1])*sin(theta)
    R[1][0] = u[1]*u[0]*(1-cos(theta))+u[2]*sin(theta)
    R[1][1] = cos(theta)+u[1]**2*(1-cos(theta))
    R[1][2] = u[1]*u[2]*(1-cos(theta))-u[0]*sin(theta)
    R[1][3] = (rp[1]*(u[0]**2+u[2]**2)-u[1]*(rp[0]*u[0]+rp[2]*u[2]))*(1-cos(theta))
    R[1][3] += (rp[2]*u[0]-rp[0]*u[2])*sin(theta)
    R[2][0] = u[2]*u[0]*(1-cos(theta))-u[1]*sin(theta)
    R[2][1] = u[2]*u[1]*(1-cos(theta))+u[0]*sin(theta)
    R[2][2] = cos(theta)+u[2]**2*(1-cos(theta))
    R[2][3] = (rp[2]*(u[0]**2+u[1]**2)-u[2]*(rp[0]*u[0]+rp[1]*u[1]))*(1-cos(theta))
    R[2][3] += (rp[0]*u[1]-rp[1]*u[0])*sin(theta)
    R[3][3] = 1
    # get new point
    rn[0] = R[0][0]*r[0]+R[0][1]*r[1]+R[0][2]*r[2] + R[0][3]
    rn[1] = R[1][0]*r[0]+R[1][1]*r[1]+R[1][2]*r[2] + R[1][3]
    rn[2] = R[2][0]*r[0]+R[2][1]*r[1]+R[2][2]*r[2] + R[2][3]
    return rn
    
##################################################
####### converts to spherical coordinates ########
######### translates/rotates and converts ########
################ back to cartesian ###############
##################################################
def PointTranslateSph(Rp,p0,D):    
    # INPUT
    #   - Rp: origin [x,y,z]
    #   - p0: point to be translated [x,y,z]
    #   - D: [radial distance, polar phi, azimuthal theta]
    # OUTPUT
    #   - p: translated point [x,y,z] 
    # translate to origin
    ps=[0,0,0]
    ps[0] = p0[0] - Rp[0]
    ps[1] = p0[1] - Rp[1]
    ps[2] = p0[2] - Rp[2]  
    # get initial spherical coords
    r0 = norm(ps) 
    if (r0 < 1e-16):
        phi0 = 0.5*pi
        theta0 = 0 
    else :
        phi0 = arccos(ps[2]/r0) # z/r
        theta0 = arctan2(ps[1],ps[0]) # y/x
    # get new point
    p = [0,0,0]
    p[0] = (D[0])*sin(phi0+D[2])*cos(theta0+D[1]) + Rp[0]
    p[1] = (D[0])*sin(phi0+D[2])*sin(theta0+D[1]) + Rp[1]
    p[2] = (D[0])*cos(phi0+D[2]) + Rp[2]
    return p

##################################################
####### converts to spherical coordinates ########
######### translates/rotates and converts ########
################ back to cartesian ###############
##################################################
def PointTranslatetoPSph(Rp,p0,D):    
    # INPUT
    #   - Rp: reference point [x,y,z]
    #   - p0: point to be translated [x,y,z]
    #   - D: [radial distance, polar theta, azimuthal phi]
    # OUTPUT
    #   - p: translated point [x,y,z] 
    # translate to origin
    ps=[0,0,0]
    ps[0] = p0[0] - Rp[0]
    ps[1] = p0[1] - Rp[1]
    ps[2] = p0[2] - Rp[2]  
    # get current spherical coords
    r0 = norm(ps) 
    if (r0 < 1e-16):
        phi0 = 0.5*pi
        theta0 = 0 
    else :
        phi0 = arccos(ps[2]/r0) #changed sign TIM IOANNIDIS
        theta0 = arctan2(ps[1],ps[0]) # atan doesn't work due to signs
    # get translation vector
    p = [0,0,0]
    p[0] = D[0]*sin(phi0+D[2])*cos(theta0+D[1]) 
    p[1] = D[0]*sin(phi0+D[2])*sin(theta0+D[1]) 
    p[2] = D[0]*cos(phi0+D[2]) 
    return p

#############################################
############# contains rotation #############
########## matrix about x,y,z axes ##########
#############################################
def PointRotateSph(Rp,p0,D):
    # INPUT
    #   - Rp: reference point [x,y,z]
    #   - p0: point to be rotated [x,y,z]
    #   - D: [theta-x, theta-y, theta-z]
    # OUTPUT
    #   - p: rotated point
    # translate to origin (reference)
    ps=[0,0,0]
    ps[0] = p0[0] - Rp[0]
    ps[1] = p0[1] - Rp[1]
    ps[2] = p0[2] - Rp[2]  
    # build 3D rotation matrices about x,y,z axes
    Mx=[[1, 0, 0],[0, cos(D[0]), -sin(D[0])],[0, sin(D[0]), cos(D[0])]]
    My=[[cos(D[1]), 0, sin(D[1])],[0, 1, 0],[-sin(D[1]), 0, cos(D[1])]]
    Mz=[[cos(D[2]), -sin(D[2]), 0],[sin(D[2]), cos(D[2]), 0],[0, 0, 1]]
    # get full rotation matrix
    M = array(mat(Mx)*mat(My)*mat(Mz))
    p=[0.0, 0.0, 0.0]
    # rotate atom and translate it back from origin
    p[0] = M[0][0]*ps[0] + M[0][1]*ps[1] + M[0][2]*ps[2] + Rp[0]
    p[1] = M[1][0]*ps[0] + M[1][1]*ps[1] + M[1][2]*ps[2] + Rp[1]
    p[2] = M[2][0]*ps[0] + M[2][1]*ps[1] + M[2][2]*ps[2] + Rp[2]
    return p
    
############################################
##### reflects molecule through plane ######
##### with normal u and a point rp #########
############################################
def reflect_through_plane(mol,u,Rp):
    # INPUT
    #   - mol: molecule to be manipulated
    #   - u: normal vector to plane
    #   - rp: reference point on plane
    # OUTPUT
    #   - mol: reflected molecule
    un = norm(u)
    if (un > 1e-16):
        u[0] = u[0]/un
        u[1] = u[1]/un
        u[2] = u[2]/un
    for atom in mol.atoms:
        # Get new point after rotation
        Rt = ReflectPlane(u,atom.coords(),Rp)
        atom.setcoords(Rt)
    return mol

###############################################
########## rotates molecule about #############
####### arbitrary axis with direction #########
############# u through point rp ##############
################ by angle theta ###############
###############################################
def rotate_around_axis(mol,Rp,u,theta):
    # INPUT
    #   - mol: molecule to be manipulated
    #   - Rp: reference point [x,y,z]
    #   - u: direction vector [vx,vy,vz]
    #   - theta: angle to rotate in degrees
    # OUTPUT
    #   - mol: translated molecule
    un = norm(u)
    theta = (theta/180.0)*pi
    if (un > 1e-16):
        u[0] = u[0]/un
        u[1] = u[1]/un
        u[2] = u[2]/un
    for atom in mol.atoms:
        # Get new point after rotation
        Rt = PointRotateAxis(u,Rp,atom.coords(),theta)
        atom.setcoords(Rt)
    return mol

#########################################################
########## sets distance of atom in molecule ############
############## from reference point #####################
#########################################################
def setPdistance(mol, Rr, Rp, bond):
    # INPUT
    #   - mol: molecule to be manipulated
    #   - Rr: coordinates of atom in molecule
    #   - Rp: reference point [x,y,z] not in molecule
    #   - bond: final bond length between Rr, Rp
    # OUTPUT
    #   - mol: translated molecule
    # get float bond length
    bl = float(bond)
    # get center of mass
    # get unit vector through line r = r0 + t*u
    u = [a-b for a,b in zip(Rr,Rp)]
    t = bl/norm(u) # get t as t=bl/norm(r1-r0)
    # get shift for centermass
    dxyz = [0,0,0]
    dxyz[0] = Rp[0]+t*u[0]-Rr[0]
    dxyz[1] = Rp[1]+t*u[1]-Rr[1]
    dxyz[2] = Rp[2]+t*u[2]-Rr[2]
    # translate molecule
    mol.translate(dxyz)
    return mol
    
#########################################################
########## sets distance of atom in molecule ############
########## from reference point on axis u ###############
#########################################################
def setPdistanceu(mol, Rr, Rp, bond, u):
    # INPUT
    #   - mol: molecule to be manipulated
    #   - Rr: coordinates of atom in molecule
    #   - Rp: reference point [x,y,z] not in molecule
    #   - bond: final bond length between Rr, Rp
    # OUTPUT
    #   - mol: translated molecule
    # get float bond length
    bl = float(bond)
    # get unit vector through line r = r0 + t*u
    t = bl/norm(u) # get t as t=bl/norm(r1-r0)
    # get shift for centermass
    dxyz = [0,0,0]
    dxyz[0] = Rp[0]+t*u[0]-Rr[0]
    dxyz[1] = Rp[1]+t*u[1]-Rr[1]
    dxyz[2] = Rp[2]+t*u[2]-Rr[2]
    # translate molecule
    mol.translate(dxyz)
    return mol
    
#################################################
########## sets distance of molecule ############
############## from reference point #############
################################################
def setcmdistance(mol, Rp, bond):
    # INPUT
    #   - mol: molecule to be manipulated
    #   - Rp: reference point [x,y,z]
    #   - bond: final bond length between Rp, center of mass
    # OUTPUT
    #   - mol: translated molecule
    # get float bond length
    bl = float(bond)
    # get center of mass
    cm = mol.centermass()
    # get unit vector through line r = r0 + t*u
    u = [a-b for a,b in zip(cm,Rp)]
    t = bl/norm(u) # get t as t=bl/norm(r1-r0)
    # get shift for centermass
    dxyz = [0,0,0]
    dxyz[0] = Rp[0]+t*u[0]-cm[0]
    dxyz[1] = Rp[1]+t*u[1]-cm[1]
    dxyz[2] = Rp[2]+t*u[2]-cm[2]
    # translate molecule
    mol.translate(dxyz)
    return mol

##############################################
######## translates/rotates molecule #########
########### about reference point ############
##############################################
def protate(mol, Rr, D):
    # INPUT
    #   - mol: molecule to be manipulated
    #   - Rr: reference point [x,y,z]
    #   - D: [radial distance, polar theta, azimuthal phi]
    # OUTPUT
    #   - mol: translated molecule
    # rotate/translate about reference point
    # convert to rad
    D[0] = float(D[0])
    D[1] = (float(D[1])/180.0)*pi
    D[2] = (float(D[2])/180.0)*pi
    # rotate/translate about reference point
    # get center of mass
    pmc = mol.centermass()
    # get translation vector that corresponds to new coords
    Rt = PointTranslateSph(Rr,pmc,D)
    # translate molecule
    mol.translate(Rt)
    return mol


##############################################
######## translates/rotates molecule #########
########### about reference point ############
##############################################
def protateref(mol, Rr, Rref, D):
    # INPUT
    #   - mol: molecule to be manipulated
    #   - Rr: reference point [x,y,z]
    #   - Rref: reference point on molecule [x,y,z]
    #   - D: [radial distance, polar theta, azimuthal phi]
    # OUTPUT
    #   - mol: translated molecule
    # rotate/translate about reference point
    # convert to rad
    D[0] = float(D[0])
    D[1] = (float(D[1])/180.0)*pi
    D[2] = (float(D[2])/180.0)*pi
    # rotate/translate about reference point
    # get translation vector that corresponds to new coords
    Rt = PointTranslateSph(Rr,Rref,D)
    # translate molecule
    mol.translate(Rt)
    return mol

########################################
########## rotates molecule ############
######## about center of mass ##########
########################################
def cmrotate(mol, D):
    # INPUT
    #   - mol: molecule to be manipulated
    #   - D: [theta-x, theta-y, theta-z]
    # OUTPUT
    #   - mol: translated molecule
    # convert to rad
    D[0] = (float(D[0])/180.0)*pi
    D[1] = (float(D[1])/180.0)*pi
    D[2] = (float(D[2])/180.0)*pi
    # perform rotation
    pmc = mol.centermass()
    for atom in mol.atoms:
        # Get new point after rotation
        Rt = PointRotateSph(pmc,atom.coords(),D)
        atom.setcoords(Rt)
    return mol

########################################
########## rotates molecule ############
######## about center of mass ##########
########################################
def rotateRef(mol, Ref, D):
    # INPUT
    #   - mol: molecule to be manipulated
    #   - Ref: reference point on the molecule
    #   - D: [theta-x, theta-y, theta-z]
    # OUTPUT
    #   - mol: translated molecule
    # convert to rad
    D[0] = (float(D[0])/180.0)*pi
    D[1] = (float(D[1])/180.0)*pi
    D[2] = (float(D[2])/180.0)*pi
    # perform rotation
    pmc = mol.centermass()
    for atom in mol.atoms:
        # Get new point after rotation
        Rt = PointRotateSph(Ref,atom.coords(),D)
        atom.setcoords(Rt)
    return mol
    
###########################################
########## translates molecule ############
########### and aligns to axis ############
###########################################
def aligntoaxis(mol,Rr,Rp,u):
    # INPUT
    #   - Rr: point to be aligned
    #   - mol: molecule to be manipulated
    #   - Rp: reference point through axis
    #   - u: target axis for alignment
    # OUTPUT
    #   - mol: aligned molecule
    # get current distance
    d0 = distance(Rp,Rr)
    # normalize u
    t =d0/norm(u) # get t as t=bl/norm(r1-r0)
    # get shift for point
    dxyz = [0,0,0]
    dxyz[0] = Rp[0]+t*u[0]-Rr[0]
    dxyz[1] = Rp[1]+t*u[1]-Rr[1]
    dxyz[2] = Rp[2]+t*u[2]-Rr[2]
    # translate molecule
    mol.translate(dxyz)
    return mol

########################################
########## rotates molecule ############
########## and aligns to axis ##########
########################################
def aligntoaxis2(mol,Rr,Rp,u,d):
    # INPUT
    #   - Rr: point to be aligned
    #   - mol: molecule to be manipulated
    #   - Rp: reference point through axis
    #   - u: target axis for alignment
    #   - d: final distance
    # OUTPUT
    #   - mol: aligned molecule
    # normalize u
    t =d/norm(u) # get t as t=bl/norm(r1-r0)
    # get shift for point
    dxyz = [0,0,0]
    dxyz[0] = Rp[0]+t*u[0]-Rr[0]
    dxyz[1] = Rp[1]+t*u[1]-Rr[1]
    dxyz[2] = Rp[2]+t*u[2]-Rr[2]
    # translate molecule
    mol.translate(dxyz)
    return mol
    
###########################################
########## translates point ############
########### and aligns to axis ############
###########################################
def alignPtoaxis(Rr,Rp,u,d):
    # INPUT
    #   - Rr: point to be aligned
    #   - Rp: reference point through axis
    #   - u: target axis for alignment
    #   - d: final distance target
    # OUTPUT
    #   - dxyz: final coordinates
    # normalize u
    t =d/norm(u) # get t as t=bl/norm(r1-r0)
    # get shift for point
    dxyz = [0,0,0]
    dxyz[0] = Rp[0]+t*u[0]
    dxyz[1] = Rp[1]+t*u[1]
    dxyz[2] = Rp[2]+t*u[2]
    return dxyz
        
########################################
########## rotates molecule ############
######## about reference point #########
########################################
def pmrotate(mol, Rp, D):
    # INPUT
    #   - mol: molecule to be manipulated
    #   - Rp: reference point [x,y,z]
    #   - D: [theta-x, theta-y, theta-z]
    # OUTPUT
    #   - mol: translated molecule
    # convert to rad
    D[0] = (float(D[0])/180.0)*pi
    D[1] = (float(D[1])/180.0)*pi
    D[2] = (float(D[2])/180.0)*pi
    # perform rotation
    for atom in mol.atoms:
        # Get new point after rotation
        Rt = PointRotateSph(Rp,atom.coords(),D)
        atom.setcoords(Rt)
    return mol   
