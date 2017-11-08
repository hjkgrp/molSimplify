# Written by JP Janet for HJK Group
# Dpt of Chemical Engineering, MIT
import os, sys, copy
import glob, re, math, random, string, numpy
from math import pi
from scipy.spatial import Delaunay, ConvexHull
from molSimplify.Scripts.geometry import *
from molSimplify.Classes.atom3D import *
from molSimplify.Classes.mol3D import*
from molSimplify.Classes.globalvars import globalvars
from operator import add
from molSimplify.Scripts.periodic_QE import *
from molSimplify.Scripts.cellbuilder_tools import *
###############################
def d_fix(unit_cell,cell_vector):
    fixed_cell = mol3D()
    fixed_cell.copymol3D(unit_cell)
    mind = 100
    for i,atoms in enumerate(fixed_cell.getAtoms()):
        this_distance = mdistance(atoms.coords(),[0,0,0])
        print("min d is " + str(mind))
        print("atom at " + str(atoms.coords()) )
        if this_distance < mind:
            mind = this_distance
            minatom = atoms
            minind = i
            print('this was saved')
        print("\n\n")
    a = cell_vector[0]

    b = cell_vector[1]
    c = cell_vector[2]
    dx = c[0]
    dy = c[1]
    dz = c[2]
    trans_vect =(dx,dy,dz)
    new_atom =atom3D(minatom.symbol(),minatom.coords())
    new_atom.translate(trans_vect)
    fixed_cell.addAtom(new_atom)
    fixed_cell.deleteatoms([minind])
    return fixed_cell
#####################################################
def cut_cell_to_index(unit_cell,cell_vector,miller_index):
    ## determine the plane:
        cut_cell = mol3D()
        cut_cell.copymol3D(unit_cell)
        h,k,l = miller_index
        #print('h,k,l',str(h) + ' ' + str(k) +  ' ' +  str(l))
        disc,p,q = xgcd(k,l)
        #print('p,q',str(p) + ' ' + str(q))
        cell_vector = numpy.array(cell_vector)
        k1= numpy.dot(p*(k*cell_vector[0]-h*cell_vector[1]) + q*(l*cell_vector[0] - h*cell_vector[2]),l*cell_vector[1] - k*cell_vector[2])
        k2= numpy.dot(l*(k*cell_vector[0]-h*cell_vector[1]) -k*(l*cell_vector[0] - h*cell_vector[2]),l*cell_vector[1] - k*cell_vector[2])
        #print('k1',k1)
        #print('k2',k2)
        tol = 1e-3
        if abs(k2)> tol:
            c = -1*int(round(k1/k2))
            p,q = p+c*l,q - c*k
        v1 = p*numpy.array(k*cell_vector[0]-h*cell_vector[1]) + q*numpy.array(l*cell_vector[0] - h*cell_vector[2])
        v2 = numpy.array(l*cell_vector[1]-k*cell_vector[2])
        disc,a,b = xgcd(p*k + q*l,h)
        v3 = numpy.array(b*cell_vector[0] + a*p*cell_vector[1]  + a*q*cell_vector[2])

        non_zero_indices  = list()
        zero_indices  = list()
        for i in [0,1,2]:
            if not (miller_index[i] == 0):
                non_zero_indices.append(i)
            else:
                zero_indices.append(i)

        print('nz ind',non_zero_indices)
        if len(non_zero_indices)==3:
            zint = 1/(miller_index[2]*cell_vector[2][2])
            yint = 1/(miller_index[1]*cell_vector[1][1])
            xint = 1/(miller_index[0]*cell_vector[0][0])
            #w = [0,0,0]
            # w[2] = zint
            #w[1] = -w[2]/yint
            #w[0] = -w[2]/xint
            plane_normal = numpy.cross(v1,v2)
        elif len(non_zero_indices)==2:
      #      print('\n\n\n\n')
       #     print(cell_vector)
       #     print("\n\n")
            vec1 = [0,0,0]
            vec1[non_zero_indices[0]] = cell_vector[non_zero_indices[0]][non_zero_indices[0]]
            vec2 = [0,0,0]
            vec2[non_zero_indices[1]] = cell_vector[non_zero_indices[1]][non_zero_indices[1]]
            vec3 = [0,0,0]
            vec3[zero_indices[0]] = cell_vector[zero_indices[0]][zero_indices[0]]
         #   print('vec1',vec1)
         #   print('vec2',vec2)
         #   print('vec3',vec3)
            plane_normal = numpy.cross(v1,v2)
        elif len(non_zero_indices) == 1:

            v1 = cell_vector[zero_indices[0]]
            v2 = cell_vector[zero_indices[1]]
            v3 = cell_vector[non_zero_indices[0]]
            plane_normal = numpy.cross(v1,v2)
        print(miller_index)
        print('plane normal is ', plane_normal)
        angle = vecangle(plane_normal,[0,0,1])
        u =  numpy.cross(plane_normal,[0,0,1])
        return v1,v2,v3,angle,u

##################################
def concave_hull(points,alpha):
    ## points should be tuples
    de = Delaunay(points)
    for i in de.simplices:
        tmp = []
        j = [points[c] for c in i]
    #    print(i)
    #    print(j)
    #print(de)

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
    #print('cand list is ' + str(cand_list))
    #print('surface_coord_list  ' + str(surface_coord_list))
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
def choose_nearest_neighbour(target_site,avail_sites_dict,occupied_sites_dict,super_cell,super_cell_vector,debug=False):
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
    weight = 0 #favours adjaceny to point over distance from other occupied sites
    # get the nearest site to target
    score = 100000 # weighted assessment, lower is better
    avail_sites_list = avail_sites_dict.keys()
    occupied_sites_list = occupied_sites_dict.keys()
    if debug:
         print('********** choosing nearest neighbour sites ********')
    if (len(avail_sites_list) > 1 ): # more than 1 option, pick closest to target site 
        for indices in avail_sites_list:
                if debug:
		    print('checking site  ' + str(indices) + ' at ' + str(avail_sites_dict[indices]) + ' relative to ' + str(target_site))
                distance_to_target =  distance(target_site,avail_sites_dict[indices]) # NOT the torus distance - must be two cells in one unit
                distance_to_nearest_occupied =0 
                for neighbours in occupied_sites_list:
                    # get distance to nearest neighbour
                    distance_to_nearest_occupied = max(distance_2d_torus(avail_sites_dict[indices],occupied_sites_dict[neighbours],extents),distance_to_nearest_occupied)
                this_score =(1 - weight)*distance_to_target -  weight*distance_to_nearest_occupied
                if debug:
		    print('this score is  '+ str(this_score))
                if this_score < score:
                    score = this_score
                    if debug:
                         print('New lowest score  at ' +str(indices)+ ' has  score =' + str(score) + '\n')
                    nn_site = indices
    elif (len(avail_sites_list) == 1):
        nn_site = avail_sites_list[0]
    else:
        emsg = ('error: no free site is possible')
        print(emsg)
    if debug:
         print('**** final choice  = ' + str(avail_sites_dict[nn_site]) + ' at ' + str(score))
    return nn_site
#####################################
def choose_best_site(avail_sites_dict,occupied_sites_dict,centroid,super_cell,super_cell_vector,weight = 0.5, method = 'linear',debug=False):

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
    skipalign = 0
    score = 100000 # weighted assessment, lower is better
    avail_sites_list = avail_sites_dict.keys()
    random.shuffle(avail_sites_list)
    occupied_sites_list = occupied_sites_dict.keys()
    if debug:	
        print('extents = ' + str(extents))
	print('centroid is at '+ str(centroid))
 	print('ac sites dict  = '+str(avail_sites_dict))
        print('oc sites list  = '+str(occupied_sites_list))
        print('ac sites list  = '+str(avail_sites_list))
        print('weight = ' +str(weight) )
    if (len(avail_sites_list) > 1 ): # more than 1 option, pick closest to center of plane 
        for indices in avail_sites_list:
                #distance_to_center =  distance_2d_torus(centroid,avail_sites_dict[indices],extents)
                distance_to_center =  distance(centroid,avail_sites_dict[indices])

                distance_to_nearest_occupied = 1000
                for neighbours in occupied_sites_list:
                    # get distance to nearest neighbour
		    if debug:
                    	print('Neighbour:' + str(occupied_sites_dict[neighbours]) + ' point is at ' + str(avail_sites_dict[indices]))  
                    distance_to_nearest_occupied = min(distance_2d_torus(avail_sites_dict[indices],occupied_sites_dict[neighbours],extents),distance_to_nearest_occupied)
		if debug:
                	print('dist to nearest = '   + str(distance_to_nearest_occupied) + ' at ' + str(avail_sites_dict[indices]) )
                if (method == 'linear'):
                    this_score =(1 - weight)*distance_to_center -  weight*distance_to_nearest_occupied
                elif (method == 'log'):
                    this_score = (1 - weight)*abs(numpy.log(distance_to_center)) - weight*abs(numpy.log(distance_to_nearest_occupied))
		if debug:
                	print('the score here : '+ str(this_score) + ' oc sites  ' + str(occupied_sites_dict) )
                if this_score < score:
                    score = this_score
                    target_site = indices
	            if debug:
                    	print('target site is ' + str(indices) + ' at ' +str(super_cell.getAtom(indices).coords()))  
    elif (len(avail_sites_list) == 1):
        target_site = avail_sites_list[0]
    else:
        emsg = ('error: no free site is possible')
        print(emsg) 
    print('choosing site ' + str(target_site) + ' at  '+ str(avail_sites_dict[target_site]) + ' score: '+ str(score) + ' oc sites ' + str(occupied_sites_dict) + '\n')
    return target_site
#####################################
def align_payload_to_multi_site(payload,surface_coord_list,cand_list,bind_dist,debug=False):
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
    payload_coord = center_of_sym([new_payload.getAtom(i).coords() for i in cand_list ])
    surface_coord = center_of_sym(surface_coord_list)

    vec1 =  vecdiff(surface_coord,payload.centersym())
    vec2 =  vecdiff(payload_coord,new_payload.centersym())
    rotate_angle = vecangle(vec1,vec2)
    theta,u = rotation_params(payload_coord,new_payload.centersym(),surface_coord)
    if debug:
        print('\n vec1 is '+ str(vec1))
 	print('vec2 is '+ str(vec2) + '\n')
    	print(cand_list)
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
        print('starting rotation for coplanar case')
        for rotate_angle in range(-100,100):
            this_payload = mol3D()
            this_payload.copymol3D(final_payload)
            this_payload = rotate_around_axis(this_payload,this_payload.centersym(),new_u,float(rotate_angle)/10) #fine grained check
            this_cost = multialign_objective_function(this_payload,surface_coord_list,cand_list,bind_dist)
            if (this_cost < (cost)):
                #print('current cost = ' + str(this_cost) + ', the max is ' + str(cost))
		if debug:
                	print('accepting rotate at theta  = ' + str(rotate_angle))
                cost = this_cost
                final_payload = this_payload
	print('placement complete')
    return final_payload

##################################
def combine_multi_aligned_payload_with_cell(super_cell,super_cell_vector,payload,cand_list,surface_coord_list,bind_dist,duplicate = False,control_angle = False,debug=False):
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
    if debug:
   	print('trying to align mol: ' + str(cand_list))
    	print('and sites ' + str(surface_coord_list))

    ##########################

    extents = find_extents_cv(super_cell_vector)
    # the generalized distance descriptors
    payload_coord = center_of_sym([new_payload.getAtom(i).coords() for i in cand_list ])
    surface_coord = center_of_sym(surface_coord_list)

    vec =  vecdiff(surface_coord,payload_coord)
    cost =  multialign_objective_function(new_payload,surface_coord_list,cand_list,bind_dist)
    final_payload = new_payload
    distances_list = []
    for indices in enumerate(cand_list):
        v1=(surface_coord_list[indices[0]])
        v2=final_payload.getAtom(int(indices[1])).coords()
        distances_list.append((distance(v1,v2)))
    if debug:
        print('\n\n Target distance was  ' + str(bind_dist)+', achieved ' + str(distances_list))
        print('the cm of sites: ' + str(surface_coord))
        print('the cm of mol: ' + str(payload_coord))
        print('cost before rotation =' + str(cost))
    distances_list = []
    for indices in enumerate(cand_list):
        v1=(surface_coord_list[indices[0]])
        v2=final_payload.getAtom(int(indices[1])).coords()
        distances_list.append((distance(v1,v2)))
    if debug:
        print('\n\n Target distance was  ' + str(bind_dist)+', achieved ' + str(distances_list))

    if not control_angle:
        print('starting align rotation')
        for rotate_angle in range(0,360):
            this_payload = mol3D()
            this_payload.copymol3D(new_payload)
            this_payload = rotate_around_axis(this_payload,this_payload.centersym(),vec,rotate_angle)
            this_cost = multialign_objective_function(this_payload,surface_coord_list,cand_list,bind_dist)
            if (this_cost < (cost)):
               cost = this_cost
               final_payload = this_payload
    if debug:
    	print('cost after rotation =' + str(cost))
    distances_list = []

    for indices in enumerate(cand_list):
        v1=(surface_coord_list[indices[0]])
        v2=final_payload.getAtom(int(indices[1])).coords()
        distances_list.append((distance(v1,v2)))
    if debug:
        print('\n\n Target distance was  ' + str(bind_dist)+', achieved ' + str(distances_list))

    # lower into positon
    # step size:
    factor = 0.20
    deltaZ =factor*(distance(payload_coord,surface_coord)-bind_dist)
    this_step_accepted = True
    num_bad_steps= 0 
    break_flag = False
    maxits = 250
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
        if debug:
            print('cost  = ' + str(this_cost) +'/' +str(cost)+ '  i = ' + str(its) + '  dz =  ' + str(deltaZ) + ' dist  '+ str(this_dist) + ' b step  = '+ str(num_bad_steps) + ' nxt dz = ' + str(this_deltaZ))
        if (this_cost < (cost)) and (this_dist > 0.75) and (deltaZ > 1e-4):
            if debug:
                print('accepting down shift at i  = ' + str(its))
            cost = this_cost
            del final_payload
            final_payload = mol3D()
            if ( this_payload.mindist(combined_cell) < 1.5):
		if debug:
                    print('ff on at iteration '  + str(its))
                distorted_payload = mol3D()
                distorted_payload.copymol3D(this_payload)
                print('Warning, a force-field relaxation is in progress. For large molecules (Natoms > 50), this may take a few minutes. Please be patient.')
                distorted_payload = force_field_relax_with_slab(super_cell,this_payload,cand_list,its)
                if debug:
                    print(this_payload.getAtom(0).symbol() + ' at ' + str(this_payload.getAtom(cand_list[0]).coords()) + 'target at ' + str(surface_coord_list[0]))
                    print(distorted_payload.getAtom(0).symbol() + ' at ' + str(distorted_payload.getAtom(cand_list[0]).coords()) + 'target at ' + str(surface_coord_list[0]))
                final_payload.copymol3D(distorted_payload)
                if debug:
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
    if debug:
        print('\n\n exit after '  + str(its) + ' iterations')
        print('target distance  = ' + str(bind_dist) + ', average deviation =  ' + str(sqrt(cost)/len(cand_list)))
    distances_list = []
    for indices in enumerate(cand_list):
        v1=(surface_coord_list[indices[0]])
        v2=final_payload.getAtom(int(indices[1])).coords()
        distances_list.append((distance(v1,v2)))
    if debug:
        print(' Target distance was  ' + str(bind_dist)+', achieved ' + str(distances_list))
    min_dist = final_payload.mindist(combined_cell)
   # now, rotate to maximize spacing, based on mask length
    rotate_on = False
    if len(cand_list) == 1:
      rotate_on = True
    elif len(cand_list) ==2:
        vec = vecdiff(final_payload.getAtom(cand_list[0]).coords(),final_payload.getAtom(cand_list[1]).coords())
        rotate_on = True
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
             this_dist = min(periodic_mindist(this_payload,combined_cell,extents),periodic_selfdist(this_payload,extents),this_payload.mindist(combined_cell))
             if (this_dist > (min_dist + 1e-3)):
		 if debug:
                     print('current dist = ' + str(this_dist) + ', the max is ' + str(min_dist))
                     print('accepting rotate at theta  = ' + str(rotate_angle))
                 min_dist = this_dist
                 final_payload = this_payload
    if len(cand_list) > 1:
    # now, distort molecule based on FF to optimize bond length
        print('\n begining distortion ')
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
	         if debug:
                 	print('accepting distort')

    distances_list = []
    for indices in enumerate(cand_list):
        v1=(surface_coord_list[indices[0]])
        v2=final_payload.getAtom(int(indices[1])).coords()
        distances_list.append((distance(v1,v2)))

    print('Target distance was  ' + str(bind_dist)+', achieved ' + str(distances_list))


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
    print('minimum inter-cell adsorbate self-atom distance is ' + str(min_intercell_d))
    combined_cell.combine(final_payload)
    return combined_cell
###################################
def molecule_placement_supervisor(super_cell,super_cell_vector,target_molecule,method,target_atom_type,align_dist,surface_atom_type = False,control_angle = False,align_ind = False, align_axis = False,
                                  duplicate = False, number_of_placements = 1, coverage = False,weighting_method = 'linear',weight = 0.5,masklength = 1, surface_atom_ind = False,debug=False):
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
    if (method == 'alignpair') and not (surface_atom_type or surface_atom_ind):
       print('Must provide surface binding atom type to use alignpair')
       print(' using centered placemented instead')
       method = 'center'
    print('\n')
    print('the method is',method)
    if (method == 'alignpair'): # get all vaccancies 
        avail_sites_dict = dict()
        occupied_sites_dict = dict()
        if not surface_atom_ind:
            if debug:
            	print('surface_atom_type',surface_atom_type)
            avail_sites_list =  find_all_surface_atoms(super_cell,tol=0.75,type_of_atom = surface_atom_type)
            avail_sites_dict = dict()
            for indices in avail_sites_list:
                avail_sites_dict[indices] = super_cell.getAtom(indices).coords()
            occupied_sites_dict = dict()
            # calculate max number of sites that need to be filled 
            max_sites =int(numpy.floor(float(len(avail_sites_dict.keys()))/masklength))
            if coverage:
                number_of_placements = int(numpy.ceil(max_sites*coverage))
                print('Coverage requested = ' + str(coverage))
	if debug:
        	print('masklengh is ' + str(masklength))
        if surface_atom_ind:
            print('using surface_atom_ind' + str(surfacce_atom_ind))
            for indices in surface_atom_ind:
                avail_sites_dict[indices] = super_cell.getAtom(indices).coords()
            avail_sites_list = [i for i in surface_atom_ind]
            if coverage:
                print('cannot use coverage with surface_atom_ind')
                coverage = False

    ######## prepare and allocate
    loaded_cell = mol3D()
    loaded_cell.copymol3D(super_cell)
    debug_cell = mol3D()
    debug_cell.copymol3D(loaded_cell)
    ####### begin actual work
    print('begining the placement loop')
    for placements in range(0,number_of_placements):
        sites_list = list() # list to hod all of the target sites on the surface
        if (method == 'center'):
            align_coord = centered_align_coord(super_cell_vector)
            sites_list.append(align_coord)
        elif (method == 'staggered'):
            align_coord = staggered2_align_coord(super_cell)
            sites_list.append(align_coord)

        elif (method == 'alignpair'):
            best_site = choose_best_site(avail_sites_dict,occupied_sites_dict,centered_align_coord(super_cell_vector),super_cell,super_cell_vector,weight,weighting_method,debug=debug)
            align_coord = super_cell.getAtom(best_site).coords()
            occupied_sites_dict[best_site] = avail_sites_dict.pop(best_site) # this transfers the site to occupied
            sites_list.append(align_coord)
            if masklength != 1: # this is if we need multiple sites
                mask_target = align_coord
                for iterates in range(1,masklength):
		    if debug:
                    	print('in loop for ' + str(iterates))
                    nn_site = choose_nearest_neighbour(mask_target,avail_sites_dict,occupied_sites_dict,super_cell,super_cell_vector,debug=False)
                    align_coord = super_cell.getAtom(nn_site).coords()
                    sites_list.append(align_coord)
                    occupied_sites_dict[nn_site] = avail_sites_dict.pop(nn_site) # this transfers the site to occupied
		    mask_target = center_of_sym(sites_list)
                align_coord = center_of_sym(sites_list)
		if debug:
			print('oc sites chosen are  = '+str(occupied_sites_dict.keys()))

        else:
            emsg = 'unkown method of molecule placement ' + method
            print(emsg)
            return emsg
        print('Target for align is ' + str(align_coord))
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
	    if debug:
            	print('cand _ind = ' + str(cand_ind))
            cand_list = [cand_ind]
        else:
            cand_ind = target_atom_type 
            cand_list = [(int(i)-1) for i in cand_ind]
	    if debug:
            	print('target molecule mask on ' + str(target_atom_type))
            	print('candidate list is ' +str(cand_list))
        ######## rotate for optimal approach
        payload = align_payload_to_multi_site(payload,sites_list,cand_list,align_dist,debug) # align
        if control_angle:
            print('begining controlled rotation, targeting angle ' + str(control_angle) + ' to  line ' + str(align_axis))
            print('aligning '+ payload.getAtom(cand_ind).symbol() + ' with ' + payload.getAtom(align_ind).symbol())
            payload = axes_angle_align(payload,cand_ind,align_ind,align_axis,control_angle)
	if debug:
            print('payload cysm '+ str(payload.centersym()))
        #######################################
        temp_pay2 = mol3D()
        temp_pay2.copymol3D(payload)
        temp_pay2.translate([0,0,-5])
        debug_cell.combine(temp_pay2)
        ####### lower payload to distance, rotate to avoid conflicr
        loaded_cell = combine_multi_aligned_payload_with_cell(loaded_cell,super_cell_vector,payload,cand_list,sites_list,align_dist,duplicate,control_angle,debug)

        ########################
        temp_pay3 = mol3D()
        temp_pay3.copymol3D(payload)
        debug_cell.combine(temp_pay3)
    #    debug_cell.writexyz('db3.xyz')
        print('number of atoms = ' + str(loaded_cell.natoms))
        #print("\n")
    ###### run tests
    overlap_flag = loaded_cell.sanitycheck(0)
    min_dist = loaded_cell.mindistmol()
    if (number_of_placements > 1): 
        print('preparing ' + str(number_of_placements) + ' placements ')
        effectvie_coverage =  float(number_of_placements)/float(max_sites)
        print('giving effectvie coverage of ' + str(effectvie_coverage) + '\n')
    print('Is there overalp? ' + str(overlap_flag))
    min_intercell_d = closest_torus_point(payload,extents)

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
    print('******** cell builder on ********')
    ###################################
    ###################################
    ############# INPUT ###############
    ######### Default values #########
    ### Invocation
    slab_gen = False 
    place_on_slab = False 
    ### Required Input: slab generation
    unit_cell = False
    cell_vector = False
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
    angle_surface_axis = False
    
    # for non-orthogonal unit cells
    ortho_cell = False

    # duplication
    duplicate = False
    
    # debug
    debug = False

    # passivate 
    passivate =  False

    # freeze layers
    freeze = False

    # expose a certain atom type
    expose_type  = False

    # shave extra layers 
    shave_extra_layers  = False
    

    # overwrite surface_atom_ind
    surface_atom_ind = False
    
    ###### Now attempt input ####
    import_success = True
    emsg = list()
    multi_placement_centering_overide = False
    miller_flag  = False
    if (args.slab_gen): #0
        slab_gen = True
    if (args.unit_cell): #1 
	print('importing unit cell')
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
    if (args.freeze): #7
        freeze = args.freeze
    if (args.debug): #8
        debug = True
    if (args.expose_type): #9
        expose_type = args.expose_type
    if (args.shave_extra_layers): #10
        shave_extra_layers = args.shave_extra_layers
     # ## parse placement options
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
    if (args.object_align): #4
       object_align = args.object_align
    if (args.align_method): #5
        align_method = args.align_method
    if (args.surface_atom_type):#6
       surface_atom_type = args.surface_atom_type
    if (args.surface_atom_ind):#7
        surface_atom_ind = args.surface_atom_ind

    if (args.num_surface_atoms): #8
       num_surface_atoms = args.num_surface_atoms
    if (args.num_placements): #9
       num_placements = args.num_placements
    if (args.coverage):#10
       coverage = args.coverage
    if (args.multi_placement_centering):#12
       multi_placement_centering = args.multi_placement_centering
       multi_placement_centering_overide= True
    if (args.control_angle):#13
       control_angle = args.control_angle
    if (args.angle_control_partner): #14
       angle_control_partner = args.angle_control_partner
    if (args.angle_surface_axis): #14
       angle_surface_axis = args.angle_surface_axis
       print('ang_surf_axis  '  +str(angle_surface_axis))
    if (args.duplicate):#15
       duplicate = True
        ### check inputs
    if slab_gen and not (slab_size or duplication_vector):
        emsg="Size of slab required (-slab_size or -duplication_vector)"
        print(emsg)
        return emsg 
    if slab_gen and not ((unit_cell and cell_vector) or cif_path):
        emsg="Unit cell info required! (-cif_path or -unit_cell and cell_vector)"
        print(emsg)
        return emsg 

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
    if place_on_slab and not align_dist and (align_distance_method !="chemisorption"):
        emsg.append('No placement distance given, defaulting to covalent radii')
	align_distance_method = "chemisorption"
        print(emsg)
    if place_on_slab and align_dist and not align_distance_method:
        print("using custom align distance of " + str(align_dist))
        align_distance_method = "custom"
    if num_placements >1 or coverage:
	weight = 1
    else:
	weight = 0.90

    #if args.target_atom_type:
    #    if not args.target_atom_type in elements:
    #        masklength = len(args.target_atom_type)
    #        print("Target masking with length  " +  str(masklength)) 
    #    else:
    #        masklength = 1
    #else:
    #    masklength = 1


    ## resolve align distance
    if align_distance_method == "chemisorption":
        globs = globalvars()
        if surface_atom_type in globs.elementsbynum():
        	surf_rad = globs.amass()[surface_atom_type][2]
        else:
		surf_rad = 1.5 
		print('unknown surface atom type, using 1.5A as distance')
       	if object_align in globs.elementsbynum():
        	obj_rad = globs.amass()[object_align][2]
        else:
		obj_rad = 1.0
		print('unknown object atom type, using 1.0A as distance')
        align_dist = obj_rad + surf_rad
        print('Chemisorption align distance set to  ' + str(align_dist))
    if miller_flag:
        non_zero_indices  = list()
        zero_indices  = list()
        for i in [0,1,2]:
            if not (miller_index[i] == 0):
                non_zero_indices.append(i)
            else:
                zero_indices.append(i)



    ## Main calls
    if slab_gen:
        print('Generating a new slab...')
        print(rootdir)
        if not os.path.exists(rootdir + 'slab'):
                os.makedirs(rootdir + 'slab')

    

        
        if cif_path:
                print('testing cif')
           # try:
                unit_cell,cell_vector = import_from_cif(cif_path)
                if debug:
                    print('cell vector from cif is')
                    print(cell_vector)
          #  except:
          #      emsg.append('unable to import cif at ' + str(cif_path))
          #      print(emsg)
          #      return emsg
                
        #### testing
        unit_cell.writexyz(rootdir + 'slab/before_COB.xyz')
        old_basis = cell_vector
        print('loaded')
        
        if miller_flag:
            print('miller index on '+ str(miller_index) + ' ' + str(miller_flag))
            point_coefficients = [get_basis_coefficients(at.coords(),cell_vector) for at in unit_cell.getAtoms()]
            point_coefficients = threshold_basis(point_coefficients,1E-6)
            print('coords in old UC')
            for j in point_coefficients:
                print(j)
            if debug:
                unit_cell.writexyz(rootdir + 'slab/step_0.xyz')
                print('\n\n')
                print('cell vector was ')
                print(cell_vector[0])
                print(cell_vector[1])
                print(cell_vector[2])
                print('\n**********************\n')
            v1,v2,v3,angle,u = cut_cell_to_index(unit_cell,cell_vector,miller_index)

            old_cv = cell_vector
            cell_vector = [v1,v2,v3]  # change basis of cell to reflect cut, will rotate after gen
            new_basis = [v1,v2,v3] 
            print('old basis:')
            print old_cv
            print('new basis:')
            print new_basis
            print('\n')
            point_coefficients = [get_basis_coefficients(at.coords(),new_basis) for at in unit_cell.getAtoms()]
            point_coefficients = threshold_basis(point_coefficients,1E-6)
            print('coords in transformed UC:' )
            print(point_coefficients)
            #for i in range(0,len(point_coefficients)):
            #    for j in [0,1,2]:
             #       if point_coefficients[i][j]<0:
             #           point_coefficients[i][j] += 1      
            new_coords = [evaluate_basis_coefficients(points,new_basis) for points in point_coefficients]
            for i,coords in enumerate(new_coords):
                unit_cell.getAtom(i).setcoords(coords)
            
            print('coords in final UC:' )
            print(point_coefficients)
        
        
        ## find out how many units to use
        max_dims = [numpy.linalg.norm(i) for i in cell_vector]
        print('vector norms of actual cell vector are' + str(max_dims))
        if slab_size:
            duplication_vector = [int(numpy.ceil(slab_size[i]/max_dims[i])) for i in [0,1,2]]
            print('duplication vector set to ' +str(duplication_vector))
            
        ## keep track of the enlarged cell vector for the slab:
        ext_duplication_vector = cell_vector
        ext_duplication_vector = [[i*duplication_vector[0] for i in ext_duplication_vector[0]],
                         [i*duplication_vector[1] for i in ext_duplication_vector[1]],
                         [i*duplication_vector[2] for i in ext_duplication_vector[2]]]
        
        if miller_index:
            duplication_vector[2] += 2 ## add some extra height to trim off 
                                       ## this is a hack to prevent bumpy bottoms
                                       ## when duplicating cell vectors were
                                       ## elements in the top an bottom layes
                                       ## are bonded/close
                                       
        ## perfrom duplication
        ####################################################################
        super_cell = unit_to_super(unit_cell,cell_vector,duplication_vector)
        ####################################################################
        
        if debug:
            super_cell.writexyz(rootdir + 'slab/after_enlargement.xyz') 
        

        if debug:
            print('ext_dup vector is: ')
            print(ext_duplication_vector[0])
            print(ext_duplication_vector[1])
            print(ext_duplication_vector[2])
        
        
        ## lower the cell into the xy plane
        #############################################################
        #############################################################
        if miller_index:
            print('rotating angle ' + str(angle) + ' around ' + str(u))##
            super_cell = rotate_around_axis(super_cell,[0,0,0],u,angle)##
            #decoy = rotate_around_axis(decoy,[0,0,0],u,angle)##
            cell_vector =  [PointRotateAxis(u,[0,0,0],list(i),numpy.pi*angle/(180)) for i in cell_vector]
            ext_duplication_vector =  [PointRotateAxis(u,[0,0,0],list(i),numpy.pi*angle/(180)) for i in ext_duplication_vector]
            # threshold:
            cell_vector=threshold_basis(cell_vector,1E-6)
            ext_duplication_vector = threshold_basis(ext_duplication_vector,1E-6)
            u_keep = 0 + u
            angle_keep  = 0+angle
        #############################################################
        
        if debug:
            print('cell vector is: ')
            print(cell_vector[0])
            print(cell_vector[1])
            print(cell_vector[2])
            print('ext_dup vector is: ')
            print(ext_duplication_vector[0])
            print(ext_duplication_vector[1])
            print(ext_duplication_vector[2])
            super_cell.writexyz(rootdir + 'slab/after_rotate.xyz') 
        
        if miller_index: ## get rid of the extra padding we added:
                  super_cell= shave_under_layer(super_cell)
                  super_cell= shave_under_layer(super_cell)
                  super_cell= shave_under_layer(super_cell)
                  super_cell= shave_surface_layer(super_cell)
                  super_cell= shave_surface_layer(super_cell)
                  super_cell=zero_z(super_cell)
        if debug:
            super_cell.writexyz(rootdir + 'slab/after_millercut.xyz')          
        
        ## check angle between v1 and x for aligining nicely
        angle = -1*vecangle(cell_vector[0],[1,0,0])
        if debug:
            print('x-axis angle is  ' + str(angle))
        if abs(angle) > 5 and False:
           print('angle is '+ str(angle))
           u = [0,0,1]
           print('aligning  with x-axis')
           print('rotating angle ' + str(angle) + ' around ' + str(u))
           super_cell = rotate_around_axis(super_cell,[0,0,0],u,angle)
           super_cell.writexyz(rootdir + 'slab/after_x_align.xyz')
           cell_vector =  [PointRotateAxis(u,[0,0,0],list(i),numpy.pi*angle/(180)) for i in cell_vector]
           ext_duplication_vector =  [PointRotateAxis(u,[0,0,0],list(i),numpy.pi*angle/(180)) for i in ext_duplication_vector]
           # threshold:
           cell_vector=threshold_basis(cell_vector,1E-6)
           ext_duplication_vector = threshold_basis(ext_duplication_vector,1E-6)
           
        stop_flag = False
        if slab_size:
            counter = 0
            while not stop_flag:
                counter += 1
                zmax = 0 
                for atoms in super_cell.getAtoms():
                    coords = atoms.coords()
                    if (coords[2] > zmax):
                        zmax = coords[2]
                if (zmax <= 1.1*slab_size[2]):
                        stop_flag = True
                else:
                    if debug:
                        print('cutting due to zmax')
                    super_cell= shave_surface_layer(super_cell)
                if counter > 20:
                        print('stopping after 10 cuts, zmax not obtained')
                        stop_flag = True
        if debug:
            super_cell.writexyz(rootdir + 'slab/after_size_control.xyz')   
        ## measure and recored slab vectors
        super_cell_dim = find_extents(super_cell)
        super_cell_vector = copy.copy(ext_duplication_vector)

        ## check if passivation needed
        if passivate:
            pass ## not implemented
        ## check if atoms should be frozen
        if freeze:
            if isinstance(freeze,int):
                print('freezing')
                super_cell = freeze_bottom_n_layers(super_cell,freeze)
            else:
                super_cell = freeze_bottom_n_layers(super_cell,1)
                
        ## check if a different surface atom should be exposed:
        if expose_type:
            super_cell = check_top_layer_correct(super_cell,expose_type)
        if shave_extra_layers:
            for i in range(0,int(shave_extra_layers)):
                print('shaving ' + str(shave_extra_layers) + ' layers')
                super_cell =shave_surface_layer(super_cell,TOL =1e-2)


        ## move in all negative positions
        #if all(super_cell_vector[0]) <= 0: ## all signs are the same:
        #    super_cell_vector[0] = [-1*i for i in super_cell_vector[0]]
        point_coefficients = [get_basis_coefficients(at.coords(),super_cell_vector) for at in super_cell.getAtoms()]
        #print('coords in final slab:' )
        #print(point_coefficients)
        point_coefficients = threshold_basis(point_coefficients,1E-6)
        #for i in range(0,len(point_coefficients)):
        #    for j in [0,1,2]:
        #        if point_coefficients[i][j]<0:
        #            point_coefficients[i][j] += 1
        #        if point_coefficients[i][j] >1:
        #            point_coefficients[i][j] -= 1
        new_coords = [evaluate_basis_coefficients(points,super_cell_vector) for points in point_coefficients]
        
        #for j,points in enumerate(point_coefficients):
            #if min(new_coords[i])<0:
                ### try shift one cv in each direction:
                #potential_new_point = []
            
        for i,coords in enumerate(new_coords):
            super_cell.getAtom(i).setcoords(coords)
        #point_coefficients = [get_basis_coefficients(at.coords(),super_cell_vector) for at in super_cell.getAtoms()]
        print('coords in final slab:' )
        print(point_coefficients)
        ######
        rest = super_cell.sanitycheck(silence = False)
        print('result of collision check is  ' + str(rest))
        
        #################################
        ## write slab output
        super_cell.writexyz(rootdir + 'slab/super' +''.join( [str(i) for i in duplication_vector])+'.xyz')
        print ('\n Created a supercell in ' + str(rootdir) + '\n')
        
        
        ## let's check if the periodicity is correct
        super_duper_cell = unit_to_super(super_cell,cell_vector,[2,2,1])
        super_duper_cell.writexyz(rootdir + 'slab/SD.xyz')
        
        ## get some vapourspace
        final_cv = copy.deepcopy(super_cell_vector)
        final_cv[2][2] = float(final_cv[2][2]) + 20  
        print('final cell vector, inc vapour space is :')
        print(final_cv)
        write_periodic_mol3d_to_qe(super_cell,final_cv,rootdir + 'slab/slab.in')
    elif not slab_gen: #placement only, skip slabbing!
        super_cell = unit_cell
        super_cell_vector = cell_vector

    if place_on_slab:
        if slab_gen:
            print('\n\n ************************ starting placement ***************** \n\n')
        if not slab_gen:
            print('\n\n ************************ placement on existing slab  ***************** \n\n')
            new_dup_vector = cell_vector
            super_cell_vector = cell_vector
            print('this supercell vector is:')
            print(super_cell_vector)
 
        if control_angle:
            print('control angle on')
            print(angle_surface_axis)
            angle_surface_axis.append(0)
            print(angle_surface_axis)
        loaded_cell =  molecule_placement_supervisor(super_cell,super_cell_vector,target_molecule,
                                                 align_method,object_align,align_dist,surface_atom_type,
                                                 control_angle = control_angle, align_ind = angle_control_partner, align_axis = angle_surface_axis,
                                                 duplicate = duplicate, number_of_placements = num_placements, coverage = coverage,
                                                 weighting_method = 'linear',weight=weight, masklength = num_surface_atoms, surface_atom_ind = surface_atom_ind,debug=debug)
        if not os.path.exists(rootdir + 'loaded_slab'):
                os.makedirs(rootdir + 'loaded_slab')
        if freeze and not slab_gen: #freezing happens at gen time
            if isinstance(freeze,int):
                print('freezing')
                loaded_cell = freeze_bottom_n_layers(loaded_cell,freeze)
            else:
                loaded_cell = freeze_bottom_n_layers(loaded_cell,1)


        loaded_cell.writexyz(rootdir + 'loaded_slab/loaded.xyz')
        super_duper_cell = unit_to_super(loaded_cell,new_dup_vector,[2,2,1])

        super_duper_cell.writexyz(rootdir + 'loaded_slab/SD.xyz')
        write_periodic_mol3d_to_qe(loaded_cell,new_dup_vector,rootdir + 'loaded_slab/loaded_slab.in')

  

