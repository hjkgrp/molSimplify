# Written by JP Janet for HJK Group
# Dpt of Chemical Engineering, MIT

##########################################################
######## This script contains a neural network  ##########
#####  trained on octahedral metal-ligand          #######
########   bond distances and spin propensity  ###########
##########################################################


## import 
from pybrain.structure import FeedForwardNetwork,TanhLayer,LinearLayer,BiasUnit,SigmoidLayer, FullConnection
import numpy as np
import csv
from pkg_resources import resource_filename, Requirement
from molSimplify.Classes.globalvars import *
import sys,os
def simple_network_builder(layers,partial_path):
    n = FeedForwardNetwork()
    ## create the network
    inlayer = LinearLayer(layers[0], name = "In")
    hidden_one = TanhLayer(layers[1], name = "Hidden 1")
    hidden_two = TanhLayer(layers[2], name  ="Hidden 2")
    b1 = BiasUnit(name="Bias")
    output = LinearLayer(1,name = "Out")
    n.addInputModule(inlayer)
    n.addModule(hidden_one)
    n.addModule(hidden_two)
    n.addModule(b1)
    n.addOutputModule(output)
    in_to_one = FullConnection(inlayer,hidden_one)
    one_to_two = FullConnection(hidden_one,hidden_two)
    two_to_out = FullConnection(hidden_two,output)
    b1_to_one = FullConnection(b1,hidden_one)
    b2_to_two = FullConnection(b1,hidden_two)
    b3_to_output = FullConnection(b1,output)
    ### load weights and biases
    in_to_one._setParameters(np.array((csv_loader(partial_path + '_w1.csv'))))
    one_to_two._setParameters(np.array(csv_loader(partial_path + '_w2.csv')))
    two_to_out._setParameters(np.array(csv_loader(partial_path + '_w3.csv')))
    b1_to_one._setParameters(np.array(csv_loader(partial_path + '_b1.csv')))
    b2_to_two._setParameters(np.array(csv_loader(partial_path + '_b2.csv')))
    b3_to_output._setParameters(np.array(csv_loader(partial_path + '_b3.csv')))

    ### connect the network topology
    n.addConnection(in_to_one)
    n.addConnection(one_to_two)
    n.addConnection(two_to_out)
#    n.sortModules()

    n.addConnection(b1_to_one)
    n.addConnection(b2_to_two)
    n.addConnection(b3_to_output)

    ### finalize network object
    n.sortModules()

    return n





def csv_loader(path):
    #print('in csv loader')
    path_to_file = resource_filename(Requirement.parse("molSimplify"),"molSimplify/python_nn/" + path)
    with open(path_to_file,'r') as csvfile:
        csv_lines = csv.reader(csvfile,delimiter= ',')
        ret_list = list()
        for lines in csv_lines:
            this_line = [float(a) for a in lines]
            ret_list += this_line
    return ret_list
def matrix_loader(path,rownames=False):
    ## loads matrix with rowname option
    if rownames:
        path_to_file = resource_filename(Requirement.parse("molSimplify"),"molSimplify/python_nn/" + path)
        with open(path_to_file, "r") as f:
            csv_lines = list(csv.reader(f))
            row_names = [row[0] for row in csv_lines]
            mat = [row[1:] for row in csv_lines]
        return mat,row_names
    else:
        path_to_file = resource_filename(Requirement.parse("molSimplify"),"molSimplify/python_nn/" + path)
        with open(path_to_file,'r') as csvfile:
            csv_lines = csv.reader(csvfile,delimiter= ',')
            mat = [a for a in csv_lines]
        return mat


#n = network_builder([25,50,51],"nn_split")
def simple_splitting_ann(excitation):
    globs=globalvars()
    path_to_file = resource_filename(Requirement.parse("molSimplify"),"molSimplify/python_nn/" + "ms_split")
    #print('path to ANN data: ',path_to_file)
    n = simple_network_builder([25,50,50],"ms_split") 
    excitation,sp_center,sp_shift = excitation_standardizer(excitation,'split')
    print(excitation)
    print('center is ' + str(sp_center))
    print('scale is '+ str(sp_shift))
    print(excitation)
    result = n.activate(excitation)

    
    #print('result is ' + str(result))
    result = (result*sp_shift) + sp_center
    #print('result is ' + str(result))
    return result
def simple_slope_ann(slope_excitation):
    globs=globalvars()
    path_to_file = resource_filename(Requirement.parse("molSimplify"),"molSimplify/python_nn/" + "ms_slope")
    #print('path to ANN data: ',path_to_file)
    n = simple_network_builder([24,50,50],"ms_slope") ## no alpha value
    #print(slope_excitation)
    slope_excitation,sl_center,sl_shift = excitation_standardizer(slope_excitation,'slope')
    #print(slope_excitation)
    result = n.activate(slope_excitation)
   # print('result is ' + str(result))
   # print('center is ' + str(sl_center) + ' shift  '+ str(sl_shift))
    result = (result*sl_shift) + sl_center
    #print('result is ' + str(result))
    return result
def simple_ls_ann(excitation):
    n = simple_network_builder([25,50,50],"ms_ls")
    slope_excitation,ls_center,ls_shift = excitation_standardizer(excitation,'ls')
    result = n.activate(excitation)
    result = result*ls_shift + ls_center
    return result
def simple_hs_ann(excitation):
    n = simple_network_builder([25,50,50],"ms_hs")
    slope_excitation,hs_center,hs_shift = excitation_standardizer(excitation,'hs')
    result = n.activate(excitation)
    result = result*hs_shift + hs_center
    return result

def excitation_standardizer(excitation,tag):
    ## this function implements 
    ## a scale-and-center
    ## type of normalization
    ## that may help predictions 
    ## currently testing for
    ## splitting and slope only 

    centers = csv_loader(tag+"_center.csv")
    shifts = csv_loader(tag+"_scale.csv")
    descriptor_centers = np.array(centers[1:])
    descriptor_shifts = np.array(shifts[1:])
    sp_center =  centers[0]
    sp_shift =  shifts[0]   
    excitation = np.array(excitation)
    mean = np.mean(excitation)
    sd = np.std(excitation,ddof=0)
    excitation = (excitation - descriptor_centers)
    excitation = np.divide(excitation,descriptor_shifts)
    mean = np.mean(excitation)
    sd = np.std(excitation,ddof=0)
    return(excitation,sp_center,sp_shift )

def find_eu_dist(excitation):
    # returns euclidean distance to nearest trainning 
    # vector in desciptor space
    mat,rownames = matrix_loader('train_data.csv',rownames=True)
    train_mat = np.array(mat,dtype='float64')
    min_dist = 1000
    excitation,sp_center,sp_shift = excitation_standardizer(excitation,'split')
    for i,rows in enumerate(train_mat):
        np.subtract(rows,np.array(excitation))
        this_dist = np.linalg.norm(np.subtract(rows,np.array(excitation)))/3
        if this_dist < min_dist:
            min_dist = this_dist
            best_row = rownames[i]
            min_row = rows
    print('min dist is ' +str(min_dist))
    print(min_row)
    return min_dist,best_row

