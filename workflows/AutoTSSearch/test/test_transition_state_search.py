import sys
import os
import time


base_dir = os.getcwd()# Path to directory where the desired complex .xyz file is located
sys.path.append(os.path.expanduser("~/molSimplify/molSimplify/Classes/"))
sys.path.append(os.path.expanduser("~/molSimplify/molSimplify/Scripts/"))
sys.path.append(os.path.expanduser("~/ts/core/"))
sys.path.append(os.path.expanduser(f"{base_dir}"))
from mol3D import mol3D
from workflow import ts_optimization_workflow_substrate

test3D = mol3D()
test3D.readfromxyz(f"{base_dir}/fe_4_1_Se2S2C12H24_4ba342db217dc92c7e4e3e6bd1ef22cf_0_Nnegbenzene_-1_oxo_-2-2.xyz")
metal_idx = test3D.findMetal()

path_to_tmc = f"{base_dir}/fe_4_1_Se2S2C12H24_4ba342db217dc92c7e4e3e6bd1ef22cf_0_Nnegbenzene_-1_oxo_-2-2.xyz"
charge = 1
spinmult = 1
bonding_atom = 0 # atom index of the substrate bonding atom
bonding_site = test3D.findMetal()[0] # index of atom in tmc to serve as bonding site
#cluster='custom'
path_to_substrate = f"{base_dir}/n2o.xyz"
#custom_cluster_path = f"{base_dir}/cluster_config.py"
cluster='sge'
ts_optimization_workflow_substrate(path_to_tmc, charge, spinmult, bonding_atom,
                                   bonding_site, cluster, path_to_substrate,
                                   custom_cluster_path = custom_cluster_path)

