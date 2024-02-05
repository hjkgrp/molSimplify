
import numpy as np
import json
from importlib_resources import files as resource_files
from molSimplify.Classes.mol3D import mol3D


def get_percentile_csd_geometrics(geometrics_csd, geodict, geotype, maxdent,
                                  metrics=['oct_angle_devi_max', 'max_del_sig_angle',
                                           'dist_del_all', 'dist_del_all_relative'],
                                  path2metric="molSimplify/Informatics/geometrics_csd.json"):
    '''
    Obtain the percentile rank of a geometric dict compared to the entire CSD complexes.
    geometrics_csd: dict, {"geotype": "metric": [each complex's metric]}. If not specified as a dict, it will be loaded by path2metric.
    geodict: dict, geometric dict for a complex. Obtained by IsStructure().
    geotype: str, type of geometry
    metrics: list, a list of geometric considered.
    path2metric: str, the molSimplify path to load geometrics_csd if geometrics_csd is not specified
    '''
    jsonpath = resource_files("molSimplify").joinpath(path2metric)
    if not isinstance(geometrics_csd, dict):
        print("loading csd geometrics...")
        with open(jsonpath, "r") as f:
            geometrics_csd = json.load(f)
    percentile_dict = {}
    for k in metrics:
        percentile_dict[k] = [round(geodict[k], 2),
                              round(sum(np.abs(geometrics_csd[geotype]["all"][k]) < geodict[k]) / float(len(geometrics_csd[geotype]["all"][k])) * 100),
                              round(sum(np.abs(geometrics_csd[geotype][str(maxdent)][k]) < geodict[k]) / float(len(geometrics_csd[geotype][str(maxdent)][k])) * 100 + 1e-4)]
    return percentile_dict


def get_percentile_from_mol2(mol2string,
                             geometrics_csd,
                             metrics=['oct_angle_devi_max', 'max_del_sig_angle',
                                      'dist_del_all', 'dist_del_all_relative'],
                             path2metric="molSimplify/Informatics/geometrics_csd.json"):
    '''
    Get the geometric percentile rank given a mol2string.
    mol2string: str, str in the mol2 file
    geometrics_csd: dict, {"geotype": "metric": [each complex's metric]}. If not specified as a dict, it will be loaded by path2metric.
    metrics: list, a list of geometric considered.
    path2metric: str, the molSimplify path to load geometrics_csd if geometrics_csd is not specified
    '''
    mol = mol3D()
    mol.readfrommol2(mol2string, readstring=True)
    eqsym, maxdent, ligdents, homoleptic, ligsymmetry = mol.get_symmetry_denticity()
    results = mol.get_geometry_type()
    geotype = results['geometry']
    if geotype in ["sandwich", "edge"]:
        print("cannot deal with sandwich or edge compounds now.")
        d = {}
        for k in metrics:
            d[k] = False
        return d
    return get_percentile_csd_geometrics(geometrics_csd=geometrics_csd, geodict=results['summary'][geotype], geotype=geotype, maxdent=maxdent, metrics=metrics, path2metric=path2metric)
