import sys
import os
import glob
import shutil
import time
sys.path.append(os.path.expanduser("~/molSimplify/molSimplify/Classes/"))
sys.path.append(os.path.expanduser("~/molSimplify/molSimplify/Scripts/"))
sys.path.append(os.path.expanduser("~/ts/core/"))
from orca_bl_scan import orca_bond_length_scan
from orca_opt import check_orca_gibbs_free_energy, orca_geometry_optimization, orca_transition_state_optimization, orca_single_point_energy
from terachem import terachem_geo_optimize
from structure_gen import intermediate_generation_substrate, find_terminal_ligand, intermediate_gen, find_oxo_n2_ligand, read_substrate_xyz, find_moieties
from io_utils import wait_avogadro_optimization
from mol3D import mol3D

def write_substrates_xyz(path_to_substrates_xyz, substrate):
    """
    Write the substrate .xyz files. These species are specific to the methane to methanol mechanism, molSimplify offers more substrates .xyz files at "molSimplify/Substrates"
    """
    xyz_dict = {
        "n2o": ["3", "", "O 0 0 -1.309", "N 0 0 0", "N 0 0 1.12"],
        "n2": ["2", "", "N 0 0 0", "N 0 0 1.12"],
        "ch4": ["5", "", "C 0 0 0", "H 0.6405128476 0.6405128476 0.6405128476", "H -0.6405128476 -0.6405128476 0.6405128476", "H -0.6405128476 0.6405128476 -0.6405128476", "H 0.6405128476 -0.6405128476 -0.6405128476"],
        "ch3": ["4", "", "C 0 0 0", "H 0.6405128476 0.6405128476 0.6405128476", "H -0.6405128476 -0.6405128476 0.6405128476", "H -0.6405128476 0.6405128476 -0.6405128476"],
        "ch3oh": ["6", "", "C 0 0 0", "H 0.6405128476 0.6405128476 0.6405128476", "H -0.6405128476 -0.6405128476 0.6405128476", "H -0.6405128476 0.6405128476 -0.6405128476", "O 0.7704864785 -0.8286749243 -0.8286749243", "H 1.7259703357 -0.6385103581 -0.6385103581"]
    }
    with open(f"{path_to_substrates_xyz}/{substrate}.xyz", "w") as f:
        f.write(f"XYZ coordinates for {substrate}\n")
        f.write("\n".join(xyz_dict[substrate]))

def ts_optimization_workflow_substrate(path_to_tmc_xyz, charge, spinmult, bonding_atom, bonding_site, cluster, path_to_substrate_xyz = None, mech_step= None, scan_type='formation', running_directory=None, custom_cluster_path=None):
    """
    Parameters:
    ----------

    path_to_tmc_xyz : str
        Path to the TMC .xyz file.
    charge : int
        Charge of the system.
    spinmult : int
        Spin multiplicity of the system.
    bonding_atom : str
        Atom involved in the bond index.
    bonding_site : str
        Site of the bond atom index.
    cluster : str
        Cluster system to use.
    path_to_substrate_xyz : str, optional
        Path to the substrate .xyz file.
    mech_step : str, optional
        Mechanistic step to consider, e.g., 'oxo', 'hat'.
    scan_type : str, optional
        Type of scan to perform, 'formation' or 'breaking'.
    running_directory : str, optional
        Directory to run the workflow in.
    custom_cluster_path : str, optional
        Path for the custom cluster system.
    Returns:
    -------
    None
    """
    base_dir = os.getcwd()# Path to directory where the desired complex .xyz file is located
    start_time = time.time()
    # Filenames
    orca_bl_name = "orca_bl_scan"
    terachem_opt_name = "terachem_geo"
    ts_guess_name = "TSguess"
    orca_ts_name= "orca_TS_opt"
    # Create intermediate
    if path_to_substrate_xyz:
        intermediate_generation_substrate(path_to_substrate_xyz, path_to_tmc_xyz, bonding_atom, bonding_site, outfile_name="substrate_bound.xyz")
    if mech_step:
        ind, terachem_opt_name = intermediate_gen(mech_step)
    wait_avogadro_optimization(f"{terachem_opt_name}_ff_guess.xyz")
    # Run Terachem geometry optimization
    ind = [bonding_atom, bonding_site]
    terachem_geo_optimize(
        f"{terachem_opt_name}_ff_guess.xyz", 
        f"{terachem_opt_name}", 
        cluster,
        charge,
        spinmult,
        ind,
        options={
            "custom_cluster_path": custom_cluster_path,
            "wait_time": 5
        })
    # Check Structures
    if mech_step == 'oxo':
        ligand_test = find_oxo_n2_ligand(f"{base_dir}/{terachem_opt_name}_opt.xyz")
    else:
        substrate_data = read_substrate_xyz(path_to_substrate_xyz, bonding_atom)
        moiety = []
        for atom in substrate_data:
            moiety.append((atom["parent"] if atom["parent"] is not None else 0, atom["element"]))
        ligand_test = find_moieties(f"{base_dir}/{terachem_opt_name}_opt.xyz", moiety)
    if not ligand_test:
        orca_bl_input_xyz = f"{base_dir}/{terachem_opt_name}_ff_guess.xyz"
    else:
        orca_bl_input_xyz = f"{base_dir}/{terachem_opt_name}_opt.xyz"
    # Run ORCA bond length scan
    orca_bond_length_scan(f"{orca_bl_input_xyz}",
                        orca_bl_name, charge, spinmult,
                        mech_step=mech_step,
                        ts_guess_name=ts_guess_name,
                        cluster=cluster,
                        step_no=5, 
                        options={
                            "scan_type": scan_type,
                            "custom_cluster_path": custom_cluster_path
                        }
                    )
    # Run ORCA TS optimization
    orca_transition_state_optimization(f"{base_dir}/{ts_guess_name}.xyz",
                                        orca_ts_name,
                                        charge, spinmult,
                                        involved_atoms=ind,
                                        cluster=cluster,
                                        options = {
                                            "custom_cluster_path": custom_cluster_path,
                                            "mech_step": mech_step,
                                            "wait_time": 5,
                                        })
    end_time = time.time()
    elapsed_seconds = int(end_time - start_time)
    hours = elapsed_seconds // 3600
    minutes = (elapsed_seconds) % 3600 // 60
    seconds = (elapsed_seconds) % 60
    print(f"TS optimization workflow for {mech_step} completed.")
    print(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")


def m2m_full_reaction_workflow(path_to_tmc_xyz, path_to_n2o_xyz, path_to_ch4_xyz, charge, spinmult, bonding_atom, binding_site, mech_step=None, path_to_substrates = None):
    """
    Parameters:
    ----------
    path_to_tmc_xyz : str
        Path to the TMC .xyz file.
    charge : int
        Charge of the system.
    spinmult : int
        Spin multiplicity of the system.
    bonding_atom : str
        Atom involved in the bond.
    bonding_site : str
        Site of the bond.
    path_to_substrate_xyz : str, optional
        Path to the substrate .xyz file.
    mech_step : str, optional
        Mechanistic step to consider.
    Returns:
    -------
    None
    """
    # Optimize substrates (if necessary) and extract Gibbs Free Energy with Orca Single Point Energy Calculation
    base_dir = os.getcwd()# Path to directory where the desired complex .xyz file is located
    reaction_species = []
    substrates = [["n2o", 0, 1], ["n2", 0, 1], ["ch4", 0, 1], ["ch3", 0, 2], ["ch3oh", 0, 1]]
    if os.path.exists(path_to_substrates):
        print(f"Substrate directory found: {path_to_substrates}")
        for substrate in substrates:
            # Check directory for completed calculations
            filepath = f"{path_to_substrates}/{substrate[0]}/orca_opt.out"
            gibbs_free_energy = check_orca_gibbs_free_energy(filepath)
            print(f"Gibbs Free Energy for {substrate[0]}: {gibbs_free_energy}")
            reaction_species.append({'species': substrate[0], 'gibbs_free_energy': gibbs_free_energy})
    elif not os.path.exists(path_to_substrates):
        # Create substrate directories
        os.makedirs(path_to_substrates, exist_ok=True)
        for substrate in substrates:
            os.makedirs(f"{path_to_substrates}/{substrate[0]}", exist_ok=True)
            write_substrates_xyz(f"{path_to_substrates}/{substrate[0]}", substrate)
            orca_geometry_optimization(f"{path_to_substrates}/{substrate[0]}/{substrate[0]}.xyz", "orca_opt", substrate[1], substrate[2], atoms_w_unpaired_electrons=[0], wait_time=3, queue_system="slurm")
            gibbs_free_energy = check_orca_gibbs_free_energy(f"{path_to_substrates}/{substrate[0]}/orca_opt.out")
            print(f"Gibbs Free Energy for {substrate[0]}: {gibbs_free_energy}")
            reaction_species.append({'species': substrate[0], 'gibbs_free_energy': gibbs_free_energy})
    # Run oxo formation step transition state search
    os.makedirs(f"{base_dir}/oxo_formation", exist_ok=True)
    ts_optimization_workflow(path_to_tmc_xyz, charge, spinmult, 2, binding_site, path_to_substrate_xyz=f"{path_to_substrates}/n2o/orca_opt.xyz", running_directory=f"{base_dir}/oxo_formation", scan_type='breaking')
    # Remove N2 and optimize oxo intermediate
    os.makedirs(f"{base_dir}/oxo", exist_ok=True) # Careful about spin
    
    # Run hydrogen atom transfer step transition state search
    os.makedirs(f"{base_dir}/hat", exist_ok=True)
    path_to_oxo_xyz = f"{base_dir}/oxo/orca_opt.xyz"  # Use optimized oxo xyz as the starting structure to this optimization
    ts_optimization_workflow(path_to_oxo_xyz, charge, spinmult, 1, binding_site, path_to_substrate_xyz=f"{path_to_substrates}/ch4/orca_opt.xyz", running_directory=f"{base_dir}/hat", scan_type='formation')
    # Remove CH3 and optimize hydroxo intermediate
    os.makedirs(f"{base_dir}/hydroxo", exist_ok=True)# Careful about spin

    # Run rebound step transition state search
    os.makedirs(f"{base_dir}/rebound", exist_ok=True)
    path_to_hydroxo_xyz = f"{base_dir}/hydroxo/orca_opt.xyz" # Use optimized hydroxo xyz as the starting structure to this optimization
    ts_optimization_workflow(path_to_hydroxo_xyz, charge, spinmult, 0, binding_site, path_to_substrate_xyz=f"{path_to_substrates}/ch3/orca_opt.xyz", running_directory=f"{base_dir}/rebound", scan_type='formation')

    # Optimize methanol bound intermediate

    # Run release step transition state search
    