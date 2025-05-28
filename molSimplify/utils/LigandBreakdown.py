from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.atom3D import atom3D
import networkx as nx
import numpy as np

def ligand_breakdown(mol,dummy_bool=False):
    # get networkx graph representations
    g=mol3D_to_networkx(mol,get_symbols=False,get_bond_order=False,get_bond_distance=False)

    # get metal_indices
    metal_indices=mol.findMetal(transition_metals_only=True)

    # remove all metals from copied graph and get all disconnected graphs from the copy
    g.remove_nodes_from(metal_indices)
    disconnected_graphs = list(nx.connected_components(g))

    list_of_ligands=[]
    # for each disconnected graph
    for set_of_nodes in disconnected_graphs:
        metal_indices=mol.findMetal(transition_metals_only=True)
        bridging_data=bridging_ligand_helper(mol,set_of_nodes,metal_indices)

        if dummy_bool:
            submol=mol.create_mol_with_inds(list(set_of_nodes)+metal_indices)
            # get metal_indices
            metal_indices=submol.findMetal(transition_metals_only=True)
            for metal_index in metal_indices:
                submol.getAtom(metal_index).mutate(newType='X')
            list_of_ligands.append((submol,bridging_data))
        else:
            # create a mol3D of just the atoms in the subgraph and get the new mol2
            submol=mol.create_mol_with_inds(list(set_of_nodes))
            list_of_ligands.append((submol,bridging_data))
    return list_of_ligands

def mol3D_to_networkx(mol,get_symbols:bool=True,get_bond_order:bool=True,get_bond_distance:bool=False):
    g = nx.Graph()

    # get every index of atoms in mol3D object
    for atom_ind in range(0,mol.natoms):
        # set each atom as a node in the graph, and add symbol information if wanted
        data={}
        if get_symbols:
            data['symbol']=mol.getAtom(atom_ind).symbol()
        g.add_node(atom_ind,**data)

    # get every bond in mol3D object
    bond_info=mol.bo_dict
    for bond in bond_info:
        # set each bond as an edge in the graph, and add symbol information if wanted
        data={}
        if get_bond_order:
            data['bond_order']=bond_info[bond]
        if get_bond_distance:
            distance=mol.getAtom(bond[0]).distance(mol.getAtom(bond[1]))
            data['bond_distance']=distance
        g.add_edge(bond[0],bond[1],**data)
    return g

def bridging_ligand_helper(mol,set_of_nodes,metal_indices):
    submol=mol.create_mol_with_inds(list(set_of_nodes)+metal_indices)
    submol.nx_graph = submol.mol3D_to_networkx()
    # get new metal indicies
    submol_metal_indices=submol.findMetal(transition_metals_only=True)
    # get disconnected graphs
    disconnected_graphs = list(nx.connected_components(submol.nx_graph))
    # check if at least two metals present in each disconnected graph
    ret={}
    for d_graph in disconnected_graphs:
        # if length is 1 then it is more than isolated metal
        if len(d_graph) > 1:
            # count each metal, see if they are in metal indices
            metal_count=0
            for node in d_graph:
                if node in submol_metal_indices:
                    metal_count+=1
            ret['Bridging_Ligand']=metal_count>1
            ret['Connected_Metals']=metal_count
    return ret
