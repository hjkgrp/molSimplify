from molSimplify.Classes.mol3D import mol3D
import networkx as nx

def get_hash(mol):
    nx_graph = mol.nx_graph
    graph_hash = nx.weisfeiler_lehman_graph_hash(nx_graph)

    graph_attr = nx.Graph()
    attributed = []
    for i, row in enumerate(mol.graph):
        for j, column in enumerate(row):
            if mol.graph[i][j] == 1:
                temp_label = (mol.getAtom(i).symbol()+mol.getAtom(j).symbol())
                temp_label = ''.join(sorted(temp_label))
                attributed.append((i, j, {'label': temp_label}))
    graph_attr.add_edges_from(attributed)
    graph_hash_attr = nx.weisfeiler_lehman_graph_hash(graph_attr, edge_attr="label")

    return graph_hash, graph_hash_attr
