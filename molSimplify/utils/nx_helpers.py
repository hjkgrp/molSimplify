import networkx as nx

def graph_from_mol2_string(m2_string, positions=False):
    lines = m2_string.splitlines(True)

    # Read counts line:
    sp = lines[2].split()
    n_atoms = int(sp[0])
    n_bonds = int(sp[1])

    g = nx.Graph()

    atom_start = lines.index("@<TRIPOS>ATOM\n") + 1
    for i, line in enumerate(lines[atom_start : atom_start + n_atoms]):
        sp = line.split()
        sym = sp[5].split(".")[0]
        data = dict(symbol=sym)
        if positions:
            pos = [float(p) for p in sp[2:5]]
            data.update(position=pos)
        g.add_node(i, **data)

    bond_start = lines.index("@<TRIPOS>BOND\n") + 1
    for line in lines[bond_start : bond_start + n_bonds]:
        sp = line.split()
        # Subtract 1 because of zero indexing vs. one indexing
        g.add_edge(int(sp[1]) - 1, int(sp[2]) - 1)

    return g
