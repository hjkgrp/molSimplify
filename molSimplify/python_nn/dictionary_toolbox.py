
def write_dictionary(dictionary, path):
    emsg = False
    try:
        with open(path, 'w') as f:
            for keys in list(dictionary.keys()):
                f.write(str(keys).strip("\n") + ',' + str(dictionary[keys]) + '\n')
    except FileNotFoundError:
        emsg = "Error, could not write dictionary space: " + path
    return emsg


def read_dictionary(path):
    emsg = False
    dictionary = dict()
    try:
        with open(path, 'r') as f:
            for lines in f:
                ll = lines.split(",")
                key = ll[0]
                value = ll[1].rstrip("\n")
                dictionary[key] = value
    except (FileNotFoundError, IndexError):
        emsg = "Error, could not read dictionary space: " + path
    return emsg, dictionary


if __name__ == "__main__":
    sfd = {
        "split_energy": [-54.19, 142.71],
        "slope": [-174.20, 161.58],
        "ls_min": [1.8146, 0.6910],
        "hs_min": [1.8882, 0.6956],
        "ox": [2, 1],
        "alpha": [0, 0.3],
        "ax_charge": [-2, 2],
        "eq_charge": [-2, 2],
        "ax_dent": [1, 1],
        "eq_dent": [1, 3],
        "sum_delen": [-5.34, 12.54],
        "max_delen": [-0.89, 2.09],
        "ax_bo": [0, 3],
        "eq_bo": [0.00, 3],
        "ax_ki": [0.00, 4.29],
        "eq_ki": [0.00, 6.96]
    }
    write_dictionary(sfd, 'scaling.csv')

    rect = read_dictionary('scaling.csv')
