from molSimplify.Scripts.generator import startgen_pythonic


def test_joption_pythonic(tmpdir, resource_path_root):
    out_dir = "cr_thd_2_cl_4_s_1/cr_thd_2_cl_4_s_1_conf_1/jobscript"
    input_dict_homo = {
        '-core': "cr",
        '-coord': str(4),
        '-oxstate': str(2),
        '-lig': str("cl"),
        '-geometry': "thd",
        '-ligocc': "4",
        '-rundir': str(tmpdir),
        '-runtyp': "minimize",
        '-keepHs': "yes",
        '-spin': str(1),
        '-jname': "{}_{}_{}_{}_hs_{}".format("cr", "thd", 2, "cl", 0),
        '-modules': "cuda,terachem",
        '-joption': "-fin terachem_input, -fin *.xyz, -fout scr/"
    }
    startgen_pythonic(input_dict_homo, write=1)
    with open(str(tmpdir) + "/" + out_dir, 'r') as f_in:
        data1 = f_in.readlines()
    with open(resource_path_root / "refs" / "joption_pythonic_jobscript", 'r') as f_in:
        data2 = f_in.readlines()
    for i, j in zip(data1, data2):
        assert i == j
