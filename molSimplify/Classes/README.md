To members of the Kulik group: If you have custom data that you want to use for yourself but do not want to push to the main repo, e.g. custom ligands for molSimplify ligand construction, you can make a file `.molSimplify` and include in it a line like `CUSTOM_DATA_PATH=/Users/your_user/molSimplify_custom_data`. In that folder you specified, you can place a folder `Ligands`, and molSimplify will look there first before looking at the Ligands folder in the central repo. This functionality is implemented in `globalvars.py`.