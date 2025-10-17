# @file __main__.py
# Gateway script to rest of program
#
# Written by Tim Ioannidis and Roland St Michel for HJK Group
#
# Dpt of Chemical Engineering, MIT

# !/usr/bin/env python
'''
    Copyright 2017 Kulik Lab @ MIT

    This file is part of molSimplify.
    molSimplify is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published
    by the Free Software Foundation, either version 3 of the License,
    or (at your option) any later version.

    molSimplify is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with molSimplify. If not, see http://www.gnu.org/licenses/.
'''
# fix OB bug: https://github.com/openbabel/openbabel/issues/1983
import sys
import matplotlib.pyplot as plt
import os
import argparse
if not ('win' in sys.platform):
    flags = sys.getdlopenflags()
if not ('win' in sys.platform):
    sys.setdlopenflags(flags)
from pathlib import Path
from molSimplify.Scripts.inparse import (parseinputs_advanced, parseinputs_slabgen,
                                         parseinputs_db, parseinputs_inputgen,
                                         parseinputs_postproc, parseinputs_random,
                                         parseinputs_binding, parseinputs_tsgen,
                                         parseinputs_customcore, parseinputs_naming,
                                         parseinputs_ligdict, parseinputs_basic,
                                         parseCLI)
from molSimplify.Scripts.generator import startgen
from molSimplify.Classes.globalvars import globalvars
from molSimplify.utils.tensorflow import tensorflow_silence


# subcommand deps (safe to import here)
import json
try:
    from openbabel import pybel  # for optional write-out
except Exception:
    pybel = None  # we'll guard usage below

# our builders
from molSimplify.Scripts.enhanced_structgen import (
    create_ligand_list,
    generate_complex,
)

from molSimplify.Scripts.enhanced_structgen_functionality import check_badjob


globs = globalvars()
# Basic help description string
DescString_basic = (
    'Welcome to molSimplify. Only basic usage is described here.\n'
    'For help on advanced modules, please refer to our documentation at '
    'molsimplify.mit.edu or provide additional commands to -h, as below:\n'
    '-h advanced: advanced structure generation help\n'
    '-h slabgen: slab builder help\n'
    # '-h chainb: chain builder help\n'
    '-h autocorr: automated correlation analysis help\n'
    '-h db: database search help\n'
    '-h inputgen: quantum chemistry code input file generation help\n'
    '-h postproc: post-processing help\n'
    '-h random: random generation help\n'
    '-h binding: binding species (second molecule) generation help\n'
    '-h customcore: custom core functionalization help\n'
    '-h tsgen: transition state generation help\n'
    '-h naming: custom filename help\n'
    '-h liganddict: ligands.dict help\n'
)
# Advanced help description string
DescString_advanced = 'Printing advanced structure generation help.'
# Slab builder help description string
DescString_slabgen = 'Printing slab builder help.'
# Chain builder help description string
DescString_chainb = 'Printing chain builder help.'
# Automated correlation analysis description string
DescString_autocorr = 'Printing automated correlation analysis help.'
# Database search help description string
DescString_db = 'Printing database search help.'
# Input file generation help description string
DescString_inputgen = 'Printing quantum chemistry code input file generation help.'
# Post-processing help description string
DescString_postproc = 'Printing post-processing help.'
# Random generation help description string
DescString_random = 'Printing random generation help.'
# Binding species placement help description string
DescString_binding = 'Printing binding species (second molecule) generation help.'
# Transition state generation help description string
DescString_tsgen = 'Printing transition state generation help.'
# Ligand replacement help description string
DescString_customcore = 'Printing ligand replacement help.'
# Custom file naming help description string
DescString_naming = 'Printing custom filename help.'
# Ligand dictionary help description string
DescString_ligdict = 'Printing ligand dictionary help.'

# Main function
#  @param args Argument namespace
def main(args=None):
    # issue a call to test TF, this is needed to keep
    # ordering between openbabel and TF calls consistent
    # on some sytems
    if globs.testTF():
        print('TensorFlow connection successful.')
        tensorflow_silence()
    else:
        print('TensorFlow connection failed.')

    if args is None:
        args = sys.argv[1:]

    # ------------------------ subcommand: build-complex ------------------------
    # Usage:
    #   python -m molSimplify build-complex \
    #       --ligand biimidazole \
    #       --ligand "c1ccccc1" \
    #       --usercatoms None \
    #       --usercatoms "[[0,1,2,3,4,5]]" \
    #       --occupancy 2 \
    #       --occupancy 2 \
    #       --out complex.mol2 --verbose
    if 'build-complex' in args:
        # Extract only the subcommand args (strip the token 'build-complex')
        subargv = [a for a in args if a != 'build-complex']

        import argparse

        def _maybe_none(x: str):
            return None if x is None or str(x).strip().lower() in {"none", "null", ""} else x

        def _parse_usercatoms(s: str):
            """
            Accepts:
              - "None" / "none" / ""  -> None
              - JSON or Python-like lists, e.g. "[[0,1,2,3,4,5]]" or "[0,1]"
            """
            if s is None:
                return None
            s = s.strip()
            if s.lower() in {"", "none", "null"}:
                return None
            try:
                return json.loads(s)
            except Exception:
                from ast import literal_eval
                return literal_eval(s)

        parser = argparse.ArgumentParser(
            prog="molSimplify build-complex",
            description="Build a coordination complex from ligands (wrapper around generate_complex)."
        )

        # Repeated flags collect into lists in positional order
        parser.add_argument("--ligand", dest="ligands", action="append", required=True,
                            help="Ligand identifier or SMILES (repeat flag for multiple ligands).")
        parser.add_argument("--usercatoms", dest="usercatoms", action="append", default=None,
                            help='Coordinating atom indices per ligand (e.g. "[[0,1,2]]"). '
                                 'Use "None" to auto-detect / dictionary.')
        parser.add_argument("--occupancy", dest="occupancies", action="append", type=int, default=None,
                            help="Occupancy per ligand (repeat to match --ligand count).")
        parser.add_argument("--isomer", dest="isomers", action="append", default=None,
                            help='Isomer tag per ligand (or "None").')

        # Common build knobs (pass-through to generate_complex)
        parser.add_argument("--metal", default="Fe")
        parser.add_argument("--geometry", default="octahedral")
        parser.add_argument("--voxel-size", type=float, default=0.5)
        parser.add_argument("--vdw-scale", type=float, default=0.8)
        parser.add_argument("--clash-weight", type=float, default=10.0)
        parser.add_argument("--nudge-alpha", type=float, default=0.1)
        parser.add_argument("--max-steps", type=int, default=500)
        parser.add_argument("--ff-name", default="UFF")
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--smart-generation", action="store_true", default=True)
        parser.add_argument("--no-smart-generation", dest="smart_generation", action="store_false")

        # Orientation terms
        parser.add_argument("--orientation-weight", type=float, default=6.0)
        parser.add_argument("--orientation-k-neighbors", type=int, default=4)
        parser.add_argument("--orientation-hinge", type=float, default=0.5)
        parser.add_argument("--orientation-cap", type=float, default=1.0)

        # Haptic multi-bond behavior
        parser.add_argument("--multibond-haptics", action="store_true", default=True)
        parser.add_argument("--no-multibond-haptics", dest="multibond_haptics", action="store_false")
        parser.add_argument("--multibond-bond-order", type=int, default=1)
        parser.add_argument("--multibond-prefer-nearest-metal", action="store_true", default=True)
        parser.add_argument("--no-multibond-prefer-nearest-metal", dest="multibond_prefer_nearest_metal", action="store_false")

        # Sterics terms
        parser.add_argument("--run-sterics", action="store_true", default=True)

        # Visualization/output
        parser.add_argument("--vis-save-dir", default=None)
        parser.add_argument("--vis-stride", type=int, default=1)
        parser.add_argument("--vis-view", default="22,-60",
                            help="Comma-separated elevation,azimuth (e.g., '22,-60').")
        parser.add_argument("--vis-prefix", default="kabsch")

        # Run directory management
        parser.add_argument("--run-dir", default="runs",
                            help="Base directory for outputs (default: runs).")
        parser.add_argument("--run-name", default=None,
                            help="Optional name for the run subfolder; default is a timestamp.")


        pargs = parser.parse_args(subargv)

        # Normalize lists: usercatoms/occupancies/isomers can be missing; create_ligand_list handles None
        ligands = pargs.ligands
        if pargs.usercatoms is not None:
            usercatoms_list = [_parse_usercatoms(_maybe_none(s)) for s in pargs.usercatoms]
        else:
            usercatoms_list = None

        occupancies = pargs.occupancies if pargs.occupancies is not None else None
        isomers = [ _maybe_none(s) for s in pargs.isomers ] if pargs.isomers is not None else None

        # Sanity: lengths (create_ligand_list will assert too)
        if usercatoms_list is not None and len(usercatoms_list) != len(ligands):
            parser.error("--usercatoms count must match --ligand count.")
        if occupancies is not None and len(occupancies) != len(ligands):
            parser.error("--occupancy count must match --ligand count.")
        if isomers is not None and len(isomers) != len(ligands):
            parser.error("--isomer count must match --ligand count.")

        # Build ligand tuples via existing helper
        ligand_list = create_ligand_list(
            userligand_list=ligands,
            usercatoms_list=usercatoms_list,
            occupancy_list=occupancies,
            isomer_list=isomers
        )

        # Parse vis view tuple
        try:
            elev, azim = [float(x) for x in pargs.vis_view.split(",")]
            vis_view = (elev, azim)
        except Exception:
            vis_view = (22, -60)

        # Run build
        mol, clash, severity, fig = generate_complex(
            ligand_list,
            metals=pargs.metal,
            voxel_size=pargs.voxel_size,
            vdw_scale=pargs.vdw_scale,
            clash_weight=pargs.clash_weight,
            nudge_alpha=pargs.nudge_alpha,
            geometry=pargs.geometry,
            coords=None,
            max_steps=pargs.max_steps,
            ff_name=pargs.ff_name,
            vis_save_dir=pargs.vis_save_dir,
            vis_stride=pargs.vis_stride,
            vis_view=vis_view,
            vis_prefix=pargs.vis_prefix,
            manual=False,
            manual_list=None,
            smart_generation=pargs.smart_generation,
            verbose=pargs.verbose,
            orientation_weight=pargs.orientation_weight,
            orientation_k_neighbors=pargs.orientation_k_neighbors,
            orientation_hinge=pargs.orientation_hinge,
            orientation_cap=pargs.orientation_cap,
            multibond_haptics=pargs.multibond_haptics,
            multibond_bond_order=pargs.multibond_bond_order,
            multibond_prefer_nearest_metal=pargs.multibond_prefer_nearest_metal,
            run_sterics=pargs.run_sterics,
        )

    # -------------------- auto-build run name --------------------
    import re, hashlib

    def _slug(s: str) -> str:
        """
        Make a filesystem-safe token from a ligand identifier (name or SMILES).
        - Lowercase, strip leading/trailing spaces
        - Replace any run of non [a-z0-9] with a single '-'
        - Trim repeated dashes
        - Fallback to short hash if empty
        """
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "-", s)         # collapse non-alnum to '-'
        s = re.sub(r"-{2,}", "-", s).strip("-")   # remove dup dashes + edge dashes
        if not s:
            s = "lig-" + hashlib.md5((s or 'x').encode()).hexdigest()[:6]
        return s

    # Build effective occupancies: use 1 where user omitted / passed None.
    if occupancies is None:
        effective_occ = [1] * len(ligands)
    else:
        # pad/trim to ligands length defensively, coerce None -> 1
        tmp = list(occupancies) + [1] * max(0, len(ligands) - len(occupancies))
        effective_occ = [ (o if (o is not None) else 1) for o in tmp[:len(ligands)] ]

    # Assemble name: {metal}_{lig1}_{occ1}_{lig2}_{occ2}...
    parts = [str(pargs.metal)]
    for lig, occ in zip(ligands, effective_occ):
        parts.append(_slug(str(lig)))
        parts.append(str(int(occ)))  # make sure it's an int-like string

    run_name = "_".join(parts)

    # Optional: limit extreme length while keeping uniqueness
    MAX_LEN = 120
    if len(run_name) > MAX_LEN:
        tail_hash = hashlib.md5(run_name.encode()).hexdigest()[:8]
        run_name = run_name[: (MAX_LEN - 9)] + "_" + tail_hash
    # -------------------------------------------------------------

    # -------------------- create run directory & handoff --------------------
    base_dir = os.path.abspath(pargs.run_dir)
    os.makedirs(base_dir, exist_ok=True)

    run_dir = os.path.join(base_dir, run_name)

    # Ensure unique folder if name collides (append _1, _2, ...)
    suffix = 1
    candidate = run_dir
    while os.path.exists(candidate):
        candidate = f"{run_dir}_{suffix}"
        suffix += 1
    run_dir = candidate
    os.makedirs(run_dir, exist_ok=False)

    # Save inputs actually used for reproducibility
    metadata = {
        "run_name": run_name,
        "run_dir": run_dir,
        "ligands": ligands,
        "effective_occupancies": effective_occ,
        "usercatoms": usercatoms_list,
        "isomers": isomers,
        "metal": pargs.metal,
        "geometry": pargs.geometry,
        "voxel_size": pargs.voxel_size,
        "vdw_scale": pargs.vdw_scale,
        "clash_weight": pargs.clash_weight,
        "nudge_alpha": pargs.nudge_alpha,
        "max_steps": pargs.max_steps,
        "ff_name": pargs.ff_name,
        "orientation_weight": pargs.orientation_weight,
        "orientation_k_neighbors": pargs.orientation_k_neighbors,
        "orientation_hinge": pargs.orientation_hinge,
        "orientation_cap": pargs.orientation_cap,
        "multibond_haptics": pargs.multibond_haptics,
        "multibond_bond_order": pargs.multibond_bond_order,
        "multibond_prefer_nearest_metal": pargs.multibond_prefer_nearest_metal,
        "run_sterics": pargs.run_sterics,
        "vis_save_dir": pargs.vis_save_dir,
        "vis_stride": pargs.vis_stride,
        "vis_view": vis_view,
        "vis_prefix": pargs.vis_prefix,
        "verbose": pargs.verbose,
    }

    # add files to runs
    with open(os.path.join(run_dir, "input_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    # xyz structural file
    mol.writexyz(os.path.join(run_dir, "complex.xyz"))
    # mol2 structural file
    mol.writemol2_bodict(ignore_dummy_atoms=False, write_bond_orders=True, return_string=False, output_file=os.path.join(run_dir, "complex.mol2"))
    # sterics report
    fig.savefig(os.path.join(run_dir, "sterics.png"), dpi=300)
    plt.close(fig)
    # convert tuple keys → string like "4-43"
    json_safe = {f"{i}-{j}": v for (i, j), v in severity.items()}
    out_path = Path(run_dir) / "steric_clashes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(json_safe, indent=2))
    # status
    overlap, same_order = check_badjob(mol)
    if overlap == True or same_order == False:
        status = f"Badjob! Overlap: {overlap}, Order: {same_order}"
    else:
        status = "Success"
    status_path = os.path.join(run_dir, "status.log")
    with open(status_path, "a") as f:
        f.write(status + "\n")

    print(f"[ok] Run directory ready: {run_dir}")
    return
    # -----------------------------------------------------------------------
    # ---------------------- end subcommand: build-complex ----------------------


    ## print help ###
    if '-h' in args or '-H' in args or '--help' in args:
        if 'advanced' in args:
            parser = argparse.ArgumentParser(description=DescString_advanced)
            parseinputs_advanced(parser)
        if 'slabgen' in args:
            parser = argparse.ArgumentParser(description=DescString_slabgen)
            parseinputs_slabgen(parser)
        #    elif 'chainb' in args:
        #        parser = argparse.ArgumentParser(description=DescString_chainb)
        #        parseinputs_chainb(parser)
        #    elif 'autocorr' in args:
        #        parser = argparse.ArgumentParser(description=DescString_autocorr)
        #        parseinputs_autocorr(parser)
        elif 'db' in args:
            parser = argparse.ArgumentParser(description=DescString_db)
            parseinputs_db(parser)
        elif 'inputgen' in args:
            parser = argparse.ArgumentParser(description=DescString_inputgen)
            parseinputs_inputgen(parser)
        elif 'postproc' in args:
            parser = argparse.ArgumentParser(description=DescString_postproc)
            parseinputs_postproc(parser)
        elif 'random' in args:
            parser = argparse.ArgumentParser(description=DescString_random)
            parseinputs_random(parser)
        elif 'binding' in args:
            parser = argparse.ArgumentParser(description=DescString_binding)
            parseinputs_binding(parser)
        elif 'tsgen' in args:
            parser = argparse.ArgumentParser(description=DescString_tsgen)
            parseinputs_tsgen(parser)
        elif 'customcore' in args:
            parser = argparse.ArgumentParser(description=DescString_customcore)
            parseinputs_customcore(parser)
        elif 'naming' in args:
            parser = argparse.ArgumentParser(description=DescString_naming)
            parseinputs_naming(parser)
        elif 'liganddict' in args:
            # The formatter class allows for the display of new lines.
            parser = argparse.ArgumentParser(description=DescString_ligdict,
                                             formatter_class=argparse.RawTextHelpFormatter)
            parseinputs_ligdict(parser)
        else:
            # print basic help
            parser = argparse.ArgumentParser(description=DescString_basic,
                                             formatter_class=argparse.RawDescriptionHelpFormatter)
            parseinputs_basic(parser)
        return
    elif len(args) == 0:
        print('No arguments supplied. GUI is no longer supported. Exiting.')
    ## if input file is specified ###
    elif '-i' in args:
        print('Input file detected, reading arguments from input file.')
        print('molSimplify is starting!')
        # run from commandline
        startgen(sys.argv, False)
    ## grab from commandline arguments ###
    else:
        print('No input file detected, reading arguments from commandline.')
        print('molSimplify is starting!')
        # create input file from commandline
        infile = parseCLI([_f for _f in args if _f])
        args = ['main.py', '-i', infile]
        startgen(args, False)


if __name__ == '__main__':
    main()
