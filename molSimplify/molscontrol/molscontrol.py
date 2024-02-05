"""
molscontrol.py
An automonous on-the-fly job control system for DFT geometry optimization aided by machine learning techniques.

Written by Chenru Duan, Kulik Group at MIT.
crduan@mit.edu, hjkulik@mit.edu
"""

import sys
import tensorflow as tf
from tensorflow import keras
from molSimplify.molscontrol.io_tools import get_configure
from molSimplify.molscontrol.dynamic_classifier import dft_control
import argparse


def main():
    """
    The main function of job controls.

    Parameters
    ----------
    pid: the process (DFT geometry optimization).

    Returns
    -------
    None
    """
    # try:
    #     pid = sys.argv[1]
    # except:
    #     pid = False
    #     print("NO PID to control. Should be in a test mode.")
    parser = argparse.ArgumentParser(description='molscontrol parser')
    parser.add_argument('--pid', action="store", default=False)
    parser.add_argument('--config', action="store", default="configure.json", type=str)
    args = parser.parse_args()
    print("pid: ", args.pid)
    print("molscontrol configure file: ", args.config)
    kwargs = get_configure(args.config)
    kwargs.update({"pid": args.pid})
    dftjob = dft_control(**kwargs)
    stop = False
    while not stop:
        stop = dftjob.update_and_predict()


if __name__ == "__main__":
    main()
