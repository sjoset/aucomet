#!/usr/bin/env python3

import os
import sys
import dill as pickle
import hashlib
# import json
import pprint

import logging as log
from argparse import ArgumentParser
# import importlib.metadata as impm

import pyvectorial as pyv

__author__ = 'Shawn Oset'
__version__ = '0.1'


myred = "#c74a77"
mybred = "#dbafad"
mygreen = "#afac7c"
mybgreen = "#dbd89c"
mypeach = "#dbb89c"
mybpeach = "#e9d4c3"
myblue = "#688894"
mybblue = "#a4b7be"
myblack = "#301e2a"
mybblack = "#82787f"
mywhite = "#d8d7dc"
mybwhite = "#e7e7ea"


def process_args():

    # Parse command-line arguments
    parser = ArgumentParser(
        usage='%(prog)s [options] [inputfile]',
        description=__doc__,
        prog=os.path.basename(sys.argv[0])
    )
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='increase verbosity level')
    parser.add_argument(
            'parameterfile', nargs=1, help='YAML file with production and molecule data'
            )  # the nargs=? specifies 0 or 1 arguments: it is optional

    args = parser.parse_args()

    # handle verbosity
    if args.verbose >= 2:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    elif args.verbose == 1:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    return args


def file_string_id_from_parameters(vmc):

    base_q = vmc.production.base_q.value
    p_tau_d = vmc.parent.tau_d.value
    f_tau_T = vmc.fragment.tau_T.value

    return f"q_{base_q}_ptau_d_{p_tau_d:06.1f}_ftau_T_{f_tau_T:06.1f}"


def dill_from_coma(coma, coma_file) -> None:
    with open(coma_file, 'wb') as comapicklefile:
        pickle.dump(coma, comapicklefile)

def coma_from_dill(coma_file: str):
    with open(coma_file, 'rb') as coma_dill:
        return pickle.load(coma_dill)


def hash_vmc(vmc: pyv.VectorialModelConfig):

    # sbpy_ver = impm.version("sbpy")

    return hashlib.sha256(pprint.pformat(vmc).encode('utf-8')).hexdigest()


def main():

    args = process_args()

    log.debug("Loading input from %s ....", args.parameterfile[0])

    vmc = pyv.vm_configs_from_yaml(args.parameterfile[0])[0]
    print(hash_vmc(vmc))


if __name__ == '__main__':
    sys.exit(main())
