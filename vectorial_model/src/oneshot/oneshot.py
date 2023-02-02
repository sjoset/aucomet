#!/usr/bin/env python3

import os
import sys
# import dill as pickle
# import time
import pathlib

import logging as log
# import numpy as np
# import astropy.units as u
from astropy.visualization import quantity_support
from argparse import ArgumentParser

import pyvectorial as pyv

__author__ = 'Shawn Oset'
__version__ = '0.1'


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


def main():

    # astropy units/quantities support in plots
    quantity_support()

    args = process_args()
    log.debug("Loading input from %s ....", args.parameterfile[0])

    vmc_set = pyv.vm_configs_from_yaml(args.parameterfile[0])

    out_table = pyv.build_calculation_table(vmc_set)

    output_fits_file = pathlib.Path('output.fits')
    log.info("Table building complete, writing results to %s ...", output_fits_file)
    out_table.write(output_fits_file, format='fits')


if __name__ == '__main__':
    sys.exit(main())
