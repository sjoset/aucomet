#!/usr/bin/env python3

import os
import sys
import pathlib

import logging as log
from argparse import ArgumentParser

import pyvectorial as pyv

__author__ = 'Shawn Oset'
__version__ = '0.1'


def process_args():

    # Parse command-line arguments
    parser = ArgumentParser(
        usage='%(prog)s [options] [inputfile] [outputfile]',
        description=__doc__,
        prog=os.path.basename(sys.argv[0])
    )
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='increase verbosity level')
    parser.add_argument(
            'parameterfile', nargs=1, help='YAML file with production and molecule data'
            )  # the nargs=? specifies 0 or 1 arguments: it is optional
    parser.add_argument('output', nargs=1, type=str, help='FITS output filename')

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

    args = process_args()
    yaml_config_file = pathlib.PurePath(args.parameterfile[0])
    output_fits_file = pathlib.PurePath(args.output[0])
    log.info("Loading yaml from %s and saving to %s ...", yaml_config_file, output_fits_file)

    vmc_set = pyv.vm_configs_from_yaml(yaml_config_file)
    out_table = pyv.build_calculation_table(vmc_set)

    log.info("Table building complete, writing results to %s ...", output_fits_file)
    out_table.write(output_fits_file, format='fits')


if __name__ == '__main__':
    sys.exit(main())
