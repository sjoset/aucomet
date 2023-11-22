#!/usr/bin/env python3

import os
import sys
import pathlib
import warnings

import logging as log
from argparse import ArgumentParser
import astropy.units as u
from astropy.table import QTable

import sbpy.activity as sba
import pyvectorial as pyv

__author__ = "Shawn Oset"
__version__ = "0.1"


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
    parser.add_argument(
        "fitsinput", nargs=1, help="fits file that contains calculation table"
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
    # sometimes the aperture counts get a little complainy
    warnings.filterwarnings("ignore")
    args = process_args()
    table_file = pathlib.PurePath(args.fitsinput[0])

    # read in table from FITS
    test_table = QTable.read(table_file, format="fits")

    for row in test_table:
        vmr = pyv.unpickle_from_base64(row["b64_encoded_vmr"])
        pyv.interpolate_volume_density(vmr)
        pyv.column_density_from_abel(vmr)
        pyv.interpolate_column_density(vmr)

        aps = (
            sba.CircularAperture(10000 * u.km),
            sba.CircularAperture(vmr.max_grid_radius),
            sba.AnnularAperture((10000, 100000) * u.km),
            sba.RectangularAperture((10000, 10000) * u.km),
            pyv.UncenteredRectangularAperture((-5000, -5000, 5000, 5000) * u.km),
            pyv.UncenteredRectangularAperture((10000.0, 0.0, 30000.0, 100000.0) * u.km),
        )

        for ap in aps:
            print(ap)
            N = pyv.total_number_in_aperture(vmr, ap)
            print(N)
            if vmr.coma is not None:
                print(vmr.coma.total_number(ap))
            print("")


if __name__ == "__main__":
    sys.exit(main())
