#!/usr/bin/env python3

import os
import sys

# import dill as pickle
# import time
import pathlib

import logging as log
import numpy as np
import pandas as pd
import astropy.units as u
from scipy.interpolate import CubicSpline
from argparse import ArgumentParser

from abel.hansenlaw import hansenlaw_transform

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
        "csv_file", nargs=1, help="csv file with r,volume density"
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


def sample_space(start: u.Quantity, end: u.Quantity, **kwargs):
    """
    Wrapper for numpy's linspace that respects astropy's unit system,
    with resulting samples and sample step in meters
    """
    xs, step = np.linspace(
        start.to_value(u.m), end.to_value(u.m), retstep=True, **kwargs
    )
    xs = xs * u.m
    step = step * u.m

    return xs, step


def main():
    args = process_args()
    log.debug("Loading input from %s ....", args.csv_file[0])
    csv_file = pathlib.PurePath(args.csv_file[0])

    vdens_dataframe = pd.read_csv(filepath_or_buffer=csv_file)  # type: ignore
    vdens_interp = CubicSpline(
        vdens_dataframe["r"],
        vdens_dataframe["volume_density"],
        bc_type="natural",
    )

    rs, step = sample_space(
        1000 * u.m,
        vdens_dataframe["r"].iloc[-1] * u.m,
        endpoint=True,
        num=70000,
    )

    n_rs = vdens_interp(rs.to_value(u.m))

    abel_cds = (
        hansenlaw_transform(
            n_rs,
            dr=step.to_value(u.m),
            direction="forward",
        )
        / u.m**2
    )

    for i, (r, cd) in enumerate(zip(rs, abel_cds)):
        if (not i % 1000) or (r.to_value(u.km) < 200):
            print(f"R: {r.to_value(u.km):8e}\tCD: {cd.to_value(1/u.cm**2):8e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
