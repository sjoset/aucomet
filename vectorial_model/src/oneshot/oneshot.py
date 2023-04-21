#!/usr/bin/env python3

import os
import sys
import pathlib
import contextlib

import logging as log
import astropy.units as u
import pyvectorial as pyv

# import plotly.graph_objects as go
from plotly.subplots import make_subplots

from astropy.visualization import quantity_support
from argparse import ArgumentParser

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
        "parameterfile", nargs=1, help="YAML file with production and molecule data"
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


def remove_file_silent_fail(f: pathlib.Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(f)


def main():
    # astropy units/quantities support in plots
    quantity_support()

    args = process_args()
    log.debug("Loading input from %s ....", args.parameterfile[0])

    vmc_set = pyv.vm_configs_from_yaml(args.parameterfile[0])

    out_table = pyv.build_calculation_table(vmc_set)

    output_fits_file = pathlib.Path("output.fits")
    remove_file_silent_fail(output_fits_file)
    log.info("Table building complete, writing results to %s ...", output_fits_file)
    out_table.write(output_fits_file, format="fits")

    # print collision sphere radii
    coma_list = [pyv.unpickle_from_base64(row["b64_encoded_coma"]) for row in out_table]
    for coma in coma_list:
        print(coma.vmr.collision_sphere_radius.to(u.km))
        vmr = pyv.get_result_from_coma(coma)
        fig = make_subplots(rows=1, cols=1)
        fig.update_xaxes(type="log", tickformat="0.1e", exponentformat="e")
        fig.update_yaxes(type="log", tickformat="0.1e", exponentformat="e")
        fig.add_trace(
            pyv.plotly_column_density_plot(
                vmr, dist_units=u.km, cdens_units=(1 / u.cm**2)
            )
        )
        for r, cd, vd in zip(
            vmr.column_density_grid, vmr.column_density, vmr.volume_density
        ):
            print(
                f"r: {r.to_value(u.km):8e}\tCD: {cd.to_value(1/u.cm**2):8e}\tVD: {vd.to_value(1/u.m**3):8e}"
            )
        fig.show()


if __name__ == "__main__":
    sys.exit(main())
