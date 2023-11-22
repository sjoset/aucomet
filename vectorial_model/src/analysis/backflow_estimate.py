#!/usr/bin/env python3

import os
import sys
import pathlib
import warnings

import logging as log
import numpy as np
from argparse import ArgumentParser
import astropy.units as u
from astropy.table import QTable

# from astropy.visualization import quantity_support
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def calculate_backflow_estimate(coma: sba.VectorialModel) -> np.float64:
    # Estimates the backflow of fragments into the collision zone by taking the volume density at the innermost edge of the model
    # and imagine it flowing radially inward at some average speed, which will be v_photo times some constant that we can figure
    # out from the geometry
    # TODO: figure out this average properly

    innermost_vol_dens = coma.vmr.volume_density[0]
    # average_fragment_radial_speed = (coma.fragment.v_photo * u.m/u.s) / np.sqrt(2)
    average_fragment_radial_speed = (coma.fragment.v_photo * u.m / u.s) / 2  # type: ignore

    backflow_estimate = (
        innermost_vol_dens
        * 4
        * np.pi
        * coma.vmr.collision_sphere_radius**2
        * average_fragment_radial_speed
    )
    return backflow_estimate


def add_backflow_estimate(qt: QTable) -> None:
    num_rows = len(qt)
    backflow_estimates = []

    for i, row in enumerate(qt):  # type: ignore
        # vmc = pyv.unpickle_from_base64(row['b64_encoded_vmc'])
        coma = pyv.unpickle_from_base64(row["b64_encoded_coma"])
        backflow_estimates.append(calculate_backflow_estimate(coma))
        print(f"{i*100/num_rows:4.1f} % complete")
        # print(f"{i*100/num_rows:4.1f} % complete\r", end='')

    qt.add_column(backflow_estimates, name="fragment_backflow_estimate")


def backflow_plot(gtab: QTable, **kwargs) -> go.Scatter:
    xs = gtab["base_q"].to_value(1 / u.s)  # type: ignore
    ys = gtab["fragment_backflow_estimate"]
    tracename = f"parent τ: {gtab['parent_tau_d'][0]:6.0e}, fragment τ: {gtab['fragment_tau_T'][0]:6.0e}"  # type: ignore

    return go.Scatter(x=xs, y=ys, name=tracename, **kwargs)


def backflow_plots_combined(
    table_file: pathlib.PurePath, data_table: QTable, **kwargs
) -> None:
    fig = make_subplots(rows=1, cols=1)
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    same_params = data_table.group_by(
        ["parent_tau_d", "parent_tau_T", "fragment_tau_T", "v_photo", "v_outflow"]
    )
    for gtab in same_params.groups:
        fig.add_trace(backflow_plot(gtab), **kwargs)

    fig.update_layout(
        title=f"Fragment backflow estimate, dataset: {table_file.stem}",
        xaxis_title="Model input Q(H2O)",
        yaxis_title="Fragment backflow, OH/s",
    )
    fig.show()


def main():
    # sometimes the aperture counts get a little complainy
    warnings.filterwarnings("ignore")

    args = process_args()
    table_file = pathlib.PurePath(args.fitsinput[0])

    # read in table from FITS
    test_table = QTable.read(table_file, format="fits")

    print("Computing backflow estimates ...")
    add_backflow_estimate(test_table)

    test_table.sort(keys="base_q")

    print("Constructing results table ...")
    output_table = test_table

    output_table.remove_columns(
        names=["b64_encoded_vmc", "b64_encoded_coma", "vmc_hash"]
    )

    backflow_plots_combined(table_file, output_table)

    print("Writing results ...")
    output_table_file = pathlib.PurePath("analysis_backflow.ecsv")
    output_table.write(output_table_file, format="ascii.ecsv", overwrite=True)

    print("Complete!")


if __name__ == "__main__":
    sys.exit(main())
