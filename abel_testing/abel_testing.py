#!/usr/bin/env python3

import os
import sys

# import dill as pickle
# import time
import pathlib

import logging as log
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
from astropy.table import QTable
import astropy.units as u
from astropy.visualization import quantity_support
from argparse import ArgumentParser

import pyvectorial as pyv

from abel.direct import direct_transform
from abel.basex import basex_transform

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


def get_vmr_from_table_row(table_row):
    # vmc = pyv.unpickle_from_base64(table_row["b64_encoded_vmc"])
    coma = pyv.unpickle_from_base64(table_row["b64_encoded_coma"])
    vmr = pyv.get_result_from_coma(coma)

    return vmr


def add_model_column_density_data(vmr, fig, row, col):
    fig.add_trace(
        pyv.plotly_column_density_plot(
            vmr, dist_units=u.km, cdens_units=1 / u.cm**2, opacity=0.5, mode="markers"
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        pyv.plotly_column_density_interpolation_plot(
            vmr, dist_units=u.km, cdens_units=1 / u.cm**2, mode="lines"
        ),
        row=row,
        col=col,
    )


def add_model_volume_density_data(vmr, fig, row, col):
    fig.add_trace(
        pyv.plotly_volume_density_plot(
            vmr, dist_units=u.km, vdens_units=1 / u.cm**3, opacity=0.5, mode="markers"
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        pyv.plotly_volume_density_interpolation_plot(
            vmr, dist_units=u.km, vdens_units=1 / u.cm**3, mode="lines"
        ),
        row=row,
        col=col,
    )


def column_density_via_direct_abel_transform(vmr: pyv.VectorialModelResult):
    # We need an array of values from small r to large r with the fragment volume density
    rs, step = np.linspace(
        3 * vmr.collision_sphere_radius.to_value(u.km),
        vmr.max_grid_radius.to_value(u.km),
        endpoint=True,
        num=1000,
        retstep=True,
    )
    rs = rs * u.km
    step = step * u.km

    n_rs = vmr.volume_density_interpolation(rs.to_value(u.m))

    forward_abel = (
        direct_transform(
            n_rs, dr=step.to_value(u.m), direction="forward", correction=False
        )
        / u.m**2
    )

    return rs, forward_abel


def column_density_via_basex_abel_transform(vmr: pyv.VectorialModelResult):
    rs, step = np.linspace(
        3 * vmr.collision_sphere_radius.to_value(u.km),
        vmr.max_grid_radius.to_value(u.km),
        endpoint=True,
        num=200,
        retstep=True,
    )
    rs = rs * u.km
    step = step * u.km

    n_rs = vmr.volume_density_interpolation(rs.to_value(u.m))

    forward_abel = (
        basex_transform(
            n_rs,
            verbose=True,
            basis_dir=None,
            dr=step.to_value(u.m),
            direction="forward",
        )
        / u.m**2
    )

    return rs, forward_abel


def add_abel_transform(abel_rs, abel_cds, fig, **kwargs):
    curve = go.Scatter(x=abel_rs.to(u.km), y=abel_cds.to(1 / u.cm**2))

    fig.add_trace(curve, **kwargs)


def add_abel_transform_comparison(vmr, abel_rs, abel_cds, fig, **kwargs):
    rs = abel_rs
    model_cds = vmr.column_density_interpolation(rs.to_value(u.m))

    cd_ratios = model_cds / abel_cds

    curve = go.Scatter(x=rs, y=cd_ratios)

    fig.add_trace(curve, **kwargs)


def main():
    # astropy units/quantities support in plots
    quantity_support()

    args = process_args()
    log.debug("Loading input from %s ....", args.fitsinput[0])
    table_file = pathlib.PurePath(args.fitsinput[0])

    test_table = QTable.read(table_file, format="fits")

    vmr = get_vmr_from_table_row(test_table[0])

    fig = make_subplots(rows=1, cols=5)
    # add_model_volume_density_data(vmr, fig, row=1, col=1)
    add_model_column_density_data(vmr, fig, row=1, col=1)

    abel_rs, abel_cds = column_density_via_direct_abel_transform(vmr)
    add_abel_transform(abel_rs, abel_cds, fig, row=1, col=2)
    add_abel_transform_comparison(vmr, abel_rs, abel_cds, fig, row=1, col=3)

    abel_rs, abel_cds = column_density_via_basex_abel_transform(vmr)
    add_abel_transform(abel_rs, abel_cds, fig, row=1, col=4)
    add_abel_transform_comparison(vmr, abel_rs, abel_cds, fig, row=1, col=5)

    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(
    #             nticks=4,
    #             range=[-100, 100],
    #         ),
    #         yaxis=dict(
    #             nticks=4,
    #             range=[-50, 100],
    #         ),
    #         zaxis=dict(
    #             nticks=4,
    #             range=[-100, 100],
    #         ),
    #         xaxis_title="aoeu",
    #     ),
    #     width=1800,
    #     margin=dict(r=20, l=10, b=10, t=10),
    # )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    fig.show()


if __name__ == "__main__":
    sys.exit(main())
