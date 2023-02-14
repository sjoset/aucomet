#!/usr/bin/env python3

import os
import sys

# import dill as pickle
# import time
import pathlib

import logging as log
import numpy as np
import astropy.units as u
import pyvectorial as pyv

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from astropy.table import QTable
from argparse import ArgumentParser

from abel.direct import direct_transform
from abel.basex import basex_transform
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
    """
    Takes a row of a vectorial model calculation table, and returns the results in a VectorialModelResult
    """
    # vmc = pyv.unpickle_from_base64(table_row["b64_encoded_vmc"])
    coma = pyv.unpickle_from_base64(table_row["b64_encoded_coma"])
    vmr = pyv.get_result_from_coma(coma)

    return vmr


def add_model_column_density_interpolation(vmr, fig, row, col):
    """
    Takes a vectorial model result and draws its column density interpolation curve on the given plotly figure
    """
    fig.add_trace(
        pyv.plotly_column_density_interpolation_plot(
            vmr, dist_units=u.km, cdens_units=1 / u.cm**2, mode="lines"
        ),
        row=row,
        col=col,
    )


def add_model_column_density_data(vmr, fig, row, col):
    """
    Takes a vectorial model result and draws its column density on the given plotly figure
    """
    fig.add_trace(
        pyv.plotly_column_density_plot(
            vmr, dist_units=u.km, cdens_units=1 / u.cm**2, opacity=0.5, mode="markers"
        ),
        row=row,
        col=col,
    )


def add_volume_density_interpolation(vmr, fig, row, col):
    """
    Takes a vectorial model result and draws its volume density interpolation curve on the given plotly figure
    """
    fig.add_trace(
        pyv.plotly_volume_density_interpolation_plot(
            vmr, dist_units=u.km, vdens_units=1 / u.cm**3, mode="lines"
        ),
        row=row,
        col=col,
    )


def add_model_volume_density_data(vmr, fig, **kwargs):
    """
    Takes a vectorial model result and draws its volume density on the given plotly figure
    """
    fig.add_trace(
        pyv.plotly_volume_density_plot(
            vmr, dist_units=u.km, vdens_units=1 / u.cm**3, opacity=0.5, mode="markers"
        ),
        **kwargs,
    )


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


def column_density_via_direct_abel_transform(
    vmr: pyv.VectorialModelResult, gridsize: int
):
    """
    Takes a vectorial model result and uses the volume density in a "direct" Abel
    transform to produce np.array(rs), np.array(column densities at rs)
    """

    rs, step = sample_space(
        2 * vmr.collision_sphere_radius,
        vmr.max_grid_radius,
        endpoint=True,
        num=gridsize,
    )

    n_rs = vmr.volume_density_interpolation(rs.to_value(u.m))

    forward_abel = (
        direct_transform(
            n_rs, dr=step.to_value(u.m), direction="forward", correction=True
        )
        / u.m**2
    )

    return rs, forward_abel


def column_density_via_hansenlaw_abel_transform(
    vmr: pyv.VectorialModelResult, gridsize: int
):
    """
    Takes a vectorial model result and uses the volume density in a "hansenlaw" Abel
    transform to produce np.array(rs), np.array(column densities at rs)
    """

    rs, step = sample_space(
        2 * vmr.collision_sphere_radius,
        vmr.max_grid_radius,
        endpoint=True,
        num=gridsize,
    )

    n_rs = vmr.volume_density_interpolation(rs.to_value(u.m))

    forward_abel = (
        hansenlaw_transform(
            n_rs,
            dr=step.to_value(u.m),
            direction="forward",
        )
        / u.m**2
    )

    return rs, forward_abel


def column_density_via_basex_abel_transform(
    vmr: pyv.VectorialModelResult, gridsize: int
):
    """
    Takes a vectorial model result and uses the volume density in a "basex" Abel
    transform to produce np.array(rs), np.array(column densities at rs)
    """

    rs, step = sample_space(
        2 * vmr.collision_sphere_radius,
        vmr.max_grid_radius,
        endpoint=True,
        num=gridsize,
    )

    n_rs = vmr.volume_density_interpolation(rs.to_value(u.m))

    forward_abel = (
        basex_transform(
            n_rs,
            # sigma=0.5,
            verbose=True,
            basis_dir=".",
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

    curve = go.Scatter(x=rs.to_value(u.km), y=cd_ratios)

    fig.add_trace(curve, **kwargs)


def style_and_label_figure(fig, vmr):
    fig.update_xaxes(type="log", tickformat="0.1e", exponentformat="e")
    fig.update_yaxes(type="log", tickformat="0.1e", exponentformat="e")

    title_text = f"Abel transforms and Column Density: collision sphere radius r_c = {vmr.collision_sphere_radius.to_value(u.km):1.2f} km, log Q = 30, no calculations done inside r_c"
    # title_text = f"Abel transforms and Column Density: collision sphere radius r_c = {vmr.collision_sphere_radius.to_value(u.km):1.2f} km, log Q = 30, calculations done inside r_c"
    fig.update_layout(
        autosize=False,
        width=1800,
        height=1200,
        margin=dict(l=50, r=100, b=100, t=100, pad=4),
        paper_bgcolor="#d8d7de",
        plot_bgcolor="#e7e7ea",
        # paper_bgcolor="LightSteelBlue",
        title_text=title_text,
        showlegend=False,
    )

    fig["layout"]["xaxis"]["title"] = "Radius, km"
    for i in range(1, 7):
        fig["layout"]["xaxis" + str(i)]["title"] = "Radius, km"

    fig["layout"]["yaxis"]["title"] = "Column density, 1/cm^2"
    for i in range(1, 7):
        fig["layout"]["yaxis1"]["title"] = "Column density, 1/cm^2"

    annotation_args = {
        "xref": "x domain",
        "yref": "y domain",
        "x": 0.9,
        "y": 0.9,
        "showarrow": False,
        "font": {"family": "monospace", "size": 12, "color": "#301e2a"},
    }
    for row, col, xfrm in [
        (1, 1, "model output"),
        (1, 2, "direct"),
        (2, 1, "hansenlaw"),
        (2, 3, "basex"),
    ]:
        fig.add_annotation(
            text=xfrm,
            row=row,
            col=col,
            **annotation_args,
        )

    for row, col, xfrm in [(1, 3, "direct"), (2, 2, "hansenlaw"), (2, 4, "basex")]:
        fig.add_annotation(
            text=f"model / {xfrm}",
            row=row,
            col=col,
            **annotation_args,
        )


def main():
    args = process_args()
    log.debug("Loading input from %s ....", args.fitsinput[0])
    table_file = pathlib.PurePath(args.fitsinput[0])

    test_table = QTable.read(table_file, format="fits")

    vmr = get_vmr_from_table_row(test_table[0])

    fig = make_subplots(rows=2, cols=4)

    add_model_column_density_data(vmr, fig, row=1, col=1)
    add_model_column_density_interpolation(vmr, fig, row=1, col=1)

    abel_rs_direct, abel_cds_direct = column_density_via_direct_abel_transform(
        vmr, gridsize=1000
    )
    add_abel_transform(abel_rs_direct, abel_cds_direct, fig, row=1, col=2)
    add_model_column_density_data(vmr, fig, row=1, col=2)
    add_abel_transform_comparison(
        vmr, abel_rs_direct, abel_cds_direct, fig, row=1, col=3
    )

    abel_rs_hansenlaw, abel_cds_hansenlaw = column_density_via_hansenlaw_abel_transform(
        vmr, gridsize=70000
    )
    print(abel_rs_hansenlaw)
    add_abel_transform(abel_rs_hansenlaw, abel_cds_hansenlaw, fig, row=2, col=1)
    add_model_column_density_data(vmr, fig, row=2, col=1)
    add_abel_transform_comparison(
        vmr, abel_rs_hansenlaw, abel_cds_hansenlaw, fig, row=2, col=2
    )

    abel_rs_basex, abel_cds_basex = column_density_via_basex_abel_transform(
        vmr, gridsize=200
    )
    add_abel_transform(abel_rs_basex, abel_cds_basex, fig, row=2, col=3)
    add_model_column_density_data(vmr, fig, row=2, col=3)
    add_abel_transform_comparison(vmr, abel_rs_basex, abel_cds_basex, fig, row=2, col=4)

    style_and_label_figure(fig, vmr)

    fig.show()

    fig.write_image("out.pdf")


if __name__ == "__main__":
    sys.exit(main())
