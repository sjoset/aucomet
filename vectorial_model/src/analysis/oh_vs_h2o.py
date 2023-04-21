#!/usr/bin/env python3

import os
import sys
import pathlib
import warnings

import logging as log
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


def add_qOH_column(qt: QTable) -> None:
    num_rows = len(qt)
    # add a column of fragment counts inside large aperture divided by time to permanent flow regime for q(OH) per second?
    qOHs = []
    # column for empirical relation in cochran & schleicher 98 between Q(H2O) and Q(OH)
    emp_qH2Os = []
    model_to_emp_ratios = []

    for i, row in enumerate(qt):  # type: ignore
        vmc = pyv.unpickle_from_base64(row["b64_encoded_vmc"])
        coma = pyv.unpickle_from_base64(row["b64_encoded_coma"])

        count_in_largest_ap = coma.total_number(
            sba.CircularAperture(coma.vmr.max_grid_radius)
        )
        qOH = count_in_largest_ap / coma.vmr.t_perm_flow.to_value(u.s)
        # TODO: cite the origin of this 1.361 empirical relation
        emp_qH2O = 1.361 * qOH / u.s
        model_to_emp_ratio = vmc.production.base_q / emp_qH2O

        qOHs.append(qOH)
        emp_qH2Os.append(emp_qH2O)
        model_to_emp_ratios.append(model_to_emp_ratio)
        print(f"{i*100/num_rows:4.1f} % complete\r", end="")

    print("")
    qt.add_column(qOHs, name="qOH_max_aperture")
    qt.add_column(emp_qH2Os, name="emp_qH2Os")
    qt.add_column(model_to_emp_ratios, name="model_to_emp_ratio")


def qOH_vs_qH2O_plot(gtab: QTable, **kwargs) -> go.Scatter:
    xs = gtab["base_q"].to_value(1 / u.s)  # type: ignore
    ys = gtab["model_to_emp_ratio"]
    tracename = f"parent τ: {gtab['parent_tau_d'][0]:6.0f}, fragment τ: {gtab['fragment_tau_T'][0]:6.0f}"  # type: ignore

    return go.Scatter(x=xs, y=ys, name=tracename, **kwargs)


def qOH_vs_qH2O_plots_combined(
    table_file: pathlib.PurePath, data_table: QTable, **kwargs
) -> None:
    fig = make_subplots(rows=1, cols=1)
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    same_params = data_table.group_by(
        ["parent_tau_d", "parent_tau_T", "fragment_tau_T", "v_photo", "v_outflow"]
    )
    for gtab in same_params.groups:
        fig.add_trace(qOH_vs_qH2O_plot(gtab), **kwargs)

    fig.update_layout(
        title=f"Vectorial model Q(H2O) vs Empirical relation Q(H2O), dataset: {table_file.stem}",
        xaxis_title="Model input Q(H2O)",
        yaxis_title="Model input Q(H2O)/Empirical Q(H2O)",
    )
    fig.show()


def main():
    # sometimes the aperture counts get a little complainy
    warnings.filterwarnings("ignore")
    args = process_args()
    table_file = pathlib.PurePath(args.fitsinput[0])

    # read in table from FITS
    test_table = QTable.read(table_file, format="fits")

    print("Computing q(OH) ...")
    add_qOH_column(test_table)

    test_table.sort(keys="base_q")

    print("Constructing results table ...")
    output_table = test_table

    output_table.remove_columns(
        names=["b64_encoded_vmc", "b64_encoded_coma", "vmc_hash"]
    )

    qOH_vs_qH2O_plots_combined(table_file, output_table)

    print("Writing results ...")
    output_table_file = pathlib.PurePath("analysis.ecsv")
    output_table.write(output_table_file, format="ascii.ecsv", overwrite=True)

    print("Complete!")


if __name__ == "__main__":
    sys.exit(main())
