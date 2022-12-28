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
            'fitsinput', nargs=1, help='fits file that contains calculation table'
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


def add_vmc_columns(qt: QTable) -> None:

    # extract VectorialModelConfig and add columns based on some parameters
    base_q_list = []
    ptd_list = []
    ptT_list = []
    voutflow_list = []
    ftT_list = []
    vphoto_list = []

    for row in qt:
        vmc = pyv.unpickle_from_base64(row['b64_encoded_vmc'])
        base_q_list.append(vmc.production.base_q)
        voutflow_list.append(vmc.parent.v_outflow)
        ptd_list.append(vmc.parent.tau_d)
        ptT_list.append(vmc.parent.tau_T)
        ftT_list.append(vmc.fragment.tau_T)
        vphoto_list.append(vmc.fragment.v_photo)

    qt.add_column(base_q_list, name='base_q')
    qt.add_column(ptd_list, name='parent_tau_d')
    qt.add_column(ptT_list, name='parent_tau_T')
    qt.add_column(voutflow_list, name='v_outflow')
    qt.add_column(ftT_list, name='fragment_tau_T')
    qt.add_column(vphoto_list, name='v_photo')


def calculate_backflow_estimate(coma: sba.VectorialModel) -> np.float64:

    # TODO: can't we just use the innermost points (r_min, whatever theta) without re-computing?

    # TODO: make this theta space work and don't just grab angular_grid
    # num_backflow_samples = 10
    # thetas = np.geomspace(0.0001, np.pi, num=num_backflow_samples, endpoint=False)

    # make a list of grid points along the collision sphere at different thetas
    thetas = coma.angular_grid

    rs = np.empty_like(thetas)
    rs.fill(np.min(coma.grid.radial_points))
    # rs.fill(coma.vmr.collision_sphere_radius.to_value(u.m) * 2)

    # from the vectorial model math
    integration_factor = (
            (1 / (4.0 * np.pi * coma.parent.tau_d)) *
            coma.d_alpha / (4.0 * np.pi)
            )

    # so we can use numpy magic
    sputter_func = np.vectorize(coma._fragment_sputter)
    # fragment densities at r, theta due to one outflow axis along positive z axis
    backflow_sputters = integration_factor * sputter_func(rs, thetas)

    # fragment densities due to a set of outflow axis pointing out along theta, weighted by spherical coordinate surface area element
    # that this axis occupies
    solid_angle_backflows = np.sin(thetas) * backflow_sputters

    # add up all of these densities due to the outflow axes, and include factor for phi
    total_backflow = 2.0 * np.pi * np.sum(solid_angle_backflows)

    return total_backflow


def add_backflow_estimate(qt: QTable) -> None:

    num_rows = len(qt)
    backflow_estimates = []

    for i, row in enumerate(qt):
        # vmc = pyv.unpickle_from_base64(row['b64_encoded_vmc'])
        coma = pyv.unpickle_from_base64(row['b64_encoded_coma'])
        backflow_estimates.append(calculate_backflow_estimate(coma))
        print(f"{i*100/num_rows:4.1f} % complete\r", end='')

    qt.add_column(backflow_estimates, name='fragment_backflow_estimate')


def backflow_plot(gtab: QTable, **kwargs) -> go.Scatter:

    xs = gtab['base_q'].to_value(1/u.s)
    ys = gtab['fragment_backflow_estimate']
    tracename = f"parent τ: {gtab['parent_tau_d'][0]:6.0e}, fragment τ: {gtab['fragment_tau_T'][0]:6.0e}"

    return go.Scatter(x=xs, y=ys, name=tracename, **kwargs)


def backflow_plots_combined(table_file: pathlib.PurePath, data_table: QTable, **kwargs) -> None:

    fig = make_subplots(rows=1, cols=1)
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    same_params = data_table.group_by(['parent_tau_d', 'parent_tau_T', 'fragment_tau_T', 'v_photo', 'v_outflow'])
    for gtab in same_params.groups:
        fig.add_trace(backflow_plot(gtab), **kwargs)
    
    fig.update_layout(
            title=f"Fragment backflow estimate, dataset: {table_file.stem}",
            xaxis_title="Model input Q(H2O)",
            yaxis_title="Fragment backflow, OH/s"
            )
    fig.show()


def main():
    
    # # sometimes the aperture counts get a little complainy
    # warnings.filterwarnings("ignore")
    args = process_args()
    table_file = pathlib.PurePath(args.fitsinput[0])

    # read in table from FITS
    test_table = QTable.read(table_file, format='fits')

    # add some analysis data to table and sort by base_q
    print("Adding columns for model input parameters ...")
    add_vmc_columns(test_table)

    print("Computing backflow estimates ...")
    add_backflow_estimate(test_table)

    test_table.sort(keys='base_q')

    print("Constructing results table ...")
    output_table = test_table

    output_table.remove_columns(names=['b64_encoded_vmc', 'b64_encoded_coma', 'vmc_hash'])

    # qOH_vs_qH2O_plots_combined(table_file, output_table)
    backflow_plots_combined(table_file, output_table)

    print("Writing results ...")
    output_table_file = pathlib.PurePath('analysis.ecsv')
    output_table.write(output_table_file, format='ascii.ecsv', overwrite=True)

    print("Complete!")


if __name__ == '__main__':
    sys.exit(main())
