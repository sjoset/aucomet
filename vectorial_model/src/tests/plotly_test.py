#!/usr/bin/env python3
import os
import sys
import dill as pickle

import logging as log
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import astropy.units as u
from astropy.visualization import quantity_support
from argparse import ArgumentParser

import pyvectorial as pyv

__author__ = 'Shawn Oset'
__version__ = '0.1'


myred = "#c74a77"
mybred = "#dbafad"
mygreen = "#afac7c"
mybgreen = "#dbd89c"
mypeach = "#dbb89c"
mybpeach = "#e9d4c3"
myblue = "#688894"
mybblue = "#a4b7be"
myblack = "#301e2a"
mybblack = "#82787f"
mywhite = "#d8d7dc"
mybwhite = "#e7e7ea"


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


def dill_from_coma(coma, coma_file) -> None:
    with open(coma_file, 'wb') as comapicklefile:
        pickle.dump(coma, comapicklefile)

def coma_from_dill(coma_file: str):
    with open(coma_file, 'rb') as coma_dill:
        return pickle.load(coma_dill)


def test_volume_and_column_density_plots(vmr: pyv.VectorialModelResult):

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(pyv.plotly_volume_density_plot(vmr, dist_units=u.km, vdens_units=1/u.cm**3, opacity=0.5, mode='markers'),
            row=1, col=1)
    fig.add_trace(pyv.plotly_volume_density_interpolation_plot(vmr, dist_units=u.km, vdens_units=1/u.cm**3, mode='lines'),
            row=1, col=1)
    fig.add_trace(pyv.plotly_column_density_plot(vmr, dist_units=u.km, cdens_units=1/u.cm**2, opacity=0.5, mode='markers'),
            row=1, col=2)
    fig.add_trace(pyv.plotly_column_density_interpolation_plot(vmr, dist_units=u.km, cdens_units=1/u.cm**2, mode='lines'),
            row=1, col=2)
    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[-100,100],),
                         yaxis = dict(nticks=4, range=[-50,100],),
                         zaxis = dict(nticks=4, range=[-100,100],),
            xaxis_title="aoeu",),
           
        width=1800,
        margin=dict(r=20, l=10, b=10, t=10))
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    fig.show()


def test_column_density_3d_plot(vmr: pyv.VectorialModelResult):

    fig = go.Figure()

    fig.add_trace(pyv.plotly_column_density_plot_3d(vmr))

    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4,),
                         yaxis = dict(nticks=4,),
                         zaxis = dict(nticks=4,),
            xaxis_title="aoeu",),
        width=1800,
        margin=dict(r=20, l=10, b=10, t=10))

    fig.show()
    # ax.view_init(90, 90)


def test_fragment_sputter_contour_plot(vmr: pyv.VectorialModelResult):

    fig = go.Figure()

    sputter, outflow, max_coord = pyv.plotly_fragment_sputter_contour_plot(vmr, dist_units=u.km, sputter_units=1/u.cm**3, within_r=3000*u.km, mirrored=True)
    fig.add_trace(sputter)
    if outflow:
        fig.add_trace(outflow)

    fig.data[1].line.color = myblue

    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[-max_coord, max_coord],),
            yaxis = dict(nticks=4, range=[-max_coord, max_coord],),
            # zaxis = dict(nticks=4, range=[-100,100],),
            xaxis_title="aoeu",),
        # width=1800,
        margin=dict(r=20, l=10, b=10, t=10))

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.show()


def test_fragment_sputter_plot(vmr: pyv.VectorialModelResult):

    fig = go.Figure()

    sputter, outflow, max_coord = pyv.plotly_fragment_sputter_plot(vmr, dist_units=u.km, sputter_units=1/u.cm**3, within_r=3000*u.km, mirrored=True, marker_colorscale="Viridis")
    fig.add_trace(sputter)
    if outflow:
        fig.add_trace(outflow)

    fig.data[1].line.color = myblue
    max_coord *= 1.1

    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[-max_coord, max_coord],),
            yaxis = dict(nticks=4, range=[-max_coord, max_coord],),
            # zaxis = dict(nticks=4, range=[-100,100],),
            xaxis_title="aoeu",),
        # width=1800,
        margin=dict(r=20, l=10, b=10, t=10))

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.show()


def test_plotting_q_t(coma):

    fig = go.Figure()
    fig.add_trace(pyv.plotly_q_t_plot(coma, time_units=u.hour))
    fig.show()


def main():

    # astropy units/quantities support in plots
    quantity_support()

    # coma = coma_from_dill('coma_pickle.vm')
    # coma = coma_from_dill('coma_pickle_gaussian.vm')
    # coma = coma_from_dill('coma_pickle_sine.vm')
    # coma = coma_from_dill('coma_pickle_squarepulse.vm')

    # for cpic in ['coma_pickle.vm', 'coma_pickle_gaussian.vm', 'coma_pickle_sine.vm', 'coma_pickle_squarepulse.vm']:
    for cpic in ['coma_pickle_sine.vm']:

        coma = coma_from_dill(cpic)
        vmr = pyv.get_result_from_coma(coma)

        pyv.print_volume_density(vmr)
        pyv.print_column_density(vmr)
        pyv.show_fragment_agreement(vmr)
        pyv.show_aperture_checks(coma)
        print(f"Collision sphere radius: {vmr.collision_sphere_radius}")

        test_volume_and_column_density_plots(vmr)
        test_column_density_3d_plot(vmr)
        test_fragment_sputter_contour_plot(vmr)
        test_fragment_sputter_plot(vmr)
        test_plotting_q_t(coma)


if __name__ == '__main__':
    sys.exit(main())
