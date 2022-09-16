#!/usr/bin/env python3

import os
import sys
import dill as pickle

import logging as log
import numpy as np
import astropy.units as u
from astropy.table import QTable
from astropy.visualization import quantity_support
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
# import scipy.interpolate

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


def file_string_id_from_parameters(vmc):

    base_q = vmc.production.base_q.value
    p_tau_d = vmc.parent.tau_d.value
    f_tau_T = vmc.fragment.tau_T.value

    return f"q_{base_q}_ptau_d_{p_tau_d:06.1f}_ftau_T_{f_tau_T:06.1f}"


def dill_from_coma(coma, coma_file) -> None:
    with open(coma_file, 'wb') as comapicklefile:
        pickle.dump(coma, comapicklefile)

def coma_from_dill(coma_file: str):
    with open(coma_file, 'rb') as coma_dill:
        return pickle.load(coma_dill)


def test_volume_and_column_density_plots(vmr: pyv.VectorialModelResult):

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

    pyv.column_density_interpolation_plot(vmr, ax1, r_units=u.km, cdens_units=1/u.cm**2, color=mybred)
    pyv.column_density_plot(vmr, ax1, r_units=u.km, cdens_units=1/u.cm**2, color=myred, marker='o')

    pyv.volume_density_interpolation_plot(vmr, ax2, color=mybblue)
    pyv.volume_density_plot(vmr, ax2, r_units=u.km, vdens_units=1/u.cm**3, color=myblue, marker='*')

    # ax1.set_xlim([0, 2000]*u.km)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.show()


def test_column_density_3d_plot(vmr: pyv.VectorialModelResult):

    _ = plt.figure()
    ax = plt.axes(projection='3d')
    pyv.column_density_plot_3d(vmr, ax, divisions=1000, center=(50000, 50000)*u.m, width=100000000 * u.m, height=100000000 * u.m, dist_units=u.km, cdens_units=1/u.cm**2, cmap='inferno', antialiased=False)
    ax.grid(False)
    ax.view_init(90, 90)
    plt.show()


def test_fragment_sputter_contour_plot(vmr: pyv.VectorialModelResult):

    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    pyv.fragment_sputter_contour_plot(vmr, ax, dist_units=u.km, within_r=2000*u.km, mirrored=False, colormap='viridis')
    ax.set_ylim(bottom=0)
    plt.show()


def test_fragment_sputter_plot(vmr: pyv.VectorialModelResult):

    _ = plt.figure()
    ax = plt.axes(projection='3d')
    pyv.fragment_sputter_plot(vmr, ax, dist_units=u.km, within_r=3000*u.km)
    plt.show()


def test_plotting_q_t(coma):

    ts = (np.linspace(-40, 40, num=1000) * u.hour).to_value(u.s)
    f_q = np.vectorize(coma.q_t)
    qs = f_q(ts)
    # qs = f_q(ts) + coma.base_q
    t_h = (ts * u.s).to(u.hour)
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(t_h, qs)
    plt.show()


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
