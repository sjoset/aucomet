#!/usr/bin/env python3

import os
import sys

import logging as log
import astropy.units as u
from astropy.visualization import quantity_support
from argparse import ArgumentParser
import sbpy.activity as sba
import numpy as np
import matplotlib.pyplot as plt

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


def run_haser(vmc):

    Q = vmc.production.base_q
    v = vmc.parent.v_outflow

    # gamma_p = sba.photo_lengthscale('H2O', source='CS93')
    # gamma_f = sba.photo_lengthscale('OH', source='CS93')

    gamma_p = vmc.parent.v_outflow * vmc.parent.tau_d
    gamma_f = vmc.fragment.v_photo * vmc.fragment.tau_T

    return sba.Haser(Q, v, gamma_p, gamma_f)


def plot_cdens_comparison(vmr: pyv.VectorialModelResult, haser, show_plots=True, out_file=None):

    rs, c_ratios = compare_haser(vmr, haser)
    r_units = u.km
    plt.style.use('Solarize_Light2')

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    ax.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
    ax.set(ylabel="Column density ratio, vectorial/Haser")
    fig.suptitle("Calculated column density ratios, vectorial/Haser")

    ax.set_xscale('log')
    ax.set_ylim([0.0, 1.5])
    # ax.set_ylim([0.5, 1.5])
    ax.plot(rs, c_ratios, color="#688894",  linewidth=2.0)

    plt.legend(loc='upper right', frameon=False)

    if out_file:
        plt.savefig(out_file + '.png')
    if show_plots:
        plt.show()
    plt.close(fig)


def compare_haser(vmr: pyv.VectorialModelResult, haser):

    print(f"Max grid: {vmr.max_grid_radius}")
    end_power = np.log10(vmr.max_grid_radius.to(u.km).value)
    rs = np.logspace(3, end_power) * u.km

    v_cdens = (vmr.column_density_interpolation(rs.to(u.m)) * (1/u.m**2)).to(1/u.cm**2)
    h_cdens = haser.column_density(rs).to(1/(u.cm**2))

    cd_ratios = v_cdens / h_cdens
    cds = list(zip(v_cdens, h_cdens))
    percent_diff = np.abs((v_cdens - h_cdens)/h_cdens)
    print("Column densities:")
    for (v, h) in cds:
        print(f"{v:5.3e}\t\t{h:5.3e}\t\t{(v/h):5.3e}")

    avg_diff = np.mean(percent_diff)
    print(f"Average difference: {avg_diff}")

    return rs, cd_ratios


def file_string_id_from_parameters(vmc):

    base_q = vmc.production.base_q.value
    p_tau_d = vmc.parent.tau_d.value
    f_tau_T = vmc.fragment.tau_T.value

    return f"q_{base_q}_ptau_d_{p_tau_d:06.1f}_ftau_T_{f_tau_T:06.1f}"


def main():

    # TODO: add curve fitting to find gamma_f and gamma_p from vectorial model

    # astropy units/quantities support in plots
    quantity_support()

    args = process_args()

    log.debug("Loading input from %s ....", args.parameterfile[0])

    # Read in our stuff
    vmc_set = pyv.vm_configs_from_yaml(args.parameterfile[0])

    for vmc in vmc_set:

        coma = pyv.run_vmodel(vmc)
        vmr = pyv.get_result_from_coma(coma)
        hcoma = run_haser(vmc)

        plot_cdens_comparison(vmr, hcoma)

        # plt, _, ax1, ax2 = pyv.column_density_plots(coma.vmodel, r_units=u.km, cd_units=1/u.cm**2, frag_name=vmc.fragment.name, show_plots=False)
        # plt, _, ax1, ax2 = pyv.column_density_plots(vmc, vmr, r_units=u.km, cd_units=1/u.cm**2, show_plots=False)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
        pyv.mpl_column_density_plot(vmr, ax1, r_units=u.km, cdens_units=1/u.cm**2)
        ax1.plot(np.logspace(1, 7) * u.km, hcoma.column_density(np.logspace(1, 7) * u.km))
        ax2.plot(np.logspace(1, 7) * u.km, hcoma.column_density(np.logspace(1, 7) * u.km))
        plt.show()


if __name__ == '__main__':
    sys.exit(main())
