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

    gamma_p = sba.photo_lengthscale('H2O', source='CS93')
    gamma_f = sba.photo_lengthscale('OH', source='CS93')

    return sba.Haser(Q, v, gamma_p, gamma_f)


def plot_cdens_comparison(vmodel, haser, show_plots=True, out_file=None):

    rs, c_ratios = compare_haser(vmodel, haser)
    r_units = u.km
    plt.style.use('Solarize_Light2')

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    ax.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
    ax.set(ylabel="Column density ratio, vectorial/Haser")
    fig.suptitle("Calculated column density ratios, vectorial/Haser")

    ax.set_xscale('log')
    ax.set_ylim([0.5, 1.5])
    ax.plot(rs, c_ratios, color="#688894",  linewidth=2.0)

    plt.legend(loc='upper right', frameon=False)

    if out_file:
        plt.savefig(out_file + '.png')
    if show_plots:
        plt.show()
    plt.close(fig)


def compare_haser(vmodel, haser):

    end_power = np.log10(vmodel['max_grid_radius'].to(u.km).value)
    rs = np.logspace(3, end_power) * u.km

    v_cdens = (vmodel['column_density_interpolation'](rs.to(u.m)) * (1/u.m**2)).to(1/u.cm**2)
    h_cdens = haser.column_density(rs).to(1/(u.cm**2))
    # print(v_cdens)
    # print(h_cdens)

    cd_ratios = v_cdens / h_cdens
    cds = list(zip(v_cdens, h_cdens))
    percent_diff = np.abs((v_cdens - h_cdens)/h_cdens)
    print("Column densities:")
    for (v, h) in cds:
        print(f"{v:5.3e}\t\t{h:5.3e}\t\t{(v/h):5.3e}")

    avg_diff = np.mean(percent_diff)
    print(f"Average difference: {avg_diff}")

    return rs, cd_ratios


def main():

    # astropy units/quantities support in plots
    quantity_support()

    args = process_args()

    log.debug("Loading input from %s ....", args.parameterfile[0])
    # Read in our stuff
    vmc, _ = pyv.vm_config_from_yaml(args.parameterfile[0])

    coma = pyv.run_vmodel(vmc)
    hcoma = run_haser(vmc)

    plot_cdens_comparison(coma.vmodel, hcoma)

    if vmc.etc['print_radial_density']:
        pyv.print_radial_density(coma.vmodel)
    if vmc.etc['print_column_density']:
        pyv.print_column_density(coma.vmodel)
    if vmc.etc['show_agreement_check']:
        pyv.show_fragment_agreement(coma.vmodel)
    if vmc.etc['show_aperture_checks']:
        pyv.show_aperture_checks(coma)

    if vmc.etc['show_radial_plots']:
        pyv.vmplotter.radial_density_plots(coma.vmodel, r_units=u.km, voldens_units=1/u.cm**3, frag_name=vmc.fragment.name)
    if vmc.etc['show_column_density_plots']:
        pyv.vmplotter.column_density_plots(coma.vmodel, r_units=u.km, cd_units=1/u.cm**2, frag_name=vmc.fragment.name)
    if vmc.etc['show_3d_column_density_centered']:
        pyv.vmplotter.column_density_plot_3d(coma.vmodel, x_min=-100000*u.km, x_max=100000*u.km,
                                             y_min=-100000*u.km, y_max=100000*u.km,
                                             grid_step_x=1000, grid_step_y=1000,
                                             r_units=u.km, cd_units=1/u.cm**2,
                                             frag_name=vmc.fragment.name)
    if vmc.etc['show_3d_column_density_off_center']:
        pyv.vmplotter.column_density_plot_3d(coma.vmodel, x_min=-100000*u.km, x_max=10000*u.km,
                                             y_min=-100000*u.km, y_max=10000*u.km,
                                             grid_step_x=1000, grid_step_y=1000,
                                             r_units=u.km, cd_units=1/u.cm**2,
                                             frag_name=vmc.fragment.name)

    pyv.save_vmodel(vmc, coma.vmodel, 'vmout')


if __name__ == '__main__':
    sys.exit(main())
