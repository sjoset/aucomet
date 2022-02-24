#!/usr/bin/env python3

import os
import sys

import logging as log
import astropy.units as u
from astropy.visualization import quantity_support
from argparse import ArgumentParser
import sbpy.activity as sba
import numpy as np

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


def run_haser(input_yaml):

    Q = input_yaml['production']['base_q']
    v = input_yaml['parent']['v_outflow']
    gamma_p = sba.photo_lengthscale('H2O', source='CS93')
    gamma_f = sba.photo_lengthscale('OH', source='CS93')

    return sba.Haser(Q, v, gamma_p, gamma_f)


def compare_haser(vmodel, haser):

    end_power = np.log10(vmodel['max_grid_radius'].to(u.km).value)
    rs = np.logspace(3, end_power) * u.km

    v_cdens = (vmodel['column_density_interpolation'](rs.to(u.m)) * (1/u.m**2)).to(1/u.cm**2)
    h_cdens = haser.column_density(rs).to(1/(u.cm**2))
    # print(v_cdens)
    # print(h_cdens)

    cds = list(zip(v_cdens, h_cdens))
    percent_diff = np.abs((v_cdens - h_cdens)/v_cdens)
    print("Column densities:")
    for (v, h) in cds:
        print(f"{v:5.3e}\t\t{h:5.3e}\t\t{(v/h):5.3e}")

    avg_diff = np.mean(percent_diff)
    print(f"Average difference: {avg_diff}")


def main():

    # astropy units/quantities support in plots
    quantity_support()

    args = process_args()

    log.debug("Loading input from %s ....", args.parameterfile[0])
    # Read in our stuff
    input_yaml, raw_yaml = pyv.get_input_yaml(args.parameterfile[0])

    coma = pyv.run_vmodel(input_yaml)
    hcoma = run_haser(input_yaml)

    print("comparing..")
    compare_haser(coma.vmodel, hcoma)

    if input_yaml['printing']['print_radial_density']:
        pyv.print_radial_density(coma.vmodel)
    if input_yaml['printing']['print_column_density']:
        pyv.print_column_density(coma.vmodel)

    if input_yaml['printing']['show_agreement_check']:
        pyv.show_fragment_agreement(coma.vmodel)

    if input_yaml['printing']['show_aperture_checks']:
        pyv.show_aperture_checks(coma)

    # Show any requested plots
    frag_name = input_yaml['fragment']['name']

    if input_yaml['plotting']['show_radial_plots']:
        pyv.vmplotter.radial_density_plots(coma.vmodel, r_units=u.km, voldens_units=1/u.cm**3, frag_name=frag_name)

    if input_yaml['plotting']['show_column_density_plots']:
        pyv.vmplotter.column_density_plots(coma.vmodel, r_units=u.km, cd_units=1/u.cm**2, frag_name=frag_name)

    if input_yaml['plotting']['show_3d_column_density_centered']:
        pyv.vmplotter.column_density_plot_3d(coma.vmodel, x_min=-100000*u.km, x_max=100000*u.km,
                                             y_min=-100000*u.km, y_max=100000*u.km,
                                             grid_step_x=1000, grid_step_y=1000,
                                             r_units=u.km, cd_units=1/u.cm**2,
                                             frag_name=frag_name)

    if input_yaml['plotting']['show_3d_column_density_off_center']:
        pyv.vmplotter.column_density_plot_3d(coma.vmodel, x_min=-100000*u.km, x_max=10000*u.km,
                                             y_min=-100000*u.km, y_max=10000*u.km,
                                             grid_step_x=1000, grid_step_y=1000,
                                             r_units=u.km, cd_units=1/u.cm**2,
                                             frag_name=frag_name)

    # save_vmodel(input_yaml_copy, coma, 'vmout')
    pyv.save_vmodel(raw_yaml, coma.vmodel, 'vmout')


if __name__ == '__main__':
    sys.exit(main())
