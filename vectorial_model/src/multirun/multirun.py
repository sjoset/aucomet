#!/usr/bin/env python3

import os
import sys

import logging as log
import astropy.units as u
from astropy.visualization import quantity_support
from argparse import ArgumentParser

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


def file_string_id_from_parameters(vmc):

    base_q = vmc.production.base_q.value
    p_tau_d = vmc.parent.tau_d.value
    f_tau_T = vmc.fragment.tau_T.value

    return f"q_{base_q}_ptau_d_{p_tau_d:06.1f}_ftau_T_{f_tau_T:06.1f}"


def main():

    # astropy units/quantities support in plots
    quantity_support()

    args = process_args()

    log.debug("Loading input from %s ....", args.parameterfile[0])

    # Read in our stuff
    vmc_set = pyv.vm_configs_from_yaml(args.parameterfile[0])

    for vmc in vmc_set:

        coma = pyv.run_vmodel(vmc)
        vmr = pyv.get_result_from_coma(coma)

        if vmc.etc['print_radial_density']:
            pyv.print_radial_density(vmr)
        if vmc.etc['print_column_density']:
            pyv.print_column_density(vmr)
        if vmc.etc['show_agreement_check']:
            pyv.show_fragment_agreement(vmr)
        if vmc.etc['show_aperture_checks']:
            pyv.show_aperture_checks(coma)

        if vmc.etc['show_radial_plots']:
            pyv.radial_density_plots(vmc, vmr, r_units=u.km, voldens_units=1/u.cm**3)
        if vmc.etc['show_column_density_plots']:
            pyv.column_density_plots(vmc, vmr, r_units=u.km, cd_units=1/u.cm**2)
        if vmc.etc['show_3d_column_density_centered']:
            pyv.column_density_plot_3d(vmc, vmr, x_min=-100000*u.km,
                    x_max=100000*u.km, y_min=-100000*u.km, y_max=100000*u.km,
                    grid_step_x=1000, grid_step_y=1000, r_units=u.km,
                    cd_units=1/u.cm**2)
        if vmc.etc['show_3d_column_density_off_center']:
            pyv.column_density_plot_3d(vmc, vmr, x_min=-100000*u.km,
                    x_max=10000*u.km, y_min=-100000*u.km, y_max=10000*u.km,
                    grid_step_x=1000, grid_step_y=1000, r_units=u.km,
                    cd_units=1/u.cm**2)

        pyv.save_results(vmc, vmr, 'test')


if __name__ == '__main__':
    sys.exit(main())
