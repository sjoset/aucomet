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
            'vmfile', nargs=1, help='yaml of finished vectorial model run, or pickled .vm'
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


def handle_pickled_vm(pickle_file):

    vmc = pyv.VectorialModelConfig(production=None, parent=None, fragment=pyv.Fragment(name='unknown', v_photo=None, tau_T=None), comet=None, grid=None, etc=None)
    vmr = pyv.read_results(pickle_file)

    # no directions about what to print stored in vmodel pickle, so do everything
    pyv.print_radial_density(vmr)
    pyv.print_column_density(vmr)
    pyv.show_fragment_agreement(vmr)

    pyv.radial_density_plots(vmc, vmr, r_units=u.km, voldens_units=1/u.cm**3)
    pyv.column_density_plots(vmc, vmr, r_units=u.km, cd_units=1/u.cm**2)
    pyv.column_density_plot_3d(vmc, vmr, x_min=-100000*u.km, x_max=100000*u.km,
                               y_min=-100000*u.km, y_max=100000*u.km,
                               grid_step_x=1000, grid_step_y=1000,
                               r_units=u.km, cd_units=1/u.cm**2)
    pyv.column_density_plot_3d(vmc, vmr, x_min=-100000*u.km, x_max=10000*u.km,
                               y_min=-100000*u.km, y_max=10000*u.km,
                               grid_step_x=1000, grid_step_y=1000,
                               r_units=u.km, cd_units=1/u.cm**2)


def handle_vm_yaml(yamlfile):

    log.debug("Loading input from %s ....", yamlfile)
    # Read in our stuff - we are assuming that the given yaml file only has one config in it
    vmc = pyv.vm_configs_from_yaml(yamlfile)[0]
    vmr = pyv.read_results(vmc.etc['pyv_coma_pickle'])

    if vmc.etc['print_radial_density']:
        pyv.print_radial_density(vmr)
    if vmc.etc['print_column_density']:
        pyv.print_column_density(vmr)
    if vmc.etc['print_binned_times']:
        pyv.print_binned_times(vmc)
    if vmc.etc['show_agreement_check']:
        pyv.show_fragment_agreement(vmr)
    # This uses the class functions so we can't do this until we save coma objects
    # if vmc.etc['show_aperture_checks']:
    #     pyv.show_aperture_checks(coma)

    if vmc.etc['show_radial_plots']:
        pyv.vmplotter.radial_density_plots(vmc, vmr, r_units=u.km, voldens_units=1/u.cm**3)
    if vmc.etc['show_column_density_plots']:
        pyv.vmplotter.column_density_plots(vmc, vmr, r_units=u.km, cd_units=1/u.cm**2)
    if vmc.etc['show_3d_column_density_centered']:
        pyv.vmplotter.column_density_plot_3d(vmc, vmr, x_min=-100000*u.km, x_max=100000*u.km,
                                             y_min=-100000*u.km, y_max=100000*u.km,
                                             grid_step_x=1000, grid_step_y=1000,
                                             r_units=u.km, cd_units=1/u.cm**2)
    if vmc.etc['show_3d_column_density_off_center']:
        pyv.vmplotter.column_density_plot_3d(vmc, vmr, x_min=-100000*u.km, x_max=10000*u.km,
                                             y_min=-100000*u.km, y_max=10000*u.km,
                                             grid_step_x=1000, grid_step_y=1000,
                                             r_units=u.km, cd_units=1/u.cm**2)


def main():

    # astropy units/quantities support in plots
    quantity_support()

    args = process_args()

    fname = args.vmfile[0]
    _, fext = os.path.splitext(fname)

    # Assume the file is a yaml file if the extension isn't .vm
    if fext == '.vmr':
        handle_pickled_vm(fname)
    else:
        handle_vm_yaml(fname)


if __name__ == '__main__':
    sys.exit(main())
