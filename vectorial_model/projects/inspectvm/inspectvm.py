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

    vmodel = pyv.read_vmodel(pickle_file)

    # no directions about what to print stored in vmodel pickle, so do everything
    pyv.print_radial_density(vmodel)
    pyv.print_column_density(vmodel)
    pyv.show_fragment_agreement(vmodel)
    frag_name = 'unknown'

    pyv.radial_density_plots(vmodel, r_units=u.km, voldens_units=1/u.cm**3, frag_name=frag_name)
    pyv.column_density_plots(vmodel, r_units=u.km, cd_units=1/u.cm**2, frag_name=frag_name)
    pyv.column_density_plot_3d(vmodel, x_min=-100000*u.km, x_max=100000*u.km,
                               y_min=-100000*u.km, y_max=100000*u.km,
                               grid_step_x=1000, grid_step_y=1000,
                               r_units=u.km, cd_units=1/u.cm**2,
                               frag_name=frag_name)
    pyv.column_density_plot_3d(vmodel, x_min=-100000*u.km, x_max=10000*u.km,
                               y_min=-100000*u.km, y_max=10000*u.km,
                               grid_step_x=1000, grid_step_y=1000,
                               r_units=u.km, cd_units=1/u.cm**2,
                               frag_name=frag_name)


def handle_vm_yaml(yamlfile):

    log.debug("Loading input from %s ....", yamlfile)
    # Read in our stuff
    input_yaml = pyv.read_yaml_from_file(yamlfile)
    vmodel = pyv.read_vmodel(input_yaml['pyvectorial_info']['vmodel_pickle'])

    if input_yaml['printing']['print_radial_density']:
        pyv.print_radial_density(vmodel)
    if input_yaml['printing']['print_column_density']:
        pyv.print_column_density(vmodel)

    if input_yaml['printing']['show_agreement_check']:
        pyv.show_fragment_agreement(vmodel)

    # This uses the class functions so we can't do this
    # if input_yaml['printing']['show_aperture_checks']:
    #     pyv.show_aperture_checks(coma)

    # Show any requested plots
    frag_name = input_yaml['fragment']['name']

    if input_yaml['plotting']['show_radial_plots']:
        pyv.radial_density_plots(vmodel, r_units=u.km, voldens_units=1/u.cm**3, frag_name=frag_name)

    if input_yaml['plotting']['show_column_density_plots']:
        pyv.vmplotter.column_density_plots(vmodel, r_units=u.km, cd_units=1/u.cm**2, frag_name=frag_name)

    if input_yaml['plotting']['show_3d_column_density_centered']:
        pyv.vmplotter.column_density_plot_3d(vmodel, x_min=-100000*u.km, x_max=100000*u.km,
                                             y_min=-100000*u.km, y_max=100000*u.km,
                                             grid_step_x=1000, grid_step_y=1000,
                                             r_units=u.km, cd_units=1/u.cm**2,
                                             frag_name=frag_name)

    if input_yaml['plotting']['show_3d_column_density_off_center']:
        pyv.vmplotter.column_density_plot_3d(vmodel, x_min=-100000*u.km, x_max=10000*u.km,
                                             y_min=-100000*u.km, y_max=10000*u.km,
                                             grid_step_x=1000, grid_step_y=1000,
                                             r_units=u.km, cd_units=1/u.cm**2,
                                             frag_name=frag_name)


def main():

    # astropy units/quantities support in plots
    quantity_support()

    args = process_args()

    fname = args.vmfile[0]
    _, fext = os.path.splitext(fname)

    if fext == '.vm':
        handle_pickled_vm(fname)
    elif fext == '.yaml':
        handle_vm_yaml(fname)
    else:
        print("Invalid file type")


if __name__ == '__main__':
    sys.exit(main())
