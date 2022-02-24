#!/usr/bin/env python3

import os
import sys
import copy

import logging as log
import astropy.units as u
import itertools
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


def get_parameter_sets(input_yaml):

    # holds a list of all combinations of our changing parameters
    varying_parameters = []

    # list of dictionaries built from these changing parameters
    # to be handed off to run one at a time
    psets = []

    # Look at these inputs and build all possible combinations for running
    allowed_variations = [
        input_yaml['production']['base_q'],
        input_yaml['parent']['tau_d'],
        input_yaml['fragment']['tau_T']
        ]

    # check if it is a list, and if not, make it a list of length 1
    # TODO: this doesn't handle lists of one only one element
    for av in allowed_variations:
        if av.size != 1:
            varying_parameters.append(av)
        else:
            varying_parameters.append([av])

    for element in itertools.product(*varying_parameters):
        new_yaml = copy.deepcopy(input_yaml)

        # Update this copy with the values we are varying
        # format is a tuple with format (base_q, parent_tau_T, parent_tau_d, tau_T)
        new_yaml['production']['base_q'] = element[0]
        new_yaml['parent']['tau_d'] = element[1]
        new_yaml['parent']['tau_T'] = element[1] * new_yaml['parent']['T_to_d_ratio']
        new_yaml['fragment']['tau_T'] = element[2]

        psets.append(new_yaml)

    return psets


def file_string_id_from_parameters(input_yaml):

    base_q = input_yaml['production']['base_q'].value
    p_tau_d = input_yaml['parent']['tau_d'].value
    f_tau_T = input_yaml['fragment']['tau_T'].value

    return f"q_{base_q}_ptau_d_{p_tau_d:06.1f}_ftau_T_{f_tau_T:06.1f}"


def main():

    # astropy units/quantities support in plots
    quantity_support()

    args = process_args()

    log.debug("Loading input from %s ....", args.parameterfile[0])
    # Read in our stuff
    input_yaml, raw_yaml = pyv.get_input_yaml(args.parameterfile[0])

    for pset in get_parameter_sets(input_yaml):

        vpset = copy.deepcopy(pset)
        coma = pyv.run_vmodel(vpset)

        if pset['printing']['print_radial_density']:
            pyv.print_radial_density(coma.vmodel)
        if pset['printing']['print_column_density']:
            pyv.print_column_density(coma.vmodel)

        if pset['printing']['show_agreement_check']:
            pyv.show_fragment_agreement(coma.vmodel)

        if pset['printing']['show_aperture_checks']:
            pyv.show_aperture_checks(coma)

        # Show any requested plots
        frag_name = pset['fragment']['name']

        if pset['plotting']['show_radial_plots']:
            pyv.vmplotter.radial_density_plots(coma.vmodel, r_units=u.km, voldens_units=1/u.cm**3, frag_name=frag_name)

        if pset['plotting']['show_column_density_plots']:
            pyv.vmplotter.column_density_plots(coma.vmodel, r_units=u.km, cd_units=1/u.cm**2, frag_name=frag_name)

        if pset['plotting']['show_3d_column_density_centered']:
            pyv.vmplotter.column_density_plot_3d(coma.vmodel, x_min=-100000*u.km, x_max=100000*u.km,
                                                 y_min=-100000*u.km, y_max=100000*u.km,
                                                 grid_step_x=1000, grid_step_y=1000,
                                                 r_units=u.km, cd_units=1/u.cm**2,
                                                 frag_name=frag_name)

        if pset['plotting']['show_3d_column_density_off_center']:
            pyv.vmplotter.column_density_plot_3d(coma.vmodel, x_min=-100000*u.km, x_max=10000*u.km,
                                                 y_min=-100000*u.km, y_max=10000*u.km,
                                                 grid_step_x=1000, grid_step_y=1000,
                                                 r_units=u.km, cd_units=1/u.cm**2,
                                                 frag_name=frag_name)

        psave = pyv.strip_input_of_units(pset)
        # The data has already been transformed, so if we re-use the dumped yaml, we don't want to transform it again
        psave['position']['transform_method'] = None

        pyv.save_vmodel(psave, coma.vmodel, 'vmout_'+file_string_id_from_parameters(pset))
        # pyv.save_vmodel(pyv.strip_input_of_units(pset), coma.vmodel, 'vmout_'+file_string_id_from_parameters(pset))


if __name__ == '__main__':
    sys.exit(main())
