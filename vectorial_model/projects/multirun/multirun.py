#!/usr/bin/env python3

import os
import sys
import copy

import logging as log
import astropy.units as u
import numpy as np
import itertools
from astropy.visualization import quantity_support
from argparse import ArgumentParser

import pyvectorial as pyv
from pyvectorial import VectorialModelConfig 

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


def get_vmconfig_sets(vmc: VectorialModelConfig) -> list[VectorialModelConfig]:

    # holds a list of all combinations of changing parameters in vmc,
    # values given as lists instead of single values
    varying_parameters = []

    # list of VectorialModelConfigs built from these changing parameters
    psets = []

    # Look at these inputs and build all possible combinations for running
    allowed_variations = [
        vmc.production.base_q,
        vmc.parent.tau_d,
        vmc.fragment.tau_T
        ]

    # check if it is a list, and if not, make it a list of length 1
    # build list of lists of changing values for input to itertools.product
    for av in allowed_variations:
        print(av)
        # already a list, add it
        if isinstance(av.value, np.ndarray):
            varying_parameters.append(av)
        else:
            # Single value specified so make a 1-element list
            varying_parameters.append([av])

    # # check if it is a list, and if not, make it a list of length 1
    # # build list of lists of changing values for input to itertools.product
    # for av in allowed_variations:
    #     # already a list, add it
    #     if av.size != 1 or isinstance(av.value, np.ndarray):
    #         varying_parameters.append(av)
    #     else:
    #         # Single value specified so make a 1-element list
    #         varying_parameters.append([av])

    for element in itertools.product(*varying_parameters):

        print(element)

        # Make copy to append to our list
        new_vmc = copy.deepcopy(vmc)

        # Update this copy with the values we are varying
        # format is a tuple with format (base_q, parent_tau_d, fragment_tau_T)
        new_vmc.production.base_q = element[0]
        new_vmc.parent.tau_d = element[1]
        new_vmc.parent.tau_T = element[1] * new_vmc.parent.T_to_d_ratio
        new_vmc.fragment.tau_T = element[2]

        psets.append(new_vmc)

    return psets


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
    vmc_in, _ = pyv.vm_config_from_yaml(args.parameterfile[0], init_ratio = False)

    for vmc in get_vmconfig_sets(vmc_in):

        # TODO: test if run_vmodel modifies its config input
        vmc_copy = copy.deepcopy(vmc)
        coma = pyv.run_vmodel(vmc_copy)

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

        pyv.save_vmodel(vmc, coma.vmodel, 'vmout_'+file_string_id_from_parameters(vmc))


if __name__ == '__main__':
    sys.exit(main())
