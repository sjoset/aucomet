#!/usr/bin/env python3

import os
import sys
# import pprint

import astropy.units as u
from astropy.visualization import quantity_support
import sbpy.activity as sba
from sbpy.data import Phys
from argparse import ArgumentParser

import vmplotter
from input_transformer import transform_input_yaml, tag_input_with_units
from parameters import read_parameters_from_file
from utils import print_radial_density, print_column_density
from timedependentproduction import TimeDependentProduction

__author__ = 'Shawn Oset'
__version__ = '0.0'


def print_binned_times(production_dict):
    print("")
    print("Binned time production summary")
    print("------------------------------")
    for q, t in zip(production_dict['q_t'], production_dict['times_at_productions']):
        t_u = t.to(u.day).value
        print(f"Q: {q}\t\tt_start (days ago): {t_u}")
    print("")


def run_vmodel(input_yaml):
    """
        Given input dictionary read from yaml file, run vectorial model and return the coma object
    """

    print("Calculating fragment density using vectorial model ...")

    # apply proper units to the input
    tag_input_with_units(input_yaml)

    # apply any transformations to the input data for heliocentric distance
    print(f"Transforming parent v_outflow and tau_d using {input_yaml['position']['transform_method']}...")
    transform_input_yaml(input_yaml)

    # build parent and fragment inputs
    parent = Phys.from_dict(input_yaml['parent'])
    fragment = Phys.from_dict(input_yaml['fragment'])

    coma = None
    q_t = None

    if 'time_variation_type' in input_yaml['production'].keys():
        t_var_type = input_yaml['production']['time_variation_type']

        # handle each type of supported time dependence
        if t_var_type == "binned":
            print("Found binned production ...")
            if input_yaml['printing']['print_binned_times']:
                print_binned_times(input_yaml['production'])
            # call the older-style binned production constructor
            coma = sba.VectorialModel.binned_production(qs=input_yaml['production']['q_t'],
                                                        parent=parent, fragment=fragment,
                                                        ts=input_yaml['production']['times_at_productions'],
                                                        radial_points=input_yaml['grid']['radial_points'],
                                                        angular_points=input_yaml['grid']['angular_points'],
                                                        radial_substeps=input_yaml['grid']['radial_substeps'],
                                                        print_progress=input_yaml['printing']['print_progress']
                                                        )
        # TODO: print out each parameter with units
        elif t_var_type == "gaussian":
            print("Found gaussian production ...")
            # set up q_t here
            prod_var = TimeDependentProduction(t_var_type)
            q_t = prod_var.create_production(amplitude=input_yaml['production']['amplitude']*(1/u.s),
                                             t_max=input_yaml['production']['t_max']*u.hour,
                                             std_dev=input_yaml['production']['std_dev']*u.hour)
        elif t_var_type == "sine wave":
            print("Found sine wave production ...")
            prod_var = TimeDependentProduction(t_var_type)
            q_t = prod_var.create_production(amplitude=input_yaml['production']['amplitude']*(1/u.s),
                                             period=input_yaml['production']['period']*u.hour,
                                             delta=input_yaml['production']['delta']*u.hour)
        elif t_var_type == "square pulse":
            print("Found square pulse production ...")
            prod_var = TimeDependentProduction(t_var_type)
            q_t = prod_var.create_production(amplitude=input_yaml['production']['amplitude']*(1/u.s),
                                             t_start=input_yaml['production']['t_start']*u.hour,
                                             duration=input_yaml['production']['duration']*u.hour)
        else:
            print("Unknown production variation type!  Defaulting to steady production.")

    # if the binned constructor above hasn't been called, we have work to do
    if coma is None:
        # did we come up with a valid time dependence?
        if q_t is None:
            print(f"No valid time dependence specified, assuming steady "
                  f"production of {input_yaml['production']['base_q']}.")

        coma = sba.VectorialModel(base_q=input_yaml['production']['base_q'],
                                  q_t=q_t,
                                  parent=parent, fragment=fragment,
                                  radial_points=input_yaml['grid']['radial_points'],
                                  angular_points=input_yaml['grid']['angular_points'],
                                  radial_substeps=input_yaml['grid']['radial_substeps'],
                                  print_progress=input_yaml['printing']['print_progress'])

    return coma


def show_aperture_checks(coma, f_theory):
    # Set aperture to entire comet to see if we get all of the fragments as an answer
    print("Percent of total fragments recovered by integrating column density over")
    ap1 = sba.RectangularAperture((coma.vmodel['max_grid_radius'].value, coma.vmodel['max_grid_radius'].value) * u.m)
    print("\tLarge rectangular aperture: ",
          coma.total_number(ap1)*100/f_theory)
    ap2 = sba.CircularAperture((coma.vmodel['max_grid_radius'].value) * u.m)
    print("\tLarge circular aperture: ", coma.total_number(ap2)*100/f_theory)
    ap3 = sba.AnnularAperture([500000, coma.vmodel['max_grid_radius'].value] * u.m)
    print("\tAnnular aperture, inner radius 500000 km, outer radius of entire grid:\n\t",
          coma.total_number(ap3)*100/f_theory)


# TODO: Cite the original parameters, print out the data nicely, maybe dump
# graphs with their filenames included, and write it out as a summary
def report_results(orig_params, coma, output_file):
    # whole_comet_aperture = sba.CircularAperture((coma.vmodel['max_grid_radius']))
    # total_fragments = coma.total_number(whole_comet_aperture)

    # # t_perm = coma.vmodel['t_perm_flow'].to(u.s).value
    # print("Total fragments in the coma by count in the grid: ", total_fragments)
    # # q_frag = total_fragments/(orig_params['fragment']['tau_T'] * 0.95)
    # q_frag = total_fragments/(orig_params['fragment']['tau_T'] * 1.00)
    # # q_par = total_fragments/(orig_params['parent']['tau_T'] * 0.99)
    # q_par = total_fragments/(orig_params['parent']['tau_T'] * 1.0)
    # print("Q_OH:\t", q_frag)
    # print("Q_H2O:\t", q_par)
    # print("Fragments to parents ratio at steady state:", q_frag/q_par)
    # lifetime_ratio = orig_params['fragment']['tau_T']/orig_params['parent']['tau_T']
    # print("Fragment to parent lifetime ratio:", lifetime_ratio)
    # print("Expected ratio of fragments to parents: ", 1/lifetime_ratio)
    # print(coma.vmodel['max_grid_radius'])
    pass


def main():

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

    print(f'Loading input from {args.parameterfile[0]} ...')
    # Read in our stuff
    input_yaml = read_parameters_from_file(args.parameterfile[0])

    # astropy units/quantities support in plots
    quantity_support()

    # # save an unaltered copy for result summary later
    # TODO: yes, we do need to do this
    # input_yaml_copy = copy.deepcopy(input_yaml)

    coma = run_vmodel(input_yaml)

    if input_yaml['printing']['print_radial_density']:
        print_radial_density(coma)
    if input_yaml['printing']['print_column_density']:
        print_column_density(coma)

    f_theory = coma.vmodel['num_fragments_theory']
    f_grid = coma.vmodel['num_fragments_grid']
    if input_yaml['printing']['show_agreement_check']:
        print("Theoretical total number of fragments in coma: ", f_theory)
        print("Total number of fragments from density grid integration: ", f_grid)

    if input_yaml['printing']['show_aperture_checks']:
        show_aperture_checks(coma, f_theory)

    # Show any requested plots
    frag_name = input_yaml['fragment']['name']

    if input_yaml['plotting']['show_radial_plots']:
        vmplotter.radial_density_plots(coma, r_units=u.km, voldens_units=1/u.cm**3, frag_name=frag_name)

    if input_yaml['plotting']['show_column_density_plots']:
        vmplotter.column_density_plots(coma, r_units=u.km, cd_units=1/u.cm**2, frag_name=frag_name)

    if input_yaml['plotting']['show_3d_column_density_centered']:
        vmplotter.column_density_plot_3d(coma, x_min=-100000*u.km, x_max=100000*u.km,
                                         y_min=-100000*u.km, y_max=100000*u.km,
                                         grid_step_x=1000, grid_step_y=1000,
                                         r_units=u.km, cd_units=1/u.cm**2,
                                         frag_name=frag_name)

    if input_yaml['plotting']['show_3d_column_density_off_center']:
        vmplotter.column_density_plot_3d(coma, x_min=-100000*u.km, x_max=10000*u.km,
                                         y_min=-100000*u.km, y_max=10000*u.km,
                                         grid_step_x=1000, grid_step_y=1000,
                                         r_units=u.km, cd_units=1/u.cm**2,
                                         frag_name=frag_name)

    report_results(input_yaml, coma, 'vmodel_output')


if __name__ == '__main__':
    sys.exit(main())
