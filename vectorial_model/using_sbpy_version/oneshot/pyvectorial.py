#!/usr/bin/env python3

import copy
import numpy
import os
import sys
import yaml
# import pprint

import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
import sbpy.activity as sba
from sbpy.data import Phys
from argparse import ArgumentParser

from vm_plotter import vm_plotter

__author__ = 'Shawn Oset'
__version__ = '0.0'

solarbluecol = np.array([38, 139, 220]) / 255.
solarblue = (solarbluecol[0], solarbluecol[1], solarbluecol[2], 1)
solargreencol = np.array([133, 153, 0]) / 255.
solargreen = (solargreencol[0], solargreencol[1], solargreencol[2], 1)
solarblackcol = np.array([0, 43, 54]) / 255.
solarblack = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)
solarwhitecol = np.array([238, 232, 213]) / 255.
solarwhite = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)


def read_parameters_from_file(filepath):
    """Read the YAML file with all of the input parameters in it"""
    with open(filepath, 'r') as stream:
        try:
            paramYAMLData = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return paramYAMLData


# def show_radial_plots(coma, r_units, volUnits, frag_name):
#     """ Show the radial density of the fragment species """

#     x_min_logplot = 2
#     x_max_logplot = 8

#     x_min_linear = (0 * u.km).to(u.m)
#     x_max_linear = (2000 * u.km).to(u.m)

#     lin_interp_x = np.linspace(x_min_linear.value, x_max_linear.value, num=200)
#     lin_interp_y = coma.vmodel['r_dens_interpolation'](lin_interp_x)/(u.m**3)
#     lin_interp_x *= u.m
#     lin_interp_x.to(r_units)

#     log_interp_x = np.logspace(x_min_logplot, x_max_logplot, num=200)
#     log_interp_y = coma.vmodel['r_dens_interpolation'](log_interp_x)/(u.m**3)
#     log_interp_x *= u.m
#     log_interp_x.to(r_units)

#     plt.style.use('Solarize_Light2')

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

#     ax1.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
#     ax1.set(ylabel=f"Fragment density, {volUnits.unit.to_string()}")
#     ax2.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
#     ax2.set(ylabel=f"Fragment density, {volUnits.unit.to_string()}")
#     fig.suptitle(f"Calculated radial density of {frag_name}")

#     ax1.set_xlim([x_min_linear, x_max_linear])
#     ax1.plot(lin_interp_x, lin_interp_y, color="red",  linewidth=1.5, linestyle="-", label="cubic spline")
#     ax1.plot(coma.vmodel['radial_grid'], coma.vmodel['radial_density'].to(volUnits), 'bo', label="model")
#     ax1.plot(coma.vmodel['radial_grid'], coma.vmodel['radial_density'].to(volUnits), 'g--', label="linear interpolation")

#     ax2.set_xscale('log')
#     ax2.set_yscale('log')
#     ax2.loglog(coma.vmodel['fast_radial_grid'], coma.vmodel['radial_density'].to(volUnits), 'bo', label="model")
#     ax2.loglog(coma.vmodel['fast_radial_grid'], coma.vmodel['radial_density'].to(volUnits), 'g--', label="linear interpolation")
#     ax2.loglog(log_interp_x, log_interp_y, color="red",  linewidth=1.5, linestyle="-", label="cubic spline")

#     ax1.set_ylim(bottom=0)
#     ax2.set_ylim(bottom=0.1)

#     # Mark the beginning of the collision sphere
#     ax1.axvline(x=coma.vmodel['collision_sphere_radius'], color=solarblue)
#     ax2.axvline(x=coma.vmodel['collision_sphere_radius'], color=solarblue)

#     # Mark the collision sphere
#     plt.text(coma.vmodel['collision_sphere_radius']*2, lin_interp_y[0]*2, 'Collision Sphere Edge', color=solarblue)

#     plt.legend(loc='upper right', frameon=False)
#     plt.show()


def show_column_density_plots(coma, r_units, cd_units, frag_name):
    """ Show the radial density of the fragment species """

    x_min_logplot = 2
    x_max_logplot = 8

    x_min_linear = (0 * u.km).to(u.m)
    x_max_linear = (2000 * u.km).to(u.m)

    lin_interp_x = np.linspace(x_min_linear.value, x_max_linear.value, num=200)
    lin_interp_y = coma.vmodel['column_density_interpolation'](lin_interp_x)/(u.m**2)
    lin_interp_x *= u.m
    lin_interp_x.to(r_units)

    log_interp_x = np.logspace(x_min_logplot, x_max_logplot, num=200)
    log_interp_y = coma.vmodel['column_density_interpolation'](log_interp_x)/(u.m**2)
    log_interp_x *= u.m
    log_interp_x.to(r_units)

    plt.style.use('Solarize_Light2')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
    ax1.set(ylabel=f"Fragment column density, {cd_units.unit.to_string()}")
    ax2.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
    ax2.set(ylabel=f"Fragment column density, {cd_units.unit.to_string()}")
    fig.suptitle(f"Calculated column density of {frag_name}")

    ax1.set_xlim([x_min_linear, x_max_linear])
    ax1.plot(lin_interp_x, lin_interp_y, color="red",  linewidth=2.5, linestyle="-", label="cubic spline")
    ax1.plot(coma.vmodel['column_density_grid'], coma.vmodel['column_densities'], 'bo', label="model")
    ax1.plot(coma.vmodel['column_density_grid'], coma.vmodel['column_densities'], 'g--', label="linear interpolation")

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.loglog(coma.vmodel['column_density_grid'], coma.vmodel['column_densities'], 'bo', label="model")
    ax2.loglog(coma.vmodel['column_density_grid'], coma.vmodel['column_densities'], 'g--',
               label="linear interpolation", linewidth=2.5)
    ax2.loglog(log_interp_x, log_interp_y, color="red",  linewidth=0.5, linestyle="-", label="cubic spline")

    ax1.set_ylim(bottom=0)

    ax2.set_xlim(right=coma.vmodel['max_grid_radius'])

    # Mark the beginning of the collision sphere
    ax1.axvline(x=coma.vmodel['collision_sphere_radius'], color=solarblue)
    ax2.axvline(x=coma.vmodel['collision_sphere_radius'], color=solarblue)

    # Only plot as far as the maximum radius of our grid on log-log plot
    ax2.axvline(x=coma.vmodel['max_grid_radius'])

    # Mark the collision sphere
    plt.text(coma.vmodel['collision_sphere_radius']*1.1, lin_interp_y[0]/10, 'Collision Sphere Edge', color=solarblue)

    plt.legend(loc='upper right', frameon=False)
    plt.show()


def show_3d_column_density_plot(coma, x_min, x_max, y_min, y_max, grid_step_x, grid_step_y, r_units, cd_units, frag_name):
    """ 3D plot of column density """

    x = np.linspace(x_min.to(u.m).value, x_max.to(u.m).value, grid_step_x)
    y = np.linspace(y_min.to(u.m).value, y_max.to(u.m).value, grid_step_y)
    xv, yv = np.meshgrid(x, y)
    z = coma.vmodel['column_density_interpolation'](np.sqrt(xv**2 + yv**2))
    # column_density_interpolation spits out m^-2
    fz = (z/u.m**2).to(cd_units)

    xu = np.linspace(x_min.to(r_units), x_max.to(r_units), grid_step_x)
    yu = np.linspace(y_min.to(r_units), y_max.to(r_units), grid_step_y)
    xvu, yvu = np.meshgrid(xu, yu)

    plt.style.use('Solarize_Light2')
    plt.style.use('dark_background')
    plt.rcParams['grid.color'] = "black"

    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes(projection='3d')
    # ax.grid(False)
    surf = ax.plot_surface(xvu, yvu, fz, cmap='inferno', vmin=0, edgecolor='none')

    plt.gca().set_zlim(bottom=0)

    ax.set_xlabel(f'Distance, ({r_units.to_string()})')
    ax.set_ylabel(f'Distance, ({r_units.to_string()})')
    ax.set_zlabel(f"Column density, {cd_units.unit.to_string()}")
    plt.title(f"Calculated column density of {frag_name}")

    ax.w_xaxis.set_pane_color(solargreen)
    ax.w_yaxis.set_pane_color(solarblue)
    ax.w_zaxis.set_pane_color(solarblack)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(90, 90)
    plt.show()


def print_radial_density(coma):
    print("\n\nRadius (km) vs Fragment density (1/cm3)\n---------------------------------------")
    for i in range(0, coma.radial_points):
        print(f"{coma.vmodel['radial_grid'][i].to(u.km):10.1f} : {coma.vmodel['radial_density'][i].to(1/(u.cm**3)):8.4f}")


def print_column_density(coma):
    print("\nRadius (km) vs Column density (1/cm2)\n-------------------------------------")
    cds = list(zip(coma.vmodel['column_density_grid'], coma.vmodel['column_densities']))
    for pair in cds:
        print(f'{pair[0].to(u.km):7.0f} :\t{pair[1].to(1/(u.cm*u.cm)):5.3e}')


def print_binned_times(production_dict):
    print("")
    print("Binned time production summary")
    print("------------------------------")
    for q, t in zip(production_dict['q_t'], production_dict['times_at_productions']):
        t_u = t.to(u.day).value
        print(f"Q: {q}\t\tt_start (days ago): {t_u}")
    print("")


def tag_input_with_units(input_yaml):

    # apply units to data loaded from file
    input_yaml['production']['base_q'] *= (1/u.s)

    # handle the variation types
    if 'time_variation_type' in input_yaml['production'].keys():
        if input_yaml['production']['time_variation_type'] == "binned":
            bin_required = set(['q_t', 'times_at_productions'])
            has_reqs = bin_required.issubset(input_yaml['production'].keys())
            if not has_reqs:
                print("Required keys for binned production not found in production section!")
                print(f"Need {list(bin_required)}.")
                print("Exiting.")
                exit(1)
            input_yaml['production']['q_t'] *= (1/u.s)
            input_yaml['production']['times_at_productions'] *= u.day

    # parent
    input_yaml['parent']['v_outflow'] *= u.km/u.s
    input_yaml['parent']['tau_T'] *= u.s
    input_yaml['parent']['tau_d'] *= u.s
    input_yaml['parent']['sigma'] *= u.cm**2

    # fragment
    input_yaml['fragment']['v_photo'] *= u.km/u.s
    input_yaml['fragment']['tau_T'] *= u.s

    # positional info
    input_yaml['position']['d_heliocentric'] *= u.AU


def transform_input(input_yaml):

    tr_method = input_yaml['position']['transform_method']

    if tr_method == 'cochran_schleicher_93':
        rh = input_yaml['position']['d_heliocentric'].to(u.AU).value
        sqrh = np.sqrt(rh)
        print(f"Transforming parent v_outflow and tau_d using {tr_method} at heliocentric distance {rh}.")

        v_old = copy.deepcopy(input_yaml['parent']['v_outflow'])
        tau_d_old = copy.deepcopy(input_yaml['parent']['tau_d'])
        input_yaml['parent']['v_outflow'] *= 0.85/sqrh
        input_yaml['parent']['tau_d'] *= rh**2
        print(f"\tParent outflow: {v_old} -> {input_yaml['parent']['v_outflow']}")
        print(f"\tParent tau_d: {tau_d_old} -> {input_yaml['parent']['tau_d']}")


def run_vmodel(input_yaml):

    print("Calculating fragment density using vectorial model ...")

    # apply proper units to the input
    tag_input_with_units(input_yaml)

    # apply any transformations to the input data for heliocentric distance
    transform_input(input_yaml)
    print(input_yaml['parent']['tau_d'])

    # build parent and fragment inputs
    parent = Phys.from_dict(input_yaml['parent'])
    fragment = Phys.from_dict(input_yaml['fragment'])

    coma = None
    q_t = None

    if 'time_variation_type' in input_yaml['production'].keys():
        t_var_type = input_yaml['production']['time_variation_type']

        # handle each type of supported time dependence
        if t_var_type == "binned":
            if input_yaml['printing']['print_time_dependence']:
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
        elif t_var_type == "gaussian":
            # set up q_t here
            pass
        elif t_var_type == "sinusoidal":
            pass
        elif t_var_type == "square pulse":
            pass
        else:
            print("Unknown production variation type!  Defaulting to steady production.")

    if q_t is None:
        print(f"No valid time dependence specified, assuming steady"
              f"production of {input_yaml['production']['base_q']}.")

    # check if the binned constructor has been called above
    if coma is None:
        coma = sba.VectorialModel(base_q=input_yaml['production']['base_q'],
                                  q_t=q_t,
                                  parent=parent, fragment=fragment,
                                  radial_points=input_yaml['grid']['radial_points'],
                                  angular_points=input_yaml['grid']['angular_points'],
                                  radial_substeps=input_yaml['grid']['radial_substeps'],
                                  print_progress=input_yaml['printing']['print_progress'])

    return coma


# TODO: Cite the original parameters, print out the data nicely, maybe dump
# graphs with their filenames included, and write it out as a summary
def print_results(orig_params, coma, output_file):
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

    # Show some plots
    frag_name = input_yaml['fragment']['name']

    if input_yaml['plotting']['show_radial_plots']:
        # vm_plotter.show_radial_plots(coma, u.km, 1/u.cm**3, frag_name)
        vm_plotter.radial_density_plots(coma, r_units=u.km, voldens_units=1/u.cm**3, frag_name='OH')

    if input_yaml['plotting']['show_column_density_plots']:
        show_column_density_plots(coma, u.km, 1/u.cm**2, frag_name)

    if input_yaml['plotting']['show_3d_column_density_centered']:
        show_3d_column_density_plot(coma, -100000*u.km, 100000*u.km, -100000*u.km, 100000*u.km, 1000, 1000, u.km,
                                    1/u.cm**2, frag_name)

    if input_yaml['plotting']['show_3d_column_density_off_center']:
        show_3d_column_density_plot(coma, -100000*u.km, 10000*u.km, -100000*u.km, 10000*u.km, 1000, 100, u.km,
                                    1/u.cm**2, frag_name)

    # pprint.pprint(coma.vmodel)
    print_results(input_yaml, coma, 'vmodel_output')


if __name__ == '__main__':
    ys.exit(main())
