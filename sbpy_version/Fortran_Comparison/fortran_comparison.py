#!/usr/bin/env python3

import copy
import numpy
import os
import sys
import yaml
import subprocess

import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
# import matplotlib.pyplot as plt
import sbpy.activity as sba
from sbpy.data import Phys
from argparse import ArgumentParser
from contextlib import redirect_stdout

from VMPlot import vmplot

__author__ = 'Shawn Oset'
__version__ = '0.0'


def read_parameters_from_file(filepath):
    """Read the YAML file with all of the input parameters in it"""
    with open(filepath, 'r') as stream:
        try:
            paramYAMLData = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return paramYAMLData


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
    input_yaml['misc']['r_helio'] *= u.AU


def transform_input(input_yaml):

    tr_method = input_yaml['misc']['transform_method']

    if tr_method == 'cochran_schleicher_93':
        rh = input_yaml['misc']['r_helio'].to(u.AU).value
        sqrh = np.sqrt(rh)
        print(f"Transforming parent v_outflow and tau_d using {tr_method} at heliocentric distance {rh}.")

        v_old = copy.deepcopy(input_yaml['parent']['v_outflow'])
        tau_d_old = copy.deepcopy(input_yaml['parent']['tau_d'])
        input_yaml['parent']['v_outflow'] *= 0.85/sqrh
        input_yaml['parent']['tau_d'] *= rh**2
        print(f"\tParent outflow: {v_old} -> {input_yaml['parent']['v_outflow']}")
        print(f"\tParent tau_d: {tau_d_old} -> {input_yaml['parent']['tau_d']}")
    elif tr_method == 'festou_fortran':
        rh = input_yaml['misc']['r_helio'].to(u.AU).value
        print(f"Transforming all lifetimes using {tr_method} at heliocentric disance {rh}.")
        ptau_d_old = copy.deepcopy(input_yaml['parent']['tau_d'])
        ptau_T_old = copy.deepcopy(input_yaml['parent']['tau_T'])
        ftau_T_old = copy.deepcopy(input_yaml['fragment']['tau_T'])
        input_yaml['parent']['tau_d'] *= rh**2
        input_yaml['parent']['tau_T'] *= rh**2
        input_yaml['fragment']['tau_T'] *= rh**2
        print(f"\tParent tau_d: {ptau_d_old} -> {input_yaml['parent']['tau_d']}")
        print(f"\tParent tau_T: {ptau_T_old} -> {input_yaml['parent']['tau_T']}")
        print(f"\tFragment tau_T: {ftau_T_old} -> {input_yaml['fragment']['tau_T']}")
    else:
        print("No valid transforms found - moving on.")


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


def print_fragment_agreement_check(coma):

    f_theory = coma.vmodel['num_fragments_theory']
    f_grid = coma.vmodel['num_fragments_grid']
    print("Theoretical total number of fragments in coma: ", f_theory)
    print("Total number of fragments from density grid integration: ", f_grid)


def print_aperture_checks(coma):

    f_theory = coma.vmodel['num_fragments_theory']

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


def produce_fortran_fparam(input_yaml):

    fparam_outfile = input_yaml['fortran_version']['infile']

    with open(fparam_outfile, 'w') as out_file:
        with redirect_stdout(out_file):
            print(f"{input_yaml['misc']['comet_name']}")
            print(f"{input_yaml['misc']['r_helio']}  {input_yaml['misc']['delta']}")
            # length of production array: only base production rate for 60 days
            print("1")
            print(f"{input_yaml['production']['base_q']}  60.0")
            # fill in dummy values for the rest of the array
            for _ in range(19):
                print("0.0 61")
            # parent info - speed, total & photo lifetime, destruction level
            print(f"{input_yaml['parent']['v_outflow']}")
            print(f"{input_yaml['parent']['tau_T']}")
            print(f"{input_yaml['parent']['tau_d']}")
            print("95.0")
            # fragment info - gfactor, speed, total lifetime, destruction level
            print(f"{input_yaml['fragment']['name']}")
            print(f"{input_yaml['misc']['gfactor']}")
            print(f"{input_yaml['fragment']['v_photo']}")
            print(f"{input_yaml['fragment']['tau_T']}")
            print("99.0")
            # Custom aperture size, unused for our purposes so these are dummy values
            print("  900.000000       3000.00000")


def run_fortran_vmodel(fortran_vmodel_binary):

    print("\n------------------------------------------------------")
    print(f"Running fortran version at {fortran_vmodel_binary} ...")
    # my vm.f consumes 14 enters before the calculation
    enter_key_string = "\n" * 14
    p1 = subprocess.Popen(["echo", enter_key_string], stdout=subprocess.PIPE)
    p2 = subprocess.run(f"{fortran_vmodel_binary}", stdin=p1.stdout, stdout=open(os.devnull, 'wb'))
    print(f"Complete!  Return code {p2.returncode}.")


def read_fortran_vm_output(fort16_file):

    # Volume density is on line 15 - 27
    fort16_voldens = range(14, 27)
    # Column density is on line 53 - 70
    fort16_coldens = range(52, 70)
    vol_grid_points = []
    col_grid_points = []
    col_dens = []
    vol_dens = []
    with open(fort16_file) as in_file:
        for i, line in enumerate(in_file):
            if i in fort16_voldens:
                vals = [float(x) for x in line.split()]
                vol_grid_points.extend(vals[0::2])
                vol_dens.extend(vals[1::2])
            if i in fort16_coldens:
                vals = [float(x) for x in line.split()]
                col_grid_points.extend(vals[0::2])
                col_dens.extend(vals[1::2])

    return vol_grid_points, vol_dens, col_grid_points, col_dens


def print_comparison(coma, vol_grid_points, vol_dens, col_grid_points,
                     col_dens, out_file=None):

    # fortran outputs grids in km and densities in 1/cm^2, 1/cm^3
    vol_grid_points *= u.km
    col_grid_points *= u.km
    vol_dens *= 1/u.cm**3
    col_dens *= 1/u.cm**2
    v_ratios = []
    c_ratios = []
    # Sometimes python grid is smaller than fortran version
    max_r = coma.vmodel['max_grid_radius'].to(u.m).value

    print("\nvolume density at fortran gridpoints, py/fort")
    for i, r in enumerate(vol_grid_points):
        r_m = r.to(u.m).value
        if r_m > max_r:
            continue
        v_ratios.append(((coma.vmodel['r_dens_interpolation'](r_m) *
                        (1/u.m**3)) / vol_dens[i]).decompose().value)

    print(v_ratios)
    print(f"Average: {np.average(v_ratios)}\t\tMax: {np.amax(v_ratios)}\t\tMin: {np.amin(v_ratios)}")

    print("\ncolumn density at fortran gridpoints, py/fort")
    for i, r in enumerate(col_grid_points):
        r_m = r.to(u.m).value
        if r_m > max_r:
            continue
        c_ratios.append(((coma.vmodel['column_density_interpolation'](r_m) *
                        (1/u.m**2)) / col_dens[i]).decompose().value)

    print(c_ratios)
    print(f"Average: {np.average(c_ratios)}\t\tMax: {np.amax(c_ratios)}\t\tMin: {np.amin(c_ratios)}")

    if out_file is not None:
        with open(out_file, 'w') as f:
            print("\nvolume density at fortran gridpoints, py/fort", file=f)
            print(v_ratios, file=f)
            print(f"Average: {np.average(v_ratios)}\t\tMax:\
                    {np.amax(v_ratios)}\t\tMin: {np.amin(v_ratios)}", file=f)
            print("\ncolumn density at fortran gridpoints, py/fort", file=f)
            print(c_ratios, file=f)
            print(f"Average: {np.average(c_ratios)}\t\tMax:\
                    {np.amax(c_ratios)}\t\tMin: {np.amin(c_ratios)}", file=f)


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

    # save an unaltered copy for result summary later
    input_yaml_copy = copy.deepcopy(input_yaml)

    # actually do it
    coma = run_vmodel(input_yaml)

    # handle optional printing
    if input_yaml['printing']['print_radial_density']:
        print_radial_density(coma)
    if input_yaml['printing']['print_column_density']:
        print_column_density(coma)
    if input_yaml['printing']['show_fragment_agreement_check']:
        print_fragment_agreement_check(coma)
    if input_yaml['printing']['show_aperture_checks']:
        print_aperture_checks(coma)

    # Show some plots
    frag_name = input_yaml['fragment']['name']

    if input_yaml['plotting']['show_radial_plots']:
        vmplot.radial_density_plots(coma, r_units=u.km, voldens_units=1/u.cm**3, frag_name=frag_name)

    if input_yaml['plotting']['show_column_density_plots']:
        vmplot.column_density_plots(coma, u.km, 1/u.cm**2, frag_name)

    if input_yaml['plotting']['show_3d_column_density_centered']:
        vmplot.column_density_plot_3d(coma, -100000*u.km, 100000*u.km, -100000*u.km, 100000*u.km, 1000, 1000, u.km,
                                      1/u.cm**2, frag_name)

    if input_yaml['plotting']['show_3d_column_density_off_center']:
        vmplot.column_density_plot_3d(coma, -100000*u.km, 10000*u.km, -100000*u.km, 10000*u.km, 1000, 100, u.km,
                                      1/u.cm**2, frag_name)

    # convert our input to something fortran version can digest and run it
    produce_fortran_fparam(input_yaml_copy)
    run_fortran_vmodel(input_yaml['fortran_version']['vmodel_binary'])
    vgrid, vdens, cgrid, cdens = read_fortran_vm_output(input_yaml['fortran_version']['outfile'])
    print_comparison(coma, vgrid, vdens, cgrid, cdens, 'comparison_out')


if __name__ == '__main__':
    sys.exit(main())