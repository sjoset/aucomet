#!/usr/bin/env python3

# import copy
import numpy
import os
import sys
# import yaml
import logging as log

import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
# from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from contextlib import redirect_stdout

# import matplotlib.cm as cmx
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import Normalize

import pyvectorial as pyv

__author__ = 'Shawn Oset'
__version__ = '0.1'


# text_blue = (104, 136, 148)
# text_peach = (219, 184, 156)
# text_green = (175, 172, 124)
# text_red = (199, 74, 119)
# text_bblue = (164, 183, 190)
# text_bpeach = (233, 212, 195)
# text_bgreen = (219, 216, 156)
# text_bred = (219, 175, 173)


# def colored(color_tuple, text):
#     r, g, b = color_tuple
#     return f"\033[38;2;{r};{g};{b}m{text} \033[38;2;255;255;255m"


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


def calculate_comparisons(coma, vol_grid_points, vol_dens, col_grid_points, col_dens):

    v_list = []
    c_list = []
    # Sometimes python grid is smaller than fortran version
    max_r = coma.vmodel['max_grid_radius'].to(u.m).value

    for i, r in enumerate(vol_grid_points):
        r_m = r.to(u.m).value
        if r_m > max_r:
            continue
        ratio = ((coma.vmodel['r_dens_interpolation'](r_m) * (1/u.m**3)) / vol_dens[i]).decompose().value
        v_list.append([r_m, ratio])

    v_array = np.array(v_list)
    vrs = v_array[:, 0]
    v_ratios = v_array[:, 1]

    for i, r in enumerate(col_grid_points):
        r_m = r.to(u.m).value
        if r_m > max_r:
            continue
        ratio = ((coma.vmodel['column_density_interpolation'](r_m) * (1/u.m**2)) / col_dens[i]).decompose().value
        c_list.append([r_m, ratio])

    c_array = np.array(c_list)
    crs = c_array[:, 0]
    c_ratios = c_array[:, 1]

    return vrs, v_ratios, crs, c_ratios


def print_comparison(coma, vol_grid_points, vol_dens, col_grid_points, col_dens, out_file=None):

    vrs, v_ratios, crs, c_ratios = calculate_comparisons(coma, vol_grid_points, vol_dens, col_grid_points, col_dens)
    print("\nvolume density at fortran gridpoints, py/fort")

    print(v_ratios)
    for r, vr in zip(vrs, v_ratios):
        print(f"r: {r/1000} km\tpy/fort: {vr}")
    print(f"Average: {np.average(v_ratios)}\t\tMax: {np.amax(v_ratios)}\t\tMin: {np.amin(v_ratios)}")

    print("\ncolumn density at fortran gridpoints, py/fort")
    print(c_ratios)
    for r, cr in zip(crs, c_ratios):
        print(f"r: {r/1000} km\tpy/fort: {cr}")
    print(f"Average: {np.average(c_ratios)}\t\tMax: {np.amax(c_ratios)}\t\tMin: {np.amin(c_ratios)}")

    plot_cdens_comparison(crs, c_ratios, out_file)

    if out_file is not None:
        with open(out_file, 'w') as f:
            with redirect_stdout(f):
                print("\nvolume density at fortran gridpoints, py/fort")
                print(v_ratios)
                print(f"Average: {np.average(v_ratios)}\t\tMax:\
                        {np.amax(v_ratios)}\t\tMin: {np.amin(v_ratios)}")
                for r, vr in zip(vrs, v_ratios):
                    print(f"r: {r/1000} km\tpy/fort: {vr}")
                print("\ncolumn density at fortran gridpoints, py/fort")
                print(c_ratios)
                print(f"Average: {np.average(c_ratios)}\t\tMax:\
                    {np.amax(c_ratios)}\t\tMin: {np.amin(c_ratios)}")
                for r, cr in zip(crs, c_ratios):
                    print(f"r: {r/1000} km\tpy/fort: {cr}")


def plot_cdens_comparison(rs, c_ratios, out_file=None):

    r_units = u.m
    plt.style.use('Solarize_Light2')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
    ax.set(ylabel="Column density ratio, py/fortran")
    fig.suptitle("Calculated column density ratios, python/fortran")
    # print(rs, c_ratios)
    # print(np.min(rs), np.max(rs))

    # ax.set_xbound(lower=np.min(rs), upper=np.max(rs))
    ax.set_xscale('log')
    ax.set_ylim([0, 1.5])
    ax.plot(rs, c_ratios, color="#688894",  linewidth=2.0)
    # ax.plot(rs, c_ratios, color="#688894",  linewidth=2.0, linestyle="-")
    # ax.plot(vmodel['column_density_grid'], vmodel['column_densities'], 'o', color=model_color, label="model")
    # ax1.plot(vmodel['column_density_grid'], vmodel['column_densities'], '--', color=linear_color,
    #          label="linear interpolation", linewidth=1.0)

    plt.legend(loc='upper right', frameon=False)

    if out_file:
        plt.savefig(out_file + '.png')
    plt.show()


# def plot_sputter_python_interpolated(f_sputter, vmodel):

#     # TODO: figure out how to make this less ugly
#     xs, ys, zs = build_sputter_python(vmodel)
#     xs = xs/1e3
#     ys = ys/1e3
#     zs /= 1e9
#     # xs, ys, zs = build_sputter_fortran(f_sputter)

#     xi = np.logspace(-1, 2, 700)
#     yi = np.logspace(-1, 2, 700)
#     # xi = np.linspace(0, 100, 500)
#     # yi = np.linspace(-100, 400, 1000)
#     xi, yi = np.meshgrid(xi, yi)

#     zi = griddata((xs, ys), zs, (xi, yi), method='cubic')
#     plt.figure()
#     plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
#     plt.contourf(xi, yi, zi, 500, cmap='magma')
#     plt.colorbar()
#     plt.show()


def dump_python_fragment_sputter(vmodel, out_file):

    with open(out_file, 'w') as f:
        with redirect_stdout(f):
            print(f"Coma size: {vmodel['coma_radius'].to(u.km):.6e}")
            print(f"Collision sphere size: {vmodel['collision_sphere_radius'].to(u.cm):.6e}")
            print(f"Max grid radius: {vmodel['max_grid_radius'].to(u.km):.6e}")
            print("Python fragment sputter\n\n")
            print("radius\tangle\tfragment sputter density")
            lastr = 0 * u.m
            for i, j in np.ndindex(vmodel['density_grid'].shape):
                r = vmodel['radial_grid'][i]
                theta = vmodel['angular_grid'][j]
                if lastr != r:
                    print("")
                print(f"{r:.5e} {theta:5.9f} {vmodel['density_grid'][i][j]:9.9e}")
                lastr = r


def main():

    # astropy units/quantities support in plots
    quantity_support()

    args = process_args()

    log.debug("Loading input from %s ....", args.parameterfile[0])
    # Read in our stuff
    input_yaml, raw_yaml = pyv.get_input_yaml(args.parameterfile[0])

    # output filenames
    out_file = 'c_ftau_' + str(raw_yaml['fragment']['tau_T'])

    # actually do it
    coma = pyv.run_vmodel(input_yaml)

    # handle optional printing
    if input_yaml['printing']['print_radial_density']:
        pyv.print_radial_density(coma.vmodel)
    if input_yaml['printing']['print_column_density']:
        pyv.print_column_density(coma.vmodel)
    if input_yaml['printing']['show_fragment_agreement_check']:
        pyv.show_fragment_agreement(coma.vmodel)
    if input_yaml['printing']['show_aperture_checks']:
        pyv.show_aperture_checks(coma)

    # Show some plots
    frag_name = input_yaml['fragment']['name']

    if input_yaml['plotting']['show_radial_plots']:
        pyv.radial_density_plots(coma.vmodel, r_units=u.km, voldens_units=1/u.cm**3, frag_name=frag_name)

    if input_yaml['plotting']['show_column_density_plots']:
        pyv.column_density_plots(coma.vmodel, u.km, 1/u.cm**2, frag_name)

    if input_yaml['plotting']['show_3d_column_density_centered']:
        pyv.column_density_plot_3d(coma.vmodel, -100000*u.km, 100000*u.km, -100000*u.km, 100000*u.km, 1000, 1000, u.km,
                                   1/u.cm**2, frag_name)

    if input_yaml['plotting']['show_3d_column_density_off_center']:
        pyv.column_density_plot_3d(coma.vmodel, -100000*u.km, 10000*u.km, -100000*u.km, 10000*u.km, 1000, 100, u.km,
                                   1/u.cm**2, frag_name)

    pyv.tag_input_with_units(raw_yaml)
    # convert our input to something fortran version can digest and run it
    pyv.produce_fortran_fparam(raw_yaml)
    pyv.run_fortran_vmodel(input_yaml['fortran_version']['vmodel_binary'])
    vgrid, vdens, cgrid, cdens, sputter = pyv.read_fortran_vm_output(input_yaml['fortran_version']['out_file'], read_sputter=True)

    print_comparison(coma, vgrid, vdens, cgrid, cdens, out_file)

    # pyv.plot_sputter_fortran(sputter)
    # pyv.plot_sputter_python(coma.vmodel, mirrored=True, trisurf=True)
    # pyv.plot_sputters(sputter, coma.vmodel)

    # plot_sputter_python_interpolated(sputter, coma.vmodel)

    dump_python_fragment_sputter(coma.vmodel, out_file + '_pysputter')


if __name__ == '__main__':
    sys.exit(main())
