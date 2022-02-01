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
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from contextlib import redirect_stdout

import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

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


def print_comparison(coma, vol_grid_points, vol_dens, col_grid_points,
                     col_dens, out_file=None):

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
            with redirect_stdout(f):
                print("\nvolume density at fortran gridpoints, py/fort")
                print(v_ratios)
                print(f"Average: {np.average(v_ratios)}\t\tMax:\
                        {np.amax(v_ratios)}\t\tMin: {np.amin(v_ratios)}")
                print("\ncolumn density at fortran gridpoints, py/fort")
                print(c_ratios)
                print(f"Average: {np.average(c_ratios)}\t\tMax:\
                    {np.amax(c_ratios)}\t\tMin: {np.amin(c_ratios)}")


def build_spray_fortran(spray):

    # get close to the nucleus - fortran distances are in km
    spray = np.array([x for x in spray if x[0] < 10000])

    rs = spray[:, 0]
    thetas = spray[:, 1]
    zs = spray[:, 2]

    xs = rs*np.sin(thetas)
    ys = rs*np.cos(thetas)

    return xs, ys, zs


def plot_spray_fortran(spray):

    xs, ys, zs = build_spray_fortran(spray)

    colorsMap = 'jet'
    cm = plt.get_cmap(colorsMap)
    cNorm = Normalize(vmin=min(zs), vmax=max(zs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(xs, ys, zs, color='white', edgecolors='grey', alpha=0.5)
    ax.scatter(xs, ys, zs, c=scalarMap.to_rgba(zs))
    scalarMap.set_array(zs)
    fig.colorbar(scalarMap)
    plt.show()


def build_spray_python(vmodel):

    vm_rs = vmodel['fast_radial_grid']
    vm_thetas = vmodel['angular_grid']

    spraylist = []
    for (i, j), vdens in np.ndenumerate(vmodel['density_grid']):
        spraylist.append([vm_rs[i], vm_thetas[j], vdens])
    spray = np.array(spraylist)

    # get close to the nucleus
    spray = np.array([x for x in spray if x[0] < 10000000])

    rs = spray[:, 0]
    thetas = spray[:, 1]
    zs = spray[:, 2]

    xs = rs*np.sin(thetas)
    ys = rs*np.cos(thetas)

    return xs, ys, zs


def plot_spray_python(vmodel):

    xs, ys, zs = build_spray_python(vmodel)

    colorsMap = 'jet'
    cm = plt.get_cmap(colorsMap)
    cNorm = Normalize(vmin=min(zs), vmax=max(zs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(xs, ys, zs, color='white', edgecolors='grey', alpha=0.5)
    ax.scatter(xs, ys, zs, c=scalarMap.to_rgba(zs))
    scalarMap.set_array(zs)
    fig.colorbar(scalarMap)
    plt.show()


def plot_sprays(f_spray, vmodel):

    pxs, pys, pzs = build_spray_python(vmodel)
    # convert python distances to km from m
    pxs = pxs/1000
    pys = pys/1000
    # convert python density to 1/cm**3 from 1/m**3
    pzs = pzs/1e6
    fxs, fys, fzs = build_spray_fortran(f_spray)

    colorsMap = 'jet'
    cm = plt.get_cmap(colorsMap)
    cNorm = Normalize(vmin=min(pzs), vmax=max(pzs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.plot_trisurf(xs, ys, zs, color='white', edgecolors='grey', alpha=0.5)
    ax.scatter(pxs, pys, pzs, c=scalarMap.to_rgba(pzs))
    # ax.scatter(fxs, fys, fzs, c=scalarMap.to_rgba(fzs))
    ax.scatter(fxs, fys, fzs, color='red')
    scalarMap.set_array(pzs)
    fig.colorbar(scalarMap)
    plt.show()


def plot_spray_python_interpolated(f_spray, vmodel):

    # TODO: figure out how to make this less ugly
    xs, ys, zs = build_spray_python(vmodel)
    xs = xs/1e3
    ys = ys/1e3
    zs /= 1e9
    # xs, ys, zs = build_spray_fortran(f_spray)

    xi = np.logspace(-1, 2, 700)
    yi = np.logspace(-1, 2, 700)
    # xi = np.linspace(0, 100, 500)
    # yi = np.linspace(-100, 400, 1000)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((xs, ys), zs, (xi, yi), method='cubic')
    plt.figure()
    plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
    plt.contourf(xi, yi, zi, 500, cmap='magma')
    plt.colorbar()
    plt.show()


def main():

    # astropy units/quantities support in plots
    quantity_support()

    args = process_args()

    log.debug("Loading input from %s ....", args.parameterfile[0])
    # Read in our stuff
    input_yaml, raw_yaml = pyv.get_input_yaml(args.parameterfile[0])

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
    vgrid, vdens, cgrid, cdens, spray = pyv.read_fortran_vm_output(input_yaml['fortran_version']['outfile'], read_spray=True)

    print_comparison(coma, vgrid, vdens, cgrid, cdens, 'comparison_out')

    # plot_spray_fortran(spray)
    # plot_spray_python(coma.vmodel)
    plot_sprays(spray, coma.vmodel)
    # plot_spray_python_interpolated(spray, coma.vmodel)


if __name__ == '__main__':
    sys.exit(main())
