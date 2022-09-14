#!/usr/bin/env python3

import os
import sys

import logging as log
import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate
from scipy import special, optimize

import pyvectorial as pyv
from matplotlib.colors import Normalize

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


def mesh_fragment_sputter(fsc, dist_units, sputter_units, within_r=None, mirrored=False):

    myblue = "#688894"

    if mirrored:
        fsc = pyv.mirror_sputter(fsc)

    if isinstance(fsc, pyv.FragmentSputterPolar):
        fsc = pyv.cartesian_sputter_from_polar(fsc)

    xs = fsc.xs.to(dist_units)
    ys = fsc.ys.to(dist_units)
    zs = fsc.fragment_density.to(sputter_units)

    within_limit = np.sqrt(xs**2 + ys**2) < (within_r.to(dist_units))

    xs = xs[within_limit]
    ys = ys[within_limit]
    zs = zs[within_limit]

    fig = plt.figure(figsize=(20, 20))

    # plt.style.use('Solarize_Light2')
    # colors_map = 'viridis'

    # cNorm = Normalize(vmin=np.min(zs.value), vmax=np.max(zs.value))
    # scalar_map = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(colors_map))

    # ax = Axes3D(fig)
    ax = plt.gca()
    ax.set(xlabel=f"x, {dist_units.to_string()}")
    ax.set(ylabel=f"y, {dist_units.to_string()}")
    # ax.set(zlabel=f"fragment volume density, {sputter_units.unit.to_string()}")
    fig.suptitle("Fragment sputter")
    # ax.set_box_aspect((1, 1, 1))

    # highlight the outflow axis, along positive y
    origin = [0, 0, 0]
    outflow_max = [0, within_r.to(dist_units).value, 0]
    ax.plot(origin, outflow_max, color=myblue, lw=2, label='outflow axis')

    x_mesh, y_mesh = np.meshgrid(np.unique(xs), np.unique(ys))
    fs_rbf = scipy.interpolate.Rbf(xs, ys, zs, function='cubic')
    frag_mesh = fs_rbf(x_mesh, y_mesh)

    ax.contourf(x_mesh, y_mesh, frag_mesh, levels=np.arange(np.min(frag_mesh), np.max(frag_mesh), 50), cmap=cmx.viridis)

    ax.contour(x_mesh, y_mesh, frag_mesh, levels=np.arange(np.min(frag_mesh), np.max(frag_mesh), 50), colors='black', linewidths=0.5)
    # ax.contour(x_mesh, y_mesh, frag_mesh, levels=np.arange(np.min(frag_mesh), np.max(frag_mesh), 50), cmap=cmx.viridis)

    # ax.scatter(xs, ys, zs, c=scalar_map.to_rgba(zs.value))
    # scalar_map.set_array(fsc.fragment_density.to(sputter_units).value)
    # plt.legend(loc='upper right', frameon=False)

    plt.show()


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
        # if vmc.etc['show_fragment_sputter']:
        #     pyv.plot_fragment_sputter(vmr.fragment_sputter, dist_units=u.km,
        #             sputter_units=1/u.cm**3, within_r=1000*u.km)

        if vmc.etc['show_fragment_sputter']:
            mesh_fragment_sputter(vmr.fragment_sputter, dist_units=u.km,
                    sputter_units=1/u.cm**3, within_r=1000*u.km, mirrored=True)

        pyv.save_results(vmc, vmr, 'test')
        print(f"Collision sphere radius: {vmr.collision_sphere_radius}")


if __name__ == '__main__':
    sys.exit(main())
