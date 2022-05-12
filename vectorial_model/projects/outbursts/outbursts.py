#!/usr/bin/env python3

import os
import sys

import logging as log
import astropy.units as u
import numpy as np
import matplotlib as mpl
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


def file_string_id_from_parameters(vmc, frame_number, varying_production_key):

    base_q = vmc.production.base_q.value
    p_tau_d = vmc.parent.tau_d.value
    f_tau_T = vmc.fragment.tau_T.value

    return f"{frame_number:03d}_q_{base_q}_ptau_d_{p_tau_d:06.1f}_ftau_T_{f_tau_T:06.1f}_{varying_production_key}_{vmc.production.params[varying_production_key].value:02.1f}"


def outburst_radial_density_plots(vmodel, frag_name, out_file, variation_type, varying_production_key, varying_production_value, max_voldens=None):

    # plt, fig, ax1, ax2
    plt, _, ax1, ax2 = pyv.vmplotter.radial_density_plots(vmodel, r_units=u.km, voldens_units=1/u.cm**3, frag_name=frag_name, show_plots=False)
    ax1.set_title(f"{variation_type}: {varying_production_key} = {varying_production_value:05.2f} ago")
    ax2.set_title(f"{variation_type}: {varying_production_key} = {varying_production_value:05.2f} ago")
    ax1.set_ylim([1e2, max_voldens])
    ax2.set_ylim([1e2, max_voldens])

    plt.savefig(out_file)
    plt.close()


def outburst_column_density_plots(vmodel, frag_name, out_file, variation_type, varying_production_key, varying_production_value, min_coldens=None, max_coldens=None):

    # plt, fig, ax1, ax2
    plt, _, ax1, ax2 = pyv.vmplotter.column_density_plots(vmodel, r_units=u.km, cd_units=1/u.cm**2, frag_name=frag_name, show_plots=False)
    ax1.set_title(f"{variation_type} {varying_production_key} = {varying_production_value:05.2f} ago")
    ax2.set_title(f"{variation_type} {varying_production_key} = {varying_production_value:05.2f} ago")
    ax1.set_ylim([min_coldens, max_coldens])
    ax2.set_ylim([min_coldens, max_coldens])

    plt.savefig(out_file)
    plt.close()


def outburst_column_density_plot_3d(vmodel, x_min, x_max, y_min, y_max, grid_step_x, grid_step_y,
                                    r_units, cdens_units, frag_name, out_file,
                                    variation_type, varying_production_key, varying_production_value,
                                    min_coldens=None, max_coldens=None):

    # plt, fig, ax, surf
    plt, _, ax, _ = pyv.column_density_plot_3d(vmodel, x_min, x_max, y_min, y_max,
            grid_step_x, grid_step_y, r_units, cdens_units,
            frag_name, show_plots=False,
            vmin=min_coldens.to(cdens_units).value/2.0, vmax=max_coldens.to(cdens_units).value/2.0)
    ax.set_title(f"{variation_type} {varying_production_key} = {varying_production_value:05.2f} ago")
    # ax.set_zlim([min_coldens, max_coldens])

    plt.savefig(out_file)
    plt.close()


def main():

    quantity_support()

    args = process_args()

    log.debug("Loading input from %s ....", args.parameterfile[0])

    vmc_set = pyv.vm_configs_from_yaml(args.parameterfile[0])

    for variable_key in ['t_max', 'delta', 't_start']:
        if variable_key in vmc_set[0].production.params:
            varying_production_key = variable_key

    # TODO: fix the vertical scale from one frame to the next in all plots
    comae = [None] * len(vmc_set)
    max_voldens = 0 * (1/u.m**3)
    min_coldens = np.Infinity
    max_coldens = 0 * (1/u.m**2)

    for index, vmc in enumerate(vmc_set):

        varying_production_value = vmc.production.params[varying_production_key]

        print(f"Calculating for {varying_production_key} = {varying_production_value:05.2f} :")
        comae[index] = pyv.run_vmodel(vmc)
        print("")

        if comae[index].vmodel['radial_density'][0] > max_voldens:
            max_voldens = comae[index].vmodel['radial_density'][0]
        if comae[index].vmodel['column_densities'][-1] < min_coldens:
            min_coldens = comae[index].vmodel['column_densities'][-1]
        if comae[index].vmodel['column_densities'][0] > max_coldens:
            max_coldens = comae[index].vmodel['column_densities'][0]

    # print(f"{max_coldens=}")
    # Make some headroom on the graph
    min_coldens /= 2.0
    max_coldens *= 7.0/6.0
    max_voldens *= 9.0/6.0

    for frame_number, (vmc, coma) in enumerate(list(zip(vmc_set, comae))):

        varying_production_value = vmc.production.params[varying_production_key]

        if vmc.etc['print_radial_density']:
            pyv.print_radial_density(coma.vmodel)
        if vmc.etc['print_column_density']:
            pyv.print_column_density(coma.vmodel)
        if vmc.etc['show_agreement_check']:
            pyv.show_fragment_agreement(coma.vmodel)
        if vmc.etc['show_aperture_checks']:
            pyv.show_aperture_checks(coma)

        plotbasename = file_string_id_from_parameters(vmc, frame_number=frame_number, varying_production_key=varying_production_key)

        # save the model for inspection later
        # pyv.save_vmodel(vmc, coma.vmodel, 'vmout_'+file_string_id_from_parameters(vmc, varying_production_key))

        print(f"Generating plots for {plotbasename} ...")

        outburst_radial_density_plots(coma.vmodel, frag_name=vmc.fragment.name, out_file=plotbasename+'_rdens.png',
                variation_type=vmc.production.time_variation_type,
                varying_production_key=varying_production_key,
                varying_production_value=varying_production_value,
                max_voldens=max_voldens)
        outburst_column_density_plots(coma.vmodel, frag_name=vmc.fragment.name, out_file=plotbasename+'_coldens2D.png',
                variation_type=vmc.production.time_variation_type,
                varying_production_key=varying_production_key,
                varying_production_value=varying_production_value,
                min_coldens=min_coldens,
                max_coldens=max_coldens)
        outburst_column_density_plot_3d(coma.vmodel, x_min=-100000*u.km, x_max=100000*u.km,
                                        y_min=-100000*u.km, y_max=100000*u.km, grid_step_x=1000,
                                        grid_step_y=1000, r_units=u.km, cdens_units=1/u.cm**2, frag_name=vmc.fragment.name,
                                        out_file=plotbasename+'_coldens3D_view1.png',
                                        variation_type=vmc.production.time_variation_type,
                                        varying_production_key=varying_production_key,
                                        varying_production_value=varying_production_value,
                                        min_coldens=min_coldens,
                                        max_coldens=max_coldens
                                        )
        mpl.rcParams.update(mpl.rcParamsDefault)
        outburst_column_density_plot_3d(coma.vmodel, x_min=-100000*u.km, x_max=10000*u.km,
                                        y_min=-100000*u.km, y_max=10000*u.km, grid_step_x=1000,
                                        grid_step_y=1000, r_units=u.km, cdens_units=1/u.cm**2, frag_name=vmc.fragment.name,
                                        out_file=plotbasename+'_coldens3D_view2.png',
                                        variation_type=vmc.production.time_variation_type,
                                        varying_production_key=varying_production_key,
                                        varying_production_value=varying_production_value,
                                        min_coldens=min_coldens,
                                        max_coldens=max_coldens
                                        )
        mpl.rcParams.update(mpl.rcParamsDefault)

        print("---------------------------------------")
        print("")

        # debug_times_hour = np.arange((day_end*u.day).to(u.hour).value, step=1)
        # debug_times_seconds = np.array(list(map(lambda x:
        #                                (x*u.hour).to(u.s).value,
        #                                debug_times_hour)))
        # debug_prods = q_t(debug_times_seconds)
        # print(np.c_[debug_times_hour, debug_prods])


if __name__ == '__main__':
    sys.exit(main())
