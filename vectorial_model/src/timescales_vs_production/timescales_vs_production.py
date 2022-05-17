#!/usr/bin/env python3

import os
import sys
import copy
import time

import logging as log
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support
import sbpy.activity as sba
# from sbpy.data import Phys
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


def file_string_id_from_parameters(vmc):

    base_q = vmc.production.base_q.value
    p_tau_d = vmc.parent.tau_d.value

    return f"q_{base_q}_ptau_d_{p_tau_d:06.1f}"


def get_aperture(vmc: pyv.VectorialModelConfig) -> sba.Aperture:

    ap = None

    # get the type
    ap_type = vmc.etc.get('aperture_type')
    if ap_type is None:
        return None

    # get the dimensions as a list and convert to tuple
    ap_dimensions = tuple(vmc.etc.get('aperture_dimensions'))
    ap_dimensions = ap_dimensions * u.km

    if ap_type == 'circular':
        ap = sba.CircularAperture(ap_dimensions)
    elif ap_type == 'rectangular':
        ap = sba.RectangularAperture(ap_dimensions)
    elif ap_type == 'annular':
        ap = sba.AnnularAperture(ap_dimensions)

    return ap


def generateAggregatePlots(t_vs_q_data, out_file=None):

    # Grab each set of values in a flat array for scatter plotting, with productions in log10
    model_qs = np.log10(t_vs_q_data[:, :, 0].flatten())
    timescales = t_vs_q_data[:, :, 1].flatten()
    qs = np.log10(t_vs_q_data[:, :, 2].flatten())

    # Make sure to clear the dark background setting from the 3D plots by defaulting everything here
    plt.style.use('default')
    plt.style.use('Solarize_Light2')

    fig = plt.figure(figsize=(30, 30))
    ax = plt.axes(projection='3d')

    # 3d scatter plot
    ax.scatter(model_qs, timescales, qs)

    # add linear best fit curves for each run
    for model_production in t_vs_q_data:
        # Each column in the data structure has the relevant data
        mqs = np.log10(model_production[:, 0])
        ts = model_production[:, 1]
        cps = np.log10(model_production[:, 2])
        coeffs = np.polyfit(ts, cps, 1)
        fitfunction = np.poly1d(coeffs)
        fit_ys = fitfunction(ts)
        ax.plot3D(mqs, ts, fit_ys, color="#c74a77")
        ax.plot3D(mqs, ts, cps, color="#afac7c")

    ax.set(xlabel="Model production, log10 Q(H2O)")
    ax.set(ylabel="Dissociative lifetime of H2O, (s)")
    ax.set(zlabel="Calculated production, log10 Q(H2O)")
    fig.suptitle("Calculated productions against varying lifetimes for range of model Q(H2O)")

    # Initialize 3d view at these angles for saving the plot
    ax.view_init(20, 30)

    if out_file:
        plt.savefig(out_file)
    plt.show()
    plt.close()


def main():

    """
        Vary the dummy input production and the water lifetimes and save plotting data
    """

    quantity_support()

    args = process_args()
    log.debug("Loading input from %s ....", args.parameterfile[0])

    # Read in our stuff
    vmc_set = pyv.vm_configs_from_yaml(args.parameterfile[0])

    # sort by production value to group results
    vmc_set = sorted(vmc_set, key=lambda vmc: vmc.production.base_q)

    num_qs = len(set(map(lambda x: x.production.base_q, vmc_set)))
    num_timescales = len(set(map(lambda x: x.parent.tau_d, vmc_set)))

    results_list = []
    for i, vmc in enumerate(vmc_set):

        vmc_check = copy.deepcopy(vmc)

        print(f"Q: {vmc.production.base_q}, p_tau_d: {vmc.parent.tau_d}")

        vm_run_t0 = time.time()
        coma = pyv.run_vmodel(vmc)
        vm_run_t1 = time.time()

        assumed_aperture_count = vmc.etc.get('assumed_aperture_count')
        ap = get_aperture(vmc)

        # calculate the number of fragments actually in the aperture and what
        # the production would need to be to produce the assumed count
        aperture_count = coma.total_number(ap)
        calculated_q = (assumed_aperture_count / aperture_count) * vmc.production.base_q

        # run another model with this calculated production to see if it actually reproduces
        # the assumed count
        vmc_check.production.base_q = calculated_q
        vm_check_t0 = time.time()
        coma_check = pyv.run_vmodel(vmc_check)
        vm_check_t1 = time.time()
        aperture_count_check = coma_check.total_number(ap)

        # how well did we reproduce the assumed count?
        aperture_accuracy = 100 * aperture_count_check / assumed_aperture_count

        results_list.append([vmc.production.base_q.value, vmc.parent.tau_d.value, calculated_q.value])

        vmc.etc['aperture_accuracy'] = aperture_accuracy
        vmc.etc['vmodel_run_time'] = [vm_run_t1 - vm_run_t0, vm_check_t1 - vm_check_t0]
        pyv.save_vmodel(vmc, coma.vmodel, 'vmout_'+file_string_id_from_parameters(vmc))
        print(f"Total progress: {(100 * (i+1)) / (num_qs*num_timescales):4.1f} %")
    
    results_array = np.array(results_list).reshape((num_qs, num_timescales, 3))
    with open('linear_fits', 'w') as fit_file:
        log.info(f"Saving linear fits to {fit_file} ...")
        for model_production in results_array:
            this_q = model_production[0][0]
            ts = model_production[:, 1]
            cqs = model_production[:, 2]
            m, b = np.polyfit(ts, cqs, 1)
            print(f"For {this_q:7.3e}, slope best fit: {m:7.3e} with intercept {b:7.3e}", file=fit_file)

    with open('output.npdata', 'wb') as np_file:
        np.save(np_file, results_array)

    generateAggregatePlots(results_array)


if __name__ == '__main__':
    sys.exit(main())
