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
from argparse import ArgumentParser
from itertools import product

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
    f_tau_T = vmc.fragment.tau_T.value

    return f"q_{base_q}_ptau_d_{p_tau_d:06.1f}_ftau_T_{f_tau_T:06.1f}"


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


def main():

    """
        Vary the dummy input production and the parent lifetimes and save plotting data
    """

    quantity_support()

    args = process_args()
    log.debug("Loading input from %s ....", args.parameterfile[0])

    # Read in our template settings
    vmc_base = pyv.vm_configs_from_yaml(args.parameterfile[0])[0]

    # sort by production value to group results
    # vmc_set = sorted(vmc_set, key=lambda vmc: vmc.production.base_q)
    log_q_begin = 27
    log_q_end = 31
    num_qs = 5
    # num_qs = log_q_end - log_q_begin + 1
    input_qs = np.logspace(log_q_begin, log_q_end, num=num_qs, endpoint=True) * 1/u.s

    # parent tau_d
    num_p_tau_ds = 1
    # num_p_tau_ds = 10
    p_tau_d_begin = sba.photo_timescale('H2O') / 2
    p_tau_d_end = sba.photo_timescale('H2O') * 2
    p_tau_ds = np.linspace(p_tau_d_begin, p_tau_d_end, num=num_p_tau_ds , endpoint=True)

    # fragment tau_T
    num_f_tau_Ts = 1
    # num_f_tau_Ts = 10
    f_tau_T_begin = sba.photo_timescale('OH') / 2
    f_tau_T_end = sba.photo_timescale('OH') * 2
    f_tau_Ts = np.linspace(f_tau_T_begin, f_tau_T_end, num=num_f_tau_Ts , endpoint=True)

    total_runs = num_qs * num_p_tau_ds * num_f_tau_Ts
    runs_completed = 0

    results_list = []
    for input_q, p_tau, f_tau in product(input_qs, p_tau_ds, f_tau_Ts):

        # set q, p_tau, f_tau for this run
        vmc = copy.deepcopy(vmc_base)
        vmc.production.base_q = input_q
        vmc.parent.tau_d = p_tau
        vmc.fragment.tau_T = f_tau
        vmc_check = copy.deepcopy(vmc)

        print(f"{input_q=}, {f_tau=}, {p_tau=}")

        # run the model and track execution time
        vm_run_t0 = time.time()
        coma = pyv.run_vmodel(vmc)
        vm_run_t1 = time.time()
        vm_run_time = vm_run_t1 - vm_run_t0

        # create aperture from settings in vmc
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
        vm_check_time = vm_check_t1 - vm_check_t0
        aperture_count_check = coma_check.total_number(ap)

        # how well did we reproduce the assumed count?
        aperture_accuracy = 100 * aperture_count_check / assumed_aperture_count

        results_list.append([
            vmc.production.base_q.value, vmc.parent.tau_d.value, vmc.fragment.tau_T.value, 
            calculated_q.value, aperture_accuracy, vm_run_time, vm_check_time
            ])

        vmc.etc['aperture_accuracy'] = aperture_accuracy
        vmc.etc['vmodel_run_time'] = [vm_run_time, vm_check_time]
        pyv.save_results(vmc, pyv.get_result_from_coma(coma), 'vmout_'+file_string_id_from_parameters(vmc))

        runs_completed += 1
        print(f"Total progress: {(100 * runs_completed) / total_runs:4.1f} %")
    
    results_array = np.array(results_list).reshape((total_runs, 7))

    with open('linear_fits', 'w') as fit_file:
        for pt in p_tau_ds:
            for ft in f_tau_Ts:
                # select only rows with given taus
                pmask = (results_array[:, 1] == pt.value)
                fmask = (results_array[:, 2] == ft.value)
                mask = np.logical_and(pmask, fmask)

                qs = results_array[mask, 0]
                cqs = results_array[mask, 3]
                m, b = np.polyfit(qs, cqs, 1)
                print(f"With p_tau={pt}, f_tau={ft}, q vs. calculated q slope best fit: {m:7.3e} with intercept {b:7.3e}", file=fit_file)

    with open('output.npdata', 'wb') as np_file:
        np.save(np_file, results_array)


if __name__ == '__main__':
    sys.exit(main())
