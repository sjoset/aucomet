
import os
import sys
import pathlib
import copy
import contextlib
import multiprocessing
import warnings
import time
import gc

from argparse import ArgumentParser
from typing import List
from itertools import product
from astropy.table import vstack, QTable
from multiprocessing import Pool

import logging as log
import importlib.metadata as impm
import numpy as np
import astropy.units as u
import pyvectorial as pyv

__author__ = 'Shawn Oset'
__version__ = '0.1'


def generate_vmc_set_h2o_small(base_q, r_h) -> List[pyv.VectorialModelConfig]:

    r_h_AU = r_h.to_value(u.AU)
    base_vmc = generate_base_vmc_h2o()

    # scale lifetimes up by r_h^2
    p_tau_ds = np.linspace(50000 * u.s, 100000 * u.s, num=2, endpoint=True) * r_h_AU**2
    f_tau_Ts = np.linspace(100000 * u.s, 220000 * u.s, num=2, endpoint=True) * r_h_AU**2

    vmc_set = []
    for element in product(p_tau_ds, f_tau_Ts):
        new_vmc = copy.deepcopy(base_vmc)
        new_vmc.parent.tau_d = element[0]
        new_vmc.parent.tau_T = element[0] * base_vmc.parent.T_to_d_ratio
        new_vmc.fragment.tau_T = element[1]
        new_vmc.production.base_q = base_q

        # use empirical formula in CS93 for outflow
        new_vmc.parent.v_outflow = (0.85 / np.sqrt(r_h_AU ** 2)) * u.km/u.s

        vmc_set.append(new_vmc)

    return vmc_set


def generate_h2o_fits_file_small(output_fits_file: pathlib.PurePath, r_h, delete_intermediates: bool = False, parallelism=1) -> None:

    # create separate intermediate files for each production and concatenate at the end

    # list of PurePath filenames of intermediate files
    out_file_list = []
    # list of intermediate QTable objects
    out_table_list = []

    qs = np.logspace(28.0, 30.5, num=10, endpoint=True) / u.s
    # qs = np.logspace(28.0, 30.5, num=2, endpoint=True) / u.s
    for i, base_q in enumerate(qs):
        print(f"Current q: {base_q:3.1e}, {i*100/len(qs)} %")
        vmc_set = generate_vmc_set_h2o_small(base_q, r_h=r_h)
        out_table = build_calculation_table(vmc_set, parallelism=parallelism)
        out_filename = pathlib.PurePath(output_fits_file.stem + '_' + str(base_q.to_value(1/u.s)) + '.fits')
        out_file_list.append(out_filename)

        log.info("Table building for base production %s complete, writing results to %s ...", base_q, out_filename)
        remove_file_silent_fail(out_filename)
        out_table.write(out_filename, format='fits')
        out_table_list.append(out_table)

    remove_file_silent_fail(output_fits_file)
    final_table = vstack(out_table_list)
    final_table.write(output_fits_file, format='fits')

    if delete_intermediates:
        print(f"Deleting intermediate files...")
        for file_to_del in out_file_list:
            print(f"\t😵 {file_to_del}")
            remove_file_silent_fail(file_to_del)
