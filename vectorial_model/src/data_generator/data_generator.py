#!/usr/bin/env python3

import os
import sys
import pathlib
import copy
import contextlib
import multiprocessing
import warnings

import logging as log
from argparse import ArgumentParser
from typing import List
from itertools import product
from astropy.table import vstack, QTable

import numpy as np
import astropy.units as u
import pyvectorial as pyv

__author__ = 'Shawn Oset'
__version__ = '0.1'


def process_args():

    # Parse command-line arguments
    parser = ArgumentParser(
        usage='%(prog)s [options] [inputfile] [outputfile]',
        description=__doc__,
        prog=os.path.basename(sys.argv[0])
    )
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='increase verbosity level')
    # parser.add_argument('output', nargs=1, type=str, help='FITS output filename')

    args = parser.parse_args()

    # handle verbosity
    if args.verbose >= 2:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    elif args.verbose == 1:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    return args


# need this function because pathlib.Path.unlink() throws an error, possible bug or bad install
def remove_file_silent_fail(f: pathlib.PurePath) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(f)


def generate_base_vmc_h2o() -> pyv.VectorialModelConfig:

    grid = pyv.Grid(radial_points=150, angular_points=80, radial_substeps=80)
    comet = pyv.Comet(name='NA', rh=1.0 * u.AU, delta=1.0 * u.AU, transform_method=None, transform_applied=False)
    production = pyv.Production(base_q=1.e28/u.s, time_variation_type=None, params=None)

    parent = pyv.Parent(
            name='parent name',
            # v_outflow=0.85 * u.km/u.s,
            v_outflow=0.85/np.sqrt((comet.rh.to_value(u.AU))**2) * u.km/u.s,
            tau_d=86000 * u.s,
            tau_T=86000 * 0.93 * u.s,
            sigma=3e-16 * u.cm**2,
            T_to_d_ratio=0.93
            )

    fragment = pyv.Fragment(
            name='fragment name',
            v_photo=1.05 * u.km/u.s,
            tau_T=160000 * u.s
            )

    etc = {'print_progress': False}

    vmc = pyv.VectorialModelConfig(
            production=production,
            parent=parent,
            fragment=fragment,
            comet=comet,
            grid=grid,
            etc=etc
            )

    return vmc


def generate_vmc_set_h2o(base_q, r_h) -> List[pyv.VectorialModelConfig]:

    r_h_AU = r_h.to_value(u.AU)
    base_vmc = generate_base_vmc_h2o()

    # scale lifetimes up by r_h^2
    p_tau_ds = np.linspace(50000 * u.s, 100000 * u.s, num=10, endpoint=True) * r_h_AU**2
    f_tau_Ts = np.linspace(100000 * u.s, 220000 * u.s, num=10, endpoint=True) * r_h_AU**2

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


def generate_h2o_fits_file(output_fits_file: pathlib.PurePath, r_h, delete_intermediates: bool = False, parallelism=1) -> None:

    # create separate intermediate files for each production and concatenate at the end

    # list of PurePath filenames of intermediate files
    out_file_list = []
    # list of intermediate QTable objects
    out_table_list = []

    qs = np.logspace(28.0, 30.5, num=31, endpoint=True) / u.s
    for i, base_q in enumerate(qs):
        print(f"Current q: {base_q:3.1e}, {i*100/len(qs)} %")
        vmc_set = generate_vmc_set_h2o(base_q, r_h=r_h)
        out_table = pyv.build_calculation_table(vmc_set, parallelism=parallelism)
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
        out_table = pyv.build_calculation_table(vmc_set, parallelism=parallelism)
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


def make_h2o_dataset_table():
    # returns an astropy QTable of AU, associated filenames for the calculations at this AU,
    # and whether or not that file has already been generated

    aus = np.linspace(1.0, 3.0, num=5, endpoint=True)
    output_fits_filenames = [pathlib.Path(f"h2o_{au:3.2f}_au.fits") for au in aus]
    fits_file_exists = [a.is_file() for a in output_fits_filenames]

    return QTable([aus * u.AU, output_fits_filenames, fits_file_exists], names=('r_h', 'filename', 'exists'),
                  meta={'name': "List of datasets to generate for h2o",
                        'parent_molecule': 'h2o'})


def main():

    # suppress warnings cluttering up the output
    warnings.filterwarnings("ignore")
    process_args()

    # generate_h2o_fits_file_small(output_fits_file, r_h=1.0 * u.AU, delete_intermediates=True, parallelism=4)
    # generate_h2o_fits_file_small(pathlib.PurePath('h2osmall_2au.fits'), r_h=2.0 * u.AU, delete_intermediates=True, parallelism=4)
    # generate_h2o_fits_file(output_fits_file, r_h=1.0 * u.AU, delete_intermediates=True, parallelism=4)
    # generate_h2o_fits_file(output_fits_file, r_h=2.0 * u.AU, delete_intermediates=True, parallelism=4)

    parallelism = max(1, multiprocessing.cpu_count()-1)
    print(f"Max CPUs: {multiprocessing.cpu_count()}\tWill use: {parallelism} concurrent processes")

    """
        Water section
    """
    h2o_dataset_table = make_h2o_dataset_table()

    print(f"Datasets for {h2o_dataset_table.meta['parent_molecule']}:")
    print(h2o_dataset_table)

    # fill generate_now column with False and ask which data to generate, if any
    h2o_dataset_table.add_column(False, name='generate_now')

    # do all of them, or pick which ones?
    if input(f"Generate all missing data for {h2o_dataset_table.meta['parent_molecule']}? [N/y] ") in ['y', 'Y']:
        for row in h2o_dataset_table:
            row['generate_now'] = not row['exists']
    else:
        for row in h2o_dataset_table:
            if row['exists']:
                continue
            if input(f"Generate data for {row['r_h']}? [N/y] ") in ['y', 'Y']:
                row['generate_now'] = True
                print("OK")

    print("Current settings:")
    print(h2o_dataset_table)
    if input("Calculate? [N/y] ") not in ['y', 'Y']:
        print("Quitting.")
        return
    
    for row in h2o_dataset_table:
        if row['generate_now']:
            generate_h2o_fits_file(row['filename'], row['r_h'], delete_intermediates=True, parallelism=parallelism)

if __name__ == '__main__':
    sys.exit(main())
