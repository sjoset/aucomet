#!/usr/bin/env python3

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
from typing import List, Tuple
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
def remove_file_silent_fail(f: pathlib.Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(f)


# Service function that takes a vmc, runs a model, and returns results + timing information
# We map this function over a set of VectorialModelConfigs to get our list of completed models
def run_vmodel_timed(vmc: pyv.VectorialModelConfig) -> Tuple:

    """
        Return the encoded coma (using the dill library) because python multiprocessing wants
        to pickle return values to send them back to the main calling process.  The coma can't be
        pickled by the stock python pickler so we pickle it here with dill and things are fine
    """

    t_i = time.time()
    coma_pickled = pyv.pickle_to_base64(pyv.run_vmodel(vmc))
    t_f = time.time()

    return (coma_pickled, (t_f - t_i)*u.s)


def build_calculation_table(vmc_set: List[pyv.VectorialModelConfig], parallelism: int=1) -> QTable:

    """
        Take a set of model configs, run them, and return QTable with results of input vmc,
        resulting comae, and model run time
        Uses the multiprocessing module to parallelize the model running, with the number of
        concurrent processes passed in as 'parallelism'
    """

    sbpy_ver = impm.version("sbpy")
    calculation_table = QTable(names=('b64_encoded_vmc', 'vmc_hash', 'b64_encoded_coma'), dtype=('U', 'U', 'U'), meta={'sbpy_ver': sbpy_ver})

    t_i = time.time()
    log.info("Running calculation of %s models with pool size of %s ...", len(vmc_set), parallelism)
    with Pool(parallelism) as vm_pool:
        model_results = vm_pool.map(run_vmodel_timed, vmc_set)
    t_f = time.time()
    log.info("Pooled calculations complete, time: %s", (t_f - t_i)*u.s)

    times_list = []
    for i, vmc in enumerate(vmc_set):

        pickled_coma = model_results[i][0]
        times_list.append(model_results[i][1])
        pickled_vmc = pyv.pickle_to_base64(vmc)

        calculation_table.add_row((pickled_vmc, pyv.hash_vmc(vmc), pickled_coma))

    calculation_table.add_column(times_list, name='model_run_time')

    # now that the model runs are finished, add the config info as columns to the table
    add_vmc_columns(calculation_table)

    return calculation_table


def add_vmc_columns(qt: QTable) -> None:

    """
        Take a QTable of finished vectorial model calculations and add information
        from the VectorialModelConfig as columns in the given table
    """
    vmc_list = [pyv.unpickle_from_base64(row['b64_encoded_vmc']) for row in qt]

    qt.add_column([vmc.production.base_q for vmc in vmc_list], name='base_q')

    qt.add_column([vmc.parent.name for vmc in vmc_list], name='parent_molecule')
    qt.add_column([vmc.parent.tau_d for vmc in vmc_list], name='parent_tau_d')
    qt.add_column([vmc.parent.tau_T for vmc in vmc_list], name='parent_tau_T')
    qt.add_column([vmc.parent.sigma for vmc in vmc_list], name='parent_sigma')
    qt.add_column([vmc.parent.v_outflow for vmc in vmc_list], name='v_outflow')

    qt.add_column([vmc.fragment.name for vmc in vmc_list], name='fragment_molecule')
    qt.add_column([vmc.fragment.tau_T for vmc in vmc_list], name='fragment_tau_T')
    qt.add_column([vmc.fragment.v_photo for vmc in vmc_list], name='v_photo')

    qt.add_column([vmc.comet.rh for vmc in vmc_list], name='r_h')

    qt.add_column([vmc.grid.radial_points for vmc in vmc_list], name='radial_points')
    qt.add_column([vmc.grid.angular_points for vmc in vmc_list], name='angular_points')
    qt.add_column([vmc.grid.radial_substeps for vmc in vmc_list], name='radial_substeps')


def generate_base_vmc_h2o() -> pyv.VectorialModelConfig:

    """
        Returns vmc with parent and fragment info filled out for water, with generic
        settings for base_q, r_h, comet name, and high grid settings
    """

    grid = pyv.Grid(radial_points=150, angular_points=80, radial_substeps=80)
    comet = pyv.Comet(name='NA', rh=1.0 * u.AU, delta=1.0 * u.AU, transform_method=None, transform_applied=False)
    production = pyv.Production(base_q=1.e28/u.s, time_variation_type=None, params=None)

    parent = pyv.Parent(
            name='h2o',
            # v_outflow=0.85 * u.km/u.s,
            v_outflow=0.85/np.sqrt((comet.rh.to_value(u.AU))**2) * u.km/u.s,
            tau_d=86000 * u.s,
            tau_T=86000 * 0.93 * u.s,
            sigma=3e-16 * u.cm**2,
            T_to_d_ratio=0.93
            )

    fragment = pyv.Fragment(
            name='oh',
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


def generate_vmc_set_h2o(base_q: u.Quantity, r_h: u.Quantity) -> List[pyv.VectorialModelConfig]:

    """
        Returns a list of VectorialModelConfigs with the parent and fragment lifetimes
        varied over a range of interest.  Scalings due to heliocentric distance are then
        applied to lifetimes, and the empirical relation between parent outflow speed
        and r_h is applied as well.
    """

    base_vmc = generate_base_vmc_h2o()

    base_vmc.comet.rh = r_h
    r_h_AU = r_h.to_value(u.AU)

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


def generate_h2o_fits_file(output_fits_file: pathlib.Path, r_h: u.Quantity, delete_intermediates: bool = False, parallelism=1) -> None:

    """
        Varies base_q over a range to be investigated, generates an output .fits file
        for each base_q, then combines the .fits files to produce a dataset for water
        at the given heliocentric distance r_h
    """

    # To combine the files, the entire dataset needs to be in memory,
    # so this shouldn't remove the intermediates files and instead check
    # if they already exist and skip them to so that we can resume calculations
    # from the last successful sub-calculation

    # list of PurePath filenames of intermediate files
    out_file_list = []
    # list of intermediate QTable objects
    out_table_list = []

    qs = np.logspace(28.0, 30.5, num=31, endpoint=True) / u.s
    for i, base_q in enumerate(qs):
        percent_complete = (i * 100)/len(qs)
        print(f"{r_h.to_value(u.AU)} AU\t\tq: {base_q:3.1e}\t\t{percent_complete:4.1f} %")

        vmc_set = generate_vmc_set_h2o(base_q, r_h=r_h)
# <<<<<<< HEAD
        out_filename = pathlib.Path(output_fits_file.stem + '_' + str(base_q.to_value(1/u.s)) + '.fits')
        if out_filename.is_file():
            print(f"Found intermediate file {out_filename}, skipping generation and reading file instead...")
            out_table = QTable.read(out_filename, format='fits')
        else:
            out_table = build_calculation_table(vmc_set, parallelism=parallelism)

# =======
#         out_table = build_calculation_table(vmc_set, parallelism=parallelism)
#         out_filename = pathlib.Path(output_fits_file.stem + '_' + str(base_q.to_value(1/u.s)) + '.fits')
# >>>>>>> c45da74 (Vectorial model: find Haser model parameters through fitting given column density data, test version of model with no collision sphere)
        out_file_list.append(out_filename)

        log.info("Table building for base production %s complete, writing results to %s ...", base_q, out_filename)
        remove_file_silent_fail(out_filename)

        out_table.write(out_filename, format='fits')
        out_table_list.append(out_table)

    remove_file_silent_fail(output_fits_file)
    final_table = vstack(out_table_list)
    final_table.write(output_fits_file, format='fits')

    if delete_intermediates:
        log.info(f"Deleting intermediate files...")
        for file_to_del in out_file_list:
            log.info("%s\t😵", file_to_del)
            remove_file_silent_fail(file_to_del)

    del final_table


def make_h2o_dataset_table() -> QTable:

    """
        Defines a list of water dataset files at a range of heliocentric distances, from 1 AU to 4 AU.
        Returns an astropy QTable with columns:
            | r_h | filename | whether 'filename' already exists |
        We use this to track which data still needs to be generated, and allows different r_h datasets to
        be generated in different sessions in case of power outages etc.
    """

    aus = np.linspace(1.0, 4.0, num=13, endpoint=True)
    output_fits_filenames = [pathlib.Path(f"h2o_{au:3.2f}_au.fits") for au in aus]
    fits_file_exists = [a.is_file() for a in output_fits_filenames]

    return QTable([aus * u.AU, output_fits_filenames, fits_file_exists], names=('r_h', 'filename', 'exists'),
                  meta={'parent_molecule': 'h2o'})


def main():

    # suppress warnings cluttering up the output
    warnings.filterwarnings("ignore")
    process_args()

    # generate_h2o_fits_file_small(pathlib.Path('h2osmall_2au.fits'), r_h=2.0 * u.AU, delete_intermediates=True, parallelism=4)

    # figure out how parallel the model running can be, leaving one core open (unless there's only a single core)
    parallelism = max(1, multiprocessing.cpu_count()-1)
    print(f"Max CPUs: {multiprocessing.cpu_count()}\tWill use: {parallelism} concurrent processes")

    # TODO:
    # This can probably be generic because we should just be filling in a table,
    # so this could be a function that takes the dataset table generating function (make_h2o_dataset_table, etc.)
    # and use that to fill in the table as requested, assuming all of our datasets are varied over r_h in some way

    # TODO:
    # This 'water section' could also be main() from another file as a plugin type of architecture, where we just discover all the
    # dataset_*.py files and ask which one to do
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

    if not any([row['generate_now'] for row in h2o_dataset_table]):
        print("No data selected for generation!  Quitting.")
        return

    print("Current settings:")
    print(h2o_dataset_table)
    if input("Calculate? [N/y] ") not in ['y', 'Y']:
        print("Quitting.")
        return

    for row in h2o_dataset_table:
        if row['generate_now']:
            generate_h2o_fits_file(row['filename'], row['r_h'], delete_intermediates=False, parallelism=parallelism)
            # call the garbage collector to clean up between generating data files
            # Not sure if this works but it might help the script finish if we generate all the data at once
            # TODO: figure out if we need this, try it with smaller dataset?
            print(f"Cleaning up: garbage collected {gc.collect()} items.")


if __name__ == '__main__':
    sys.exit(main())
