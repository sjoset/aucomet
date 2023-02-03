#!/usr/bin/env python3

import os
import sys
import pathlib
import copy
import contextlib
import multiprocessing
import warnings
import time

from argparse import ArgumentParser
from typing import List, Tuple
from itertools import product, groupby
from astropy.table import vstack, QTable
from multiprocessing import Pool
from dataclasses import dataclass

import logging as log
import importlib.metadata as impm
import numpy as np
import astropy.units as u
import pyvectorial as pyv

__author__ = 'Shawn Oset'
__version__ = '0.1'


@dataclass
class VMParameterSet:
    parameter_set_id: str
    rh_list: np.array
    base_q_list: np.array
    parent_tau_list: np.array
    fragment_tau_list: np.array
    base_vmc: pyv.VectorialModelConfig


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


vm_parameter_sets = [
        # Water, large data set for full analysis
        VMParameterSet(parameter_set_id = 'h2o',
                       rh_list = np.linspace(1.0, 4.0, num=13, endpoint=True) * u.AU,
                       base_q_list = np.logspace(28.0, 30.5, num=31, endpoint=True) / u.s,
                       parent_tau_list = np.linspace(50000 * u.s, 100000 * u.s, num=10, endpoint=True),
                       fragment_tau_list = np.linspace(100000 * u.s, 220000 * u.s, num=10, endpoint=True),
                       base_vmc = generate_base_vmc_h2o()
                       ),
        # Water, small data set for testing
        VMParameterSet(parameter_set_id = 'h2o_small',
                       rh_list = np.linspace(1.0, 4.0, num=4, endpoint=True) * u.AU,
                       base_q_list = np.logspace(28.0, 30.5, num=4, endpoint=True) / u.s,
                       parent_tau_list = np.linspace(50000 * u.s, 100000 * u.s, num=3, endpoint=True),
                       fragment_tau_list = np.linspace(100000 * u.s, 220000 * u.s, num=3, endpoint=True),
                       base_vmc = generate_base_vmc_h2o()
                       )
        ]


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


def run_vmodel_timed(vmc: pyv.VectorialModelConfig) -> Tuple:

    """
        Service function that takes a vmc, runs a model, and returns results + timing information.

        Returns the encoded coma (using the dill library) because python multiprocessing wants
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


def generate_vmc_set(vmps: VMParameterSet, r_h: u.Quantity) -> List[pyv.VectorialModelConfig]:

    """
        Returns a slice of all possible VectorialModelConfigs at the given r_h.
        Scalings due to heliocentric distance are then applied to lifetimes, and the
        empirical relation between parent outflow speed and r_h is applied as well.
    """

    base_vmc = copy.deepcopy(vmps.base_vmc)

    base_vmc.comet.rh = r_h
    r_h_AU = r_h.to_value(u.AU)

    vmc_set = []
    for element in product(vmps.base_q_list, vmps.parent_tau_list, vmps.fragment_tau_list):
        new_vmc = copy.deepcopy(base_vmc)
        new_vmc.production.base_q = element[0]
        new_vmc.parent.tau_d = element[1] * r_h_AU**2
        new_vmc.parent.tau_T = element[1] * r_h_AU**2 * base_vmc.parent.T_to_d_ratio
        new_vmc.fragment.tau_T = element[2] * r_h_AU**2

        # use empirical formula in CS93 for outflow
        new_vmc.parent.v_outflow = (0.85 / np.sqrt(r_h_AU)) * u.km/u.s

        vmc_set.append(new_vmc)

    return vmc_set


def generate_fits_file(output_fits_file: pathlib.Path, r_h: u.Quantity, vmps: VMParameterSet, delete_intermediates: bool = False, parallelism=1) -> None:

    """
        Compute all of the models at a given AU and generate a (probably large) fits file with all of the results.

    """

    # list of PurePath filenames of intermediate files
    out_file_list = []
    # list of intermediate QTable objects
    out_table_list = []

    # Take a slice of configs at this r_h
    vmc_list = generate_vmc_set(vmps=vmps, r_h=r_h)

    # split our set of vmc into sublists, where every sublist has a matching base_q
    # so we can loop through and compute all the models that have the same base_q and save the intermediate results
    grouped_by_base_q = [list(x) for _, x in groupby(vmc_list, lambda x: x.production.base_q)]

    # build one intermediate output table for each base_q and combine after all of the calculations are done
    for i, vmc_sublist in enumerate(grouped_by_base_q):
        percent_complete = (i * 100)/len(grouped_by_base_q)
        base_q = vmc_sublist[0].production.base_q

        print(f"{r_h.to_value(u.AU)} AU\t\tq: {base_q:3.1e}\t\t{percent_complete:4.1f} %")

        out_filename = pathlib.Path(output_fits_file.stem + '_' + str(base_q.to_value(1/u.s)) + '.fits')
        if out_filename.is_file():
            print(f"Found intermediate file {out_filename}, skipping generation and reading file instead...")
            out_table = QTable.read(out_filename, format='fits')
        else:
            out_table = build_calculation_table(vmc_sublist, parallelism=parallelism)

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


def make_dataset_table(vmps: VMParameterSet) -> QTable:

    """
        TODO: check comments for accuracy, do we really need to use a QTable here? Instead, named tuple? Dict?
        Defines a list of dataset files at a range of heliocentric distances given by au_list
        Returns an astropy QTable with columns:
            | r_h | filename | whether 'filename' already exists |
        We use this to track which data still needs to be generated, and allows different r_h datasets to
        be generated in different sessions in case of power outages etc.
    """

    # aus = np.linspace(1.0, 4.0, num=13, endpoint=True)
    output_fits_filenames = [pathlib.Path(f"{vmps.parameter_set_id}_{rh.to_value(u.AU):3.2f}_au.fits") for rh in vmps.rh_list]
    fits_file_exists = [a.is_file() for a in output_fits_filenames]

    return QTable([vmps.rh_list, output_fits_filenames, fits_file_exists], names=('r_h', 'filename', 'exists'))


def get_dataset_selection() -> VMParameterSet:

    """
        Allows the user to select a dataset from the global list vm_parameter_sets
    """

    dataset_names = [x.parameter_set_id for x in vm_parameter_sets]
    user_selection = None

    while user_selection == None:
        print("Select a dataset:")
        for i, dataset_name in enumerate(dataset_names):
            print(f"{i}:\t{dataset_name}")

        raw_selection = input()
        try:
            selection = int(raw_selection)
        except ValueError:
            print("Numbers only, please")
            selection = -1

        if selection in range(len(dataset_names)):
            user_selection = selection

    return vm_parameter_sets[user_selection]


def main():

    # suppress warnings cluttering up the output
    warnings.filterwarnings("ignore")
    process_args()

    # figure out how parallel the model running can be, leaving one core open (unless there's only a single core)
    parallelism = max(1, multiprocessing.cpu_count()-1)
    print(f"Max CPUs: {multiprocessing.cpu_count()}\tWill use: {parallelism} concurrent processes")

    # we define a few datasets of interest to us - which one to compute?
    vmps = get_dataset_selection()
    dataset_table = make_dataset_table(vmps)

    # TODO: clarify what we mean by dataset
    print(f"Datasets for {vmps.parameter_set_id}:")
    print(dataset_table)

    # fill generate_now column with False and ask which data to generate, if any
    dataset_table.add_column(False, name='generate_now')

    # do all of them, or pick which ones?
    if input(f"Generate all missing data for {vmps.parameter_set_id}? [N/y] ") in ['y', 'Y']:
        # mark all missing data to be generated this run
        for row in dataset_table:
            # if it exists, don't generate, and vice versa
            row['generate_now'] = not row['exists']
    else:
        # pick which missing data to run
        for row in dataset_table:
            if row['exists']:
                continue
            if input(f"Generate data for {row['r_h']}? [N/y] ") in ['y', 'Y']:
                row['generate_now'] = True
                print("OK")

    if not any([row['generate_now'] for row in dataset_table]):
        print("No data selected for generation!  Quitting.")
        return

    print("Current settings:")
    print(dataset_table)
    if input("Calculate? [N/y] ") not in ['y', 'Y']:
        print("Quitting.")
        return

    for row in dataset_table:
        if row['generate_now']:
            generate_fits_file(output_fits_file=row['filename'], r_h=row['r_h'], vmps=vmps, delete_intermediates=True, parallelism=parallelism)


if __name__ == '__main__':
    sys.exit(main())
