#!/usr/bin/env python3

import os
import sys
import pathlib
import copy
import contextlib
import multiprocessing
import warnings

from argparse import ArgumentParser
from typing import List
from itertools import product, groupby
from astropy.table import vstack, QTable

from dataclasses import dataclass

import logging as log
import numpy as np
import astropy.units as u
import pyvectorial as pyv

__author__ = "Shawn Oset"
__version__ = "0.1"


@dataclass
class GridVsAccuracyParameterSet:
    parameter_set_id: str
    radial_gridsize_list: np.ndarray
    angular_gridsize_list: np.ndarray
    radial_substep_list: np.ndarray
    base_vmc: pyv.VectorialModelConfig


def generate_base_vmc_h2o() -> pyv.VectorialModelConfig:
    """
    Returns vmc with parent and fragment info filled out for water, with generic
    settings for base_q, r_h, comet name, and high grid settings
    """

    grid = pyv.Grid(radial_points=150, angular_points=80, radial_substeps=80)
    comet = pyv.Comet(
        name="NA",
        rh=1.0 * u.AU,  # type: ignore
        delta=1.0 * u.AU,  # type: ignore
        transform_method=None,  # type: ignore
        transform_applied=False,
    )
    production = pyv.Production(
        base_q=1.0e28 / u.s, time_variation_type=None, params=None  # type: ignore
    )

    parent = pyv.Parent(
        name="h2o",
        # v_outflow=0.85 * u.km/u.s,
        v_outflow=0.85 / np.sqrt((comet.rh.to_value(u.AU)) ** 2) * u.km / u.s,  # type: ignore
        tau_d=86000 * u.s,
        tau_T=86000 * 0.93 * u.s,
        sigma=3e-16 * u.cm**2,  # type: ignore
        T_to_d_ratio=0.93,
    )

    fragment = pyv.Fragment(name="oh", v_photo=1.05 * u.km / u.s, tau_T=160000 * u.s)  # type: ignore

    etc = {"print_progress": False}

    vmc = pyv.VectorialModelConfig(
        production=production,
        parent=parent,
        fragment=fragment,
        comet=comet,
        grid=grid,
        etc=etc,
    )

    return vmc


vm_parameter_sets = [
    GridVsAccuracyParameterSet(
        parameter_set_id="gridsize_vs_accuracy",
        radial_gridsize_list=np.linspace(50, 300, num=20, endpoint=True, dtype=int),
        angular_gridsize_list=np.linspace(30, 100, num=3, endpoint=True, dtype=int),
        radial_substep_list=np.linspace(80, 300, num=20, endpoint=True, dtype=int),
        base_vmc=generate_base_vmc_h2o(),
    )
]


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile] [outputfile]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
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


def generate_vmc_set(
    vmps: GridVsAccuracyParameterSet,
) -> List[pyv.VectorialModelConfig]:
    """
    Returns a list of VectorialModelConfigs with varying grid sizes, but all other inputs are kept the same
    """
    base_vmc = copy.deepcopy(vmps.base_vmc)

    vmc_set = []
    for element in product(
        vmps.radial_gridsize_list, vmps.angular_gridsize_list, vmps.radial_substep_list
    ):
        new_vmc = copy.deepcopy(base_vmc)
        new_vmc.grid = pyv.Grid(
            radial_points=element[0],
            angular_points=element[1],
            radial_substeps=element[2],
        )
        vmc_set.append(new_vmc)

    return vmc_set


def generate_fits_file(
    output_fits_file: pathlib.Path,
    vmps: GridVsAccuracyParameterSet,
    delete_intermediates: bool = False,
    parallelism=1,
) -> None:
    """
    Compute all of the models at a given AU and generate a (probably large) fits file with all of the results.

    """

    # list of PurePath filenames of intermediate files
    out_file_list = []
    # list of intermediate QTable objects
    out_table_list = []

    vmc_list = generate_vmc_set(vmps=vmps)

    # # split our set of vmc into sublists, where every sublist has a matching base_q
    # # so we can loop through and compute all the models that have the same base_q and save the intermediate results
    grouped_by_radial_points = [
        list(x) for _, x in groupby(vmc_list, lambda x: x.grid.radial_points)
    ]

    # build one intermediate output table for each base_q and combine after all of the calculations are done
    for i, vmc_sublist in enumerate(grouped_by_radial_points):
        percent_complete = (i * 100) / len(grouped_by_radial_points)
        radial_points = vmc_sublist[0].grid.radial_points

        print(
            f"Currently running N_r = {radial_points} ({len(vmc_sublist):03d} models), {percent_complete:4.1f}% complete overall..."
        )

        out_filename = pathlib.Path(
            output_fits_file.stem + "_" + str(radial_points) + ".fits"
        )
        if out_filename.is_file():
            print(
                f"Found intermediate file {out_filename}, skipping generation and reading file instead..."
            )
            out_table = QTable.read(out_filename, format="fits")
        else:
            out_table = pyv.build_calculation_table(
                vmc_sublist, parallelism=parallelism
            )

        out_file_list.append(out_filename)

        log.info(
            "Table building for radial points %s complete, writing results to %s ...",
            radial_points,
            out_filename,
        )
        remove_file_silent_fail(out_filename)

        out_table.write(out_filename, format="fits")
        out_table_list.append(out_table)

    remove_file_silent_fail(output_fits_file)
    final_table = vstack(out_table_list)
    final_table.write(output_fits_file, format="fits")

    if delete_intermediates:
        log.info("Deleting intermediate files...")
        for file_to_del in out_file_list:
            log.info("%s\t😵", file_to_del)
            remove_file_silent_fail(file_to_del)

    del final_table


def make_dataset_table() -> QTable:
    """
    Returns an astropy QTable with columns:
        | filename | whether 'filename' already exists |
    """

    output_fits_filenames = [pathlib.Path("gridsize_vs_accuracy.fits")]
    fits_file_exists = [a.is_file() for a in output_fits_filenames]

    return QTable(
        [output_fits_filenames, fits_file_exists],
        names=("filename", "exists"),
    )


def get_dataset_selection() -> GridVsAccuracyParameterSet:
    """
    Allows the user to select a dataset from the global list vm_parameter_sets
    """

    dataset_names = [x.parameter_set_id for x in vm_parameter_sets]
    user_selection = None

    while user_selection is None:
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
    parallelism = max(1, multiprocessing.cpu_count() - 1)
    print(
        f"Max CPUs: {multiprocessing.cpu_count()}\tWill use: {parallelism} concurrent processes"
    )

    # we define a few datasets of interest to us - which one to compute?
    vmps = get_dataset_selection()
    dataset_table = make_dataset_table()

    print(f"Dataset {vmps.parameter_set_id} consists of:")
    print(dataset_table)

    # fill generate_now column with False and ask which data to generate, if any
    dataset_table.add_column(False, name="generate_now")

    # do all of them, or pick which ones?
    if input(f"Generate all missing data for {vmps.parameter_set_id}? [N/y] ") in [
        "y",
        "Y",
    ]:
        # mark all missing data to be generated this run
        for row in dataset_table:
            # if it exists, don't generate, and vice versa
            row["generate_now"] = not row["exists"]  # type: ignore
    else:
        # pick which missing data to run
        for row in dataset_table:
            if row["exists"]:  # type: ignore
                continue
            if input("Generate data?") in ["y", "Y"]:  # type: ignore
                row["generate_now"] = True  # type: ignore
                print("OK")

    if not any([row["generate_now"] for row in dataset_table]):  # type: ignore
        print("No data selected for generation!  Quitting.")
        return

    print("Current settings:")
    print(dataset_table)
    if input("Calculate? [N/y] ") not in ["y", "Y"]:
        print("Quitting.")
        return

    for row in dataset_table:
        if row["generate_now"]:  # type: ignore
            generate_fits_file(
                output_fits_file=row["filename"],  # type: ignore
                vmps=vmps,
                delete_intermediates=True,
                parallelism=parallelism,
            )


if __name__ == "__main__":
    sys.exit(main())
