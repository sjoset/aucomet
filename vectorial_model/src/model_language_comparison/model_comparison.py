#!/usr/bin/env python3

# import copy
import os
import sys
import logging as log
import copy
import pathlib

import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from contextlib import redirect_stdout

import pyvectorial as pyv

# from collections import namedtuple
# from typing import List, NamedTuple

__author__ = "Shawn Oset"
__version__ = "0.1"


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
    parser.add_argument(
        "parameterfile", nargs=1, help="YAML file with production and molecule data"
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


def calculate_comparisons(
    pvmr: pyv.VectorialModelResult, fvmr: pyv.VectorialModelResult
):
    v_list = []
    c_list = []
    # Sometimes python grid is smaller than fortran version
    max_r = pvmr.max_grid_radius.to(u.m).value

    for i, r in enumerate(fvmr.volume_density_grid):
        r_m = r.to(u.m).value
        if r_m > max_r:
            continue
        ratio = (
            (
                (pvmr.volume_density_interpolation(r_m) * (1 / u.m**3))
                / fvmr.volume_density[i]
            )
            .decompose()
            .value
        )
        v_list.append([r_m, ratio])

    v_array = np.array(v_list)
    vrs = v_array[:, 0]
    v_ratios = v_array[:, 1]

    for i, r in enumerate(fvmr.column_density_grid):
        r_m = r.to(u.m).value
        if r_m > max_r:
            continue
        ratio = (
            (
                (pvmr.column_density_interpolation(r_m) * (1 / u.m**2))
                / fvmr.column_density[i]
            )
            .decompose()
            .value
        )
        c_list.append([r_m, ratio])

    c_array = np.array(c_list)
    crs = c_array[:, 0]
    c_ratios = c_array[:, 1]

    return vrs, v_ratios, crs, c_ratios


def do_cdens_comparison(
    pvmr: pyv.VectorialModelResult,
    fvmr: pyv.VectorialModelResult,
    vmc: pyv.VectorialModelConfig,
    show_plots=False,
    out_file=None,
):
    vrs, v_ratios, crs, c_ratios = calculate_comparisons(pvmr, fvmr)
    plot_cdens_comparison(crs, c_ratios, vmc, show_plots=show_plots, out_file=out_file)

    if out_file is not None:
        with open(out_file, "w") as f:
            with redirect_stdout(f):
                print("\nvolume density at fortran gridpoints, py/fort")
                print(v_ratios)
                print(
                    f"Average: {np.average(v_ratios)}\t\tMax:\
                        {np.amax(v_ratios)}\t\tMin: {np.amin(v_ratios)}"
                )
                for r, vr in zip(vrs, v_ratios):
                    print(f"r: {r/1000} km\tpy/fort: {vr}")
                print("\ncolumn density at fortran gridpoints, py/fort")
                print(c_ratios)
                print(
                    f"Average: {np.average(c_ratios)}\t\tMax:\
                    {np.amax(c_ratios)}\t\tMin: {np.amin(c_ratios)}"
                )
                for r, cr in zip(crs, c_ratios):
                    print(f"r: {r/1000} km\tpy/fort: {cr}")


def plot_cdens_comparison(
    rs, c_ratios, vmc: pyv.VectorialModelConfig, show_plots=True, out_file=None
):
    r_units = u.km
    plt.style.use("Solarize_Light2")

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    ax.set(xlabel=f"Distance from nucleus, {r_units.to_string()}")
    ax.set(ylabel="Column density ratio, py/fortran")
    fig.suptitle(
        f"Calculated column density ratios, python/fortran\nFragment lifetime {vmc.fragment.tau_T.to(u.s).value:6.0f}"
    )

    ax.set_xscale("log")
    ax.set_ylim([0.5, 1.5])
    ax.plot(rs, c_ratios, color="#688894", linewidth=2.0)

    if out_file:
        plt.savefig(out_file + ".png")
    if show_plots:
        plt.show()
    plt.close(fig)


def dump_python_fragment_sputter(vmr: pyv.VectorialModelResult, out_file):
    with open(out_file, "w") as f:
        with redirect_stdout(f):
            print(f"Coma size: {vmr.coma_radius.to(u.km):.6e}")
            print(f"Collision sphere size: {vmr.collision_sphere_radius.to(u.km):.6e}")
            print(f"Max grid radius: {vmr.max_grid_radius.to(u.km):.6e}")
            print("Python fragment sputter\n\n")
            print("radius\tangle\tfragment sputter density")
            lastr = 0 * u.m
            for i, frag_dens in enumerate(vmr.fragment_sputter.fragment_density):
                r = vmr.fragment_sputter.rs[i].to(u.km)
                theta = vmr.fragment_sputter.thetas[i]
                if lastr != r:
                    print("")
                print(f"{r:.5e} {theta:5.9f} {frag_dens.to(1/u.cm**3):9.9e}")
                lastr = r


def file_string_id_from_parameters(vmc_prime):
    vmc = copy.deepcopy(vmc_prime)
    pyv.unapply_input_transform(vmc)

    f_tau_T = vmc.fragment.tau_T.to(u.s).value

    return f"c_ftau_{f_tau_T:06.0f}s"


def plot_fragment_sputters(vmr, fvmr, out_file, **kwargs):
    dist_units = kwargs.get("dist_units", u.m)
    sputter_units = kwargs.get("sputter_units", 1 / u.m**3)

    fig = plt.figure(figsize=(30, 10))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    pyv.mpl_fragment_sputter_plot(vmr, ax1, **kwargs)
    ax1.set(xlabel=f"x, {dist_units.to_string()}")
    ax1.set(ylabel=f"y, {dist_units.to_string()}")
    ax1.set(zlabel=f"python fragment volume density, {sputter_units.unit.to_string()}")

    pyv.mpl_fragment_sputter_plot(fvmr, ax2, color="red", **kwargs)
    ax2.set(xlabel=f"x, {dist_units.to_string()}")
    ax2.set(ylabel=f"y, {dist_units.to_string()}")
    ax2.set(zlabel=f"fortran fragment volume density, {sputter_units.unit.to_string()}")

    pyv.mpl_fragment_sputter_plot(vmr, ax3, **kwargs)
    pyv.mpl_fragment_sputter_plot(fvmr, ax3, color="red", **kwargs)
    ax3.set(xlabel=f"x, {dist_units.to_string()}")
    ax3.set(ylabel=f"y, {dist_units.to_string()}")
    ax3.set(
        zlabel=f"combined fragment volume density, {sputter_units.unit.to_string()}"
    )

    fig.suptitle("Fragment sputter density")
    plt.style.use("Solarize_Light2")
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file)
    plt.close()


def main():
    # astropy units/quantities support in plots
    quantity_support()

    args = process_args()

    log.debug("Loading input from %s ....", args.parameterfile[0])
    # Read in our stuff
    vmc_set = pyv.vm_configs_from_yaml(args.parameterfile[0])

    for vmc in vmc_set:
        # template for output filenames
        run_name_template = file_string_id_from_parameters(vmc)

        # keep backup of config
        vmc_untransformed = copy.deepcopy(vmc)
        pyv.unapply_input_transform(vmc_untransformed)

        # run the python version and get the results
        coma = pyv.run_vmodel(vmc)
        vmr = pyv.get_result_from_coma(coma)

        # run fortran version and load the results from the output file it dumps to
        pyv.run_fortran_vmodel(
            vmc_untransformed, pathlib.Path(vmc.etc["fortran_vmodel_binary"])
        )
        fvmr = pyv.get_result_from_fortran(vmc.etc["out_file"])

        # move the fortran input/output files for inspection later
        os.rename(vmc.etc["in_file"], run_name_template + "_" + vmc.etc["in_file"])
        os.rename(vmc.etc["out_file"], run_name_template + "_" + vmc.etc["out_file"])

        # output the column density comparison
        do_cdens_comparison(
            vmr, fvmr, vmc, show_plots=False, out_file=run_name_template + "_ratio"
        )
        plot_fragment_sputters(
            vmr,
            fvmr,
            dist_units=u.km,
            sputter_units=1 / u.cm**3,
            within_r=1000 * u.km,
            out_file=run_name_template + "_sputter.png",
        )
        dump_python_fragment_sputter(vmr, run_name_template + "_pysputter")

        # pyv.save_results(vmc, vmr, run_name_template + "_python_save")
        # pyv.save_results(vmc, fvmr, run_name_template + "_fortran_save")


if __name__ == "__main__":
    sys.exit(main())
