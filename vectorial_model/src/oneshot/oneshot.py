#!/usr/bin/env python3

import os
import sys
import pathlib
import contextlib

import logging as log
import astropy.units as u
import pyvectorial as pyv
import pandas as pd

from argparse import ArgumentParser
from typing import Union

__author__ = "Shawn Oset"
__version__ = "0.1"

rustbin = pathlib.Path(
    pathlib.Path.home(),
    pathlib.Path(
        "repos/aucomet/vectorial_model/src/model_language_comparison/bin/rust_vect"
    ),
)
fortranbin = pathlib.Path(
    pathlib.Path.home(),
    pathlib.Path("repos/aucomet/vectorial_model/src/model_language_comparison/bin/fvm"),
)

model_backend_configs = {
    "sbpy (python)": pyv.PythonModelExtraConfig(print_progress=True),
    "rustvec (rust)": pyv.RustModelExtraConfig(
        bin_path=rustbin,
        rust_input_filename=pathlib.Path("rust_in.yaml"),
        rust_output_filename=pathlib.Path("rust_out.txt"),
    ),
    "vm (fortran)": pyv.FortranModelExtraConfig(
        bin_path=fortranbin,
        fortran_input_filename=pathlib.Path("fparam.dat"),
        fortran_output_filename=pathlib.Path("fort.16"),
        r_h=1.0 * u.AU,  # type: ignore
    ),
}


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
    parser.add_argument(
        "parameterfile", nargs=1, help="YAML file with production and molecule data"
    )  # the nargs=? specifies 0 or 1 arguments: it is optional
    # parser.add_argument("output_fits", nargs=1, help="Filename of FITS output")

    args = parser.parse_args()

    # handle verbosity
    if args.verbose >= 2:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    elif args.verbose == 1:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    return args


def remove_file_silent_fail(f: pathlib.Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(f)


def get_backend_model_selection() -> (
    Union[
        pyv.PythonModelExtraConfig,
        pyv.FortranModelExtraConfig,
        pyv.RustModelExtraConfig,
    ]
):
    backend_selection = None
    model_backend_names = list(model_backend_configs.keys())

    while backend_selection is None:
        print("Select a model backend:")
        for i, dataset_name in enumerate(model_backend_names):
            print(f"{i}:\t{dataset_name}")

        raw_selection = input()
        try:
            selection = int(raw_selection)
        except ValueError:
            print("Numbers only, please")
            selection = -1

        if selection in range(len(model_backend_names)):
            backend_selection = model_backend_names[selection]

    return model_backend_configs[backend_selection]


def main():
    args = process_args()
    log.debug("Loading input from %s ....", args.parameterfile[0])

    r_h = 5.92

    vmc = pyv.vectorial_model_config_from_yaml(pathlib.Path(args.parameterfile[0]))
    if vmc is None:
        return

    vmc.parent.tau_d = r_h**2 * vmc.parent.tau_d
    vmc.parent.tau_T = r_h**2 * vmc.parent.tau_T
    vmc.fragment.tau_T = r_h**2 * vmc.fragment.tau_T

    if vmc is not None:
        vmc_set = [vmc]
    else:
        print(f"Failed to read {args.parameterfile}!")
        return 1

    ec = get_backend_model_selection()

    out_table = pyv.build_calculation_table(vmc_set, extra_config=ec)  # type: ignore

    vmr = pyv.unpickle_from_base64(out_table["b64_encoded_vmr"][0])  # type: ignore

    r_kms = vmr.column_density_grid.to_value(u.km)
    cds = vmr.column_density.to_value(1 / u.cm**2)
    df = pd.DataFrame({"r_kms": r_kms, "column_density_per_cm2": cds})

    output_dir = pathlib.Path("output")
    output_file_stem = pathlib.Path(args.parameterfile[0]).stem
    output_file_name = pathlib.Path(output_file_stem + ".fits")
    output_file_path = output_dir / output_file_name
    # output_fits_file = pathlib.Path(args.output_fits[0])
    remove_file_silent_fail(output_file_path)
    log.info("Table building complete, writing results to %s ...", output_file_path)
    out_table.write(output_file_path, format="fits")

    df.to_csv(output_file_path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    sys.exit(main())
