#!/usr/bin/env python3
import os
import sys
import pathlib
import dill as pickle
import hashlib
import pprint
import codecs
import time
from typing import List

import logging as log
import numpy as np
from argparse import ArgumentParser
import astropy.units as u
from astropy.table import QTable, vstack

# from astropy.visualization import quantity_support
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

import sbpy.activity as sba
import pyvectorial as pyv

__author__ = 'Shawn Oset'
__version__ = '0.1'


myred = "#c74a77"
mybred = "#dbafad"
mygreen = "#afac7c"
mybgreen = "#dbd89c"
mypeach = "#dbb89c"
mybpeach = "#e9d4c3"
myblue = "#688894"
mybblue = "#a4b7be"
myblack = "#301e2a"
mybblack = "#82787f"
mywhite = "#d8d7dc"
mybwhite = "#e7e7ea"


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


def dill_from_coma(coma, coma_file) -> None:
    with open(coma_file, 'wb') as comapicklefile:
        pickle.dump(coma, comapicklefile)


def coma_from_dill(coma_file: str):
    with open(coma_file, 'rb') as coma_dill:
        return pickle.load(coma_dill)


def pickle_to_base64(obj) -> str:

    return codecs.encode(pickle.dumps(obj), 'base64').decode()


def unpickle_from_base64(p_str: str):

    return pickle.loads(codecs.decode(p_str.encode(), 'base64'))


def hash_vmc(vmc: pyv.VectorialModelConfig):

    # sbpy_ver = impm.version("sbpy")
    return hashlib.sha256(pprint.pformat(vmc).encode('utf-8')).hexdigest()


def build_basic_calculation_table(vmc_set: List[pyv.VectorialModelConfig]) -> QTable:
    """ Take a set of model configs, run them, and return QTable with results """

    # calculation_table = QTable(names=('b64_encoded_vmc', 'vmc_hash', 'b64_encoded_coma', 'model_run_time'), dtype=('U', 'U', 'U', 'f8'))
    calculation_table = QTable(names=('b64_encoded_vmc', 'vmc_hash', 'b64_encoded_coma'), dtype=('U', 'U', 'U'))

    times_list = []
    for vmc in vmc_set:
        t_i = time.time()
        coma = pyv.run_vmodel(vmc)
        t_f = time.time()
        times_list.append((t_f - t_i) * u.s)

        print(hash_vmc(vmc))
        pickled_vmc = pickle_to_base64(vmc)
        pickled_coma = pickle_to_base64(coma)
        calculation_table.add_row((pickled_vmc, hash_vmc(vmc), pickled_coma))

    calculation_table.add_column(times_list, name='model_run_time')

    return calculation_table


def test_basic_calculation_table(calculation_table: QTable) -> None:

    for row in calculation_table:
        vmc = unpickle_from_base64(row['b64_encoded_vmc'])
        coma = unpickle_from_base64(row['b64_encoded_coma'])
        vmr = pyv.get_result_from_coma(coma)
        pyv.show_aperture_checks(coma)
        print(vmc.production.base_q)
        print(vmr.collision_sphere_radius)
        print(row['vmc_hash'])


def test_table_building_writing(yaml_config_file: pathlib.Path, out_table_file: pathlib.Path) -> None:

    vmc_set = pyv.vm_configs_from_yaml(yaml_config_file)
    out_table = build_basic_calculation_table(vmc_set)

    out_table.write(out_table_file, format='fits')


def load_calculation_table(in_table_file: pathlib.Path) -> QTable:

    a = QTable.read(in_table_file, format='fits')
    return a


def add_aperture_calc_column(qt: QTable, ap: sba.Aperture, col_name: str) -> None:

    ap_calcs = []

    for row in qt:
        coma = unpickle_from_base64(row['b64_encoded_coma'])
        ap_calcs.append(coma.total_number(ap))

    qt.add_column(ap_calcs, name=col_name)


def add_vmc_columns(qt: QTable) -> None:

    base_q_list = []
    for row in qt:
        vmc = unpickle_from_base64(row['b64_encoded_vmc'])
        base_q_list.append(vmc.production.base_q)

    qt.add_column(base_q_list, name='base_q')


# TODO: split this into 2 scripts: generating the data to a fits file, and analyzing the data from a fits file
def main():

    table_file = pathlib.PurePath('raw_calc_table.fits')

    # args = process_args()
    # test_table_building_writing(pathlib.PurePath(args.parameterfile[0]), table_file)

    test_table = load_calculation_table(table_file)
    # test_basic_calculation_table(test_table)
    add_aperture_calc_column(test_table, sba.CircularAperture(1.e6*u.km), '1e6_km_aperture_count')
    add_vmc_columns(test_table)
    test_table.sort(keys='base_q')

    ratios = test_table['1e6_km_aperture_count']/np.min(test_table['1e6_km_aperture_count']) * (np.min(test_table['base_q'])/test_table['base_q'])
    test_table.add_column(ratios, name='ratio_test')
    test_table.add_column(test_table['ratio_test']/test_table['model_run_time'], name='accuracy_per_time')

    output_table = test_table['base_q', '1e6_km_aperture_count', 'model_run_time', 'ratio_test', 'accuracy_per_time']

    output_table_file = pathlib.PurePath('analysis.csv')
    output_table.write(output_table_file, format='csv', overwrite=True)

if __name__ == '__main__':
    sys.exit(main())
