#!/usr/bin/env python3

import os
import sys
import pathlib
import warnings

import logging as log
import numpy as np
from argparse import ArgumentParser
import astropy.units as u
from astropy.table import QTable
from dataclasses import asdict

# from astropy.visualization import quantity_support
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
            'fitsinput', nargs=1, help='fits file that contains calculation table'
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


def test_calculation_table(calculation_table: QTable) -> None:

    # go through each entry and decode into usable coma objects, then extract VectorialModelResult
    # and perform some aperture checks
    for row in calculation_table:
        vmc = pyv.unpickle_from_base64(row['b64_encoded_vmc'])
        coma = pyv.unpickle_from_base64(row['b64_encoded_coma'])
        vmr = pyv.get_result_from_coma(coma)
        pyv.show_aperture_checks(coma)
        print(vmc.production.base_q)
        print(vmr.collision_sphere_radius.to(u.km))


def add_aperture_calc_column(qt: QTable, ap: sba.Aperture, col_name: str) -> None:

    # add a column of fragment counts inside given aperture to a given QTable
    ap_calcs = []

    for row in qt:
        coma = pyv.unpickle_from_base64(row['b64_encoded_coma'])
        ap_calcs.append(coma.total_number(ap))

    qt.add_column(ap_calcs, name=col_name)


def add_vmc_columns(qt: QTable) -> None:

    # extract VectorialModelConfig and add columns based on some parameters
    base_q_list = []
    ptd_list = []
    ptT_list = []
    voutflow_list = []
    ftT_list = []
    vphoto_list = []

    for row in qt:
        vmc = pyv.unpickle_from_base64(row['b64_encoded_vmc'])
        base_q_list.append(vmc.production.base_q)
        voutflow_list.append(vmc.parent.v_outflow)
        ptd_list.append(vmc.parent.tau_d)
        ptT_list.append(vmc.parent.tau_T)
        ftT_list.append(vmc.fragment.tau_T)
        vphoto_list.append(vmc.fragment.v_photo)

    qt.add_column(base_q_list, name='base_q')
    qt.add_column(ptd_list, name='parent_tau_d')
    qt.add_column(ptT_list, name='parent_tau_T')
    qt.add_column(voutflow_list, name='v_outflow')
    qt.add_column(ftT_list, name='fragment_tau_T')
    qt.add_column(vphoto_list, name='v_photo')


def add_haser_equivalents(qt: QTable) -> None:

    # calculate equivalent haser parameters and add as a column to table
    hp_list = []
    for row in qt:
        vmc = pyv.unpickle_from_base64(row['b64_encoded_vmc'])
        hp_list.append(pyv.haser_from_vectorial_cd1980(vmc))

    # turn dataclass into a dict and add each dataclass property as a column
    for k in asdict(hp_list[0]).keys():
        tmplist = []
        for hp in hp_list:
            tmplist.append(asdict(hp).get(k))
        qt.add_column(tmplist, name='haser_equiv_' + k)


def accuracy_plots(xs, ys, ys_2, **kwargs):

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=xs, y=ys, name="accuracy per time", **kwargs))
    fig.add_trace(go.Scatter(x=xs, y=ys_2, name="ratio of accuracy compared to best", **kwargs))
    fig.update_xaxes(type="log")
    fig.update_layout(
            title="Model accuracy per time",
            xaxis_title="Base production"
            # legend_title=""
            )
    fig.show()


def add_accuracy_per_time_by_1e6(qt: QTable) -> None:

    # take low production models as being most reliable, and calculate how well proportionally
    # the aperture counts are replicated with higher productions
    ratios = qt['1e6_km_aperture_count']/np.min(qt['1e6_km_aperture_count']) * (np.min(qt['base_q'])/qt['base_q'])
    qt.add_column(ratios, name='ratio_test_1e6')
    qt.add_column(qt['ratio_test_1e6']/qt['model_run_time'], name='accuracy_per_time_1e6')
    qt.add_column(qt['accuracy_per_time_1e6']/np.max(qt['accuracy_per_time_1e6']), name='accuracy_per_time_rating_1e6')


def add_accuracy_per_time_by_1e7(qt: QTable) -> None:

    # take low production models as being most reliable, and calculate how well proportionally
    # the aperture counts are replicated with higher productions
    ratios = qt['1e7_km_aperture_count']/np.min(qt['1e7_km_aperture_count']) * (np.min(qt['base_q'])/qt['base_q'])
    qt.add_column(ratios, name='ratio_test_1e7')
    qt.add_column(qt['ratio_test_1e7']/qt['model_run_time'], name='accuracy_per_time_1e7')
    qt.add_column(qt['accuracy_per_time_1e7']/np.max(qt['accuracy_per_time_1e7']), name='accuracy_per_time_rating_1e7')


def add_qOH_column(qt: QTable) -> None:

    num_rows = len(qt)
    # add a column of fragment counts inside large aperture divided by time to permanent flow regime for q(OH) per second?
    qOHs = []
    # column for empirical relation in cochran & schleicher 98 between Q(H2O) and Q(OH)
    emp_qH2Os = []
    model_to_emp_ratios = []

    for i, row in enumerate(qt):

        vmc = pyv.unpickle_from_base64(row['b64_encoded_vmc'])
        coma = pyv.unpickle_from_base64(row['b64_encoded_coma'])

        count_in_largest_ap = coma.total_number(sba.CircularAperture(coma.vmr.max_grid_radius))
        qOH = count_in_largest_ap/coma.vmr.t_perm_flow.to_value(u.s)
        emp_qH2O = 1.361 * qOH / u.s
        model_to_emp_ratio = vmc.production.base_q/emp_qH2O

        qOHs.append(qOH)
        emp_qH2Os.append(emp_qH2O)
        model_to_emp_ratios.append(model_to_emp_ratio)
        print(f"{i*100/num_rows:4.1f} % complete\r", end='')

    print("")
    qt.add_column(qOHs, name='qOH_max_aperture')
    qt.add_column(emp_qH2Os, name='emp_qH2Os')
    qt.add_column(model_to_emp_ratios, name='model_to_emp_ratios')


def main():
    
    # sometimes the aperture counts get a little complainy
    warnings.filterwarnings("ignore")
    args = process_args()
    table_file = pathlib.PurePath(args.fitsinput[0])

    # read in table from FITS
    test_table = QTable.read(table_file, format='fits')

    # add some analysis data to table and sort by base_q
    print("Adding columns for model input parameters ...")
    add_vmc_columns(test_table)

    print("Running aperture tests ...")
    # test_calculation_table(test_table)

    print("Adding haser equivalent scale lengths ...")
    add_haser_equivalents(test_table)

    print("Adding fragment counts inside apertures ...")
    # add_aperture_calc_column(test_table, sba.CircularAperture(1.e9*u.km), '1e9_km_aperture_count')
    # add_aperture_calc_column(test_table, sba.CircularAperture(1.e8*u.km), '1e8_km_aperture_count')
    # add_aperture_calc_column(test_table, sba.CircularAperture(1.e7*u.km), '1e7_km_aperture_count')
    # add_aperture_calc_column(test_table, sba.CircularAperture(1.e6*u.km), '1e6_km_aperture_count')
    # add_aperture_calc_column(test_table, sba.CircularAperture(1.e5*u.km), '1e5_km_aperture_count')

    print("Computing q(OH) ...")
    add_qOH_column(test_table)

    test_table.sort(keys='base_q')

    # print("Adding accuracy per time ...")
    # add_accuracy_per_time_by_1e6(test_table)
    # add_accuracy_per_time_by_1e7(test_table)

    print("Constructing results table ...")
    output_table = test_table

    # dump haser_equiv_q because it isn't calculated when we do haser equiv scale lengths - TODO?
    output_table.remove_columns(names=['b64_encoded_vmc', 'b64_encoded_coma', 'vmc_hash', 'haser_equiv_q'])
    # output_table.remove_columns(names=['b64_encoded_vmc', 'b64_encoded_coma', 'vmc_hash'])

    # mask = np.logical_and(output_table['base_q'] < 5.e+29/u.s, output_table['base_q'] > 1.e+29/u.s)
    # new_table = output_table[mask]
    # new_table.pprint_all()

    # same_params = output_table.group_by(['parent_tau_d', 'parent_tau_T', 'fragment_tau_T', 'v_photo', 'v_outflow'])
    # for gtab in same_params.groups:
    #     accuracy_plots(xs=gtab['base_q'].to_value(1/u.s), ys=gtab['accuracy_per_time_rating_1e6'], ys_2=gtab['ratio_test_1e6'])
    # for gtab in same_params.groups:
    #     accuracy_plots(xs=gtab['base_q'].to_value(1/u.s), ys=gtab['accuracy_per_time_rating_1e7'], ys_2=gtab['ratio_test_1e7'])

    # accuracy_plots(xs=output_table['base_q'].to_value(1/u.s), ys=output_table['accuracy_per_time_rating'], ys_2=output_table['ratio_test'])

    print("Writing results ...")
    output_table_file = pathlib.PurePath('analysis.ecsv')
    output_table.write(output_table_file, format='ascii.ecsv', overwrite=True)

    print("Complete!")


if __name__ == '__main__':
    sys.exit(main())
