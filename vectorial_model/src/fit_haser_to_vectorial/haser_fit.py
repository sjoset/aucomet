#!/usr/bin/env python3

import os
import sys

import logging as log
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

from astropy.visualization import quantity_support
from argparse import ArgumentParser

import sbpy.activity as sba
import pyvectorial as pyv

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


def file_string_id_from_parameters(vmc):
    base_q = vmc.production.base_q.value
    p_tau_d = vmc.parent.tau_d.value
    f_tau_T = vmc.fragment.tau_T.value

    return f"q_{base_q}_ptau_d_{p_tau_d:06.1f}_ftau_T_{f_tau_T:06.1f}"


def handle_output_options(vmc, vmr, coma):
    if vmc.etc["print_radial_density"]:
        pyv.print_volume_density(vmr)
    if vmc.etc["print_column_density"]:
        pyv.print_column_density(vmr)
    if vmc.etc["show_agreement_check"]:
        pyv.show_fragment_agreement(vmr)
    if vmc.etc["show_aperture_checks"]:
        pyv.show_aperture_checks(coma)

    if vmc.etc["show_radial_plots"]:
        pyv.radial_density_plots(vmc, vmr, r_units=u.km, voldens_units=1 / u.cm**3)  # type: ignore
    if vmc.etc["show_column_density_plots"]:
        pyv.column_density_plots(vmc, vmr, r_units=u.km, cd_units=1 / u.cm**2)  # type: ignore
    if vmc.etc["show_3d_column_density_centered"]:
        pyv.mpl_column_density_plot_3d(  # type: ignore
            vmc,
            vmr,
            x_min=-100000 * u.km,  # type: ignore
            x_max=100000 * u.km,  # type: ignore
            y_min=-100000 * u.km,  # type: ignore
            y_max=100000 * u.km,  # type: ignore
            grid_step_x=1000,
            grid_step_y=1000,
            r_units=u.km,
            cd_units=1 / u.cm**2,  # type: ignore
        )
    if vmc.etc["show_3d_column_density_off_center"]:
        pyv.mpl_column_density_plot_3d(
            vmc,
            vmr,
            x_min=-100000 * u.km,  # type: ignore
            x_max=10000 * u.km,  # type: ignore
            y_min=-100000 * u.km,  # type: ignore
            y_max=10000 * u.km,  # type: ignore
            grid_step_x=1000,
            grid_step_y=1000,
            r_units=u.km,
            cd_units=1 / u.cm**2,  # type: ignore
        )
    # if vmc.etc["show_fragment_sputter"]:
    #     pyv.mpl_plot_fragment_sputter(
    #         vmr.fragment_sputter,
    #         dist_units=u.km,
    #         sputter_units=1 / u.cm**3,  # type: ignore
    #         within_r=1000 * u.km,  # type: ignore
    #     )


def test_haser_q_fit_with_noise():
    # Use sbpy's Haser model with some noise added to test if haser fits recover good values

    # use water values and velocity scaling
    p = sba.photo_lengthscale("H2O")
    f = sba.photo_lengthscale("OH")
    v = 0.85 * u.km / u.s  # type: ignore

    # range of impact parameters for column densities
    rs = np.logspace(4, 7, num=10000)
    cd_sbpy = sba.Haser(Q=1.0e28 * (1 / u.s), v=v, parent=p, daughter=f)._column_density
    cds = cd_sbpy(rs)
    # add some noise to a standard Haser column density
    rng = np.random.default_rng()
    cd_noise = 1.0e19 * rng.normal(size=rs.size)

    hps = pyv.HaserParams(q=None, v_outflow=v, gamma_p=p, gamma_d=f)
    hfr = pyv.haser_q_fit_from_column_density(
        q_guess=1.0e26 * 1 / u.s, hps=hps, rs=rs, cds=(cds + cd_noise)
    )
    print(hfr)
    # print(pyv.haser_params_from_fit_result(hfr))

    plt.plot(rs, cds + cd_noise)
    plt.plot(rs, hfr.fitting_function(rs, *hfr.fitted_params))
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

    # for r in np.linspace(4, 7, num=10):
    #     print(hfr.fitting_function(r, *hfr.fitted_params)/cd_sbpy(r))


def test_haser_full_fit_with_noise():
    # Use sbpy's Haser model with some noise added to test if haser fits recover good values

    # use water values and velocity scaling
    p_model = sba.photo_lengthscale("H2O")
    f_model = sba.photo_lengthscale("OH")
    v_model = 0.85 * u.km / u.s  # type: ignore
    q_model = 1.0e28 / u.s

    # range of impact parameters for column densities
    rs = np.logspace(4, 7, num=10000)
    # get column densities at these rs
    cd_sbpy = sba.Haser(
        Q=q_model, v=v_model, parent=p_model, daughter=f_model
    )._column_density
    cds = cd_sbpy(rs)

    # add some noise to a standard Haser column density
    rng = np.random.default_rng(seed=41)
    cd_noise = 1.0e16 * rng.normal(size=rs.size)

    hfr = pyv.haser_full_fit_from_column_density(
        # q_guess=1.0e25 / u.s,
        q_guess=q_model / 2,
        # v_guess=0.1 * u.m / u.s,
        v_guess=v_model / 0.9,
        parent_guess=p_model / 0.9,
        fragment_guess=f_model / 2,
        rs=rs,
        cds=(cds + cd_noise),
    )

    haser_fit_params = pyv.haser_params_from_full_fit_result(hfr)
    print(f"{haser_fit_params.q} ===> {100 * haser_fit_params.q/q_model}%")
    print(
        f"{haser_fit_params.v_outflow.to(u.km/u.s)} ===> {(100 * haser_fit_params.v_outflow/v_model).decompose()}%"
    )
    print(
        f"{haser_fit_params.gamma_p.to(u.km)} ===> {(100 * haser_fit_params.gamma_p/p_model).decompose()}%"
    )
    print(
        f"{haser_fit_params.gamma_d.to(u.km)} ===> {(100 * haser_fit_params.gamma_d/f_model).decompose()}%"
    )

    # plt.plot(rs, cds)
    plt.plot(rs, cds + cd_noise)
    plt.plot(rs, hfr.fitting_function(rs, *hfr.fitted_params))
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

    # for r in np.linspace(4, 7, num=10):
    #     print(hfr.fitting_function(r, *hfr.fitted_params)/cd_sbpy(r))


def main():
    # astropy units/quantities support in plots
    quantity_support()

    # args = process_args()

    # log.debug("Loading input from %s ....", args.parameterfile[0])

    # test_haser_q_fit_with_noise()
    test_haser_full_fit_with_noise()

    # # Read in our stuff
    # vmc_set = pyv.vm_configs_from_yaml(args.parameterfile[0])
    #
    # parent_gammas = np.linspace(sba.photo_lengthscale('H2O')/2, sba.photo_lengthscale('H2O')*4, num=25, endpoint=True)
    # fragment_gammas = np.linspace(sba.photo_lengthscale('OH')/2, sba.photo_lengthscale('OH')*8, num=25, endpoint=True)
    #
    # for vmc in vmc_set:
    #
    #     coma = pyv.run_vmodel(vmc)
    #     vmr = pyv.get_result_from_coma(coma)
    #     # pyv.save_results(vmc, vmr, 'test')
    #     handle_output_options(vmc, vmr, coma)
    #
    #     search_results = pyv.find_best_haser_scale_lengths_q(vmc, vmr, parent_gammas, fragment_gammas)
    #     print(search_results.best_params)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     pyv.haser_search_result_plot(search_results, ax)
    #     plt.show()
    #
    #     hps = pyv.haser_from_vectorial_cd1980(vmc)
    #     print(f"Haser from vectorial cd1980 params: {hps}")
    #
    #     hfr = pyv.haser_q_fit(q_guess=1.e27 * 1/u.s, hps=hps, rs=vmr.column_density_grid, cds=vmr.column_density)
    #     print(hfr.fitted_params)
    #     plt.loglog(vmr.column_density_grid, vmr.column_density, label='vectorial')
    #     plt.plot(vmr.column_density_grid, hfr.fitting_function(vmr.column_density_grid.to_value('m'), *hfr.fitted_params), label='Monte-carlo corrected Haser')
    #     plt.legend()
    #     plt.show()
    #
    # test_hps = hps
    # test_hps.q = hfr.fitted_params[0] * 1/u.s
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # rs1 = np.logspace(0, 3) * u.km
    # rs2 = np.logspace(2, 6) * u.km
    # pyv.plot_haser_column_density(hps, ax1, rs1)
    # pyv.plot_haser_column_density(hps, ax2, rs2)
    # ax1.set_xscale('log')
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # plt.show()


if __name__ == "__main__":
    sys.exit(main())
