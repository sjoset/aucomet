#!/usr/bin/env python3

import sys
# import copy

import numpy as np
# import astropy.units as u
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
# from matplotlib import cm, colors
# from matplotlib import colors

__author__ = 'Shawn Oset'
__version__ = '0.0'

# solarbluecol = np.array([38, 139, 220]) / 255.
# solarblue = (solarbluecol[0], solarbluecol[1], solarbluecol[2], 1)
# solargreencol = np.array([133, 153, 0]) / 255.
# solargreen = (solargreencol[0], solargreencol[1], solargreencol[2], 1)
# solarblackcol = np.array([0, 43, 54]) / 255.
# solarblack = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)
# solarwhitecol = np.array([238, 232, 213]) / 255.
# solarwhite = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)


def calculated_q_vs_q_plot(all_data, p_tau, f_tau, show_plots=False, out_file=None):

    # select only data rows with given taus
    pmask = (all_data[:, 1] == p_tau)
    fmask = (all_data[:, 2] == f_tau)
    mask = np.logical_and(pmask, fmask)

    # first and fourth columns are the data we want
    qs = np.log10(all_data[mask, 0])
    cqs = np.log10(all_data[mask, 3])

    plt.style.use('default')
    plt.style.use('Solarize_Light2')

    fig, ax = plt.subplots(figsize=(15, 15))

    # 2d scatter plot
    ax.scatter(qs, cqs)

    ax.set(xlabel="Production used in model, log10 Q(H2O)")
    ax.set(ylabel="Production calculated, log10 Q(H2O)")
    fig.suptitle(f"Calculated Q vs model input Q, via fixed count in aperture\nParent lifetime {p_tau:06.1f}, fragment lifetime {f_tau:06.1f}")

    if out_file:
        plt.savefig(out_file)
    if show_plots:
        plt.show()
    plt.close(fig)


def calculated_q_vs_q_plot_3d_fixed_ptau(all_data, p_tau, show_plots=False, out_file=None):

    # select only data rows with given ptaus
    mask = (all_data[:, 1] == p_tau)

    # first and fourth columns are the data we want
    qs = np.log10(all_data[mask, 0])
    f_taus = all_data[mask, 2]
    cqs = np.log10(all_data[mask, 3])

    plt.style.use('default')
    plt.style.use('Solarize_Light2')

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.set_zlim([27.2, 29.0])
    # 3d scatter plot
    ax.scatter(qs, f_taus, cqs)
    ax.plot_trisurf(qs, f_taus, cqs, color='white', edgecolors=(0, 0, 0, 0.05), alpha=0.2)

    ax.set(xlabel="Production used in model, log10 Q(H2O)")
    ax.set(ylabel="Fragment lifetime, s")
    ax.set(zlabel="Production calculated, log10 Q(H2O)")
    fig.suptitle(f"Calculated Q vs model input Q, via fixed count in aperture\nParent lifetime {p_tau:06.1f}")

    # Initialize 3d view at these angles for saving the plot
    ax.view_init(30, 40)

    if out_file:
        plt.savefig(out_file)
    if show_plots:
        plt.show()
    plt.close(fig)


def calculated_q_vs_q_plot_3d_fixed_ftau(all_data, f_tau, show_plots=False, out_file=None):

    # select only data rows with given ptaus
    mask = (all_data[:, 2] == f_tau)

    # first and fourth columns are the data we want
    qs = np.log10(all_data[mask, 0])
    p_taus = all_data[mask, 1]
    cqs = np.log10(all_data[mask, 3])

    plt.style.use('default')
    plt.style.use('Solarize_Light2')

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.set_zlim([27.2, 29.0])
    # 3d scatter plot
    ax.scatter(qs, p_taus, cqs)
    ax.plot_trisurf(qs, p_taus, cqs, color='white', edgecolors=(0, 0, 0, 0.05), alpha=0.2)

    ax.set(xlabel="Production used in model, log10 Q(H2O)")
    ax.set(ylabel="Parent lifetime, s")
    ax.set(zlabel="Production calculated, log10 Q(H2O)")
    fig.suptitle(f"Calculated Q vs model input Q, via fixed count in aperture\nFragment lifetime {f_tau:06.1f}")

    # Initialize 3d view at these angles for saving the plot
    ax.view_init(30, 40)

    if out_file:
        plt.savefig(out_file)
    if show_plots:
        plt.show()
    plt.close(fig)


def comp_time_vs_q_plot(all_data, p_tau, f_tau, show_plots=False, out_file=None):

    # select only data rows with given taus
    pmask = (all_data[:, 1] == p_tau)
    fmask = (all_data[:, 2] == f_tau)
    mask = np.logical_and(pmask, fmask)

    # first and fourth columns are the data we want
    qs = np.log10(all_data[mask, 0])
    cts = all_data[mask, 5]

    plt.style.use('default')
    plt.style.use('Solarize_Light2')

    fig, ax = plt.subplots(figsize=(15, 15))

    # 2d scatter plot
    ax.scatter(qs, cts)

    ax.set(xlabel="Production used in model, log10 Q(H2O)")
    ax.set(ylabel="Time for calculation to complete, s")
    fig.suptitle(f"Model computation time vs model input Q\nParent lifetime {p_tau:06.1f}, fragment lifetime {f_tau:06.1f}")

    if out_file:
        plt.savefig(out_file)
    if show_plots:
        plt.show()
    plt.close(fig)


def comp_time_vs_q_plot_3d_fixed_ptau(all_data, p_tau, show_plots=False, out_file=None):

    # select only data rows with given ptaus
    mask = (all_data[:, 1] == p_tau)

    # first and fourth columns are the data we want
    qs = np.log10(all_data[mask, 0])
    f_taus = all_data[mask, 2]
    cts = all_data[mask, 5]

    plt.style.use('default')
    plt.style.use('Solarize_Light2')

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # 3d scatter plot
    ax.scatter(qs, f_taus, cts)
    ax.plot_trisurf(qs, f_taus, cts, color='white', edgecolors=(0, 0, 0, 0.05), alpha=0.2)

    ax.set(xlabel="Production used in model, log10 Q(H2O)")
    ax.set(ylabel="Fragment lifetime, s")
    ax.set(zlabel="Calculation time, s")
    fig.suptitle(f"Calculation time vs model input Q\nParent lifetime {p_tau:06.1f}")

    # Initialize 3d view at these angles for saving the plot
    ax.view_init(30, 40)

    if out_file:
        plt.savefig(out_file)
    if show_plots:
        plt.show()
    plt.close(fig)


def comp_time_vs_q_plot_3d_fixed_ftau(all_data, f_tau, show_plots=False, out_file=None):

    # select only data rows with given ptaus
    mask = (all_data[:, 2] == f_tau)

    # first and fourth columns are the data we want
    qs = np.log10(all_data[mask, 0])
    p_taus = all_data[mask, 1]
    cts = all_data[mask, 5]

    plt.style.use('default')
    plt.style.use('Solarize_Light2')

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # 3d scatter plot
    ax.scatter(qs, p_taus, cts)
    ax.plot_trisurf(qs, p_taus, cts, color='white', edgecolors=(0, 0, 0, 0.05), alpha=0.2)

    ax.set(xlabel="Production used in model, log10 Q(H2O)")
    ax.set(ylabel="Parent lifetime, s")
    ax.set(zlabel="Calculation time, s")
    fig.suptitle(f"Calculation time vs model input Q\nParent lifetime {f_tau:06.1f}")

    # Initialize 3d view at these angles for saving the plot
    ax.view_init(30, 40)

    if out_file:
        plt.savefig(out_file)
    if show_plots:
        plt.show()
    plt.close(fig)


def acc_vs_q_plot(all_data, p_tau, f_tau, show_plots=False, out_file=None):

    # select only data rows with given taus
    pmask = (all_data[:, 1] == p_tau)
    fmask = (all_data[:, 2] == f_tau)
    mask = np.logical_and(pmask, fmask)

    # first and fourth columns are the data we want
    qs = np.log10(all_data[mask, 0])
    accs = all_data[mask, 4]

    plt.style.use('default')
    plt.style.use('Solarize_Light2')

    fig, ax = plt.subplots(figsize=(15, 15))

    # 2d scatter plot
    ax.scatter(qs, accs)

    ax.set(xlabel="Production used in model, log10 Q(H2O)")
    ax.set(ylabel="Aperture count reproduction accuracy, percent")
    fig.suptitle(f"Aperture count reproduction accuracy vs model input Q, via fixed count in aperture\nParent lifetime {p_tau:06.1f}, fragment lifetime {f_tau:06.1f}")

    if out_file:
        plt.savefig(out_file)
    if show_plots:
        plt.show()
    plt.close(fig)


def acc_vs_q_plot_3d_fixed_ptau(all_data, p_tau, show_plots=False, out_file=None):

    # select only data rows with given ptaus
    mask = (all_data[:, 1] == p_tau)

    # first and fourth columns are the data we want
    qs = np.log10(all_data[mask, 0])
    f_taus = all_data[mask, 2]
    accs = all_data[mask, 4]

    plt.style.use('default')
    plt.style.use('Solarize_Light2')

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # 3d scatter plot
    ax.scatter(qs, f_taus, accs)
    ax.plot_trisurf(qs, f_taus, accs, color='white', edgecolors=(0, 0, 0, 0.05), alpha=0.2)

    ax.set(xlabel="Production used in model, log10 Q(H2O)")
    ax.set(ylabel="Fragment lifetime, s")
    ax.set(zlabel="Aperture count reproduction accuracy, percent")
    fig.suptitle(f"Aperture count reproduction accuracy vs model input Q, via fixed count in aperture\nParent lifetime {p_tau:06.1f}")

    # Initialize 3d view at these angles for saving the plot
    ax.view_init(30, 40)

    if out_file:
        plt.savefig(out_file)
    if show_plots:
        plt.show()
    plt.close(fig)


def acc_vs_q_plot_3d_fixed_ftau(all_data, f_tau, show_plots=False, out_file=None):

    # select only data rows with given ptaus
    mask = (all_data[:, 2] == f_tau)

    # first and fourth columns are the data we want
    qs = np.log10(all_data[mask, 0])
    p_taus = all_data[mask, 1]
    accs = all_data[mask, 4]

    plt.style.use('default')
    plt.style.use('Solarize_Light2')

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # 3d scatter plot
    ax.scatter(qs, p_taus, accs)
    ax.plot_trisurf(qs, p_taus, accs, color='white', edgecolors=(0, 0, 0, 0.05), alpha=0.2)

    ax.set(xlabel="Production used in model, log10 Q(H2O)")
    ax.set(ylabel="Parent lifetime, s")
    ax.set(zlabel="Aperture count reproduction accuracy, percent")
    fig.suptitle(f"Aperture count reproduction accuracy vs model input Q, via fixed count in aperture\nFragment lifetime {f_tau:06.1f}")

    # Initialize 3d view at these angles for saving the plot
    ax.view_init(30, 40)

    if out_file:
        plt.savefig(out_file)
    if show_plots:
        plt.show()
    plt.close(fig)


def do_calc_q_vs_q_plots(all_data):

    p_taus = sorted(set(all_data[:, 1]))
    f_taus = sorted(set(all_data[:, 2]))

    for pt in p_taus:
        for ft in f_taus:
            out_file = f"Calc_Q_vs_Q_ptau_{pt:06.1f}_ftau_{ft:06.1f}.png"
            print(f"Generating {out_file} ...")
            calculated_q_vs_q_plot(all_data, pt, ft, out_file=out_file)


def do_calc_q_vs_q_fixed_ptau_plots(all_data):

    p_taus = sorted(set(all_data[:, 1]))

    for pt in p_taus:
        out_file = f"Calc_Q_vs_Q_ptau_{pt:06.1f}_summary.png"
        print(f"Generating {out_file} ...")
        calculated_q_vs_q_plot_3d_fixed_ptau(all_data, pt, out_file=out_file)


def do_calc_q_vs_q_fixed_ftau_plots(all_data):

    f_taus = sorted(set(all_data[:, 2]))

    for ft in f_taus:
        out_file = f"Calc_Q_vs_Q_ftau_{ft:06.1f}_summary.png"
        print(f"Generating {out_file} ...")
        calculated_q_vs_q_plot_3d_fixed_ftau(all_data, ft, out_file=out_file)


def do_comp_time_vs_q_plots(all_data):

    p_taus = sorted(set(all_data[:, 1]))
    f_taus = sorted(set(all_data[:, 2]))

    for pt in p_taus:
        for ft in f_taus:
            out_file = f"comp_time_vs_Q_ptau_{pt:06.1f}_ftau_{ft:06.1f}.png"
            print(f"Generating {out_file} ...")
            comp_time_vs_q_plot(all_data, pt, ft, out_file=out_file)


def do_comp_time_vs_q_fixed_ptau_plots(all_data):

    p_taus = sorted(set(all_data[:, 1]))

    for pt in p_taus:
        out_file = f"comp_time_vs_Q_ptau_{pt:06.1f}_summary.png"
        print(f"Generating {out_file} ...")
        comp_time_vs_q_plot_3d_fixed_ptau(all_data, pt, out_file=out_file)


def do_comp_time_vs_q_fixed_ftau_plots(all_data):

    f_taus = sorted(set(all_data[:, 2]))

    for ft in f_taus:
        out_file = f"comp_time_vs_Q_ftau_{ft:06.1f}_summary.png"
        print(f"Generating {out_file} ...")
        comp_time_vs_q_plot_3d_fixed_ftau(all_data, ft, out_file=out_file)


def do_acc_vs_q_plots(all_data):

    p_taus = sorted(set(all_data[:, 1]))
    f_taus = sorted(set(all_data[:, 2]))

    for pt in p_taus:
        for ft in f_taus:
            out_file = f"acc_vs_Q_ptau_{pt:06.1f}_ftau_{ft:06.1f}.png"
            print(f"Generating {out_file} ...")
            acc_vs_q_plot(all_data, pt, ft, out_file=out_file)


def do_acc_vs_q_fixed_ptau_plots(all_data):

    p_taus = sorted(set(all_data[:, 1]))

    for pt in p_taus:
        out_file = f"acc_vs_Q_ptau_{pt:06.1f}_summary.png"
        print(f"Generating {out_file} ...")
        acc_vs_q_plot_3d_fixed_ptau(all_data, pt, out_file=out_file)


def do_acc_vs_q_fixed_ftau_plots(all_data):

    f_taus = sorted(set(all_data[:, 2]))

    for ft in f_taus:
        out_file = f"acc_vs_Q_ftau_{ft:06.1f}_summary.png"
        print(f"Generating {out_file} ...")
        acc_vs_q_plot_3d_fixed_ftau(all_data, ft, out_file=out_file)


def main():

    quantity_support()

    with open('output.npdata', 'rb') as np_file:
        all_data = np.load(np_file)

    do_calc_q_vs_q_plots(all_data)
    do_calc_q_vs_q_fixed_ptau_plots(all_data)
    do_calc_q_vs_q_fixed_ftau_plots(all_data)

    do_comp_time_vs_q_plots(all_data)
    do_comp_time_vs_q_fixed_ptau_plots(all_data)
    do_comp_time_vs_q_fixed_ftau_plots(all_data)

    do_acc_vs_q_plots(all_data)
    do_acc_vs_q_fixed_ptau_plots(all_data)
    do_acc_vs_q_fixed_ftau_plots(all_data)

    # qs = set(all_data[:, 0])
    # p_taus = sorted(set(all_data[:, 1]))
    # f_taus = sorted(set(all_data[:, 2]))

    # acc_vs_q_plot(all_data, p_taus[0], f_taus[0], show_plots=True)
    # acc_vs_q_plot_3d_fixed_ptau(all_data, p_taus[0], show_plots=True)
    # acc_vs_q_plot_3d_fixed_ftau(all_data, f_taus[0], show_plots=True)

if __name__ == '__main__':
    sys.exit(main())
