#!/usr/bin/env python3

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
# from matplotlib import cm, colors
# from matplotlib import colors

__author__ = 'Shawn Oset'
__version__ = '0.1'

solarbluecol = np.array([38, 139, 220]) / 255.
solarblue = (solarbluecol[0], solarbluecol[1], solarbluecol[2], 1)
solargreencol = np.array([133, 153, 0]) / 255.
solargreen = (solargreencol[0], solargreencol[1], solargreencol[2], 1)
solarblackcol = np.array([0, 43, 54]) / 255.
solarblack = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)
solarwhitecol = np.array([238, 232, 213]) / 255.
solarwhite = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)


def find_cdens_inflection_points(coma):
    """
        Look for changes in sign of second derivative of the column density,
        given a vectorial model coma and return a list of inflection points
    """

    xs = np.linspace(0, 5e8, num=100)
    concavity = coma.vmodel['column_density_interpolation'].derivative(nu=2)
    ys = concavity(xs)

    # for pair in zip(xs, ys):
    #     print(f"R: {pair[0]:08.1e}\t\tConcavity: {pair[1]:8.8f}")

    # Array of 1s or 0s marking if the sign changed from one element to the next
    sign_changes = (np.diff(np.sign(ys)) != 0)*1

    # Manually remove the weirdness near the nucleus and rezize the array
    sign_changes[0] = 0
    sign_changes = np.resize(sign_changes, 100)

    inflection_points = xs*sign_changes
    # Only want non-zero elements
    inflection_points = inflection_points[inflection_points > 0]

    # Only want inflection points outside the collision sphere
    csphere_radius = coma.vmodel['collision_sphere_radius'].to_value(u.m)
    inflection_points = inflection_points[inflection_points > csphere_radius]

    inflection_points = inflection_points * u.m
    return inflection_points


def outburst_radial_density_plot(coma, r_units, voldens_units, frag_name, out_file, days_since_start):

    plt, fig, ax1, ax2 = radial_density_plots(coma, r_units, voldens_units, frag_name, show_plots=False)

    fig.suptitle(f"Calculated radial density of {frag_name}, t = {days_since_start:05.2f} days")

    plt.savefig(out_file)
    plt.close()


def outburst_column_density_plot(coma, r_units, cdens_units, frag_name, out_file, days_since_start):

    plt, fig, ax1, ax2 = column_density_plots(coma, r_units, cdens_units, frag_name, show_plots=False)
    fig.suptitle(f"Calculated column density of {frag_name}, t = {days_since_start:05.2f} days")

    plt.savefig(out_file)
    plt.close()


def outburst_column_density_plot_3d(coma, x_min, x_max, y_min, y_max, grid_step_x, grid_step_y,
                                    r_units, cdens_units, frag_name, filename, days_since_start):

    plt, fig, ax, surf = column_density_plot_3d(coma, x_min, x_max, y_min, y_max, grid_step_x, grid_step_y, r_units, cdens_units,
                                                frag_name, show_plots=False)
    plt.title(f"Calculated column density of {frag_name}, t = {days_since_start:05.2f} days")

    plt.savefig(filename)
    plt.close()


# def generateColumnDensity3D(coma, x_min, x_max, y_min, y_max, grid_step_x, grid_step_y, r_units, cdens_units, frag_name, filename,
#                             days_since_start):
#     """ 3D plot of column density """

#     # mesh grid for units native to the interpolation function
#     x = np.linspace(x_min.to(u.m).value, x_max.to(u.m).value, grid_step_x)
#     y = np.linspace(y_min.to(u.m).value, y_max.to(u.m).value, grid_step_y)
#     xv, yv = np.meshgrid(x, y)
#     z = coma.vmodel['ColumnDensity']['Interpolator'](np.sqrt(xv**2 + yv**2))
#     # Interpolator spits out m^-2
#     fz = (z/u.m**2).to(cdens_units)

#     # mesh grid in the units requested for plotting
#     xu = np.linspace(x_min.to(r_units), x_max.to(r_units), grid_step_x)
#     yu = np.linspace(y_min.to(r_units), y_max.to(r_units), grid_step_y)
#     xvu, yvu = np.meshgrid(xu, yu)

#     plt.style.use('Solarize_Light2')
#     plt.style.use('dark_background')
#     plt.rcParams['grid.color'] = "black"

#     expectedMaxColDens = 3e13
#     normColors = colors.Normalize(vmin=0, vmax=expectedMaxColDens/1.5, clip=False)

#     fig = plt.figure(figsize=(20, 20))
#     ax = plt.axes(projection='3d')
#     surf = ax.plot_surface(xvu, yvu, fz, cmap='inferno', norm=normColors, edgecolor='none')

#     frame1 = plt.gca()
#     frame1.axes.xaxis.set_ticklabels([])
#     frame1.axes.yaxis.set_ticklabels([])
#     frame1.axes.zaxis.set_ticklabels([])

#     ax.set_zlim(bottom=0, top=expectedMaxColDens)

#     ax.set_xlabel(f'Distance, ({r_units.to_string()})')
#     ax.set_ylabel(f'Distance, ({r_units.to_string()})')
#     ax.set_zlabel(f"Column density, {cdens_units.unit.to_string()}")
#     plt.title(f"Calculated column density of {frag_name}, t = {days_since_start:05.2f} days")

#     ax.w_xaxis.set_pane_color(solargreen)
#     ax.w_yaxis.set_pane_color(solarblue)
#     ax.w_zaxis.set_pane_color(solarblack)

#     fig.colorbar(surf, shrink=0.5, aspect=5)
#     # ax.view_init(25, 45)
#     ax.view_init(90, 90)
#     # plt.show()
#     plt.savefig(filename)
#     plt.close()


def radial_density_plots(coma, r_units, voldens_units, frag_name, show_plots=True, out_file=None):

    interp_color = "#688894"
    model_color = "#c74a77"
    linear_color = "#afac7c"
    csphere_color = "#a4b7be"
    csphere_text_color = "#301e2a"
    inflection_color = "#82787f"

    x_min_logplot = 2
    x_max_logplot = 9

    x_min_linear = (0 * u.km).to(u.m)
    x_max_linear = (2000 * u.km).to(u.m)

    lin_interp_x = np.linspace(x_min_linear.value, x_max_linear.value, num=200)
    lin_interp_y = coma.vmodel['r_dens_interpolation'](lin_interp_x)/(u.m**3)
    lin_interp_x *= u.m
    lin_interp_x.to(r_units)

    log_interp_x = np.logspace(x_min_logplot, x_max_logplot, num=200)
    log_interp_y = coma.vmodel['r_dens_interpolation'](log_interp_x)/(u.m**3)
    log_interp_x *= u.m
    log_interp_x.to(r_units)

    plt.style.use('Solarize_Light2')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
    ax1.set(ylabel=f"Fragment density, {voldens_units.unit.to_string()}")
    ax2.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
    ax2.set(ylabel=f"Fragment density, {voldens_units.unit.to_string()}")
    fig.suptitle(f"Calculated radial density of {frag_name}")

    ax1.set_xlim([x_min_linear, x_max_linear])
    ax1.plot(lin_interp_x, lin_interp_y, color=interp_color,  linewidth=2.0, linestyle="-", label="cubic spline")
    ax1.plot(coma.vmodel['radial_grid'], coma.vmodel['radial_density'].to(voldens_units), 'bo', color=model_color, label="model")
    ax1.plot(coma.vmodel['radial_grid'], coma.vmodel['radial_density'].to(voldens_units), 'g--', color=linear_color,
             linewidth=1.0, label="linear interpolation")

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.loglog(log_interp_x, log_interp_y, color=interp_color,  linewidth=2.0, linestyle="-", label="cubic spline")
    ax2.loglog(coma.vmodel['fast_radial_grid'], coma.vmodel['radial_density'].to(voldens_units), 'bo',
               color=model_color, label="model")
    ax2.loglog(coma.vmodel['fast_radial_grid'], coma.vmodel['radial_density'].to(voldens_units), 'g--',
               color=linear_color, linewidth=1.0, label="linear interpolation")

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0.1)

    # Mark the beginning of the collision sphere
    ax1.axvline(x=coma.vmodel['collision_sphere_radius'], color=csphere_color)
    ax2.axvline(x=coma.vmodel['collision_sphere_radius'], color=csphere_color)

    # Text for the collision sphere
    plt.text(coma.vmodel['collision_sphere_radius']*2, lin_interp_y[0]/10, 'Collision Sphere Edge',
             color=csphere_text_color)

    plt.legend(loc='upper right', frameon=False)

    # Find possible inflection points
    for ipoint in find_cdens_inflection_points(coma):
        ax1.axvline(x=ipoint, color=inflection_color)

    if show_plots:
        plt.show()
    if out_file is not None:
        plt.savefig(out_file)

    return plt, fig, ax1, ax2


def column_density_plots(coma, r_units, cd_units, frag_name, show_plots=True, out_file=None):

    interp_color = "#688894"
    model_color = "#c74a77"
    linear_color = "#afac7c"
    csphere_color = "#a4b7be"
    csphere_text_color = "#301e2a"
    inflection_color = "#82787f"

    x_min_logplot = 2
    x_max_logplot = 9

    x_min_linear = (0 * u.km).to(u.m)
    x_max_linear = (2000 * u.km).to(u.m)

    lin_interp_x = np.linspace(x_min_linear.value, x_max_linear.value, num=200)
    lin_interp_y = coma.vmodel['column_density_interpolation'](lin_interp_x)/(u.m**2)
    lin_interp_x *= u.m
    lin_interp_x.to(r_units)

    log_interp_x = np.logspace(x_min_logplot, x_max_logplot, num=200)
    log_interp_y = coma.vmodel['column_density_interpolation'](log_interp_x)/(u.m**2)
    log_interp_x *= u.m
    log_interp_x.to(r_units)

    plt.style.use('Solarize_Light2')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
    ax1.set(ylabel=f"Fragment column density, {cd_units.unit.to_string()}")
    ax2.set(xlabel=f'Distance from nucleus, {r_units.to_string()}')
    ax2.set(ylabel=f"Fragment column density, {cd_units.unit.to_string()}")
    fig.suptitle(f"Calculated column density of {frag_name}")

    ax1.set_xlim([x_min_linear, x_max_linear])
    ax1.plot(lin_interp_x, lin_interp_y, color=interp_color,  linewidth=2.0, linestyle="-", label="cubic spline")
    ax1.plot(coma.vmodel['column_density_grid'], coma.vmodel['column_densities'], 'bo', color=model_color, label="model")
    ax1.plot(coma.vmodel['column_density_grid'], coma.vmodel['column_densities'], 'g--', color=linear_color,
             label="linear interpolation", linewidth=1.0)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.loglog(log_interp_x, log_interp_y, color=interp_color,  linewidth=2.0, linestyle="-", label="cubic spline")
    ax2.loglog(coma.vmodel['column_density_grid'], coma.vmodel['column_densities'], 'bo', color=model_color, label="model")
    ax2.loglog(coma.vmodel['column_density_grid'], coma.vmodel['column_densities'], 'g--', color=linear_color,
               label="linear interpolation", linewidth=1.0)

    ax1.set_ylim(bottom=0)

    ax2.set_xlim(right=coma.vmodel['max_grid_radius'])

    # Mark the beginning of the collision sphere
    ax1.axvline(x=coma.vmodel['collision_sphere_radius'], color=csphere_color)
    ax2.axvline(x=coma.vmodel['collision_sphere_radius'], color=csphere_color)

    # Only plot as far as the maximum radius of our grid on log-log plot
    ax2.axvline(x=coma.vmodel['max_grid_radius'])

    # Mark the collision sphere
    plt.text(coma.vmodel['collision_sphere_radius']*1.1, lin_interp_y[0]/10, 'Collision Sphere Edge',
             color=csphere_text_color)

    plt.legend(loc='upper right', frameon=False)

    # Find possible inflection points
    for ipoint in find_cdens_inflection_points(coma):
        ax1.axvline(x=ipoint, color=inflection_color)
        ax2.axvline(x=ipoint, color=inflection_color, linewidth=0.5)

    if show_plots:
        plt.show()
    if out_file is not None:
        plt.savefig(out_file)

    return plt, fig, ax1, ax2


def column_density_plot_3d(coma, x_min, x_max, y_min, y_max, grid_step_x, grid_step_y, r_units, cd_units,
                           frag_name, view_angles=(90, 90), show_plots=True, out_file=None):

    x = np.linspace(x_min.to(u.m).value, x_max.to(u.m).value, grid_step_x)
    y = np.linspace(y_min.to(u.m).value, y_max.to(u.m).value, grid_step_y)
    xv, yv = np.meshgrid(x, y)
    z = coma.vmodel['column_density_interpolation'](np.sqrt(xv**2 + yv**2))
    # column_density_interpolation spits out m^-2
    fz = (z/u.m**2).to(cd_units)

    xu = np.linspace(x_min.to(r_units), x_max.to(r_units), grid_step_x)
    yu = np.linspace(y_min.to(r_units), y_max.to(r_units), grid_step_y)
    xvu, yvu = np.meshgrid(xu, yu)

    plt.style.use('Solarize_Light2')
    plt.style.use('dark_background')
    plt.rcParams['grid.color'] = "black"

    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes(projection='3d')
    # ax.grid(False)
    surf = ax.plot_surface(xvu, yvu, fz, cmap='inferno', vmin=0, edgecolor='none')

    plt.gca().set_zlim(bottom=0)

    ax.set_xlabel(f'Distance, ({r_units.to_string()})')
    ax.set_ylabel(f'Distance, ({r_units.to_string()})')
    ax.set_zlabel(f"Column density, {cd_units.unit.to_string()}")
    plt.title(f"Calculated column density of {frag_name}")

    ax.w_xaxis.set_pane_color(solargreen)
    ax.w_yaxis.set_pane_color(solarblue)
    ax.w_zaxis.set_pane_color(solarblack)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(view_angles[0], view_angles[1])

    if show_plots:
        plt.show()
    if out_file is not None:
        plt.savefig(out_file)

    return plt, fig, ax, surf
