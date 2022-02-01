#!/usr/bin/env python3

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt


solarbluecol = np.array([38, 139, 220]) / 255.
solarblue = (solarbluecol[0], solarbluecol[1], solarbluecol[2], 1)
solargreencol = np.array([133, 153, 0]) / 255.
solargreen = (solargreencol[0], solargreencol[1], solargreencol[2], 1)
solarblackcol = np.array([0, 43, 54]) / 255.
solarblack = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)
solarwhitecol = np.array([238, 232, 213]) / 255.
solarwhite = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)


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


def find_cdens_inflection_points(vmodel):
    """
        Look for changes in sign of second derivative of the column density,
        given a vectorial model coma and return a list of inflection points
    """

    xs = np.linspace(0, 5e8, num=100)
    concavity = vmodel['column_density_interpolation'].derivative(nu=2)
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
    csphere_radius = vmodel['collision_sphere_radius'].to_value(u.m)
    inflection_points = inflection_points[inflection_points > csphere_radius]

    inflection_points = inflection_points * u.m
    return inflection_points


def radial_density_plots(vmodel, r_units, voldens_units, frag_name, show_plots=True, out_file=None):

    interp_color = myblue
    model_color = myred
    linear_color = mygreen
    csphere_color = mybblue
    csphere_text_color = myblack
    inflection_color = mybblack

    x_min_logplot = 2
    x_max_logplot = 9

    x_min_linear = (0 * u.km).to(u.m)
    x_max_linear = (2000 * u.km).to(u.m)

    lin_interp_x = np.linspace(x_min_linear.value, x_max_linear.value, num=200)
    lin_interp_y = vmodel['r_dens_interpolation'](lin_interp_x)/(u.m**3)
    lin_interp_x *= u.m
    lin_interp_x.to(r_units)

    log_interp_x = np.logspace(x_min_logplot, x_max_logplot, num=200)
    log_interp_y = vmodel['r_dens_interpolation'](log_interp_x)/(u.m**3)
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
    ax1.plot(vmodel['radial_grid'], vmodel['radial_density'].to(voldens_units), 'o', color=model_color, label="model")
    ax1.plot(vmodel['radial_grid'], vmodel['radial_density'].to(voldens_units), '--', color=linear_color,
             linewidth=1.0, label="linear interpolation")

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.loglog(log_interp_x, log_interp_y, color=interp_color,  linewidth=2.0, linestyle="-", label="cubic spline")
    ax2.loglog(vmodel['fast_radial_grid'], vmodel['radial_density'].to(voldens_units), 'o',
               color=model_color, label="model")
    ax2.loglog(vmodel['fast_radial_grid'], vmodel['radial_density'].to(voldens_units), '--',
               color=linear_color, linewidth=1.0, label="linear interpolation")

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0.1)

    # Mark the beginning of the collision sphere
    ax1.axvline(x=vmodel['collision_sphere_radius'], color=csphere_color)
    ax2.axvline(x=vmodel['collision_sphere_radius'], color=csphere_color)

    # Text for the collision sphere
    plt.text(vmodel['collision_sphere_radius']*2, lin_interp_y[0]/10, 'Collision Sphere Edge',
             color=csphere_text_color)

    plt.legend(loc='upper right', frameon=False)

    # Find possible inflection points
    for ipoint in find_cdens_inflection_points(vmodel):
        ax1.axvline(x=ipoint, color=inflection_color)

    if show_plots:
        plt.show()
    if out_file is not None:
        plt.savefig(out_file)

    return plt, fig, ax1, ax2


def column_density_plots(vmodel, r_units, cd_units, frag_name, show_plots=True, out_file=None):

    interp_color = myblue
    model_color = myred
    linear_color = mygreen
    csphere_color = mybblue
    csphere_text_color = myblack
    inflection_color = mybblack

    x_min_logplot = 4
    x_max_logplot = 11

    x_min_linear = (0 * u.km).to(u.m)
    x_max_linear = (2000 * u.km).to(u.m)

    lin_interp_x = np.linspace(x_min_linear.value, x_max_linear.value, num=200)
    lin_interp_y = vmodel['column_density_interpolation'](lin_interp_x)/(u.m**2)
    lin_interp_x *= u.m
    lin_interp_x.to(r_units)

    log_interp_x = np.logspace(x_min_logplot, x_max_logplot, num=200)
    log_interp_y = vmodel['column_density_interpolation'](log_interp_x)/(u.m**2)
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
    ax1.plot(vmodel['column_density_grid'], vmodel['column_densities'], 'o', color=model_color, label="model")
    ax1.plot(vmodel['column_density_grid'], vmodel['column_densities'], '--', color=linear_color,
             label="linear interpolation", linewidth=1.0)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.loglog(log_interp_x, log_interp_y, color=interp_color,  linewidth=2.0, linestyle="-", label="cubic spline")
    ax2.loglog(vmodel['column_density_grid'], vmodel['column_densities'], 'o', color=model_color, label="model")
    ax2.loglog(vmodel['column_density_grid'], vmodel['column_densities'], '--', color=linear_color,
               label="linear interpolation", linewidth=1.0)

    # limits for plot 1
    ax1.set_ylim(bottom=0)

    # limits for plot 2
    # ax2.set_xlim(right=vmodel['max_grid_radius'])

    # Mark the beginning of the collision sphere
    ax1.axvline(x=vmodel['collision_sphere_radius'], color=csphere_color)
    ax2.axvline(x=vmodel['collision_sphere_radius'], color=csphere_color)

    # Only plot as far as the maximum radius of our grid on log-log plot
    ax2.axvline(x=vmodel['max_grid_radius'])

    # Mark the collision sphere
    plt.text(vmodel['collision_sphere_radius']*1.1, lin_interp_y[0]/10, 'Collision Sphere Edge',
             color=csphere_text_color)

    plt.legend(loc='upper right', frameon=False)

    # Find possible inflection points
    for ipoint in find_cdens_inflection_points(vmodel):
        ax1.axvline(x=ipoint, color=inflection_color)
        ax2.axvline(x=ipoint, color=inflection_color, linewidth=0.5)

    if show_plots:
        plt.show()
    if out_file is not None:
        plt.savefig(out_file)

    return plt, fig, ax1, ax2


def column_density_plot_3d(vmodel, x_min, x_max, y_min, y_max, grid_step_x, grid_step_y, r_units, cd_units,
                           frag_name, view_angles=(90, 90), show_plots=True, out_file=None):

    x = np.linspace(x_min.to(u.m).value, x_max.to(u.m).value, grid_step_x)
    y = np.linspace(y_min.to(u.m).value, y_max.to(u.m).value, grid_step_y)
    xv, yv = np.meshgrid(x, y)
    z = vmodel['column_density_interpolation'](np.sqrt(xv**2 + yv**2))
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
