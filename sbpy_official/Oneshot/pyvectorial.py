#!/usr/bin/env python3

# import copy
import numpy
import os
import sys
import yaml

import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
import sbpy.activity as sba
from sbpy.data import Phys

from argparse import ArgumentParser

# from vmodel import VectorialModel

__author__ = 'Shawn Oset, Lauren Lyons'
__version__ = '0.0'

solarbluecol = np.array([38, 139, 220]) / 255.
solarblue = (solarbluecol[0], solarbluecol[1], solarbluecol[2], 1)
solargreencol = np.array([133, 153, 0]) / 255.
solargreen = (solargreencol[0], solargreencol[1], solargreencol[2], 1)
solarblackcol = np.array([0, 43, 54]) / 255.
solarblack = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)
solarwhitecol = np.array([238, 232, 213]) / 255.
solarwhite = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)


def read_parameters_from_file(filepath):
    """Read the YAML file with all of the input parameters in it"""
    with open(filepath, 'r') as stream:
        try:
            paramYAMLData = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return paramYAMLData


def show_radial_plots(coma, rUnits, volUnits, fragName):
    """ Show the radial density of the fragment species """

    xMin_logplot = 2
    xMax_logplot = 8

    xMin_linear = (0 * u.km).to(u.m)
    xMax_linear = (2000 * u.km).to(u.m)

    linInterpX = np.linspace(xMin_linear.value, xMax_linear.value, num=200)
    linInterpY = coma.vModel['rDensInterpolator'](linInterpX)/(u.m**3)
    linInterpX *= u.m
    linInterpX.to(rUnits)

    logInterpX = np.logspace(xMin_logplot, xMax_logplot, num=200)
    logInterpY = coma.vModel['rDensInterpolator'](logInterpX)/(u.m**3)
    logInterpX *= u.m
    logInterpX.to(rUnits)

    plt.style.use('Solarize_Light2')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.set(xlabel=f'Distance from nucleus, {rUnits.to_string()}')
    ax1.set(ylabel=f"Fragment density, {volUnits.unit.to_string()}")
    ax2.set(xlabel=f'Distance from nucleus, {rUnits.to_string()}')
    ax2.set(ylabel=f"Fragment density, {volUnits.unit.to_string()}")
    fig.suptitle(f"Calculated radial density of {fragName}")

    ax1.set_xlim([xMin_linear, xMax_linear])
    ax1.plot(linInterpX, linInterpY, color="red",  linewidth=1.5, linestyle="-", label="cubic spline")
    ax1.plot(coma.vModel['RadialGrid'], coma.vModel['RadialDensity'].to(volUnits), 'bo', label="model")
    ax1.plot(coma.vModel['RadialGrid'], coma.vModel['RadialDensity'].to(volUnits), 'g--', label="linear interpolation")

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.loglog(coma.vModel['FastRadialGrid'], coma.vModel['RadialDensity'].to(volUnits), 'bo', label="model")
    ax2.loglog(coma.vModel['FastRadialGrid'], coma.vModel['RadialDensity'].to(volUnits), 'g--', label="linear interpolation")
    ax2.loglog(logInterpX, logInterpY, color="red",  linewidth=1.5, linestyle="-", label="cubic spline")

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0.1)

    # Mark the beginning of the collision sphere
    ax1.axvline(x=coma.vModel['CollisionSphereRadius'], color=solarblue)
    ax2.axvline(x=coma.vModel['CollisionSphereRadius'], color=solarblue)

    # Mark the collision sphere
    plt.text(coma.vModel['CollisionSphereRadius']*2, linInterpY[0]*2, 'Collision Sphere Edge', color=solarblue)

    plt.legend(loc='upper right', frameon=False)
    plt.show()


def show_column_density_plots(coma, rUnits, cdUnits, fragName):
    """ Show the radial density of the fragment species """

    xMin_logplot = 2
    xMax_logplot = 8

    xMin_linear = (0 * u.km).to(u.m)
    xMax_linear = (2000 * u.km).to(u.m)

    linInterpX = np.linspace(xMin_linear.value, xMax_linear.value, num=200)
    linInterpY = coma.vModel['ColumnDensity']['Interpolator'](linInterpX)/(u.m**2)
    linInterpX *= u.m
    linInterpX.to(rUnits)

    logInterpX = np.logspace(xMin_logplot, xMax_logplot, num=200)
    logInterpY = coma.vModel['ColumnDensity']['Interpolator'](logInterpX)/(u.m**2)
    logInterpX *= u.m
    logInterpX.to(rUnits)

    plt.style.use('Solarize_Light2')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.set(xlabel=f'Distance from nucleus, {rUnits.to_string()}')
    ax1.set(ylabel=f"Fragment column density, {cdUnits.unit.to_string()}")
    ax2.set(xlabel=f'Distance from nucleus, {rUnits.to_string()}')
    ax2.set(ylabel=f"Fragment column density, {cdUnits.unit.to_string()}")
    fig.suptitle(f"Calculated column density of {fragName}")

    ax1.set_xlim([xMin_linear, xMax_linear])
    ax1.plot(linInterpX, linInterpY, color="red",  linewidth=2.5, linestyle="-", label="cubic spline")
    ax1.plot(coma.vModel['ColumnDensity']['CDGrid'], coma.vModel['ColumnDensity']['Values'], 'bo', label="model")
    ax1.plot(coma.vModel['ColumnDensity']['CDGrid'], coma.vModel['ColumnDensity']['Values'], 'g--', label="linear interpolation")

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.loglog(coma.vModel['ColumnDensity']['CDGrid'], coma.vModel['ColumnDensity']['Values'], 'bo', label="model")
    ax2.loglog(coma.vModel['ColumnDensity']['CDGrid'], coma.vModel['ColumnDensity']['Values'], 'g--',
               label="linear interpolation", linewidth=0.5)
    ax2.loglog(logInterpX, logInterpY, color="red",  linewidth=0.5, linestyle="-", label="cubic spline")

    ax1.set_ylim(bottom=0)

    ax2.set_xlim(right=coma.vModel['MaxRadiusOfGrid'])

    # Mark the beginning of the collision sphere
    ax1.axvline(x=coma.vModel['CollisionSphereRadius'], color=solarblue)
    ax2.axvline(x=coma.vModel['CollisionSphereRadius'], color=solarblue)

    # Only plot as far as the maximum radius of our grid on log-log plot
    ax2.axvline(x=coma.vModel['MaxRadiusOfGrid'])

    # Mark the collision sphere
    plt.text(coma.vModel['CollisionSphereRadius']*1.1, linInterpY[0]/10, 'Collision Sphere Edge', color=solarblue)

    plt.legend(loc='upper right', frameon=False)
    plt.show()


def show_3d_column_density_plots(coma, xMin, xMax, yMin, yMax, gridStepX, gridStepY, rUnits, cdUnits, fragName):
    """ 3D plot of column density """

    x = np.linspace(xMin.to(u.m).value, xMax.to(u.m).value, gridStepX)
    y = np.linspace(yMin.to(u.m).value, yMax.to(u.m).value, gridStepY)
    xv, yv = np.meshgrid(x, y)
    z = coma.vModel['ColumnDensity']['Interpolator'](np.sqrt(xv**2 + yv**2))
    # Interpolator spits out m^-2
    fz = (z/u.m**2).to(cdUnits)

    xu = np.linspace(xMin.to(rUnits), xMax.to(rUnits), gridStepX)
    yu = np.linspace(yMin.to(rUnits), yMax.to(rUnits), gridStepY)
    xvu, yvu = np.meshgrid(xu, yu)

    plt.style.use('Solarize_Light2')
    plt.style.use('dark_background')
    plt.rcParams['grid.color'] = "black"

    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes(projection='3d')
    # ax.grid(False)
    surf = ax.plot_surface(xvu, yvu, fz, cmap='inferno', vmin=0, edgecolor='none')

    plt.gca().set_zlim(bottom=0)

    ax.set_xlabel(f'Distance, ({rUnits.to_string()})')
    ax.set_ylabel(f'Distance, ({rUnits.to_string()})')
    ax.set_zlabel(f"Column density, {cdUnits.unit.to_string()}")
    plt.title(f"Calculated column density of {fragName}")

    ax.w_xaxis.set_pane_color(solargreen)
    ax.w_yaxis.set_pane_color(solarblue)
    ax.w_zaxis.set_pane_color(solarblack)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(90, 90)
    plt.show()


def print_radial_density(coma):
    print("\n\nRadius (km) vs Fragment density (1/cm3)\n---------------------------------------")
    for i in range(0, coma.radial_points):
        print(f"{coma.vModel['RadialGrid'][i]:10.1f} : {coma.vModel['RadialDensity'][i].to(1/(u.cm**3)):8.4f}")


def print_column_density(coma):
    print("\nRadius (km) vs Column density (1/cm2)\n-------------------------------------")
    cds = list(zip(coma.vModel['ColumnDensity']['CDGrid'], coma.vModel['ColumnDensity']['Values']))
    for pair in cds:
        print(f'{pair[0]:7.0f} :\t{pair[1].to(1/(u.cm*u.cm)):5.3e}')


def tag_input_with_units(input_yaml):

    # apply units to data loaded from file
    input_yaml['production']['rates'] *= (1/u.s)
    input_yaml['production']['times_at_productions'] *= u.day

    input_yaml['parent']['v'] *= u.km/u.s
    input_yaml['parent']['tau_T'] *= u.s
    input_yaml['parent']['tau_d'] *= u.s
    input_yaml['parent']['sigma'] *= u.cm**2

    input_yaml['fragment']['v'] *= u.km/u.s
    input_yaml['fragment']['tau_T'] *= u.s


def run_vmodel(input_yaml):

    # apply proper units to the input
    tag_input_with_units(input_yaml)

    parent = Phys.from_dict(input_yaml['parent'])
    fragment = Phys.from_dict(input_yaml['fragment'])

    # Do the calculations
    print("Calculating fragment density using vectorial model ...")
    # coma = sba.VectorialModel(input_yaml)

    coma = sba.VectorialModel(Q=input_yaml['production']['rates'],
                              dt=input_yaml['production']['times_at_productions'],
                              parent=parent, fragment=fragment,
                              radial_points=input_yaml['grid']['radial_points'],
                              angular_points=input_yaml['grid']['angular_points'],
                              radial_substeps=input_yaml['grid']['radial_substeps'],
                              angular_substeps=input_yaml['grid']['angular_substeps'],
                              print_progress=input_yaml['printing']['print_progress'])

    return coma


def main():

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

    print(f'Loading input from {args.parameterfile[0]} ...')
    # Read in our stuff
    input_yaml = read_parameters_from_file(args.parameterfile[0])

    # Check if the length ProductionRates TimeAtProductions is the same, otherwise we're in trouble
    if(len(input_yaml['production']['rates']) !=
            len(input_yaml['production']['times_at_productions'])):
        print("Mismatched lengths for production rates and times!  Exiting.")
        return 1

    # astropy units/quantities support in plots
    quantity_support()

    # # save an unaltered copy
    # input_yaml_copy = copy.deepcopy(input_yaml)

    coma = run_vmodel(input_yaml)

    if input_yaml['printing']['print_radial_density']:
        print_radial_density(coma)
    if input_yaml['printing']['print_column_density']:
        print_column_density(coma)

    fragmentTheory = coma.vModel['NumFragmentsTheory']
    fragmentGrid = coma.vModel['NumFragmentsFromGrid']
    if input_yaml['printing']['show_agreement_check']:
        print("Theoretical total number of fragments in coma: ", fragmentTheory)
        print("Total number of fragments from density grid integration: ", fragmentGrid)

    if input_yaml['printing']['show_aperture_checks']:
        # Set aperture to entire comet to see if we get all of the fragments as an answer
        ap1 = sba.RectangularAperture((coma.vModel['MaxRadiusOfGrid'].value, coma.vModel['MaxRadiusOfGrid'].value) * u.m)
        print("Percent of total fragments recovered by integrating column density over")
        print("\tLarge rectangular aperture: ",
              coma.total_number(ap1)*100/fragmentTheory)

        # Large circular aperture
        ap2 = sba.CircularAperture((coma.vModel['MaxRadiusOfGrid'].value) * u.m)
        print("\tLarge circular aperture: ", coma.total_number(ap2)*100/fragmentTheory)

        # Try out annular
        ap3 = sba.AnnularAperture([500000, coma.vModel['MaxRadiusOfGrid'].value] * u.m)
        print("\tAnnular aperture, inner radius 500000 km, outer radius of entire grid:\n\t",
              coma.total_number(ap3)*100/fragmentTheory)

    # Show some plots
    fragName = input_yaml['fragment']['name']

    if input_yaml['plotting']['show_radial_plots']:
        show_radial_plots(coma, u.km, 1/u.cm**3, fragName)

    if input_yaml['plotting']['show_column_density_plots']:
        show_column_density_plots(coma, u.km, 1/u.cm**3, fragName)

    if input_yaml['plotting']['show_3d_column_density_centered']:
        show_3d_column_density_plots(coma, -100000*u.km, 100000*u.km, -100000*u.km, 100000*u.km, 1000, 1000, u.km, 1/u.cm**2, fragName)

    if input_yaml['plotting']['show_3d_column_density_off_center']:
        show_3d_column_density_plots(coma, -100000*u.km, 10000*u.km, -100000*u.km, 10000*u.km, 1000, 100, u.km, 1/u.cm**2, fragName)


if __name__ == '__main__':
    sys.exit(main())
