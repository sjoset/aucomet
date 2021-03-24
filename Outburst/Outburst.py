#!/usr/bin/env python3

import sys
import copy

import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
# from matplotlib import cm, colors
from matplotlib import colors
import sbpy.activity as sba

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


def showRadialPlots(coma, rUnits, volUnits, fragName):
    """ Show the radial density of the fragment species """

    xMin_logplot = np.ceil(np.log10(coma.vModel['CollisionSphereRadius'].to(u.m).value))
    xMax_logplot = np.floor(np.log10(coma.vModel['MaxRadiusOfGrid'].to(u.m).value))

    xMin_linear = 0 * u.m
    xMax_linear = 2000000 * u.m

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


def showColumnDensityPlot(coma, rUnits, cdUnits, fragName, filename, daysSince):
    """ Show the radial density of the fragment species """

    # xMin_logplot = np.ceil(np.log10(coma.vModel['CollisionSphereRadius'].to(u.m).value))
    # xMax_logplot = np.floor(np.log10(coma.vModel['MaxRadiusOfGrid'].to(u.m).value))

    xMin_linear = 0 * u.m
    # xMax_linear = 2000000 * u.m
    xMax_linear = coma.vModel['MaxRadiusOfGrid'].to(u.m)

    linInterpX = np.linspace(xMin_linear.value, xMax_linear.value, num=200)
    linInterpY = coma.vModel['ColumnDensity']['Interpolator'](linInterpX)/(u.m**2)
    linInterpX *= u.m
    linInterpX.to(rUnits)

    # logInterpX = np.logspace(xMin_logplot, xMax_logplot, num=200)
    # logInterpY = coma.vModel['ColumnDensity']['Interpolator'](logInterpX)/(u.m**2)
    # logInterpX *= u.m
    # logInterpX.to(rUnits)

    plt.style.use('Solarize_Light2')

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig, ax1 = plt.subplots(1, 1, figsize=(20, 10))

    ax1.set(xlabel=f'Distance from nucleus, {rUnits.to_string()}')
    ax1.set(ylabel=f"Fragment column density, {cdUnits.unit.to_string()}")
    # ax2.set(xlabel=f'Distance from nucleus, {rUnits.to_string()}')
    # ax2.set(ylabel=f"Fragment column density, {cdUnits.unit.to_string()}")
    fig.suptitle(f"Calculated column density of {fragName}, {daysSince:02.1f} days after outburst")

    ax1.set_xlim([xMin_linear, xMax_linear])
    ax1.plot(linInterpX, linInterpY, color='#688894',  linewidth=2.5, linestyle="-", label="cubic spline")
    ax1.plot(coma.vModel['ColumnDensity']['CDGrid'], coma.vModel['ColumnDensity']['Values'], 'o', color='#c74a77', label="model")

    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # ax2.loglog(coma.vModel['ColumnDensity']['CDGrid'], coma.vModel['ColumnDensity']['Values'], 'bo', label="model")
    # ax2.loglog(coma.vModel['ColumnDensity']['CDGrid'], coma.vModel['ColumnDensity']['Values'], 'g--',
    #            label="linear interpolation", linewidth=0.5)
    # ax2.loglog(logInterpX, logInterpY, color="red",  linewidth=0.5, linestyle="-", label="cubic spline")

    ax1.set_ylim(bottom=0, top=2e17)

    # ax2.set_xlim(right=coma.vModel['MaxRadiusOfGrid'])

    # Mark the beginning of the collision sphere
    # ax1.axvline(x=coma.vModel['CollisionSphereRadius'], color=solarblue)
    # ax2.axvline(x=coma.vModel['CollisionSphereRadius'], color=solarblue)

    # Only plot as far as the maximum radius of our grid on log-log plot
    # ax2.axvline(x=coma.vModel['MaxRadiusOfGrid'])

    # Mark the collision sphere
    # plt.text(coma.vModel['CollisionSphereRadius']*1.1, linInterpY[0]/10, 'Collision Sphere Edge', color=solarblue)

    plt.legend(loc='upper right', frameon=False)
    # plt.show()
    plt.savefig(filename)
    plt.close()


def showColumnDensity3D(coma, xMin, xMax, yMin, yMax, gridStepX, gridStepY, rUnits, cdUnits, fragName, filename,
                        daysSince):
    """ 3D plot of column density """

    # mesh grid for units native to the interpolation function
    x = np.linspace(xMin.to(u.m).value, xMax.to(u.m).value, gridStepX)
    y = np.linspace(yMin.to(u.m).value, yMax.to(u.m).value, gridStepY)
    xv, yv = np.meshgrid(x, y)
    z = coma.vModel['ColumnDensity']['Interpolator'](np.sqrt(xv**2 + yv**2))
    # Interpolator spits out m^-2
    fz = (z/u.m**2).to(cdUnits)

    # mesh grid in the units requested for plotting
    xu = np.linspace(xMin.to(rUnits), xMax.to(rUnits), gridStepX)
    yu = np.linspace(yMin.to(rUnits), yMax.to(rUnits), gridStepY)
    xvu, yvu = np.meshgrid(xu, yu)

    plt.style.use('Solarize_Light2')
    plt.style.use('dark_background')
    plt.rcParams['grid.color'] = "black"

    expectedMaxColDens = 3e13
    normColors = colors.Normalize(vmin=0, vmax=expectedMaxColDens/1.5, clip=False)

    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes(projection='3d')
    # ax.grid(False)
    # surf = ax.plot_surface(xvu, yvu, fz, cmap='inferno', vmin=0, edgecolor='none')
    surf = ax.plot_surface(xvu, yvu, fz, cmap='inferno', norm=normColors, edgecolor='none')

    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    frame1.axes.zaxis.set_ticklabels([])

    # frame1.set_zlim(bottom=0)
    ax.set_zlim(bottom=0, top=expectedMaxColDens)
    # ax.set_zlim(bottom=0)

    ax.set_xlabel(f'Distance, ({rUnits.to_string()})')
    ax.set_ylabel(f'Distance, ({rUnits.to_string()})')
    ax.set_zlabel(f"Column density, {cdUnits.unit.to_string()}")
    plt.title(f"Calculated column density of {fragName}, {daysSince:02.1f} days after outburst")

    ax.w_xaxis.set_pane_color(solargreen)
    ax.w_yaxis.set_pane_color(solarblue)
    ax.w_zaxis.set_pane_color(solarblack)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.view_init(25, 45)
    ax.view_init(90, 90)
    # plt.show()
    plt.savefig(filename)
    plt.close()


def makeInputDict():

    # Fill in parameters to run vectorial model
    HeliocentricDistance = 1.4 * u.AU

    vModelInput = {}

    # # 30 days ago to 15 days ago, production was 1e28,
    # # then jumped to 3e29 from 15 days ago to now
    vModelInput['TimeAtProductions'] = [30, 15] * u.day
    vModelInput['ProductionRates'] = [1.e28, 3.e29]

    # Parent molecule is H2O
    vModelInput['Parent'] = {}
    vModelInput['Parent']['TotalLifetime'] = 86430 * u.s
    vModelInput['Parent']['DissociativeLifetime'] = 101730 * u.s

    # vModelInput['Parent']['DissociativeLifetime'] = sba.photo_timescale('H2O')
    # vModelInput['Parent']['TotalLifetime'] = vModelInput['Parent']['DissociativeLifetime']*0.8

    vModelInput['Parent']['Velocity'] = 1.0 * (u.km/u.s)

    # Fragment molecule is OH
    vModelInput['Fragment'] = {}
    # Cochran and Schleicher, 1993
    vModelInput['Fragment']['Velocity'] = 1.05 * u.km/u.s
    vModelInput['Fragment']['TotalLifetime'] = sba.photo_timescale('OH') * 0.8
    # vModelInput['Fragment']['TotalLifetime'] = 129000 * u.s

    # Adjust some things for heliocentric distance
    rs2 = HeliocentricDistance.value**2
    vModelInput['Parent']['TotalLifetime'] *= rs2
    vModelInput['Parent']['DissociativeLifetime'] *= rs2
    vModelInput['Fragment']['TotalLifetime'] *= rs2
    # Cochran and Schleicher, 1993
    vModelInput['Parent']['Velocity'] /= np.sqrt(HeliocentricDistance.value)

    # vModelInput['Grid'] = {}
    # vModelInput['Grid']['NumRadialGridpoints'] = 25
    # vModelInput['Grid']['NumAngularGridpoints'] = 40

    vModelInput['PrintDensityProgress'] = True
    return vModelInput


def main():

    quantity_support()

    fragName = 'OH'
    vModelInputBase = makeInputDict()

    for daysSince in np.arange(0, 5, step=0.25):

        vModelInput = copy.deepcopy(vModelInputBase)
        vModelInput['TimeAtProductions'] = [50, daysSince+5, daysSince] * u.day
        vModelInput['ProductionRates'] = [1.e28, 1.e29, 1.e28]
        print(vModelInput['TimeAtProductions'], vModelInput['ProductionRates'])

        print(f"Calculating for {daysSince} days since outburst:")
        coma = sba.VectorialModel(0*(1/u.s), 0 * u.m/u.s, vModelInput)

        plotfilename = f"{daysSince:05.2f}_days_since_outburst.png"
        print(f"Generating {plotfilename} ...")

        # showRadialPlots(coma, u.km, 1/u.cm**3, fragName)
        # showColumnDensityPlot(coma, u.km, 1/u.cm**3, fragName, plotfilename, daysSince)
        # showColumnDensity3D(coma, -100000*u.km, 10000*u.km, -100000*u.km, 10000*u.km, 1000, 100, u.km, 1/u.cm**2, fragName)
        showColumnDensity3D(coma, -100000*u.km, 100000*u.km, -100000*u.km, 100000*u.km, 1000, 1000, u.km, 1/u.cm**2,
                            fragName, plotfilename, daysSince)

    # generate the gif with
    # convert -delay 25 -loop 0 -layers OptimizePlus *.png outburst.gif


if __name__ == '__main__':
    sys.exit(main())
