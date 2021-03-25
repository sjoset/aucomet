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

__author__ = 'Shawn Oset'
__version__ = '0.0'

solarbluecol = np.array([38, 139, 220]) / 255.
solarblue = (solarbluecol[0], solarbluecol[1], solarbluecol[2], 1)
solargreencol = np.array([133, 153, 0]) / 255.
solargreen = (solargreencol[0], solargreencol[1], solargreencol[2], 1)
solarblackcol = np.array([0, 43, 54]) / 255.
solarblack = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)
solarwhitecol = np.array([238, 232, 213]) / 255.
solarwhite = (solarblackcol[0], solarblackcol[1], solarblackcol[2], 1)


def findColDensInflectionPoints(coma):

    xs = np.linspace(0, 5e8, num=100)
    concavity = coma.vModel['ColumnDensity']['Interpolator'].derivative(nu=2)
    ys = concavity(xs)

    # for pair in zip(xs, ys):
    #     print(f"R: {pair[0]:08.1e}\t\tConcavity: {pair[1]:8.8f}")

    # Array of 1s or 0s marking if the sign changed from one element to the next
    signChanges = (np.diff(np.sign(ys)) != 0)*1

    # Manually remove the weirdness near the nucleus and rezize the array
    signChanges[0] = 0
    signChanges = np.resize(signChanges, 100)

    radInflectionPts = xs*signChanges
    # Only want non-zero elements
    radInflectionPts = radInflectionPts[radInflectionPts > 0]

    # Only want inflection points outside the collision sphere
    cRadius = coma.vModel['CollisionSphereRadius'].to_value(u.m)
    radInflectionPts = radInflectionPts[radInflectionPts > cRadius]

    return radInflectionPts


def generateRadialPlots(coma, rUnits, volUnits, fragName, filename, daysSince):
    """ Show the radial density of the fragment species """

    xMin_linear = 0 * u.m
    xMax_linear = 2000000 * u.m

    linInterpX = np.linspace(xMin_linear.value, xMax_linear.value, num=200)
    linInterpY = coma.vModel['rDensInterpolator'](linInterpX)/(u.m**3)
    linInterpX *= u.m
    linInterpX.to(rUnits)

    # Make sure to clear the dark background setting from the 3D plots by defaulting everything here
    plt.style.use('default')
    plt.style.use('Solarize_Light2')

    fig, ax1 = plt.subplots(1, 1, figsize=(20, 10))

    ax1.set(xlabel=f'Distance from nucleus, {rUnits.to_string()}')
    ax1.set(ylabel=f"Fragment density, {volUnits.unit.to_string()}")
    fig.suptitle(f"Calculated radial density of {fragName}, {daysSince:05.2f} days after outburst")

    ax1.set_xlim([xMin_linear, xMax_linear])
    ax1.plot(linInterpX, linInterpY, color="#688894",  linewidth=2.5, linestyle="-", label="cubic spline")
    ax1.plot(coma.vModel['RadialGrid'], coma.vModel['RadialDensity'].to(volUnits), 'o', color="#c74a77", label="model")

    expectedMaxRDens = 5e11
    ax1.set_ylim(bottom=0, top=expectedMaxRDens)

    # Mark the beginning of the collision sphere
    ax1.axvline(x=coma.vModel['CollisionSphereRadius'], color=solarblue)

    plt.legend(loc='upper right', frameon=False)
    # plt.show()
    plt.savefig(filename)
    plt.close()


def generateColumnDensityPlot(coma, rUnits, cdUnits, fragName, filename, daysSince):
    """ Show the radial density of the fragment species """

    xMin_linear = 0 * u.m
    xMax_linear = coma.vModel['MaxRadiusOfGrid'].to(u.m)

    linInterpX = np.linspace(xMin_linear.value, xMax_linear.value, num=200)
    linInterpY = coma.vModel['ColumnDensity']['Interpolator'](linInterpX)/(u.m**2)
    linInterpX *= u.m
    linInterpX.to(rUnits)

    # Make sure to clear the dark background setting from the 3D plots by defaulting everything here
    plt.style.use('default')
    plt.style.use('Solarize_Light2')

    fig, ax1 = plt.subplots(1, 1, figsize=(20, 10))

    ax1.set(xlabel=f'Distance from nucleus, {rUnits.to_string()}')
    ax1.set(ylabel=f"Fragment column density, {cdUnits.unit.to_string()}")
    fig.suptitle(f"Calculated column density of {fragName}, {daysSince:05.2f} days after outburst")

    ax1.set_xlim([xMin_linear, xMax_linear])
    ax1.plot(linInterpX, linInterpY, color='#688894',  linewidth=2.5, linestyle="-", label="cubic spline")
    ax1.plot(coma.vModel['ColumnDensity']['CDGrid'], coma.vModel['ColumnDensity']['Values'], 'o', color='#c74a77', label="model")

    expectedMaxColDens = 2e17
    ax1.set_ylim(bottom=0, top=expectedMaxColDens)

    # Find possible inflection points
    for ipoint in findColDensInflectionPoints(coma):
        ax1.axvline(x=ipoint, color=solargreen)

    plt.legend(loc='upper right', frameon=False)
    # plt.show()
    plt.savefig(filename)
    plt.close()


def generateColumnDensity3D(coma, xMin, xMax, yMin, yMax, gridStepX, gridStepY, rUnits, cdUnits, fragName, filename,
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
    surf = ax.plot_surface(xvu, yvu, fz, cmap='inferno', norm=normColors, edgecolor='none')

    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    frame1.axes.zaxis.set_ticklabels([])

    ax.set_zlim(bottom=0, top=expectedMaxColDens)

    ax.set_xlabel(f'Distance, ({rUnits.to_string()})')
    ax.set_ylabel(f'Distance, ({rUnits.to_string()})')
    ax.set_zlabel(f"Column density, {cdUnits.unit.to_string()}")
    plt.title(f"Calculated column density of {fragName}, {daysSince:05.2f} days after outburst")

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
    vModelInput['Fragment']['TotalLifetime'] = sba.photo_timescale('OH') * 0.93
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

    # Outburst information
    outburstLength = 5 * u.day
    baseProduction = 1e28
    outburstProduction = 1e29

    # Long enough ago that we reach steady state by the time the outburst happens
    daysAgoProductionStarts = 50

    # Make images after the outburst until this amount of time goes by
    daysAfterToStop = 5
    dayStep = 0.2

    # Which outputs?
    radialDensityPlots = False
    coldens2DPlots = True
    coldens3DPlots = False

    for daysSince in np.arange(0, daysAfterToStop, step=dayStep):

        vModelInput = copy.deepcopy(vModelInputBase)
        vModelInput['TimeAtProductions'] = [daysAgoProductionStarts, daysSince+outburstLength.to_value(u.day), daysSince] * u.day
        vModelInput['ProductionRates'] = [baseProduction, outburstProduction, baseProduction]
        print("")
        for pair in zip(vModelInput['TimeAtProductions'], vModelInput['ProductionRates']):
            print(f"Time: {pair[0]:05.2f} ago\t\tProduction rate: {pair[1]:05.2e}")

        print(f"Calculating for {daysSince:05.2f} days since outburst:")
        coma = sba.VectorialModel(0*(1/u.s), 0 * u.m/u.s, vModelInput)
        print("")

        plotbasename = f"{daysSince:05.2f}_days_since_outburst"
        print(f"Generating plots for {plotbasename} ...")

        # The volume density seems to evolve very quickly, it settles ~0.1 days after a 5-day outburst
        if radialDensityPlots:
            generateRadialPlots(coma, u.km, 1/u.cm**3, fragName, plotbasename+'_rdens.png', daysSince)

        if coldens2DPlots:
            generateColumnDensityPlot(coma, u.km, 1/u.cm**3, fragName, plotbasename+'_coldens2D.png', daysSince)

        if coldens3DPlots:
            generateColumnDensity3D(coma, -100000*u.km, 100000*u.km, -100000*u.km, 100000*u.km, 1000, 1000, u.km, 1/u.cm**2,
                                    fragName, plotbasename+'_coldens3D_view1.png', daysSince)
            # generateColumnDensity3D(coma, -100000*u.km, 10000*u.km, -100000*u.km, 10000*u.km, 1000, 100, u.km, 1/u.cm**2,
            #                         fragName, plotbasename+'_coldens3D_view2.png', daysSince)

    # generate the gifs with
    # convert -delay 25 -loop 0 -layers OptimizePlus *rdens.png outburst_rdens.gif
    # convert -delay 25 -loop 0 -layers OptimizePlus *coldens2D.png outburst_coldens2D.gif
    # convert -delay 25 -loop 0 -layers OptimizePlus *coldens3D_view1.png outburst_coldens3D_view1.gif


if __name__ == '__main__':
    sys.exit(main())
