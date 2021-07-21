#!/usr/bin/env python3

import sys
# import copy

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


def generate2D_TvsQPlot(TvsQdata, modelProduction):

    timescales = TvsQdata[:, 1]
    Qs = TvsQdata[:, 2]

    # Make sure to clear the dark background setting from the 3D plots by defaulting everything here
    plt.style.use('default')
    plt.style.use('Solarize_Light2')

    fig, ax1 = plt.subplots(1, 1, figsize=(20, 10))

    ax1.set(xlabel="Dissociative lifetime of H2O, (s)")
    ax1.set(ylabel="Q(H2O)")
    fig.suptitle(f"Calculated productions against varying lifetimes for initial model Q = {modelProduction}")

    # ax1.set_xlim([xMin_linear, xMax_linear])
    ax1.plot(timescales, Qs, color="#688894",  linewidth=2.5, linestyle="-", label="", zorder=1)
    ax1.scatter(timescales, Qs, color="#c74a77", zorder=2)
    # ax1.plot(coma.vModel['RadialGrid'], coma.vModel['RadialDensity'].to(volUnits), 'o', color="#c74a77", label="model")

    # ax1.set_ylim(bottom=1e29, top=1e30)

    plt.legend(loc='upper right', frameon=False)
    plt.show()
    # plt.savefig(filename)
    plt.close()


def generateAggregatePlots(TvsQdata, filename):

    modelQ = TvsQdata[:, 0]
    timescales = TvsQdata[:, 1]
    Qs = TvsQdata[:, 2]

    numTimes = len(set(timescales))
    numModelQs = len(set(modelQ))
    # numPoints = numTimes * numModelQs

    # Make sure to clear the dark background setting from the 3D plots by defaulting everything here
    plt.style.use('default')
    plt.style.use('Solarize_Light2')

    # normColors = colors.Normalize(vmin=0, vmax=5e31, clip=False)

    fig = plt.figure(figsize=(30, 30))
    ax = plt.axes(projection='3d')
    surf = ax.scatter(modelQ, timescales, Qs)
    # ax.plot3D([modelQ[0]]*3, timescales[:3], Qs[:3], color="#c74a77")

    for x in list(range(0, numModelQs)):
        print(x)
        ax.plot3D(modelQ[x*numTimes:(x+1)*numTimes], timescales[x*numTimes:(x+1)*numTimes], Qs[x*numTimes:(x+1)*numTimes], color="#c74a77")

    # ax.bar3d(modelQ, timescales, np.zeros(numPoints), np.ones(len(modelQ)), np.ones(len(timescales)), Qs)
    # surf = ax.scatter(modelQ, timescales, Qs, cmap='inferno', norm=normColors, edgecolor='none')

    # ax1.set(xlabel="Dissociative lifetime of H2O, (s)")
    # ax1.set(ylabel="Q(H2O)")
    fig.suptitle("Calculated productions against varying lifetimes for range of model Q(H2O)")

    # ax1.set_xlim([xMin_linear, xMax_linear])
    # ax1.plot(timescales, Qs, color="#688894",  linewidth=2.5, linestyle="-", label="", zorder=1)
    # ax1.plot_surface(modelQ, timescales, Qs)
    # ax1.scatter(timescales, Qs, color="#c74a77", zorder=2)
    # ax1.plot(coma.vModel['RadialGrid'], coma.vModel['RadialDensity'].to(volUnits), 'o', color="#c74a77", label="model")

    # ax1.set_ylim(bottom=1e29, top=1e30)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.view_init(0, 0)

    plt.legend(loc='upper right', frameon=False)
    plt.savefig(filename)
    plt.show()
    plt.close()


def generateRadialPlots(coma, rUnits, volUnits, fragName, filename, lifetime):
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
    fig.suptitle(f"Calculated radial density of {fragName}, {lifetime:05.2f} water lifetime")

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


def generateColumnDensityPlot(coma, rUnits, cdUnits, fragName, filename, lifetime):
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
    fig.suptitle(f"Calculated column density of {fragName}, {lifetime:05.2f} water lifetime")

    ax1.set_xlim([xMin_linear, xMax_linear])
    ax1.plot(linInterpX, linInterpY, color='#688894',  linewidth=2.5, linestyle="-", label="cubic spline")
    ax1.plot(coma.vModel['ColumnDensity']['CDGrid'], coma.vModel['ColumnDensity']['Values'], 'o', color='#c74a77', label="model")

    expectedMaxColDens = 2e17
    ax1.set_ylim(bottom=0, top=expectedMaxColDens)

    plt.legend(loc='upper right', frameon=False)
    # plt.show()
    plt.savefig(filename)
    plt.close()


def generateColumnDensity3D(coma, xMin, xMax, yMin, yMax, gridStepX, gridStepY, rUnits, cdUnits, fragName, filename,
                            lifetime):
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
    plt.title(f"Calculated column density of {fragName}, {lifetime:05.2f} water lifetime")

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
    HeliocentricDistance = 1.0 * u.AU

    vModelInput = {}

    # Steady-state production for 50 days
    vModelInput['TimeAtProductions'] = [50] * u.day
    vModelInput['ProductionRates'] = [1e28]

    # Parent molecule is H2O
    vModelInput['Parent'] = {}
    vModelInput['Parent']['Velocity'] = 1.0 * (u.km/u.s)
    vModelInput['Parent']['TotalLifetime'] = 86430 * u.s
    vModelInput['Parent']['DissociativeLifetime'] = 101730 * u.s

    # vModelInput['Parent']['DissociativeLifetime'] = sba.photo_timescale('H2O')
    # vModelInput['Parent']['TotalLifetime'] = vModelInput['Parent']['DissociativeLifetime']*0.8

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

    vModelInput['Grid'] = {}
    vModelInput['Grid']['NumRadialGridpoints'] = 20
    # vModelInput['Grid']['NumRadialGridpoints'] = 100
    vModelInput['Grid']['NumAngularGridpoints'] = 20
    # vModelInput['Grid']['NumAngularGridpoints'] = 40

    vModelInput['PrintDensityProgress'] = True
    return vModelInput


def main():

    quantity_support()

    numModelProductions = 3
    numDissocLifetimes = 3

    vModelInputBase = makeInputDict()

    # Aperture to use
    ap = sba.RectangularAperture((100.0, 100.0) * u.km)

    # Ratio of total and dissociative lifetimes, Cochran 1993
    totalToDissoc = 0.93

    acceptedLifetime = sba.photo_timescale('H2O')
    # Set of water dissociative lifetimes to input into the model
    waterDisLifetimes = np.linspace(acceptedLifetime/2, acceptedLifetime*2, num=numDissocLifetimes)

    # Assuming this constant count in an aperture
    countInAperture = 1e29
    # Constant production for 50 days is enough to reach steady state
    vModelInputBase['TimeAtProductions'] = [50] * u.day

    # Initial 'guess' productions to run the model to scale up or down based on the model results
    # productions = [5e25, 1e26, 1e27, 1e28, 1e29]
    # productions = np.logspace(26, 29, num=2)
    productions = np.logspace(28, 29, num=numModelProductions)

    aggregateResults = []

    for q in productions:

        vModelInputBase['ProductionRates'] = [q]

        # Holds [[modelProduction, lifetime, correctedProduction]]
        resultsArray = []

        for h2olifetime in waterDisLifetimes:

            # vModelInput = copy.deepcopy(vModelInputBase)
            vModelInput = vModelInputBase

            vModelInput['Parent']['DissociativeLifetime'] = h2olifetime
            vModelInput['Parent']['TotalLifetime'] = h2olifetime*totalToDissoc

            print(f"Calculating for water lifetime of {h2olifetime:05.2f} and production {q}:")
            coma = sba.VectorialModel(0*(1/u.s), 0 * u.m/u.s, vModelInput)
            print("")
            print("Done.")

            # Count the number of fragments in the aperture
            tNum = coma.total_number(ap)

            # Scale up the production to produce the desired number of counts in aperture
            resultsArray.append([q, h2olifetime.to(u.s).value, (countInAperture/tNum)*vModelInput['ProductionRates'][0]])

        # generate2D_TvsQPlot(np.array(resultsArray), q)
        # print(resultsArray)
        aggregateResults += resultsArray
        # print(aggregateResults)

        # Output the results to file
        with open("results", "a") as outfile:
            print(f"Using steady production for {vModelInputBase['TimeAtProductions'][0]:05.2e} at {q:05.2e} molecules/sec", file=outfile)  # noqa: E501
            print(f"Assuming a fixed count in aperture of {countInAperture:05.2e} in 100km square aperture", file=outfile)
            print(f"Total to dissociative lifetime ratio: {totalToDissoc:03.2f}", file=outfile)
            print(f"Parent and fragment velocities: {vModelInputBase['Parent']['Velocity']:05.2e}, {vModelInputBase['Fragment']['Velocity']:05.2e}", file=outfile)  # noqa: E501

            print("Photo Tscale\tAp Count\tCalc Production", file=outfile)
            for row in resultsArray:
                for col in row:
                    print(f"{col:08.6e}\t", end='', file=outfile)
                print("", file=outfile)
            print("\n\n", file=outfile)

    generateAggregatePlots(np.array(aggregateResults), 'results.png')


if __name__ == '__main__':
    sys.exit(main())
