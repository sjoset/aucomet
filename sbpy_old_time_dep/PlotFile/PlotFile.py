#!/usr/bin/env python3

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

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


def doPlot(plotdata):
    fig, ax1 = plt.subplots(1, 1, figsize=(20, 10))

    # ax1.set(xlabel=f'Distance from nucleus, {rUnits.to_string()}')
    # ax1.set(ylabel=f"Fragment density, {volUnits.unit.to_string()}")
    # fig.suptitle(f"Calculated radial density of {fragName}, {lifetime:05.2f} water lifetime")

    # ax1.set_xlim([xMin_linear, xMax_linear])
    # ax1.plot(linInterpX, linInterpY, color="#688894",  linewidth=2.5, linestyle="-", label="cubic spline")
    # ax1.plot(coma.vModel['RadialGrid'], coma.vModel['RadialDensity'].to(volUnits), 'o', color="#c74a77", label="model")

    # expectedMaxRDens = 5e11
    # ax1.set_ylim(bottom=0, top=expectedMaxRDens)

    # Mark the beginning of the collision sphere
    # ax1.axvline(x=coma.vModel['CollisionSphereRadius'], color=solarblue)

    plt.legend(loc='upper right', frameon=False)
    # plt.show()
    # plt.savefig(filename)
    # plt.close()


def main():

    # quantity_support()

    # Parse command-line arguments
    parser = ArgumentParser(
        # usage='%(prog)s [options] [datafile]',
        description=__doc__,
        prog=os.path.basename(sys.argv[0])
    )
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-o', '--output', help='write plot to file as png, replacing file extension with png', action="store_true")
    parser.add_argument('-a', '--appendout', help='write plot to file as png, appending .png to filename', action="store_true")
    parser.add_argument('-d', '--dontshow', help='do not show the plot after it is generated', action="store_true")
    parser.add_argument(
            'datafile', nargs=1, help='file with data to plot in it'
            )  # the nargs=? specifies 0 or 1 arguments: it is optional
    args = parser.parse_args()

    print(f'Loading input from {args.datafile[0]} ...')
    # Read in our stuff
    plotdata = np.loadtxt(args.datafile[0])
    x, y = plotdata.T

    # Style the plot
    plt.style.use('Solarize_Light2')
    fig, ax1 = plt.subplots(1, 1, figsize=(20, 10))
    plt.scatter(x, y)

    if args.output:
        outfile = os.path.splitext(args.datafile[0])[0]+'.png'
    if args.appendout:
        outfile = args.datafile[0]+'.png'
    if args.output or args.appendout:
        print(f"Saving plot image as {outfile} ...")
        plt.savefig(outfile)

    if not args.dontshow:
        plt.show()
    plt.close()


if __name__ == '__main__':
    sys.exit(main())
