#!/usr/bin/env python3

import sys

import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
import sbpy.activity as sba
from sbpy.data import Phys

__author__ = 'Shawn Oset'
__version__ = '0.0'


def makeInputDict():

    # Fill in parameters to run vectorial model
    HeliocentricDistance = 1.0 * u.AU

    vModelInput = {}

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
    # vModelInput['Grid']['NumRadialGridpoints'] = 20
    vModelInput['Grid']['NumRadialGridpoints'] = 50
    # vModelInput['Grid']['NumAngularGridpoints'] = 20
    vModelInput['Grid']['NumAngularGridpoints'] = 30

    vModelInput['PrintDensityProgress'] = True
    return vModelInput


def main():

    quantity_support()

    numModelProductions = 2
    numDissocLifetimes = 3

    # Aperture to use
    ap = sba.RectangularAperture((40000.0, 40000.0) * u.km)

    # Ratio of total and dissociative lifetimes, Cochran 1993
    totalToDissoc = 0.93

    acceptedLifetime = sba.photo_timescale('H2O')
    # Set of water dissociative lifetimes to input into the model
    waterDisLifetimes = np.linspace(acceptedLifetime/2, acceptedLifetime*2, num=numDissocLifetimes, endpoint=True)

    # Assuming this constant count in an aperture
    countInAperture = 1e29

    # Constant production for 50 days is enough to reach steady state
    daysOfSteadyProduction = 50
    times_at_production = [daysOfSteadyProduction] * u.day

    # Initial 'guess' productions to run the model to scale up or down based on the model results
    # productions = np.logspace(25, 30, num=numModelProductions, endpoint=True)
    productions = np.logspace(27, 30, num=numModelProductions, endpoint=True)

    # Parent and fragment
    parent = Phys.from_dict({
        'tau_T': 86430 * u.s,
        'tau_d': 101730 * u.s,
        'v': 1 * u.km/u.s,
        'sigma': 3e-16 * u.cm**2
        })
    fragment = Phys.from_dict({
        'tau_T': sba.photo_timescale('OH') * 0.93,
        'v': 1.05 * u.km/u.s
        })

    aggregateResults = []

    for q in productions:

        production_rates = [q]

        # Holds [[modelProduction, lifetime, correctedProduction]]
        resultsArray = []

        for h2olifetime in waterDisLifetimes:

            parent['tau_d'][0] = h2olifetime
            parent['tau_T'][0] = h2olifetime * totalToDissoc

            print(f"Calculating for water lifetime of {h2olifetime:05.2f} and production {q}:")
            coma = sba.VectorialModel(Q=production_rates*(1/u.s), dt=times_at_production, parent=parent, fragment=fragment,
                                      print_progress=True)

            # Count the number of fragments in the aperture
            tNum = coma.total_number(ap)
            # print(f"In aperture: {tNum:5.2e}")

            # Scale up the production to produce the desired number of counts in aperture
            calculatedQ = (countInAperture/tNum) * production_rates[0]
            resultsArray.append([q, h2olifetime.to(u.s).value, calculatedQ])
            print(f"\nCalculated production: {calculatedQ}")

        aggregateResults.append(resultsArray)

    # Turn our list into a numpy array
    agArray = np.array(aggregateResults)

    # Save the run data as numpy binary data because plaintext saving only works for 2d or 1d arrays
    with open('output.npdata', 'wb') as outfile:
        np.save(outfile, agArray)
    print("Saved numpy data.")

    # Include details of the calculation
    with open('parameters.txt', 'w') as paramfile:
        print(f"Using steady production for {times_at_production[0]:05.2e}", file=paramfile)  # noqa: E501
        print(f"Assuming a fixed count in aperture of {countInAperture:05.2e} in {ap}", file=paramfile)
        print(f"Total to dissociative lifetime ratio: {totalToDissoc:03.2f}", file=paramfile)
        print(f"Parent and fragment velocities: {parent['v'][0]:05.2e}, {fragment['v'][0]:05.2e}", file=paramfile)  # noqa: E501


if __name__ == '__main__':
    sys.exit(main())
