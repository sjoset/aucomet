#!/usr/bin/env python3

import sys

import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
import sbpy.activity as sba
from sbpy.data import Phys

__author__ = 'Shawn Oset'
__version__ = '0.0'


def main():

    quantity_support()

    numModelProductions = 50
    numDissocLifetimes = 20

    # Aperture to use
    square_ap_size = 2000000.0
    ap = sba.RectangularAperture((square_ap_size, square_ap_size) * u.km)

    # Ratio of total and dissociative lifetimes, Cochran 1993
    totalToDissoc = 0.93

    acceptedLifetime = sba.photo_timescale('H2O')
    # Set of water dissociative lifetimes to input into the model
    waterDisLifetimes = np.linspace(acceptedLifetime/2, acceptedLifetime*2, num=numDissocLifetimes, endpoint=True)

    # Assuming this constant count of OH in an aperture
    countInAperture = 1e32

    # Constant production for 50 days is enough to reach steady state
    daysOfSteadyProduction = 50
    times_at_production = [daysOfSteadyProduction] * u.day

    # Initial 'guess' productions to run the model to scale up or down based on the model results
    productions = np.logspace(26, 30, num=numModelProductions, endpoint=True)

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
            coma = sba.VectorialModel(Q=production_rates*(1/u.s), dt=times_at_production,
                                      parent=parent, fragment=fragment, print_progress=True)
            print("")

            # Count the number of fragments in the aperture
            ap_count = coma.total_number(ap)

            # Scale up the production to produce the desired number of counts in aperture
            calculatedQ = (countInAperture/ap_count)*production_rates[0]
            resultsArray.append([q, h2olifetime.to(u.s).value, calculatedQ])
            print(f"Calculated production: {calculatedQ}")

            fragTheory = coma.vModel['NumFragmentsTheory']
            fragGrid = coma.vModel['NumFragmentsFromGrid']
            print(f"Fragment percentage captured: {(fragGrid/fragTheory):3.5f}")

            # Check if this result holds by taking the calculated production and running another model with it;
            #  the result in the aperture should be countInAperture or very close
            production_rates = [calculatedQ]

            # print(f"Calculating for water lifetime of {h2olifetime:05.2f} and production {calculatedQ}:")
            # comaCheck = sba.VectorialModel(Q=production_rates*(1/u.s), dt=times_at_production,
            #                                parent=parent, fragment=fragment, print_progress=True)
            # print("")

            # # Count the number of fragments in the aperture
            # ap_count_check = comaCheck.total_number(ap)

            # # Does this give the right number in the aperture?
            # print(f"Number in aperture at this production: {ap_count_check:3.5e}\t\tRecovered percent: {(100*ap_count_check/countInAperture):1.7f}%")
            # print("---------")

        aggregateResults.append(resultsArray)

    # Turn our list into a numpy array
    agArray = np.array(aggregateResults)

    # Save the run data as numpy binary data because plaintext saving only works for 2d or 1d arrays
    with open('output.npdata', 'wb') as outfile:
        np.save(outfile, agArray)

    # Include details of the calculation
    with open('parameters.txt', 'w') as paramfile:
        print(f"Using steady production for {times_at_production[0]:05.2e}", file=paramfile)  # noqa: E501
        print(f"Assuming a fixed count in aperture of {countInAperture:05.2e} in {ap}", file=paramfile)
        print(f"Total to dissociative lifetime ratio: {totalToDissoc:03.2f}", file=paramfile)
        print(f"Parent and fragment velocities: {parent['v'][0]:05.2e}, {fragment['v'][0]:05.2e}", file=paramfile)  # noqa: E501


if __name__ == '__main__':
    sys.exit(main())
