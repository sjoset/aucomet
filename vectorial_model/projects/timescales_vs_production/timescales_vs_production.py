#!/usr/bin/env python3

import sys

import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
import sbpy.activity as sba
from sbpy.data import Phys

import pyvectorial as pyv

__author__ = 'Shawn Oset'


def q_t(t):
    return 0


def main():

    """
        Vary the dummy input production and the water lifetimes and save plotting data
    """

    quantity_support()

    numModelProductions = 3
    numDissocLifetimes = 2

    # Aperture to use
    square_ap_size = 1.0e5
    ap = sba.RectangularAperture((square_ap_size, square_ap_size) * u.km)

    # Ratio of total and dissociative lifetimes, Cochran 1993
    totalToDissoc = 0.93

    acceptedLifetime = sba.photo_timescale('H2O')
    # Set of water dissociative lifetimes to input into the model
    waterDisLifetimes = np.linspace(acceptedLifetime/2, acceptedLifetime*2, num=numDissocLifetimes, endpoint=True)

    # Assuming this constant count of OH in an aperture
    count_in_aperture = 1e32

    # Initial 'dummy' productions to run the model to scale up or down based on the model results
    productions = np.logspace(27, 29, num=numModelProductions, endpoint=True)

    parent = Phys.from_dict({
        'tau_T': 86430 * u.s,
        'tau_d': 101730 * u.s,
        'v_outflow': 1 * u.km/u.s,
        'sigma': 3e-16 * u.cm**2
        })
    fragment = Phys.from_dict({
        'tau_T': sba.photo_timescale('OH') * 0.93,
        'v_photo': 1.05 * u.km/u.s
        })

    aggregateResults = []

    for q in productions:

        # Holds [[modelProduction, lifetime, correctedProduction]]
        resultsArray = []

        for h2olifetime in waterDisLifetimes:

            parent['tau_d'][0] = h2olifetime
            parent['tau_T'][0] = h2olifetime * totalToDissoc

            print(f"Calculating for water lifetime of {h2olifetime:05.2f} and production {q}:")
            coma = sba.VectorialModel(base_q=q*(1/u.s),
                                      parent=parent, fragment=fragment, print_progress=False)
            print("")

            # Count the number of fragments in the aperture
            ap_count = coma.total_number(ap)

            # Scale up the production to produce the desired number of counts in aperture
            calculatedQ = (count_in_aperture/ap_count)*q
            resultsArray.append([q, h2olifetime.to(u.s).value, calculatedQ])
            print(f"Calculated production: {calculatedQ}")

            fragTheory = coma.vmodel['num_fragments_theory']
            fragGrid = coma.vmodel['num_fragments_grid']
            print(f"Grid: {fragGrid}, Theory: {fragTheory}")
            print(f"Fragment percentage captured: {(fragGrid*100/fragTheory):3.5f}%")

            # Check if this result holds by taking the calculated production and running another model with it;
            #  the result in the aperture should be count_in_aperture or very close

            print(f"Calculating for water lifetime of {h2olifetime:05.2f} and production {calculatedQ}:")
            comaCheck = sba.VectorialModel(base_q=calculatedQ*(1/u.s),
                                           parent=parent, fragment=fragment, print_progress=False)
            print("")

            # Count the number of fragments in the aperture
            ap_count_check = comaCheck.total_number(ap)

            # Does this give the right number in the aperture?
            print(f"Number in aperture at this production: {ap_count_check:3.5e}\t\tRecovered percent: {(100*ap_count_check/count_in_aperture):1.7f}%")

            # generate filename for saved models
            vm_out_file = f"q_{q}_h2o_taud_{h2olifetime}"
            print(vm_out_file)

            # include some other info before we save
            coma.vmodel['tau_vs_q'] = {}
            coma.vmodel['tau_vs_q']['symmetry_check'] = (100*ap_count_check)/count_in_aperture
            coma.vmodel['tau_vs_q']['assumed_aperture_count'] = count_in_aperture
            coma.vmodel['tau_vs_q']['ap_size'] = square_ap_size
            coma.vmodel['tau_vs_q']['ap_type'] = 'square'
            pyv.pickle_vmodel(coma.vmodel, vm_out_file)

            print("---------")

        aggregateResults.append(resultsArray)

    # Turn our list into a numpy array
    ag_array = np.array(aggregateResults)

    # Save the run data as numpy binary data because plaintext saving only works for 2d or 1d arrays
    with open('output.npdata', 'wb') as outfile:
        np.save(outfile, ag_array)

    # Include details of the calculation
    with open('parameters.txt', 'w') as paramfile:
        print(f"Assuming a fixed count in aperture of {count_in_aperture:05.2e} in {ap}", file=paramfile)
        print(f"Total to dissociative lifetime ratio: {totalToDissoc:03.2f}", file=paramfile)
        print(f"Parent and fragment velocities: {parent['v_outflow'][0]:05.2e}, {fragment['v_photo'][0]:05.2e}", file=paramfile)  # noqa: E501


if __name__ == '__main__':
    sys.exit(main())
