#!/usr/bin/env python3

import sys
# import copy

import numpy as np
# import matplotlib.pyplot as plt
import astropy.units as u
from astropy.visualization import quantity_support
import sbpy.activity as sba
from sbpy.data import Phys

from VMPlot import vmplot

__author__ = 'Shawn Oset'
__version__ = '0.0'


def make_sine_q_t(amplitude, period, delta):
    period_in_secs = period.to(u.s).value

    def q_t(t):
        return 0.5 * amplitude * (np.sin((2 * np.pi * t)/period_in_secs + delta) + 1)

    return q_t


def main():

    quantity_support()

    fragName = 'OH'

    # Ratio of photodissociative lifetime to total lifetime, Cochran 1993
    total_to_photo_ratio = 0.93

    # Cochran & Schleicher '93
    water_photo_lifetime = sba.photo_timescale('H2O')

    parent = Phys.from_dict({
        'tau_T': water_photo_lifetime * total_to_photo_ratio,
        'tau_d': water_photo_lifetime,
        'v': 1 * u.km/u.s,
        'sigma': 3e-16 * u.cm**2
        })
    fragment = Phys.from_dict({
        'tau_T': sba.photo_timescale('OH') * 0.93,
        'v': 1.05 * u.km/u.s
        })

    # Baseline production
    base_q = 1e28

    # amplitude of outburst
    outburst_amplitude = 1e28
    # peak-to-peak period of outburst
    outburst_period = 20 * u.hour

    # Make images until this amount of time goes by, at this time step
    day_end = 10
    day_step = 0.2

    # Which outputs?
    radialDensityPlots = False
    coldens2DPlots = True
    coldens3DPlots = True

    for days_since_start in np.arange(0, day_end, step=day_step):

        q_t = make_sine_q_t(outburst_amplitude, outburst_period,
                            delta=(days_since_start*u.day).to(u.s).value)
        # debug_times_hour = np.arange((day_end*u.day).to(u.hour).value, step=1)
        # debug_times_seconds = np.array(list(map(lambda x:
        #                                (x*u.hour).to(u.s).value,
        #                                debug_times_hour)))
        # debug_prods = q_t(debug_times_seconds)
        # print(np.c_[debug_times_hour, debug_prods])

        print(f"Calculating for t = {days_since_start:05.2f} days:")
        coma = sba.VectorialModel(baseQ=base_q*(1/u.s), Qt=q_t,
                                  parent=parent, fragment=fragment, print_progress=True)
        print("")

        plotbasename = f"{days_since_start:05.2f}_days"
        print(f"Generating plots for {plotbasename} ...")

        # Do the requested plots
        if radialDensityPlots:
            vmplot.generateRadialPlots(coma, u.km, 1/u.cm**3, fragName,
                                       plotbasename+'_rdens.png',
                                       days_since_start)

        if coldens2DPlots:
            vmplot.generateColumnDensityPlot(coma, u.km, 1/u.cm**3, fragName,
                                             plotbasename+'_coldens2D.png',
                                             days_since_start)

        if coldens3DPlots:
            vmplot.generateColumnDensity3D(coma, -100000*u.km, 100000*u.km,
                                           -100000*u.km, 100000*u.km, 1000,
                                           1000, u.km, 1/u.cm**2, fragName,
                                           plotbasename+'_coldens3D_view1.png',
                                           days_since_start)

            vmplot.generateColumnDensity3D(coma, -100000*u.km, 10000*u.km,
                                           -100000*u.km, 10000*u.km, 1000, 100,
                                           u.km, 1/u.cm**2, fragName,
                                           plotbasename+'_coldens3D_view2.png',
                                           days_since_start)
        print("---------------------------------------")
        print("")

    # generate the gifs with
    # convert -delay 25 -loop 0 -layers OptimizePlus *rdens.png outburst_rdens.gif
    # convert -delay 25 -loop 0 -layers OptimizePlus *coldens2D.png outburst_coldens2D.gif
    # convert -delay 25 -loop 0 -layers OptimizePlus *coldens3D_view1.png outburst_coldens3D_view1.gif


if __name__ == '__main__':
    sys.exit(main())
