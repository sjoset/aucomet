#!/usr/bin/env python3

import sys
# import copy

import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
import sbpy.activity as sba
from sbpy.data import Phys

import vmplotter
from timedependentproduction import TimeDependentProduction

__author__ = 'Shawn Oset'
__version__ = '0.0'


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
        'v_outflow': 1 * u.km/u.s,
        'sigma': 3e-16 * u.cm**2
        })
    fragment = Phys.from_dict({
        'tau_T': sba.photo_timescale('OH') * 0.93,
        'v_photo': 1.05 * u.km/u.s
        })

    # Baseline production
    base_q = 1e27

    # Dictionary to hold outburst data
    outburst = {}
    outburster = TimeDependentProduction("sine wave")

    if outburster.type == "gaussian":
        # amplitude of outburst
        outburst['amplitude'] = 2e28
        # standard deviation of gaussian
        outburst['std_dev'] = 3 * u.hour
    elif outburster.type == "sine wave":
        # amplitude of outburst
        outburst['amplitude'] = 2e28
        # peak-to-peak period of outburst
        outburst['period'] = 20 * u.hour
    elif outburster.type == "square pulse":
        # amplitude of outburst
        # outburst['amplitude'] = 2e28
        outburst['amplitude'] = 5e27
        # starts at t = tstart
        outburst['duration'] = 1.0 * u.day
    else:
        print("Unsupported outburst type! Exiting.")
        return

    # Run model until this amount of time goes by, at this time step
    day_end = 1.5
    day_step = 0.5

    # Which outputs?
    radialDensityPlots = False
    coldens2DPlots = False
    coldens3DPlots = True

    for days_since_start in np.arange(0, day_end, step=day_step):

        if outburster.type == "gaussian":
            t_max = days_since_start - (day_end/2)
            q_t = outburster.create_production(
                                             amplitude=outburst['amplitude']*(1/u.s),
                                             t_max=t_max*u.day,
                                             std_dev=outburst['std_dev']
                                             )
        elif outburster.type == "sine wave":
            q_t = outburster.create_production(
                                             amplitude=outburst['amplitude']*(1/u.s),
                                             period=outburst['period'],
                                             delta=days_since_start*u.day
                                             )
        elif outburster.type == "square pulse":
            t_start = days_since_start - (day_end/2) + (outburst['duration'].value/2)
            q_t = outburster.create_production(
                                             amplitude=outburst['amplitude']*(1/u.s),
                                             t_start=t_start*u.day,
                                             duration=outburst['duration']
                                             )

        print(f"Calculating for t = {days_since_start:05.2f} days:")
        coma = sba.VectorialModel(base_q=base_q*(1/u.s),
                                  parent=parent,
                                  fragment=fragment,
                                  q_t=q_t,
                                  print_progress=True)
        print("")

        plotbasename = f"{days_since_start:05.2f}_days"
        print(f"Generating plots for {plotbasename} ...")

        # Do the requested plots
        if radialDensityPlots:
            vmplotter.outburst_radial_density_plot(coma, u.km, 1/u.cm**3, fragName,
                                                   plotbasename+'_rdens.png',
                                                   days_since_start)

        if coldens2DPlots:
            vmplotter.outburst_column_density_plot(coma, u.km, 1/u.cm**3, fragName,
                                                   plotbasename+'_coldens2D.png',
                                                   days_since_start)

        if coldens3DPlots:
            vmplotter.outburst_column_density_plot_3d(coma, -100000*u.km, 100000*u.km,
                                                      -100000*u.km, 100000*u.km, 1000,
                                                      1000, u.km, 1/u.cm**2, fragName,
                                                      plotbasename+'_coldens3D_view1.png',
                                                      days_since_start)

            vmplotter.outburst_column_density_plot_3d(coma, -100000*u.km, 10000*u.km,
                                                      -100000*u.km, 10000*u.km, 1000, 100,
                                                      u.km, 1/u.cm**2, fragName,
                                                      plotbasename+'_coldens3D_view2.png',
                                                      days_since_start)
        print("---------------------------------------")
        print("")

        # debug_times_hour = np.arange((day_end*u.day).to(u.hour).value, step=1)
        # debug_times_seconds = np.array(list(map(lambda x:
        #                                (x*u.hour).to(u.s).value,
        #                                debug_times_hour)))
        # debug_prods = q_t(debug_times_seconds)
        # print(np.c_[debug_times_hour, debug_prods])


if __name__ == '__main__':
    sys.exit(main())
