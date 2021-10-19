#!/usr/bin/env python3

import numpy
import sys

import numpy as np
import astropy.units as u
import sbpy.activity as sba
from sbpy.data import Phys
from astropy import constants as const


def test_festou92():
    """Compare to Festou et al. 1992 production rates of comet 6P/d'Arrest.
    Festou et al. 1992. The Gas Production Rate of Periodic Comet d'Arrest.
    Asteroids, Comets, Meteors 1991, 177.
    https://ui.adsabs.harvard.edu/abs/1992acm..proc..177F/abstract
    IUE observations of OH(0-0) band.
    The table appears to have a typo in the Q column for the 1.32 au
    observation: using 2.025 instead of 3.025.
    """

    # Table 2
    rh = [1.2912, 1.2949, 1.3089, 1.3200, 1.3366] * u.au
    delta = [0.7410, 0.7651, 0.8083, 0.8353, 0.8720] * u.au
    flux = ([337e-14, 280e-14, 480e-14, 522e-14, 560e-14]
            * u.erg / u.cm**2 / u.s)
    g = [2.33e-4, 2.60e-4, 3.36e-4, 3.73e-4, 4.03e-4] / u.s
    Q = [1.451e28, 1.228e28, 1.967e28, 2.025e28, 2.035e28] / u.s

    # OH (0-0) luminosity per molecule
    L_N = g / (rh / u.au)**2 * const.h * const.c / (3086 * u.AA)

    # Parent molecule is H2O
    parent = Phys.from_dict({
        # 'tau_T': np.ones(len(rh)) * 65000 * u.s,
        # 'tau_T': tau_Ts,
        'tau_T': (rh / u.au)**2 * 65000 * u.s,
        'tau_d': 72500 * (rh / u.au)**2 * u.s,
        # 'tau_d': tau_ds,
        'v_outflow': np.ones(len(rh)) * 0.85 * u.km / u.s,
        'sigma': np.ones(len(rh)) * 3e-16 * u.cm**2,
    })
    # Fragment molecule is OH
    fragment = Phys.from_dict({
        # 'tau_T': 160000 * u.s,
        'tau_T': 160000 * (rh / u.au)**2 * u.s,
        # 'v_photo': 1.05 * u.km / u.s
        'v_photo': 1.05 * np.ones(len(rh)) * u.km / u.s
    })

    # https://pds.nasa.gov/ds-view/pds/viewInstrumentProfile.jsp?INSTRUMENT_ID=LWP&INSTRUMENT_HOST_ID=IUE
    # Large-Aperture Length(arcsec)   22.51+/-0.40
    # Large-Aperture Width(arcsec)     9.91+/-0.17
    #
    # 10x20 quoted by Festou et al.
    # effective circle is 8.0 radius
    # half geometric mean is 7.07
    # lwp = core.CircularAperture(7.07 * u.arcsec)  # geometric mean
    # lwp = sba.RectangularAperture((20, 10) * u.arcsec)
    lwp = sba.RectangularAperture((9.1, 15.3) * u.arcsec)

    Q0 = 1e28 / u.s
    N = []
    comas = []
    for i in range(len(rh)):
        coma = sba.VectorialModel(base_q=Q0,
                                  parent=parent[i],
                                  fragment=fragment[i],
                                  print_progress=True)
        comas.append(coma)
        N.append(coma.total_number(lwp, eph=delta[i]))

    N = u.Quantity(N)
    Q_model = (Q0 * flux / (L_N * N) * 4 * np.pi * delta**2).to(Q.unit)
    Q_ratio = (Q/Q_model)
    print(f"Q: {Q}")
    print(f"Q_model: {Q_model}")
    print(f"Q/Q_model: {Q_ratio}")
    # assert u.allclose(Q, Q_model)
    return comas


def main():
    comas = test_festou92()

    filebase = "dArrest_"
    for i, coma in enumerate(comas):
        out_file = filebase + str(i)
        with open(out_file, 'w') as f:
            print("\n\nRadius (km) vs Fragment density (1/cm3)\n---------------------------------------", file=f)
            volume_densities = list(zip(coma.vmodel['radial_grid'], coma.vmodel['radial_density']))
            for pair in volume_densities:
                print(f'{pair[0].to(u.km):7.0f} :\t{pair[1].to(1/(u.cm**3)):5.3e}', file=f)

            print("\nRadius (km) vs Column density (1/cm2)\n-------------------------------------", file=f)
            column_densities = list(zip(coma.vmodel['column_density_grid'], coma.vmodel['column_densities']))
            for pair in column_densities:
                print(f'{pair[0].to(u.km):7.0f} :\t{pair[1].to(1/(u.cm**2)):5.3e}', file=f)


if __name__ == '__main__':
    sys.exit(main())
