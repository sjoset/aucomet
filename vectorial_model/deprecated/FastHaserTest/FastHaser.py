#!/usr/bin/env python3

import sys
import time

import numpy as np
import astropy.units as u
from scipy import special
from scipy.interpolate import CubicSpline
from astropy.visualization import quantity_support
import sbpy.activity as sba

import multiprocessing as mp


# Returns interpolator function for iK0(x) given a numpy space for the function sampling
def makeiK0(sampleSpace):

    iK0_true = special.iti0k0(sampleSpace)[1]
    iK0spline = CubicSpline(sampleSpace, iK0_true)

    return iK0spline


# Returns interpolator function for K1(x) given a numpy space for the function sampling
def makeK1(sampleSpace):

    K1_true = special.k1(sampleSpace)
    K1spline = CubicSpline(sampleSpace, K1_true)

    return K1spline


def fastK1(self, x):
    return self.newK1(x)


def fastiK0(self, x):
    return self.newiK0(x)


def timeModelCalcs(comae, aperture):

    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    results = [pool.apply_async(coma.total_number, args=(aperture,)) for coma in comae]

    resultList = [res.get(timeout=None) for res in results]
    end = time.time()
    print(resultList)

    pool.close()
    return (end - start), np.array(resultList)


def main():

    quantity_support()

    # Use a low-quality interpolation for quick function value lookup
    sampleSpace = np.logspace(-5, 2, num=5)

    K1interp = makeK1(sampleSpace)
    iK0interp = makeiK0(sampleSpace)

    # Lengthscale of H2O is 24000.0 km
    pts = sba.photo_lengthscale('H2O')

    fracStart = 0.3
    fracStop = 2.0
    fracStep = 0.3
    scaleFractions = np.arange(fracStart, fracStop, step=fracStep)
    print(f"Running Haser models on parent-daughter fractions of {scaleFractions} ...")

    # Set up the models with daughter scales as a fraction of parent
    comae = [sba.Haser(1e28/u.s, 0.85 * u.km/u.s, pts, pts*frac) for frac in scaleFractions]
    # Or use no daughters
    # comae = [sba.Haser(1e28/u.s, 0.85 * u.km/u.s, pts, None) for frac in scaleFractions]

    numModels = len(comae)

    # Apertures
    # ~1.35x faster
    # ap = sba.RectangularAperture((100, 10000)*u.km)

    # ~2.6x faster
    # ap = sba.RectangularAperture((100, 1000)*u.km)

    # ~8x faster
    ap = sba.CircularAperture(10000*u.km)
    # ap = sba.RectangularAperture((100, 100)*u.km)

    tUnpatched, resUnpatched = timeModelCalcs(comae, ap)

    # inject interpolators into Haser class
    sba.Haser.newK1 = K1interp
    sba.Haser.newiK0 = iK0interp

    # overwrite the old function
    sba.Haser._K1 = fastK1
    # Don't bother, this doesn't seem to affect speed
    # tOnePatched, resOnepatched = timeModelCalcs(comae, ap)

    # overwrite the old function
    sba.Haser._iK0 = fastiK0
    tBothPatched, resTwopatched = timeModelCalcs(comae, ap)

    print(f"Total for original Haser models: {tUnpatched} s\t\t\t\tPer model: {tUnpatched/numModels} s")
    # print(f"Total for patching K1 in the Haser model: {tOnePatched} s\t\t\tPer model: {tOnePatched/numModels} s")
    print(f"Total for patching K1 and iK0 in the Haser model: {tBothPatched} s\t\tPer model: {tBothPatched/numModels} s")

    percentError = np.abs(100*(1 - resTwopatched/resUnpatched))
    print(f"Interpolated model is off by {percentError} %")

    print(f"Execution speedup factor: {tUnpatched/tBothPatched} times faster")


if __name__ == '__main__':
    sys.exit(main())
