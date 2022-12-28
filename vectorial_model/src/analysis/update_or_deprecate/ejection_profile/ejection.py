#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def q_epsilon(outflow, vectorial):

    # adjustment = 0.0
    adjustment = 0.00001

    ratio = (outflow/vectorial)
    ratio_squared = ratio**2

    if outflow > vectorial:
        max_angle = np.arcsin(vectorial/outflow) - adjustment
    else:
        max_angle = np.pi - adjustment

    angles = np.linspace(0, max_angle, endpoint=True, num=100)

    if outflow > vectorial:
        qs = 2 * (1 + ratio_squared * np.cos(2 * angles))/(np.sqrt(1 - ratio_squared * np.sin(angles)**2))
    else:
        qs = 2 * ratio * np.cos(angles) + (1 + ratio_squared * np.cos(2 * angles))/(np.sqrt(1 - ratio_squared * np.sin(angles)**2))

    return angles, qs, max_angle


def numerator(outflow, vectorial):

    adjustment = 0.00001

    ratio = (outflow/vectorial)
    ratio_squared = ratio**2

    max_angle = np.arcsin(vectorial/outflow) - adjustment
    angles = np.linspace(0, max_angle, endpoint=True, num=100)

    qs = 1 + ratio_squared * np.cos(2 * angles)

    return angles, qs, max_angle


def denominator(outflow, vectorial):

    adjustment = 0.00001

    ratio = (outflow/vectorial)
    ratio_squared = ratio**2

    max_angle = np.arcsin(vectorial/outflow) - adjustment
    angles = np.linspace(0, max_angle, endpoint=True, num=100)

    qs = np.sqrt(1 - ratio_squared * np.sin(angles)**2)

    return angles, qs, max_angle


def main():

    # angles, qs, max_angle = numerator(1.0001, 1.0)
    # plt.plot(angles, qs)
    # angles, qs, max_angle = denominator(1.0001, 1.0)
    # plt.plot(angles, qs)
    #
    # plt.show()

    # for outflow in [0.9999, 1.0001, 100.0]:
    #     angles, qs, max_angle = q_epsilon(outflow, 1.0)
    #     plt.plot(angles, qs)
    #     integral = integrate.simpson(qs, angles)
    #     print(max_angle/np.pi, outflow, integral)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for outflow in [0.9999, 1.0001, 1.05]:
        angles, qs, max_angle = q_epsilon(outflow, 1.0)
        ax.plot(angles, qs)
        # plt.plot(angles, qs)
        integral = integrate.simpson(qs, angles)
        print(max_angle/np.pi, outflow, integral)

    # xs = qs * np.sin(angles)
    # ys = qs * np.cos(angles)
    # plt.plot(xs, ys)

    plt.show()


if __name__ == "__main__":
    main()
