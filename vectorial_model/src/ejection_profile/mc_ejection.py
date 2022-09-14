#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def calc_epsilon(outflow, vectorial, gamma):

    numerator = outflow + vectorial * np.cos(gamma)
    denominator = np.sqrt(vectorial**2 + outflow**2 + 2 * vectorial * outflow * np.cos(gamma))

    if np.isclose(np.abs(numerator), np.abs(denominator)):
        # print(f"Close: {numerator=}, {denominator=}")
        return np.arccos(np.sign(numerator*denominator))

    return np.arccos(numerator/denominator)


def calc_ejection_cone_angle(outflow, vectorial):

    # ejection angles in the dissociation frame, isotropic
    gammas = np.linspace(0, np.pi, endpoint=True, num=1000)
    
    vec_eps = np.vectorize(calc_epsilon)
    epsilons = vec_eps(outflow, vectorial, gammas)
    # epsilons = calc_epsilon(outflow, vectorial, gammas)

    return np.max(epsilons)


def do_ejection_transform_plots(outflows, vectorial):

    xs = np.linspace(0, np.pi, endpoint=True, num=10)
    gammas = np.linspace(0, np.pi, endpoint=True, num=1000)

    vec_eps = np.vectorize(calc_epsilon)

    for outflow in outflows:
        epsilons = vec_eps(outflow, vectorial, gammas)
        plt.scatter(epsilons, gammas, marker='+')

    plt.plot(xs, xs)
    plt.gca().invert_yaxis()

    plt.show()


def do_cone_angle_plots():

    outflows = np.linspace(0.8, 4.0, endpoint=True, num=100)
    # outflows = np.logspace(-2, 1, endpoint=True, num=100)
    
    vec_ecas = np.vectorize(calc_ejection_cone_angle)

    ecas = vec_ecas(outflows, 1.0)

    plt.scatter(outflows, ecas)
    plt.show()


def main():

    do_ejection_transform_plots([np.sqrt(2)/2, 2.0, 5.0, 20.0], 1.0)

    do_cone_angle_plots()

    # for g in np.linspace(0, np.pi, endpoint=True, num=100):
    #     print(f"{g=}", calc_epsilon(np.sqrt(2)/2, 1.0, g))


if __name__ == "__main__":
    main()
