
import logging as log
import astropy.units as u
import sbpy.activity as sba


def show_fragment_agreement(vmodel):
    print("\nFragment agreement check:")
    print(f"\tTheoretical total number of fragments in coma:\t\t\t {vmodel['num_fragments_theory']:.7e}")
    print(f"\tTotal number of fragments from density grid integration:\t {vmodel['num_fragments_grid']:.7e}")


def show_aperture_checks(coma):

    log.debug("Starting aperture checks ...")
    f_theory = coma.vmodel['num_fragments_theory']

    # use various large apertures to see how much we recover
    ap1 = sba.RectangularAperture((coma.vmodel['max_grid_radius'].value, coma.vmodel['max_grid_radius'].value) * u.m)
    ap2 = sba.CircularAperture((coma.vmodel['max_grid_radius'].value) * u.m)
    ap3 = sba.AnnularAperture([500000, coma.vmodel['max_grid_radius'].value] * u.m)

    rect_result = coma.total_number(ap1)*100/f_theory
    circular_result = coma.total_number(ap2)*100/f_theory
    annular_result = coma.total_number(ap3)*100/f_theory

    print("\nPercent of fragments recovered by integrating column density over")
    print(f"\tLarge rectangular aperture:\t\t\t\t\t\t{rect_result:2.2f}%")
    print(f"\tLarge circular aperture:\t\t\t\t\t\t{circular_result:2.2f}%")
    print(f"\tAnnular aperture, inner radius 500000 km, outer radius of entire grid:\t{annular_result:2.2f}%")
