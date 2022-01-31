
import astropy.units as u


def print_radial_density(coma):
    print("\n\nRadius (km) vs Fragment density (1/cm3)\n---------------------------------------")
    for i in range(0, coma.radial_points):
        print(f"{coma.vmodel['radial_grid'][i].to(u.km):10.1f} : {coma.vmodel['radial_density'][i].to(1/(u.cm**3)):8.4f}")


def print_column_density(coma):
    print("\nRadius (km) vs Column density (1/cm2)\n-------------------------------------")
    cds = list(zip(coma.vmodel['column_density_grid'], coma.vmodel['column_densities']))
    for pair in cds:
        print(f'{pair[0].to(u.km):7.0f} :\t{pair[1].to(1/(u.cm*u.cm)):5.3e}')


def print_binned_times(production_dict):
    print("")
    print("Binned time production summary")
    print("------------------------------")
    for q, t in zip(production_dict['q_t'], production_dict['times_at_productions']):
        t_u = t.to(u.day).value
        print(f"Q: {q}\t\tt_start (days ago): {t_u}")
    print("")
