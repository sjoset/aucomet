
import logging as log
import astropy.units as u
import sbpy.activity as sba
from sbpy.data import Phys

from utils import print_binned_times
from timedependentproduction import TimeDependentProduction


def run_vmodel(input_yaml):
    """
        Given input dictionary read from yaml file, run vectorial model and return the coma object
    """

    log.info("Calculating fragment density using vectorial model ...")

    # build parent and fragment inputs
    parent = Phys.from_dict(input_yaml['parent'])
    fragment = Phys.from_dict(input_yaml['fragment'])

    coma = None
    q_t = None

    # set up q_t here if we have variable production
    if 'time_variation_type' in input_yaml['production'].keys():
        t_var_type = input_yaml['production']['time_variation_type']

        # handle each type of supported time dependence
        if t_var_type == "binned":
            log.debug("Found binned production ...")
            if input_yaml['printing']['print_binned_times']:
                print_binned_times(input_yaml['production'])
            # call the older-style binned production constructor
            coma = sba.VectorialModel.binned_production(qs=input_yaml['production']['q_t'],
                                                        parent=parent, fragment=fragment,
                                                        ts=input_yaml['production']['times_at_productions'],
                                                        radial_points=input_yaml['grid']['radial_points'],
                                                        angular_points=input_yaml['grid']['angular_points'],
                                                        radial_substeps=input_yaml['grid']['radial_substeps'],
                                                        print_progress=input_yaml['printing']['print_progress']
                                                        )
        elif t_var_type == "gaussian":
            log.debug("Found gaussian production ...")
            amplitude = input_yaml['production']['amplitude']
            t_max = input_yaml['production']['t_max']
            std_dev = input_yaml['production']['std_dev']
            log.debug("Amplitude: %s molecules/s, t_max: %s hrs, std_dev: %s hrs", amplitude, t_max, std_dev)

            prod_var = TimeDependentProduction(t_var_type)
            q_t = prod_var.create_production(amplitude=amplitude*(1/u.s),
                                             t_max=t_max*u.hour,
                                             std_dev=std_dev*u.hour)
        elif t_var_type == "sine wave":
            log.debug("Found sine wave production ...")
            amplitude = input_yaml['production']['amplitude']
            period = input_yaml['production']['period']
            delta = input_yaml['production']['delta']
            log.debug("Amplitude: %s molecules/s, period: %s hrs, delta: %s hrs", amplitude, period, delta)

            prod_var = TimeDependentProduction(t_var_type)
            q_t = prod_var.create_production(amplitude=amplitude*(1/u.s),
                                             period=period*u.hour,
                                             delta=delta*u.hour)
        elif t_var_type == "square pulse":
            log.debug("Found square pulse production ...")
            amplitude = input_yaml['production']['amplitude']
            t_start = input_yaml['production']['t_start']
            duration = input_yaml['production']['duration']
            log.debug("Amplitude: %s molecules/s, t_start: %s hrs, duration: %s hrs", amplitude, t_start, duration)

            prod_var = TimeDependentProduction(t_var_type)
            q_t = prod_var.create_production(amplitude=amplitude*(1/u.s),
                                             t_start=t_start*u.hour,
                                             duration=duration*u.hour)

    # if the binned constructor above hasn't been called, we have work to do
    if coma is None:
        # did we come up with a valid time dependence?
        if q_t is None:
            # print(f"No valid time dependence specified, assuming steady "
            #       f"production of {input_yaml['production']['base_q']}.")
            log.info("No valid time dependence specified, assuming steady production of %s",
                     input_yaml['production']['base_q'])

        coma = sba.VectorialModel(base_q=input_yaml['production']['base_q'],
                                  q_t=q_t,
                                  parent=parent, fragment=fragment,
                                  radial_points=input_yaml['grid']['radial_points'],
                                  angular_points=input_yaml['grid']['angular_points'],
                                  radial_substeps=input_yaml['grid']['radial_substeps'],
                                  print_progress=input_yaml['printing']['print_progress'])

    return coma
