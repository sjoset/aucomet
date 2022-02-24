
import logging as log
# import astropy.units as u
import sbpy.activity as sba
from sbpy.data import Phys

from .utils import print_binned_times
from .timedependentproduction import TimeDependentProduction


def run_vmodel(input_yaml):
    """
        Given input dictionary with astropy units, run vectorial model and return the coma object

        Constructs parent & fragment Phys objects, along with any time dependence function
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
            prod_var = TimeDependentProduction(t_var_type)
            q_t = prod_var.create_production(**input_yaml['production'])
        elif t_var_type == "sine wave":
            prod_var = TimeDependentProduction(t_var_type)
            q_t = prod_var.create_production(**input_yaml['production'])
        elif t_var_type == "square pulse":
            prod_var = TimeDependentProduction(t_var_type)
            q_t = prod_var.create_production(**input_yaml['production'])

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
