
import yaml
import copy

import logging as log
import astropy.units as u
import numpy as np


def get_input_yaml(filepath):
    """ Return the transformed, unit-tagged dictionary based on yaml file """
    input_yaml = read_yaml_from_file(filepath)
    raw_yaml = copy.deepcopy(input_yaml)

    # apply proper units to the input
    tag_input_with_units(input_yaml)

    # apply any transformations to the input data for heliocentric distance
    transform_input_yaml(input_yaml)

    return input_yaml, raw_yaml


def read_yaml_from_file(filepath):
    """ Read YAML file from disk and return dictionary unmodified """
    with open(filepath, 'r') as stream:
        try:
            param_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return param_yaml


def dump_parameters_to_file(filepath, to_dump):
    """ Dump given dictionary to filepath """
    with open(filepath, 'w') as stream:
        try:
            yaml.safe_dump(to_dump, stream)
        except yaml.YAMLError as exc:
            print(exc)


def tag_input_with_units(input_yaml):
    """
        Takes an input dictionary and applies astropy units to relevant parameters
    """

    input_yaml['production']['base_q'] *= (1/u.s)

    p_dict = input_yaml['production']
    # handle the time-variation types
    if 'time_variation_type' in input_yaml['production'].keys():
        t_var_type = p_dict['time_variation_type']
        if t_var_type == "binned":
            bin_required = set(['q_t', 'times_at_productions'])
            has_reqs = bin_required.issubset(p_dict.keys())
            if not has_reqs:
                print("Required keys for binned production not found in production section!")
                print(f"Need {list(bin_required)}.")
                print("Exiting.")
                exit(1)
            p_dict['q_t'] *= (1/u.s)
            p_dict['times_at_productions'] *= u.day
        elif t_var_type == "gaussian":
            # TODO: add required keys for all production types
            log.debug("Found gaussian production ...")
            gauss_reqs = set(['amplitude', 't_max', 'std_dev'])
            has_reqs = gauss_reqs.issubset(p_dict.keys())
            if not has_reqs:
                print("Required keys for gaussian production not found in production section!")
                print(f"Need {list(gauss_reqs)}.")
                print("Exiting.")
                exit(1)
            p_dict['amplitude'] *= (1/u.s)
            p_dict['t_max'] *= u.hour
            p_dict['std_dev'] *= u.hour
            log.debug("Amplitude: %s, t_max: %s, std_dev: %s", p_dict['amplitude'], p_dict['t_max'], p_dict['std_dev'])
        elif t_var_type == "sine wave":
            log.debug("Found sine wave production ...")
            sine_reqs = set(['amplitude', 'period', 'delta'])
            has_reqs = sine_reqs.issubset(p_dict.keys())
            if not has_reqs:
                print("Required keys for sine wave production not found in production section!")
                print(f"Need {list(sine_reqs)}.")
                print("Exiting.")
                exit(1)
            p_dict['amplitude'] *= (1/u.s)
            p_dict['period'] *= u.hour
            p_dict['delta'] *= u.hour
            log.debug("Amplitude: %s, period: %s, delta: %s", p_dict['amplitude'], p_dict['period'], p_dict['delta'])
        elif t_var_type == "square pulse":
            log.debug("Found square pulse production ...")
            sq_reqs = set(['amplitude', 't_start', 'duration'])
            has_reqs = sq_reqs.issubset(p_dict.keys())
            if not has_reqs:
                print("Required keys for square pulse production not found in production section!")
                print(f"Need {list(sq_reqs)}.")
                print("Exiting.")
                exit(1)
            p_dict['amplitude'] *= (1/u.s)
            p_dict['t_start'] *= u.hour
            p_dict['duration'] *= u.hour
            log.debug("Amplitude: %s, t_start: %s, duration: %s", p_dict['amplitude'], p_dict['t_start'], p_dict['duration'])

    # parent
    input_yaml['parent']['v_outflow'] *= u.km/u.s
    input_yaml['parent']['tau_d'] *= u.s
    input_yaml['parent']['tau_T'] = input_yaml['parent']['tau_d'] * input_yaml['parent']['T_to_d_ratio']
    input_yaml['parent']['sigma'] *= u.cm**2

    # fragment
    input_yaml['fragment']['v_photo'] *= u.km/u.s
    input_yaml['fragment']['tau_T'] *= u.s

    # positional info
    input_yaml['position']['d_heliocentric'] *= u.AU


def strip_input_of_units(input_yaml):
    """
        Takes an input dictionary with astropy units and strips units, opposite of tag_input_with_units(),
        but returns a new dictionary instead of modifying in place
    """

    # TODO: remove the float() calls once the bug in pyyaml writing decimal numbers is fixed
    # https://github.com/yaml/pyyaml/issues/255

    new_yaml = copy.deepcopy(input_yaml)

    new_yaml['production']['base_q'] = float(new_yaml['production']['base_q'].to(1/u.s).value)

    p_new = new_yaml['production']
    # handle the time-variation types
    if 'time_variation_type' in p_new.keys():
        t_var_type = p_new['time_variation_type']
        if t_var_type == "binned":
            p_new['q_t'] = list(map(float, list(p_new['q_t'].to(1/u.s).value)))
            p_new['times_at_productions'] = list(map(float, list(p_new['times_at_productions'].to(u.day).value)))
        elif t_var_type == "gaussian":
            p_new['amplitude'] = float(p_new['amplitude'].to(1/u.s).value)
            p_new['t_max'] = float(p_new['t_max'].to(u.hour).value)
            p_new['std_dev'] = float(p_new['std_dev'].to(u.hour).value)
        elif t_var_type == "sine wave":
            p_new['amplitude'] = float(p_new['amplitude'].to(1/u.s).value)
            p_new['period'] = float(p_new['period'].to(u.hour).value)
            p_new['delta'] = float(p_new['delta'].to(u.hour).value)
        elif t_var_type == "square pulse":
            p_new['amplitude'] = float(p_new['amplitude'].to(1/u.s).value)
            p_new['t_start'] = float(p_new['t_start'].to(u.hour).value)
            p_new['duration'] = float(p_new['duration'].to(u.hour).value)

    new_yaml['parent']['v_outflow'] = float(input_yaml['parent']['v_outflow'].to(u.km/u.s).value)
    new_yaml['parent']['tau_T'] = float(input_yaml['parent']['tau_T'].to(u.s).value)
    new_yaml['parent']['tau_d'] = float(input_yaml['parent']['tau_d'].to(u.s).value)
    new_yaml['parent']['sigma'] = float(input_yaml['parent']['sigma'].to(u.cm**2).value)

    # fragment
    new_yaml['fragment']['v_photo'] = float(input_yaml['fragment']['v_photo'].to(u.km/u.s).value)
    new_yaml['fragment']['tau_T'] = float(input_yaml['fragment']['tau_T'].to(u.s).value)

    # positional info
    new_yaml['position']['d_heliocentric'] = float(input_yaml['position']['d_heliocentric'].to(u.AU).value)

    return new_yaml


def transform_input_yaml(input_yaml):
    """
        Returns a dictionary with any adjustments to the relevant data applied
        Assumes input dictionary has astropy units attached

        The particular set of transformations is controlled by transform_method in the input dictionary
    """

    # try getting transform method
    tr_method = input_yaml['position'].get('transform_method')

    # if none specified, nothing to do
    if tr_method is None:
        log.info("No valid tranformation of input data specified -- input unaltered")
        return

    log.info("Transforming input parameters using method %s", tr_method)

    if tr_method == 'cochran_schleicher_93':
        # This overwrites v_outflow with its own value!

        log.info("Reminder: cochran_schleicher_93 overwrites v_outflow of parent")
        rh = input_yaml['position']['d_heliocentric'].to(u.AU).value
        sqrh = np.sqrt(rh)

        v_old = copy.deepcopy(input_yaml['parent']['v_outflow'])
        tau_d_old = copy.deepcopy(input_yaml['parent']['tau_d'])
        input_yaml['parent']['v_outflow'] = (0.85/sqrh) * u.km/u.s
        input_yaml['parent']['tau_d'] *= rh**2
        log.debug("Parent outflow: %s --> %s", v_old, input_yaml['parent']['v_outflow'])
        log.debug("Parent tau_d: %s --> %s", tau_d_old, input_yaml['parent']['tau_d'])
    elif tr_method == 'festou_fortran':
        rh = input_yaml['position']['d_heliocentric'].to(u.AU).value
        ptau_d_old = copy.deepcopy(input_yaml['parent']['tau_d'])
        ptau_T_old = copy.deepcopy(input_yaml['parent']['tau_T'])
        ftau_T_old = copy.deepcopy(input_yaml['fragment']['tau_T'])
        input_yaml['parent']['tau_d'] *= rh**2
        input_yaml['parent']['tau_T'] *= rh**2
        input_yaml['fragment']['tau_T'] *= rh**2
        log.debug("\tParent tau_d: %s --> %s", ptau_d_old, input_yaml['parent']['tau_d'])
        log.debug("\tParent tau_T: %s --> %s", ptau_T_old, input_yaml['parent']['tau_T'])
        log.debug("\tFragment tau_T: %s --> %s", ftau_T_old, input_yaml['fragment']['tau_T'])
