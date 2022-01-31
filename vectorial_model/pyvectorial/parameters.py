
import yaml
import copy

import logging as log
import astropy.units as u
import numpy as np


def get_input_yaml(filepath):

    input_yaml = read_parameters_from_file(filepath)
    raw_yaml = copy.deepcopy(input_yaml)

    # apply proper units to the input
    _tag_input_with_units(input_yaml)

    # apply any transformations to the input data for heliocentric distance
    _transform_input_yaml(input_yaml)

    return input_yaml, raw_yaml


def read_parameters_from_file(filepath):
    """Read the YAML file with all of the input parameters in it"""
    with open(filepath, 'r') as stream:
        try:
            param_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return param_yaml


def dump_parameters_to_file(filepath, to_dump):
    """Dump given dictionary to filepath"""
    with open(filepath, 'w') as stream:
        try:
            yaml.safe_dump(to_dump, stream)
        except yaml.YAMLError as exc:
            print(exc)


def _transform_input_yaml(input_yaml):
    """
        Take input dictionary and adjust parameters based on things like heliocentric distance
        and empirical relations

        The particular set of transformations is controlled by transform_method in the input dictionary
    """

    # TODO: add fortran's method of transforming input

    # try getting transform method
    tr_method = input_yaml['position'].get('transform_method')

    # if none specified, nothing to do
    if tr_method is None:
        log.info("No valid tranformation of input data specified -- input unaltered")
        return

    if tr_method == 'cochran_schleicher_93':
        rh = input_yaml['position']['d_heliocentric'].to(u.AU).value
        sqrh = np.sqrt(rh)

        v_old = copy.deepcopy(input_yaml['parent']['v_outflow'])
        tau_d_old = copy.deepcopy(input_yaml['parent']['tau_d'])
        input_yaml['parent']['v_outflow'] *= 0.85/sqrh
        input_yaml['parent']['tau_d'] *= rh**2
        log.debug("Parent outflow: %s --> %s", v_old, input_yaml['parent']['v_outflow'])
        log.debug("Parent tau_d: %s --> %s", tau_d_old, input_yaml['parent']['tau_d'])
        # print(f"\tParent outflow: {v_old} -> {input_yaml['parent']['v_outflow']}")
        # print(f"\tParent tau_d: {tau_d_old} -> {input_yaml['parent']['tau_d']}")


def _tag_input_with_units(input_yaml):
    """
        Takes an input dictionary and applies astropy units to relevant parameters
    """

    input_yaml['production']['base_q'] *= (1/u.s)

    # handle the variation types
    if 'time_variation_type' in input_yaml['production'].keys():
        if input_yaml['production']['time_variation_type'] == "binned":
            bin_required = set(['q_t', 'times_at_productions'])
            has_reqs = bin_required.issubset(input_yaml['production'].keys())
            if not has_reqs:
                print("Required keys for binned production not found in production section!")
                print(f"Need {list(bin_required)}.")
                print("Exiting.")
                exit(1)
            input_yaml['production']['q_t'] *= (1/u.s)
            input_yaml['production']['times_at_productions'] *= u.day

    # parent
    input_yaml['parent']['v_outflow'] *= u.km/u.s
    input_yaml['parent']['tau_T'] *= u.s
    input_yaml['parent']['tau_d'] *= u.s
    input_yaml['parent']['sigma'] *= u.cm**2

    # fragment
    input_yaml['fragment']['v_photo'] *= u.km/u.s
    input_yaml['fragment']['tau_T'] *= u.s

    # positional info
    input_yaml['position']['d_heliocentric'] *= u.AU
