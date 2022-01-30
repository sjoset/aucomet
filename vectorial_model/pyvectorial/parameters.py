
import yaml


def read_parameters_from_file(filepath):
    """Read the YAML file with all of the input parameters in it"""
    with open(filepath, 'r') as stream:
        try:
            paramYAMLData = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return paramYAMLData
