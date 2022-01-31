
import pickle
import logging as log

from datetime import datetime

from parameters import dump_parameters_to_file


def save_vmodel(input_yaml, coma, base_outfile_name):

    """
    """
    parameters_file = base_outfile_name + ".yaml"
    pickle_file = base_outfile_name + ".vm"

    input_yaml['pyvectorial_info'] = {}
    # TODO: find a way to do this cleanly
    # input_yaml['pyvectorial_info']['version'] = __version__
    input_yaml['pyvectorial_info']['date_run'] = datetime.now().strftime("%m %d %Y")
    input_yaml['pyvectorial_info']['vmodel_pickle'] = pickle_file

    log.info("Writing parameters to %s ...", parameters_file)
    dump_parameters_to_file(parameters_file, input_yaml)

    log.info("Writing model results to %s ...", pickle_file)
    with open(pickle_file, 'wb') as picklejar:
        pickle.dump(coma.vmodel, picklejar)
