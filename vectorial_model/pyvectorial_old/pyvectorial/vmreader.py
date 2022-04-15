
import pickle


def read_vmodel(vmodelfile):

    with open(vmodelfile, 'rb') as vmp:
        return pickle.load(vmp)
