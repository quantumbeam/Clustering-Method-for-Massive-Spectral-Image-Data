import utils
import numpy as np

def Reduce_CandidateSpectra(spectra, ref, mul, offset, distance_threshold):
    """
    Reduce Candidate spectra by performing hierarchical clustering
    :param spectra: Candidate spectra
    :param ref: Reference counts(Noise parameter for calculating Eq.(3))
    :param mul: Constant multiplication parameter(Noise parameter)
    :param offset: Constant sum parameter(Noise parameter)
    :param distance_threshold: Threshold of hierarchical clustering
    :return: Final candidate spectra
    """
    distance_matrix = utils.Calc_distance_matrix(spectra, ref, mul, offset)

    label = utils.hierarchical_clustering(distance_matrix, distance_threshold)

    ret_spectra_list = []

    for i in range(spectra.shape[1]):
        spectra_tmp = spectra[:, label == i]

        if spectra_tmp.shape[1] != 0:
            ret_spectra_list.append(np.mean(spectra_tmp, axis=1))

    return np.array(ret_spectra_list).T