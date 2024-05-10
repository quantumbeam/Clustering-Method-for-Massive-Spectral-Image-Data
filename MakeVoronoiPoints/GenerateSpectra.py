import utils
import numpy as np
from tqdm import tqdm


def Generate_and_AddSpectra(spectra, ref, mul, offset, num_max, limit, components_num = 3):
    """
    Generate candidate spectrum by linear combination and calculating similarity between the spectrum and spectra which are already included in the candidate
    Then, if the minimum similarity is smaller than the limit, add it to the candidate.
    :param spectra: Candidate spectra
    :param ref: Reference counts(Noise parameter for calculating Eq.(3))
    :param mul: Constant multiplication parameter(Noise parameter)
    :param offset: Constant sum parameter(Noise parameter)
    :param num_max: Number of trials to generate spectrum
    :param limit: Limit value of similarity
    :return: Candidate spectra
    """
    for i in tqdm(range(num_max)):
        coefficients_tmp = np.random.uniform(0, 1, components_num)

        coefficients = np.zeros(spectra.shape[1])
        coefficients[np.random.randint(0, spectra.shape[1], components_num)] = coefficients_tmp
        coefficients = coefficients / np.sum(coefficients)
        candidate_spectrum = np.dot(spectra, coefficients)

        distance_list = []
        for j in range(spectra.shape[1]):
            pearson_mean_1, pearson_sigma_1 = utils.Calc_distribution(candidate_spectrum, spectra[:, j], ref, mul, offset)

            pearson_mean_2, pearson_sigma_2 = utils.Calc_distribution(spectra[:, j], candidate_spectrum, ref, mul, offset)

            distance_list.append((pearson_mean_1 / pearson_sigma_1 + pearson_mean_2 / pearson_sigma_2) / 2)

        distance_list = np.array(distance_list)

        if np.min(distance_list) > limit:
            spectra = np.hstack([spectra, candidate_spectrum.reshape(-1, 1)])

    return spectra
