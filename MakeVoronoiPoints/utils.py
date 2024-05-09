import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

def Count2Noise(count):
    return np.sqrt(9.3131 * count - np.ones_like(count) * 63.003)


def Calc_count(ref, mut):
    try:
        if ref.shape != mut.shape:
            ref = np.ones_like(mut) * ref
    except:
        ref = np.ones_like(mut) * ref

    return ref * np.exp(-mut)


def Calc_mutnoise(ref, mut):
    """
    Calculating noise(standard deviation) of absorption(mut) by error propagation
    :param ref: Reference counts
    :param mut: Absorption
    :return: Noise(standard deviation) of absorption
    """
    try:
        if ref.shape != mut.shape:
            ref = np.ones_like(mut) * ref
    except:
        ref = np.ones_like(mut) * ref

    count = Calc_count(ref, mut)

    ref_noise = Count2Noise(ref)
    count_noise = Count2Noise(count)

    tmp1 = np.power(ref_noise / ref, 2)
    tmp2 = np.power(count_noise / count, 2)

    return np.sqrt(tmp1 + tmp2)


def Standardization(spectrum):
    spectrum_mean = np.mean(spectrum, axis = 0)
    spectrum = spectrum  - np.broadcast_to(spectrum_mean, (spectrum.shape[0], spectrum_mean.shape[0]))

    spectrum_norm = np.linalg.norm(spectrum, axis = 0)

    spectrum_norm_broadcast = np.broadcast_to(spectrum_norm, (spectrum.shape[0], spectrum.shape[1]))


    return spectrum / spectrum_norm_broadcast


def Calc_distribution(spectrum_1, spectrum_2, ref, mul, offset):
    """
    Calculating the mean and standard deviation of the distribution defined in Eq.(5) for approximating Eq.(1) of the paper.
    :param ref: Reference counts(Noise parameter for calculating Eq.(3))
    :param mul: Constant multiplication parameter(Noise parameter)
    :param offset: Constant sum parameter(Noise parameter)
    :return: Mean and standard deviation of the distribution defined in Eq.(5)
    """
    spectrum = spectrum_1 * mul + offset

    spectrum_subtract_mean = spectrum - np.mean(spectrum)
    spectrum_norm = np.linalg.norm(spectrum_subtract_mean, 2)

    noise_spectrum = Calc_mutnoise(ref, spectrum) / spectrum_norm

    spectrum = Standardization(np.array([spectrum]).T)

    spectrum_1 = Standardization(np.array([spectrum_1]).T)[:, 0]
    spectrum_2 = Standardization(np.array([spectrum_2]).T)[:, 0]

    delta = spectrum_1 - spectrum_2

    spectrum = spectrum[:, 0]

    pearson_mean = np.dot(delta, spectrum)

    delta_power_2 = np.power(delta, 2)
    noise_spectrum_power_2 = np.power(noise_spectrum, 2)

    pearson_sigma = np.sqrt(np.dot(delta_power_2, noise_spectrum_power_2))

    return pearson_mean, pearson_sigma


def Calc_distance_matrix(spectra, ref, mul, offset):
    Distance_matrix = np.ones((spectra.shape[1], spectra.shape[1])) * 100

    for i in tqdm(range(spectra.shape[1])):
        for j in range(spectra.shape[1]):
            if i != j:
                mean, sigma = Calc_distribution(spectra[:, i], spectra[:, j], ref, mul, offset)
                Distance_matrix[i, j] = mean / sigma

    Distance_matrix = (Distance_matrix + Distance_matrix.T) / 2

    return Distance_matrix


def hierarchical_clustering(distance_matrix, distance_threshold):
    clustering = AgglomerativeClustering(n_clusters=None, affinity="precomputed", distance_threshold=distance_threshold, linkage="average")
    clustering.fit(distance_matrix)

    return clustering.labels_