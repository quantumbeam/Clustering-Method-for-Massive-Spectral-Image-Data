import GenerateSpectra
import Reduce_by_HierarchicalClustering
import pandas as pd


def main(spectra, ref, mul, offset, generate_num, generate_similarity_limit, reduce_similarity_limit, path_save):
    """
    Generate Voronoi points
    :param spectra: Initial candidate spectra (Each column represents a spectrum)
    :param ref: Reference counts(Noise parameter for calculating Eq.(3))
    :param mul: Constant multiplication parameter(Noise parameter)
    :param offset: Constant sum parameter(Noise parameter)
    :param generate_num: Number of trials to generate spectrum
    :param generate_similarity_limit: Similarity threshold when generating spectra (Note that this is percent point)
    :param reduce_similarity_limit: Similarity threshold when reducing candidate spectra (Note that this is percent point)
    :param path_save: path of save file(.csv)
    """

    print("Generating candidate spectra")
    candidate_spectra = GenerateSpectra.Generate_and_AddSpectra(spectra, ref, mul, offset,
                                                                  generate_num, generate_similarity_limit)

    print("Reducing candidate spectra")
    spectra_of_VoronoiPoints = Reduce_by_HierarchicalClustering.Reduce_CandidateSpectra(candidate_spectra, ref, mul,
                                                                                         offset, reduce_similarity_limit)

    df = pd.DataFrame(spectra_of_VoronoiPoints)
    df.to_csv(path_save)


if __name__ == "__main__":
    spectra = pd.read_csv("../Data/VoronoiPoints/NIMS_Standard_mixed_20221018.csv").values[:, 1:]

    generate_num = 1
    generate_similarity_limit = 2.326 #Percent point of 0.01
    reduce_similarity_limit = 3.719 #Percent point of 0.0001

    ref = 27000
    mul = 1.56
    offset = 0.36

    path_save = "../Data/VoronoiPoints/NIMS_Standard_mixed_20221018_2p326_addspectrum_reduce3p719.csv"

    main(spectra, ref, mul, offset, generate_num, generate_similarity_limit, reduce_similarity_limit, path_save)



