import numpy as np
import pandas as pd
import tifffile
import glob
import os
import cv2
import copy
import shutil


def Dilate_each_cluster(cluster_map, max_cluster_num):
    """
    Complement cluster map after removing isolation points
    :param cluster_map: Cluster map
    :param max_cluster_num: Number of clusters
    :return: Result of cluster map
    """

    kernel = np.ones((3, 3), np.uint8)
    cluster_map_save = copy.deepcopy(cluster_map)
    cluster_map_save_tmp = np.zeros_like(cluster_map_save)

    while (cluster_map_save != cluster_map_save_tmp).any():
        cluster_map_save_tmp = copy.deepcopy(cluster_map_save)
        for j in range(max_cluster_num):
            binary_img = np.zeros_like(cluster_map_save, dtype = np.uint8)
            binary_img[cluster_map_save == j + 1] = 1
            dilate_img = cv2.dilate(binary_img, kernel, iterations = 1)
            cluster_map_save[(cluster_map_save == 0) & (dilate_img == 1)] = j + 1

    return cluster_map_save


def main(path_of_save_folder, load_folder):
    """
    :param path_of_save_folder: Path of save folder
    :param load_folder: path of load folder
    """
    os.mkdir(path_of_save_folder)
    os.mkdir(path_of_save_folder + "/Cluster_map")

    all_count = np.array(pd.read_csv(load_folder + "/Count.csv")["count"])
    material = np.array(pd.read_csv(load_folder + "/Count.csv")["Material"])
    max_cluster_num = all_count.shape[0]

    all_count_new = np.zeros(max_cluster_num)

    All_cluster_map_path = glob.glob(load_folder  +"/Cluster_map/*")

    All_cluster_map_path.sort()

    for i in range(len(All_cluster_map_path)):
        print("Analyze : " + All_cluster_map_path[i])
        cluster_map = tifffile.imread(All_cluster_map_path[i])

        dilite_cluster_map = Dilate_each_cluster(cluster_map, max_cluster_num)

        print("Counting pixels")
        for j in range(max_cluster_num):
            all_count_new[j] += np.sum(dilite_cluster_map == j + 1)

        tifffile.imsave(path_of_save_folder + "/Cluster_map/" + os.path.splitext(os.path.basename(All_cluster_map_path[i]))[0] + ".tif", dilite_cluster_map)

    df_count = pd.DataFrame()
    df_count["Material"] = material
    df_count["count"] = all_count_new

    df_count.to_csv(path_of_save_folder + "/Count.csv")
    shutil.copy(load_folder + "/Spectra.csv", path_of_save_folder + "/Spectra.csv")
    shutil.copy("Dilate_each_cluster.py", path_of_save_folder + "/Execute_src_Dilate_each_cluster.py")


if __name__ == "__main__":
    df = pd.read_csv("param.csv")
    path_of_save_folder_tmp = df["path_of_save_folder"].values[0]

    path_of_save_folder = path_of_save_folder_tmp + "/DilateCluster"
    load_folder = path_of_save_folder_tmp + "/Shape_reduce/Clustering_map_rename"

    main(path_of_save_folder, load_folder)