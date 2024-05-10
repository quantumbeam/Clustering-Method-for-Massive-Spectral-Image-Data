using TiffImages
using Images
using PyCall
using CSV
using DataFrames
using LinearAlgebra
using Statistics
using Dates
using Base.Threads
using DSP
@pyimport os
@pyimport glob
@pyimport shutil

mutable struct VoronoiPoints_and_label
    VoronoiPoints::Matrix{Float64}
    label::Vector{String}
end


mutable struct experiment_Data
    Data::Matrix{Float64}
end


function Data_load(path_of_experiment_data)::Tuple{experiment_Data, Int64, Int64}
    data = convert(Array{Float64}, TiffImages.load(path_of_experiment_data))[:, :, :]

    vertical_size = size(data)[1]
    horizontal_size = size(data)[2]

    number_of_data = vertical_size * horizontal_size
    number_of_observed_energy_point = size(data)[3]


    return experiment_Data(transpose(reshape(data, number_of_data, number_of_observed_energy_point))), vertical_size, horizontal_size
end


function VoronoiPoints_load(path_of_voronoi_points, label)::VoronoiPoints_and_label
    if size(label)[1] == 0
        voronoi = CSV.read(path_of_voronoi_points, drop=[1], header=true, DataFrame)
        return VoronoiPoints_and_label(Matrix{Float64}(voronoi), names(voronoi))

    else
        voronoi = CSV.read(path_of_voronoi_points, drop=[1], header=true, DataFrame)[:, label]
        return VoronoiPoints_and_label(Matrix{Float64}(voronoi), label)
    end
end


function PCA(data::Matrix{Float64})::Vector{Float64}
    A = data*transpose(data)
    U, s, V = svd(A)

    return U[:, 1]/U[end, 1]
end


function Standardization(spectrum::Vector{Float64}, mean_spectrum::Float64)::Vector{Float64}
    mean_spectrum = mean(spectrum)
    spectrum .-= mean_spectrum
    spectrum .+= 0.0001
    normalize!(spectrum)

    return spectrum
end



function Clustering_near_spectrum(experiment_data::experiment_Data, voronoi_points_and_label::VoronoiPoints_and_label)
    """
    Clustering by Voronoi tessellation. It is implemented by classification into nearest-neighbour Voronoi point.
    :param experiment_data: Spectral imaging data
    :param voronoi_points_and_label: Voronoi points and corresponding labels
    :return: Clustering map and number of points of each cluster
    """
    Cluster_num = zeros(Float64, size(experiment_data.Data)[2])
    count = zeros(Int64, size(voronoi_points_and_label.VoronoiPoints)[2])

    experiment_data_copy = copy(experiment_data.Data)
    voronoi_points_and_label_copy = copy(voronoi_points_and_label.VoronoiPoints)

    mean = 0.0
    for k::Int16 = 1:size(voronoi_points_and_label_copy)[2]
        voronoi_points_and_label_copy[:, k] .= Standardization(voronoi_points_and_label_copy[:, k], mean)
    end

    for k::Int64 = 1:size(experiment_data_copy)[2]
        if all(experiment_data_copy[:, k] .== 0)
            experiment_data_copy[:, k] .= 0
        else
            experiment_data_copy[:, k] .= Standardization(experiment_data_copy[:, k], mean)
        end
    end

    cos_dis_matrix = transpose(voronoi_points_and_label_copy)*experiment_data_copy

    for k::Int64 = 1:size(experiment_data_copy)[2]
        if all(experiment_data_copy[:, k] .== 0)
            Cluster_num[k] = 0
        else
            max_index = argmax(cos_dis_matrix[:, k])
            Cluster_num[k] = max_index
            count[max_index] += 1
        end
    end

    return Cluster_num, count
end


function Convolution_kernel(img::Matrix{Float64}, kernel::Matrix{Float64})::Matrix{Float64}
    """
    Convolution for performing noise reduction by using spatial correlation.
    """
    kernel_height = size(kernel)[1]
    kernel_width = size(kernel)[2]

    img_height = size(img)[1]
    img_width = size(img)[2]

    result = conv(img, kernel)[(kernel_height÷2)*2 + 1:(kernel_height÷2)*2 + 1 + img_height - (kernel_height÷2)*2 - 1, (kernel_width÷2)*2 + 1:(kernel_width÷2)*2 + 1 + img_width - (kernel_height÷2)*2 - 1]

    return result
end


function Reduce_cluster(img::Matrix{Float64}, max_cluster_num::Int64, kernel::Matrix{Float64}, kernel_delta::Int64)::Matrix{Float64}
    """
    Noise reduction by using spatial correlation.
    :param img: Flattened spectral imaging data
    :param max_cluster_num: Number of clusters
    :param kernel: K_size x K_size (parameter of integrating spatial correlation) matrix of ones
    :param kernel_delta: kernel_delta = K_size x K_size - P_num
    :return: Result of reducing cluster by using spatial correlation
    """
    kernel_size = size(kernel)[1]
    for k::Int64 = 1:max_cluster_num

        img_copy = copy(img)
        img_copy[img_copy .!= k] .= 0.0
        img_copy[img_copy .== k] .= 1.0

        img_copy[(kernel_size÷2) + 1:end - (kernel_size÷2), (kernel_size÷2) + 1:end - (kernel_size÷2)] = Convolution_kernel(img_copy, kernel)
        img_copy[begin:begin + (kernel_size÷2) - 1, :] .= 0.0
        img_copy[end - (kernel_size÷2) + 1:end, :] .= 0.0
        img_copy[:, begin:begin + (kernel_size÷2) - 1] .= 0.0
        img_copy[:, end - (kernel_size÷2) + 1:end] .= 0.0

        img_copy[0.1 .< img_copy .< sum(kernel) - kernel_delta] .= 3.0
        img_copy[img .!= k .&& img_copy .< sum(kernel) - kernel_delta] .= 0.0
        img_copy[float64(sum(kernel) - kernel_delta) .<= img_copy .<= float64(sum(kernel) + 0.1)] .= 2.0

        while sum(img_copy .== 2.0) != 0
            check_point = (img_copy .== 2.0)

            for i::Int64 = 1:size(img_copy)[1] - (kernel_size÷2)*2
                for j::Int64 = 1:size(img_copy)[2] - (kernel_size÷2)*2

                    if img_copy[(kernel_size÷2) + i, (kernel_size÷2) + j] == 2.0
                        if img_copy[(kernel_size÷2) + i + 1, (kernel_size÷2) + j] == 3.0
                            img_copy[(kernel_size÷2) + i + 1, (kernel_size÷2) + j] = 2.0
                        end
                        if img_copy[(kernel_size÷2) + i - 1, (kernel_size÷2) + j] == 3.0
                            img_copy[(kernel_size÷2) + i - 1, (kernel_size÷2) + j] = 2.0
                        end
                        if img_copy[(kernel_size÷2) + i, (kernel_size÷2) + j + 1] == 3.0
                            img_copy[(kernel_size÷2) + i, (kernel_size÷2) + j + 1] = 2.0
                        end
                        if img_copy[(kernel_size÷2) + i, (kernel_size÷2) + j - 1] == 3.0
                            img_copy[(kernel_size÷2) + i, (kernel_size÷2) + j - 1] = 2.0
                        end
                        if img_copy[(kernel_size÷2) + i + 1, (kernel_size÷2) + j + 1] == 3.0
                            img_copy[(kernel_size÷2) + i + 1, (kernel_size÷2) + j + 1] = 2.0
                        end
                        if img_copy[(kernel_size÷2) + i + 1, (kernel_size÷2) + j - 1] == 3.0
                            img_copy[(kernel_size÷2) + i + 1, (kernel_size÷2) + j - 1] = 2.0
                        end
                        if img_copy[(kernel_size÷2) + i - 1, (kernel_size÷2) + j + 1] == 3.0
                            img_copy[(kernel_size÷2) + i - 1, (kernel_size÷2) + j + 1] = 2.0
                        end
                        if img_copy[(kernel_size÷2) + i - 1, (kernel_size÷2) + j - 1] == 3.0
                            img_copy[(kernel_size÷2) + i - 1, (kernel_size÷2) + j - 1] = 2.0
                        end
                    end

                end
            end
            img_copy[check_point] .= 1.0
        end

        img_copy[img_copy .== 3.0] .= 0.0

        img[img .== k] .= 0.0
        img[img_copy .== 1.0] .= float(k)
    end

    return img
end


function Extract_spectra(experiment_data::experiment_Data, Cluster_num::Vector{Float64}, max_cluster_num)::Matrix{Float64}
    """
    Extracting the cluster spectrum by Principal Component Analysis (PCA).
    :param experiment_data: Spectral imaging data
    :param Cluster_num: Clustering result map
    :param max_cluster_num: Number of clusters
    :return: Extracting spectra
    """
    Extract_spectra = zeros(Float64, size(experiment_data.Data[:, 1])[1], max_cluster_num)

    for k::Int16 = 1:max_cluster_num
        pca_matrix = experiment_data.Data[:, Cluster_num .== k]
        if size(pca_matrix)[2] != 0
            Extract_spectra[:, k] = PCA(pca_matrix)
        end
    end

    return Extract_spectra
end


function Rename_cluster(path_of_save_folder)
    """
    Rename cluster since the cluster number is set at intervals.
    """
    All_path = glob.glob(path_of_save_folder * "/Spectra/*")
    sort!(All_path)

    df = CSV.read(All_path[1], header=true, DataFrame)
    Material = names(df)

    Spectra = zeros(Float64, size(Material)[1], size(Matrix(df))[1], size(All_path)[1])
    for k::Int16 = 1:size(All_path)[1]
        Spectra[:, :, k] = transpose(Matrix(CSV.read(All_path[k], header=true, DataFrame)))
    end

    All_path = glob.glob(path_of_save_folder * "/Count/*")
    sort!(All_path)
    count = zeros(Int64, size(Material)[1], size(All_path)[1])
    for k::Int16 = 1:size(All_path)[1]
        count[:, k] = CSV.read(All_path[k], header=true, DataFrame)[:, "count"]
    end

    All_count = zeros(Int64, size(Material)[1])
    for k::Int16 = 1:size(Material)[1]
        All_count[k] = norm(count[k, :], 1)
    end

    count_rate = zeros(Float64, size(Material)[1], size(All_path)[1])
    for k::Int16 = 1:size(Material)[1]
        count_rate[k, :] = count[k, :]/(All_count[k] + 0.0001)
    end

    Integrate_spectra = zeros(Float64, size(Matrix(df))[1], size(Material)[1])
    for k::Int16 = 1:size(Material)[1]
        Integrate_spectra[:, k] = Spectra[k, :, :] * count_rate[k, :]
    end

    os.mkdir(path_of_save_folder * "/Clustering_map_rename")
    os.mkdir(path_of_save_folder * "/Clustering_map_rename/Cluster_map")

    println("Saving Spectra : " * path_of_save_folder * "/Clustering_map_rename/Spectra.csv")
    df = DataFrame(Integrate_spectra[:, All_count .!= 0], Material[All_count .!= 0])
    df |> CSV.write(path_of_save_folder * "/Clustering_map_rename/Spectra.csv")

    All_path = glob.glob(path_of_save_folder * "/Cluster_map/*")

    for k::Int64 = 1:size(All_path)[1]
        println("Analyze : " * All_path[k])
        img = TiffImages.load(All_path[k])
        vertical = size(img)[1]
        horizontal = size(img)[2]
        img = reshape(img, vertical*horizontal)

        i = 1.0
        for l::Int16 = 1:size(All_count)[1]
            if All_count[l] != 0
                img[img .== l] .= i
                i += 1
            end
        end

        img = TiffImages.DenseTaggedImage(reinterpret(Gray{Float64}, reshape(img, vertical, horizontal)))
        TiffImages.save(path_of_save_folder * "/Clustering_map_rename/Cluster_map/" * os.path.splitext(os.path.basename(All_path[k]))[1] * ".tif", img)
    end

    println("Saving count : " * path_of_save_folder * "/Clustering_map_rename/Count.csv")
    df_count = DataFrame()
    df_count.Material = Material[All_count .!= 0]
    df_count.count = All_count[All_count .!= 0]
    df_count |> CSV.write(path_of_save_folder * "/Clustering_map_rename/Count.csv")
end


function Load_and_clustering_one_shot(path_of_experiment_data, path_of_voronoi_points, kernel, kernel_delta, path_of_save_folder)
    """
    Performing data loading, clustering by voronoi tessellation, and reduce cluster by using spatial correlation.
    :param path_of_experiment_data: Path of spectral imaging data (.tif)
    :param path_of_voronoi_points: Path of voronoi points data (.csv)
    :param kernel: K_size x K_size (parameter of integrating spatial correlation) matrix of ones
    :param kernel_delta: kernel_delta = K_size x K_size - P_num
    :param path_of_save_folder: Path of saving folder
    """
    voronoi_and_label = VoronoiPoints_load(path_of_voronoi_points, [])
    max_cluster_num = size(voronoi_and_label.VoronoiPoints)[2]

    println("Loading experiment data")
    @time experiment_data, vertical, horizontal = Data_load(path_of_experiment_data)

    println("Clustering by voronoi tessellation")
    @time Cluster_num, count = Clustering_near_spectrum(experiment_data, voronoi_and_label)   #最近接と二番目近接の読み込み

    Cluster_num = reshape(Cluster_num, vertical, horizontal)

    println("Shape reduce")
    @time Cluster_num_reduce = Reduce_cluster(copy(Cluster_num), max_cluster_num, kernel, kernel_delta)     #最近接の形による削減
    count_reduce = zeros(Int64, max_cluster_num)
    for k::Int64 = 1:max_cluster_num
        count_reduce[k] = sum(Cluster_num_reduce .== k)
    end

    println("Extract Voronoi spectra")
    @time first_spectra = Extract_spectra(experiment_data, reshape(Cluster_num, vertical * horizontal), max_cluster_num)   #最近接スペクトルの取り出し

    println("Extract Voronoi spectra after shape reduce")
    @time first_spectra_after_reduce = Extract_spectra(experiment_data, reshape(Cluster_num_reduce, vertical * horizontal), max_cluster_num)   #形で減らした後のスペクトルの取り出し

    println("Saving Voronoi map : " * path_of_save_folder * "/Voronoi/Cluster_map/" * os.path.splitext(os.path.basename(path_of_experiment_data))[1] * "_Cluster.tif")
    TiffImages.save(path_of_save_folder * "/Voronoi/Cluster_map/" * os.path.splitext(os.path.basename(path_of_experiment_data))[1] * "_Cluster.tif", TiffImages.DenseTaggedImage(reinterpret(Gray{Float64}, Cluster_num)))


    println("Saving Voronoi count : " * path_of_save_folder * "/Voronoi/Count/" * os.path.splitext(os.path.basename(path_of_experiment_data))[1] * "_Count.csv")
    df = DataFrame()
    df.Material = voronoi_and_label.label
    df.count = count
    df |> CSV.write(path_of_save_folder * "/Voronoi/Count/" * os.path.splitext(os.path.basename(path_of_experiment_data))[1] * "_count.csv")

    println("Saving Voronoi spectra : " * path_of_save_folder * "/Voronoi/Spectra/" * os.path.splitext(os.path.basename(path_of_experiment_data))[1] * "_Spectra.csv")
    df = DataFrame(first_spectra, voronoi_and_label.label)
    df |> CSV.write(path_of_save_folder * "/Voronoi/Spectra/" * os.path.splitext(os.path.basename(path_of_experiment_data))[1] * "_Spectra.csv")


    println("Saving Shape reduce map : " * path_of_save_folder * "/Shape_reduce/Cluster_map/" * os.path.splitext(os.path.basename(path_of_experiment_data))[1] * "_Cluster.tif")
    TiffImages.save(path_of_save_folder * "/Shape_reduce/Cluster_map/" * os.path.splitext(os.path.basename(path_of_experiment_data))[1] * "_Cluster.tif", TiffImages.DenseTaggedImage(reinterpret(Gray{Float64}, Cluster_num_reduce)))

    println("Saving Shape reduce count : " * path_of_save_folder * "/Shape_reduce/Count/" * os.path.splitext(os.path.basename(path_of_experiment_data))[1] * "_Count.csv")
    df = DataFrame()
    df.Material = voronoi_and_label.label
    df.count = count_reduce
    df |> CSV.write(path_of_save_folder * "/Shape_reduce/Count/" * os.path.splitext(os.path.basename(path_of_experiment_data))[1] * "_count.csv")

    println("Saving Shape reduce spectra : " * path_of_save_folder * "/Shape_reduce/Spectra/" * os.path.splitext(os.path.basename(path_of_experiment_data))[1] * "_Spectra.csv")
    df = DataFrame(first_spectra_after_reduce, voronoi_and_label.label)
    df |> CSV.write(path_of_save_folder * "/Shape_reduce/Spectra/" * os.path.splitext(os.path.basename(path_of_experiment_data))[1] * "_Spectra.csv")
end


function Clustering_by_VoronoiTessellation(path_of_experiment_data, path_of_voronoi_points, kernel, kernel_delta, path_of_save_folder)
    """
    Performing data loading, clustering by voronoi tessellation, reduce cluster by using spatial correlation, and rename cluster.
    :param path_of_experiment_data: Path of spectral imaging data (.tif)
    :param path_of_voronoi_points: Path of voronoi points data (.csv)
    :param kernel: K_size x K_size (parameter of integrating spatial correlation) matrix of ones
    :param kernel_delta: kernel_delta = K_size x K_size - P_num
    :param path_of_save_folder: Path of saving folder
    """

    os.mkdir(path_of_save_folder)
    os.mkdir(path_of_save_folder * "/Voronoi")
    os.mkdir(path_of_save_folder * "/Voronoi/Cluster_map")
    os.mkdir(path_of_save_folder * "/Voronoi/Count")
    os.mkdir(path_of_save_folder * "/Voronoi/Spectra")
    os.mkdir(path_of_save_folder * "/Shape_reduce")
    os.mkdir(path_of_save_folder * "/Shape_reduce/Cluster_map")
    os.mkdir(path_of_save_folder * "/Shape_reduce/Count")
    os.mkdir(path_of_save_folder * "/Shape_reduce/Spectra")

    start = now()
    println("Analyze : " * path_of_experiment_data)
    Load_and_clustering_one_shot(path_of_experiment_data, path_of_voronoi_points, kernel, kernel_delta, path_of_save_folder)

    Rename_cluster(path_of_save_folder * "/Voronoi")
    Rename_cluster(path_of_save_folder * "/Shape_reduce")

    finish = now()

    shutil.copy("Clustering_VoronoiTessellation.jl", path_of_save_folder * "/Execute_src_Clustering_VoronoiTessellation.jl")
end


#####################################
df = CSV.read("param.csv", DataFrame)

path_of_experiment_data = string(df[1, "path_of_experiment_data"])
path_of_voronoi_points = string(df[1, "path_of_voronoi_points"])
kernel_size = Int(df[1, "kernel_size"])
kernel = ones(Float64, kernel_size, kernel_size)
kernel_delta = Int(df[1, "kernel_delta"])
path_of_save_folder = string(df[1, "path_of_save_folder"])

Clustering_by_VoronoiTessellation(path_of_experiment_data, path_of_voronoi_points, kernel, kernel_delta, path_of_save_folder)
