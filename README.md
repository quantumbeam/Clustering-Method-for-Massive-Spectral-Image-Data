## Rational partitioning of spectral feature space for effective clustering of massive spectral image data
Yusei Ito, Yasuo Takeichi, Hideitsu Hino and Kanta Ono

Scientific Reports 14, 22549 (2024). https://doi.org/10.1038/s41598-024-74016-0

## Citation
```text
@article{10.1038/s41598-024-74016-0, 
year = {2024}, 
title = {Rational partitioning of spectral feature space for effective clustering of massive spectral image data}, 
author = {Ito, Yusei and Takeichi, Yasuo and Hino, Hideitsu and Ono, Kanta}, 
journal = {Scientific Reports}, 
doi = {10.1038/s41598-024-74016-0}, 
pages = {22549}, 
number = {1}, 
volume = {14}
}
```

## Steps of our clustering method
1. Preparing Voronoi diagram site points.  
   The implementation is in the '/MakeVoronoiPoints' directory.
   
2. Clustering by performing Voronoi tessellation.  
   The implementation is in the '/Clustering' directory.
   
Details of each step are provided in the following sections.  

<br>

## How to prepare Voronoi diagram site points
Use 'main' function in the '/MakeVoronoiPoints/MakeVoronoiPoints.py' file. The parameters are as follows:  

- `spectra`(ndarray): Initial candidate spectra of Voronoi diagram site points. Each column represents a spectrum.  
- `ref`(float): Reference counts (Noise parameter for calculating Eq.(3)).  
- `mul`(float): Constant multiplication parameter(Noise parameter).  
- `offset`(float): Constant sum parameter (Noise parameter).  
- `generate_num`(int): Number of trials of generating spectrum.  
- `generate_similarity_limit`(float): Similarity threshold when generating spectra (Note that this is percent point).  
- `reduce_similarity_limit`(float): Similarity threshold when reducing candidate spectra by hierarchical clustering (Note that this is percent point).  
- `path_save`(str): Path of save file (.csv).

<br>

## How to perform clustering
Use 'Clustering_by_VoronoiTessellation' function in the '/Clustering/Clustering_VoronoiTessellation.jl' file. This function performs clustering and removing isolation points by using spatial correlation.  
Then, use 'main' function in the '/Clustering/Dilate_each_cluster.py' file for complementing the removing points.   
The parameters are as follows:  
<br>
**Clustering by VoronoiTessellation**
- `path_of_experiment_data`(str): Path of spectral imaging data (.tif).  
- `path_of_voronoi_points`(str): Path of voronoi points data (.csv).
- `kernel`(matrix): K_size x K_size (parameter of integrating spatial correlation) matrix of ones.
- `kernel_delta`(int): Parameter of integrating spatial correlation (kernel_delta = K_size x K_size - P_num)
- `path_of_save_folder`(str): Path of save folder.
<br>

**Dilate each cluster**  
- `path_of_save_folder`(str): Path of save folder.
- `load_folder`(str): Path of load folder.

<br>

## Test code
### Preparing test data
Download test data: [[GoogleDrive](https://drive.google.com/drive/folders/1TEAbpo5oTvh54pXha75B332jw5ADN_6o?usp=drive_link)] in the same folder.

### Preparing Voronoi diagram site points
```bash
python MakeVoronoiPoints.py
```

### Clustering by VoronoiTessellation
```bash
julia Clustering_VoronoiTessellation.jl
```
```bash
python Dilate_each_cluster.py
```
Note that the result is also in the test data folder. Running it will result in a FileExistsError, therefore you must either move or delete the resulting folder.
