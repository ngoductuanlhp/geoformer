# GeoFormer
## Geodesic-Former: a Geodesic-Guided Few-shot 3D Point Cloud Instance Segmenter

Code for the paper **Geodesic-Former: a Geodesic-Guided Few-shot 3D Point Cloud Instance Segmenter**.

## Introduction
This paper introduces a new problem in 3D point cloud: fewshot instance segmentation. Given a few annotated point clouds characterizing a target class, our goal is to segment all instances of this target class in a query point cloud. This problem has a wide range of practical applications, especially in the areas where point-wise instance label segmentation annotation is prohibitively expensive to collect. To address this problem, we present Geodesic-Former – the first geodesic-guided transformer for 3D point cloud instance segmentation. The key idea is to leverage the geodesic distance to tackle the density imbalance of LiDAR 3D point clouds. The LiDAR 3D point clouds are dense near object surface and sparse or empty elsewhere making the Euclidean distance less effective to distinguish different objects. The geodesic distance, on the other hand, is more suitable since it encodes the object’s geometry which can be used as a guiding signal for the attention mechanism in a transformer decoder to generate kernels representing distinguishing features of instances. These kernels are then used in a dynamic convolution to obtain the final instance masks. To evaluate Geodesic-Former on the new task, we propose new datasets adapted from the two common 3D point cloud instance segmentation datasets: ScannetV2 and S3DIS. Geodesic-Former consistently outperforms very strong baselines adapted from state-of-the-art 3D point cloud instance segmentation approaches with significant margins.

## Installation

### Requirements
* Python 3.7.0
* Pytorch 1.7.1
* CUDA 11.2

### Virtual Environment
```
conda create -n df python==3.7
conda activate df
```

### Install `GeoFormer`

(1) Clone the code.

(2) Install the dependent libraries.
```
conda install pytorch==1.7.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
conda install -c bioconda google-sparsehash 
```



(3) For the SparseConv, we use the repo from [PointGroup](https://github.com/llijiang/spconv/tree/740a5b717fc576b222abc169ae6047ff1e95363f)

* To compile `spconv`, firstly install the dependent libraries. 
```
conda install libboost
conda install -c daleydeng gcc-5 # need gcc-5.4 for sparseconv
```
Add the `$INCLUDE_PATH$` that contains `boost` in `lib/spconv/CMakeLists.txt`. (Not necessary if it could be found.)
```
include_directories($INCLUDE_PATH$)
```

* Clone and compile the `spconv` library.
```
cd lib/
git clone https://github.com/llijiang/spconv.git -- recursive
cd spconv/
python setup.py bdist_wheel
```

* Run `cd dist` and `pip install` the generated `.whl` file.

(4) Compile the `pointgroup_ops` library.
```
cd lib/pointgroup_ops
python setup.py develop
```
If any header files could not be found, run the following commands. 
```
python setup.py build_ext --include-dirs=$INCLUDE_PATH$
python setup.py develop
```
`$INCLUDE_PATH$` is the path to the folder containing the header files that could not be found.

(5) Compile the `pointnet2` library.
```
cd lib/pointnet2
python setup.py install
```

(6) Install FAISS:

```
conda install -c faiss-gpu cudatoolkit=10.2 # for CUDA 10.2
```

(7) Install CuGraph

```
conda install -c nvidia -c rapidsai -c numba -c conda-forge cugraph cudatoolkit=10.2
```

If you encounter problem during install CuGraph, please follow [CuGraph](https://hub.docker.com/r/rapidsai/rapidsai/) to use CuGraph in docker.

## Data Preparation

We follow [DyCo3D](https://github.com/aim-uofa/DyCo3D) to prepare ScannetV2 dataset. 
The dataset files are organized as 
```
GeoFormer
├── data
│   ├── scannetv2
│   │   ├── scenes
│   │   ├── raws
│   │   |   ├── *_vh_clean_2.ply
│   │   |   ├── *_vh_clean_2.labels.ply
│   │   |   ├── *_vh_clean_2.0.010000.segs.ply
│   │   |   ├── *[0-9].aggregation.json
│   │   ├── val_gt
```
First, copy all the required files (_vh_clean_2.ply, _vh_clean_2.labels.ply, _vh_clean_2.0.010000.segs.ply, [0-9].aggregation.json) to the folder data/scannetv2/raws

To generate *npy file for training scenes, run:
```
python3 datasets/preprocess/prepare_data_inst.py
```

A new folder 'scenes' will be created in data/scannetv2

## Traing and Testing

1. Pretrain baseline (DyCo3D) with training classes

    ```bash
    python3 train_dyco.py --config config/dyco3d.yaml --use_backbone_transformer --output_path OUTPUT_PATH 
    ```

2. Pretrain GeoFormer with training classes

    ```bash
    python3 train.py --config config/detr_scannet.yaml --output_path OUTPUT_PATH --use_backbone_transformer
    ```

3. Generate Geodesic distance graph:

    a. Generate euclid distance info

    ```bash
    python3 geodist/extract.py --config config/test_fs_detr_scannet.yaml --resume PATH_TO_PRETRAIN_WEIGHT --use_backbone_transformer --output_path extract_graph
    ```

    b. Use KNN/ball-query (FAISS) to construct connected graph

    ```bash
    python3 geodist/gen_graph.py --dataset scannetv2 --info_path scene_graph_info_train.pkl --edge_out_path edge_dict_train.pkl

    python3 geodist/gen_graph.py --dataset scannetv2 --info_path scene_graph_info_test.pkl --edge_out_path edge_dict_test.pkl
    ```

    c. Use shortest path algorithm to estimate geodesic distance

    ```bash
    python3 geodist/shortest_path.py --dataset scannetv2 --info_path scene_graph_info_train.pkl --edge_in_path edge_dict_train.pkl --geodist_out_path geo_dist_train

    python3 geodist/shortest_path.py --dataset scannetv2 --info_path scene_graph_info_test.pkl --edge_in_path edge_dict_test.pkl --geodist_out_path geo_dist_test
    ```

    Optional, if you use CuGraph in docker:
    ```bash
    # Run docker container and mount this folder to docker:
    docker run -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --shm-size 8G --mount src=PATH_TO_THIS_FOLDER, target=/rapids/GeoFormer, type=bind rapidsai/rapidsai:latest /bin/bash

    cd /rapids/GeoFormer

    python3 geodist/shortest_path.py --dataset scannetv2 --info_path scene_graph_info_train.pkl --edge_in_path edge_dict_train.pkl --geodist_out_path geo_dist_train

    python3 geodist/shortest_path.py --dataset scannetv2 --info_path scene_graph_info_test.pkl --edge_in_path edge_dict_test.pkl --geodist_out_path geo_dist_test
    ```

4. Episodic Training

    ```bash
    python3 train_fs.py --config config/fs_detr_scannet.yaml --use_backbone_transformer --output_path OUTPUT_PATH --pretrain PATH_TO_PRETRAIN_WEIGHT
    ```

5. Inference and Evaluation

    Test the pretrain model:
    ```bash
    python test.py --config config/test_detr_scannet.yaml --use_backbone_transformer  --output_path OUTPUT_PATH --resume PATH_TO_WEIGHT
    ```

    Test GeoFormer in few-shot setup:
    ```bash
    python test_fs.py --use_backbone_transformer --config config/test_fs_detr_scannet.yaml --output_path OUTPUT_PATH --resume PATH_TO_WEIGHT
    ```

    Test baseline in few-shot setup:
    ```bash
    python test_fs.py --use_backbone_transformer --config config/test_fs_dyco3d.yaml --output_path OUTPUT_PATH --resume PATH_TO_WEIGHT
    ```

## Acknowledgement
This repo is built upon [DyCo3D](https://github.com/aim-uofa/DyCo3D), [spconv](https://github.com/traveller59/spconv), [3DETR](https://github.com/facebookresearch/3detr). 

