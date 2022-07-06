## Traing and Testing

1\) Pretrain baseline (DyCo3D) with training classes

    ```bash
    python3 train_dyco.py --config config/dyco3d.yaml --use_backbone_transformer --output_path OUTPUT_PATH 
    ```

2\) Pretrain GeoFormer with training classes

    ```bash
    python3 train.py --config config/detr_scannet.yaml --output_path OUTPUT_PATH
    ```

3\) Generate Geodesic distance graph:

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

4\) Episodic Training

    ```bash
    python3 train_fs.py --config config/fs_detr_scannet.yaml --use_backbone_transformer --output_path OUTPUT_PATH --pretrain PATH_TO_PRETRAIN_WEIGHT
    ```

5\) Inference and Evaluation

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