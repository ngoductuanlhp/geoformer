#!/bin/bash -e
#SBATCH --job-name=geo
#SBATCH --output=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/slurm_out/slurm_%A.out
#SBATCH --error=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/slurm_out/slurm_%A.err

#SBATCH --gpus=1
#SBATCH --nodes=1

#SBATCH --mem-per-gpu=40G

#SBATCH --cpus-per-gpu=64

#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.tuannd42@vinai.io

# srun --container-image="harbor.vinai-systems.com#research/tuannd42:softgroup" \
# --container-mounts=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/SoftGroup:/home/ubuntu/SoftGroup \
# --container-workdir=/home/ubuntu/SoftGroup/ \
# python3 tools/train.py configs/softgroup_scannet_bbox_context_head_ballquery.yaml --exp_name thresh0.95

# srun --container-image="harbor.vinai-systems.com#research/tuannd42:pointr" \
# --container-mounts=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/geoformer:/home/ubuntu/geoformer \
# --container-workdir=/home/ubuntu/geoformer/ \
# python3 train.py --config config/detr_scannet_finetune.yaml --output_path exp/finetune_detr_geo_onlydyco --resume exp/finetune_detr_geo_onlydyco/checkpoint_last.pth

srun --container-image="harbor.vinai-systems.com#research/tuannd42:pointr" \
--container-mounts=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/geoformer:/home/ubuntu/geoformer \
--container-workdir=/home/ubuntu/geoformer/ \
python3 train.py --config config/detr_scannet_finetune.yaml --output_path exp/finetune_detr_geo_decoderdyco_long --resume exp/finetune_detr_geo_decoderdyco/checkpoint_last.pth