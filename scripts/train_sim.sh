#!/bin/bash

#SBATCH --job-name=similarity_train
#SBATCH --output=similarity_train.txt
#SBATCH --error=similarity_train.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=4-00:00:00


master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=50194
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1
module load Anaconda3
source activate cotr
conda activate base
conda activate cotr

srun python ../train_similarity.py \
--model_name base_3_shot \
--det_model_name verification \
--data_path /d/hpc/projects/FRI/pelhanj/fsc147 \
--model_path /d/hpc/projects/FRI/pelhanj/ \
--backbone resnet50 \
--swav_backbone \
--reduction 8 \
--image_size 512 \
--num_enc_layers 3 \
--num_dec_layers 3 \
--kernel_dim 3 \
--emb_dim 256 \
--num_objects 3 \
--epochs 50 \
--lr 1e-5 \
--lr_drop 220 \
--weight_decay 1e-2 \
--batch_size 32 \
--dropout 0.1 \
--num_workers 8 \
--max_grad_norm 0.1 \
--normalized_l2 \
--detection_loss_weight 0.01 \
--count_loss_weight 0.0 \
--min_count_loss_weight 0.0 \
--aux_weight 0.3 \
--tiling_p 0.4 \
--use_query_pos_emb \
--use_objectness \
--use_appearance \
--pre_norm \