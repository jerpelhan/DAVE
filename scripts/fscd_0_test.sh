#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python ../main.py \
--model_name DAVE_0_shot \
--skip_train \
--zero_shot \
--backbone resnet50 \
--swav_backbone \
--reduction 8 \
--num_enc_layers 3 \
--num_dec_layers 3 \
--kernel_dim 3 \
--emb_dim 256 \
--num_objects 3 \
--num_workers 8 \
--use_query_pos_emb \
--use_objectness \
--use_appearance \
--batch_size 1 \
--pre_norm \
--egv 0.2
