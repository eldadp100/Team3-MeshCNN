#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/cubes \
--name cubes \
--arch mesh_transformer \
--slide_verts 0.2 \
--num_aug 20 \
--lr 0.0003
