#!/usr/bin/env bash

python train.py \
-gen_bs 128 \
-dis_bs 64 \
--dataset stl10 \
--bottom_width 6 \
--img_size 48 \
--max_iter 50000 \
--gen_model autogan_cifar10_a \
--dis_model autogan_cifar10_a \
--latent_dim 128 \
--gf_dim 256 \
--df_dim 128 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 10 \
--exp_name autogan_cifar10_a2stl10