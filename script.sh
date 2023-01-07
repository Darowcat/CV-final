#!/usr/bin/env bash
    # Example script
     python -m torch.distributed.launch --nproc_per_node=1 train_v1.py \
        --lr_schedule poly \
        --lr 5e-4 \
        --poly_exp 0.9 \
        --max_iter 150000 \
        --bs_mult 16 \
        --date 1125 \
        --cl_weight 1.0 \
        --exp r50os16_gtav_pretrain_fd34_iteradain_img \
        --ckpt ./logs/ \
        --tb_path ./logs/ \
        # --wandb_name CV_final \