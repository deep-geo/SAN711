#!/usr/bin/env bash

python train.py \
--work_dir "/root/autodl-tmp/workdir" \
--run_name "sup_CEP" \
--seed 42 \
--epochs 1000 \
--batch_size 16 \
--num_workers 8 \
--eval_interval 527 \
--test_sample_rate 0.1 \
--image_size 256 \
--mask_num 5 \
--data_root "/root/autodl-tmp/ALL_Multi" \
--test_size 0.1 \
--metrics 'iou' 'dice' 'precision' 'f1_score' 'recall' 'aji' 'dq' 'sq' 'pq' \
--checkpoint "/root/autodl-tmp/workdir/models/sup_clust_edge_enc_06-02_20-42/epoch0065_test-loss0.1528_sam.pth" \
--device "cuda" \
--lr 0.0001 \
--resume "" \
--model_type "vit_b" \
--boxes_prompt \
--point_num 1 \
--edge_point_num 3 \
--iter_point 8 \
#--encoder_adapter \
#--multimask \
#--unsupervised_only \
#--prompt_path
#--save_pred
#--lr_scheduler \
#--point_list 
#autodl-tmp/workdir/models/sup_clust_edge_enc_06-02_20-42/epoch0065_test-loss0.1528_sam.pth   /root/autodl-tmp/sam_vit_b_01ec64.pth

