#!/bin/bash
# You may need to modify: model_name_or_path, dataset, vision_tower, output_dir

deepspeed llava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path path_to_stage1_llm \
    --version v1 \
    --dataset "dataset_you_want_train" \
    --vision_tower path_to_stage1_convnext\
    --mm_projector_type mlp2x_gelu \
    --mm_vision_resolution 768 \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/convllava/stage2 \
    --tune_vision_tower True \
    --tune_vit_from_layer 2 \
    --tune_entire_model False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
