#!/bin/bash

export WANDB_PROJECT=LLM_Finetune
export WANDB_MODE=offline
timestamp=$(date +"%Y%m%d_%H%M%S")

# 1. Fix LoRA_alpha to 16
#   https://datascience.stackexchange.com/questions/123229/understanding-alpha-parameter-tuning-in-lora-paper
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate_configs/deepspeed_zero3.yaml src/sft_trainer.py \
    --model_name mistralai/Mistral-7B-v0.1 \
    --template zephyr \
    --dataset_name HuggingFaceH4/ultrachat_200k \
    --messages_col_name messages \
    --use_peft true \
    --peft_lora_r 64 \
    --peft_lora_alpha 16 \
    --max_seq_length 2048 \
    --peft_lora_dropout 0.05 \
    --peft_use_rslora true \
    --peft_lora_bias none \
    --peft_lora_targets q_proj,k_proj,v_proj,o_proj \
    --train_split train_sft \
    --test_split test_sft \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --output_dir output/mistral-7b_sft_ultrachat200k_$timestamp \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --eval_strategy steps \
    --eval_steps 800 \
    --save_strategy steps \
    --save_steps 800 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.03 \
    --bf16 true \
    --packing true \
    --report_to wandb