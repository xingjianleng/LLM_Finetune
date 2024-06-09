#!/bin/bash

export WANDB_PROJECT=LLM_Finetune
export WANDB_MODE=offline

timestamp=$(date +"%Y%m%d_%H%M%S")

# Set the default master port
MASTER_PORT=29500

# Check if the master port argument is provided and update the master port
if [ -n "$1" ]; then
    MASTER_PORT=$1
fi

deepspeed --master_port $MASTER_PORT src/sft_trainer.py \
    --deepspeed configs/deepspeed_configs/ds_zero3.json \
    --model_name output/mistral-7b_sft_ultrachat200k_20240602_225731_merged  \
    --template zephyr \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --messages_col_name chosen \
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
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir output/ultrafeedback/zephyr-7b_sft_ultrafeedback_$timestamp \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.1 \
    --bf16 true \
    --dataset_text_field text \
    --packing true \
    --report_to wandb
