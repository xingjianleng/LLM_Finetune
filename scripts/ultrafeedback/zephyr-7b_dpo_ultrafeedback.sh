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

deepspeed --master_port $MASTER_PORT src/dpo_trainer.py \
    --deepspeed configs/deepspeed_configs/ds_zero3.json \
    --model_name output/ultrachat200k/mistral-7b_sft_ultrachat200k_20240602_225731_merged \
    --template zephyr \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --use_peft true \
    --peft_lora_r 64 \
    --peft_lora_alpha 16 \
    --peft_lora_dropout 0.05 \
    --peft_use_rslora true \
    --peft_lora_bias none \
    --peft_lora_targets q_proj,k_proj,v_proj,o_proj \
    --train_split train_prefs \
    --test_split test_prefs \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --learning_rate 5e-6 \
    --num_train_epochs 3 \
    --output_dir output/ultrafeedback/zephyr-7b_dpo_ultrafeedback_$timestamp \
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
    --report_to wandb \
    --beta 0.1 \
    --max_length 2048 \
    --max_prompt_length 1024 \
    --loss_type sigmoid \
    --label_smoothing 0 \
    --reference_free false \
    --sync_ref_model false \
    --ref_model_mixup_alpha 1.0 \
    --ref_model_sync_steps 2
