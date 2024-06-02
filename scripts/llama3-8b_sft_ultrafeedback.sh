#!/bin/bash

export WANDB_PROJECT=LLM_Finetune
export WANDB_MODE=offline

timestamp=$(date +"%Y%m%d_%H%M%S")
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_ARRAY[@]}

# Set the default master port
MASTER_PORT=29500

# Check if the master port argument is provided and update the master port
if [ -n "$1" ]; then
    MASTER_PORT=$1
fi

# 1. Fix LoRA_alpha to 16
#   https://datascience.stackexchange.com/questions/123229/understanding-alpha-parameter-tuning-in-lora-paper
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --num_processes $NUM_GPUS --main_process_port $MASTER_PORT --config_file configs/accelerate_configs/deepspeed_zero3.yaml src/sft_trainer.py \
    --model_name meta-llama/Meta-Llama-3-8B \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --messages_col_name chosen \
    --use_peft true \
    --peft_lora_r 128 \
    --peft_lora_alpha 16 \
    --max_seq_length 2048 \
    --peft_lora_dropout 0.0 \
    --peft_use_rslora true \
    --peft_lora_bias none \
    --peft_lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --train_split train_sft \
    --test_split test_sft \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir output/llama3-8b_sft_ultrafeedback_$timestamp \
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
