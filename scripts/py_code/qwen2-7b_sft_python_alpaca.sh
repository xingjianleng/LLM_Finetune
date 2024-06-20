#!/bin/bash

export WANDB_PROJECT=Code_DPO

timestamp=$(date +"%Y%m%d_%H%M%S")

# Set the default master port
MASTER_PORT=29500

# Check if the master port argument is provided and update the master port
if [ -n "$1" ]; then
    MASTER_PORT=$1
fi

deepspeed --master_port $MASTER_PORT src/sft_trainer.py \
    --deepspeed configs/deepspeed_configs/ds_zero3.json \
    --model_name Qwen/Qwen2-7B-Instruct \
    --template qwen2 \
    --dataset_name Vezora/Tested-22k-Python-Alpaca \
    --prompt_col_name instruction \
    --messages_col_name output \
    --use_peft true \
    --peft_lora_r 64 \
    --peft_lora_alpha 16 \
    --max_seq_length 2048 \
    --peft_lora_dropout 0.05 \
    --peft_use_rslora true \
    --peft_lora_bias none \
    --peft_lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --train_split train \
    --test_size 0.1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --gradient_checkpointing true \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir output/py_code/qwen2-7b_sft_python_alpaca_$timestamp \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --eval_strategy steps \
    --eval_steps 30 \
    --save_strategy steps \
    --save_steps 30 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.03 \
    --bf16 true \
    --dataset_text_field text \
    --packing true \
    --truncation_side right \
    --report_to wandb
