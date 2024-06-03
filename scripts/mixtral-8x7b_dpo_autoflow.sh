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
    --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --template mistral \
    --dataset_name liuyuchiLewis/autoworkflow \
    --chosen_col_name chosen \
    --rejected_col_name reject \
    --prompt_col_name whole_prompt \
    --use_peft true \
    --peft_lora_r 32 \
    --peft_lora_alpha 16 \
    --peft_lora_dropout 0.05 \
    --peft_use_rslora true \
    --peft_lora_bias none \
    --peft_lora_targets q_proj,v_proj \
    --train_split langgraph_v6prompt_twoshot \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --learning_rate 6e-6 \
    --num_train_epochs 5 \
    --output_dir output/mixtral-8x7b_dpo_autoflow_$timestamp \
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
    --beta 0.01 \
    --max_length 5290 \
    --max_prompt_length 4096 \
    --loss_type sigmoid \
    --label_smoothing 0 \
    --reference_free false \
    --sync_ref_model false \
    --ref_model_mixup_alpha 1.0 \
    --ref_model_sync_steps 2
