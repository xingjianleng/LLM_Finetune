"""
Modified from: https://gist.githubusercontent.com/lewtun/b9d46e00292d9ecdd6fd9628d53c2814/raw/113d9cc98b1556c8b49f608645e2cf269030995d/sft_trainer.py
"""
from utils import SFTArguments, prepare_dialogue, split_arg, templates
from optim import create_loraplus_optimizer

import torch
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict, disable_caching
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# Disable caching, since tokenizer is dynamic, we can't reload cache anyway
disable_caching()


def main():
    parser = HfArgumentParser(SFTArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    peft_lora_targets_parsed = split_arg(script_args.peft_lora_targets)

    # Step 1: Load the dataset
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    # Set chat template if the tokenizer does not have one
    if tokenizer.chat_template is None:
        tokenizer.chat_template = templates[script_args.template]
    dataset = load_dataset(script_args.dataset_name, split=script_args.train_split)

    # Fix tokenizer by setting pad_token to eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.test_split is not None:
        dataset = DatasetDict({
            "train": dataset,
            "test": load_dataset(script_args.dataset_name, split=script_args.test_split)
        })
    else:
        assert script_args.test_size is not None and script_args.test_size > 0
        dataset = dataset.train_test_split(test_size=script_args.test_size)

    remove_columns = dataset.column_names['train']
    dataset = dataset.map(
        prepare_dialogue,
        num_proc=4,
        load_from_cache_file=script_args.load_from_cache_file,
        fn_kwargs={"tokenizer": tokenizer, "script_args": script_args},
        remove_columns=remove_columns
    )

    # Step 2: Define the training arguments
    # NOTE: Moving this to the front to make zero.Init() call before model instantiation
    training_args = SFTConfig(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=script_args.learning_rate,
        weight_decay=script_args.weight_decay,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        save_total_limit=script_args.save_total_limit,
        bf16=script_args.bf16,
        fp16=script_args.fp16,
        tf32=script_args.tf32,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        warmup_steps=script_args.warmup_steps,
        eval_strategy=script_args.eval_strategy,
        eval_steps=script_args.eval_steps,
        save_strategy=script_args.save_strategy,
        save_steps=script_args.save_steps,
        max_seq_length=script_args.max_seq_length,
        dataset_text_field=script_args.dataset_text_field,
        packing=True,
        report_to=script_args.report_to,
    )

    # Step 3: Load the model
    torch_dtype = torch.bfloat16 if script_args.bf16 else (torch.float16 if script_args.fp16 else torch.float32)

    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # If we are using deepspeed, we need to check if we are using Zero stage 3
        if Accelerator().state.deepspeed_plugin is not None:
            assert Accelerator().state.deepspeed_plugin.zero_stage != 3, \
                "DeepSpeed Zero stage 3 is not supported with quantization"
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
    else:
        device_map = None
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
        attn_implementation=script_args.attn_implementation,
        use_cache=not script_args.gradient_checkpointing,
    )

    # Step 4: Define the LoraConfig and load the model
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            lora_dropout=script_args.peft_lora_dropout,
            use_rslora=script_args.peft_use_rslora,
            bias=script_args.peft_lora_bias,
            target_modules=peft_lora_targets_parsed,
            task_type=script_args.peft_task_type,
        )
        
        # Directly get the peft model
        model = get_peft_model(model, peft_config)
    else:
        peft_config = None

    # Step 6: Define the Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    # Step 5: Prepare LoRA+ optimizer and start training
    if script_args.loraplus_lr_ratio is not None:
        optimizer = create_loraplus_optimizer(model, training_args, script_args)
        trainer.optimizer = optimizer

    trainer.train()

    # Step 7: Save the model
    trainer.save_model(script_args.output_dir)


if __name__ == '__main__':
    main()
