"""
Modified from: https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py
"""
import json
import pathlib
from utils import ScriptArguments, templates, split_arg, prepare_dpo_dialogue, rank0_print
from optim import create_loraplus_optimizer

import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, AutoTokenizer
from trl import DPOTrainer


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    # Manually set gradient checkpointing kwargs
    script_args.gradient_checkpointing_kwargs = {"use_reentrant": script_args.use_reentrant}
    peft_lora_targets_parsed = split_arg(script_args.peft_lora_targets)

    # If we are using deepspeed, we need to check if we are using Zero stage 3
    if script_args.deepspeed is not None and (script_args.load_in_8bit or script_args.load_in_4bit):
        if type(script_args.deepspeed) == str:
            with open(script_args.deepspeed) as f:
                ds_config = json.load(f)
        elif type(script_args.deepspeed) == dict:
            ds_config = script_args.deepspeed
        else:
            raise TypeError(f"Invalid type for deepspeed: {type(script_args.deepspeed)}")
        
        # Get the DeepSpeed stage from the configuration
        zero_optimization = ds_config.get("zero_optimization", {})
        stage = zero_optimization.get("stage", "Not specified")
        assert stage != 3, \
            "DeepSpeed Zero stage 3 is not supported with quantization"

    # Step 1: Load the dataset
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    # Set chat template if the tokenizer does not have one
    if tokenizer.chat_template is None:
        assert script_args.template is not None, "Chat template must be provided"
        rank0_print(f"Setting chat template to {templates[script_args.template]}", script_args)
        tokenizer.chat_template = templates[script_args.template]
    else:
        assert templates[script_args.template] == tokenizer.chat_template, \
            (f"Tokenizer chat template {tokenizer.chat_template} does not match "
            f"provided template {templates[script_args.template]}. Double check the template argument!")
    dataset = load_dataset(script_args.dataset_name,
                           script_args.dataset_config_name,
                           split=script_args.train_split)

    # Fix tokenizer by setting pad_token to eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.test_split is not None:
        dataset = DatasetDict({
            "train": dataset,
            "test": load_dataset(script_args.dataset_name,
                                 script_args.dataset_config_name,
                                 split=script_args.test_split)
        })
    elif script_args.test_size is not None:
        # If test_size is provided, we split the dataset into train and test
        assert 0 < script_args.test_size < 1, "test_size must be a float between 0 and 1"
        dataset = dataset.train_test_split(test_size=script_args.test_size)
    else:
        # This is the case where no eval_set is used
        dataset = DatasetDict({
            "train": dataset,
        })
        # Set eval_strategy to no if no testing set is provided
        script_args.eval_strategy = "no"

    remove_columns = dataset.column_names['train']
    dataset = dataset.map(
        prepare_dpo_dialogue,
        num_proc=4,
        load_from_cache_file=script_args.load_from_cache_file,
        fn_kwargs={"tokenizer": tokenizer, "script_args": script_args},
        remove_columns=remove_columns
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in dataset.keys():
        dataset[split] = dataset[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    # Step 2: Load the model
    torch_dtype = torch.bfloat16 if script_args.bf16 else (torch.float16 if script_args.fp16 else torch.float32)

    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": script_args.device}
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

    # Step 3: Define the LoraConfig and load the model
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

    # Step 4: Define the Trainer
    # Since we always want to have a reference model with the same architecture as the
    # tuned model, we pass None as the ref_model
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        args=script_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset else None,
    )

    # Step 5: Prepare LoRA+ optimizer and start training
    if script_args.loraplus_lr_ratio is not None:
        optimizer = create_loraplus_optimizer(model, script_args)
        trainer.optimizer = optimizer

    if list(pathlib.Path(script_args.output_dir).glob("checkpoint-*")):
        rank0_print("Resuming from checkpoint...", script_args)
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Step 6: Save the model
    trainer.save_model(script_args.output_dir)


if __name__ == '__main__':
    main()
