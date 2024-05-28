"""
Modified from: https://gist.githubusercontent.com/lewtun/b9d46e00292d9ecdd6fd9628d53c2814/raw/113d9cc98b1556c8b49f608645e2cf269030995d/sft_trainer.py
"""
from dataclasses import dataclass, field
from typing import Optional
from functools import partial

import torch
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, AutoTokenizer
from trl import SFTTrainer, SFTConfig


templates = {
    "llama3": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
    "mixtral": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
}

def split_arg(arg):
    if isinstance(arg, str):
        return [item.strip() for item in arg.split(",")]
    return arg


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="HuggingFaceH4/ultrachat_200k", metadata={"help": "the dataset name"}
    )
    train_split: Optional[str] = field(default="train", metadata={"help": "the train split of the dataset"})
    test_split: Optional[str] = field(default=None, metadata={"help": "the test split of the dataset"})
    test_size: Optional[float] = field(default=0.15, metadata={"help": "the size of the test split, only used if test_split is None"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})
    weight_decay: Optional[float] = field(default=0., metadata={"help": "the weight decay"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "the training batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=4, metadata={"help": "the evaluation batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=32, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_seq_length: Optional[int] = field(default=1024, metadata={"help": "Input sequence length"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=32, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_targets: Optional[str] = field(default='all',
                                             metadata={"help": "the targets of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    peft_lora_bias: Optional[str] = field(default="none", metadata={"help": "the bias parameter of the LoRA adapters"})
    peft_lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the dropout parameter of the LoRA adapters"})
    peft_use_rslora: Optional[bool] = field(default=False, metadata={"help": "Enable RSLora"})
    peft_task_type: Optional[str] = field(default="CAUSAL_LM", metadata={"help": "the task type of the LoRA adapters"})
    optim: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the learning rate scheduler type"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    warmup_ratio: Optional[float] = field(default=0.05, metadata={"help": "the warmup ratio"})
    warmup_steps: Optional[int] = field(default=0, metadata={"help": "the number of warmup steps"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    bf16: Optional[bool] = field(default=True, metadata={"help": "Use bfloat16"})
    fp16: Optional[bool] = field(default=False, metadata={"help": "Use fp16"})
    tf32: Optional[bool] = field(default=False, metadata={"help": "Use tf32"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Use gradient checkpointing"})
    template: Optional[str] = field(default="llama3", metadata={"help": "Chat template"})
    messages_col_name: Optional[str] = field(default="messages", metadata={"help": "Column name for messages"})
    eval_strategy: Optional[str] = field(default="epoch", metadata={"help": "Evaluation strategy"})
    eval_steps: Optional[int] = field(default=500, metadata={"help": "Number of updates steps before two evaluations"})
    save_strategy: Optional[str] = field(default="steps", metadata={"help": "Save strategy"})
    save_steps: Optional[int] = field(
        default=500, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=3, metadata={"help": "Limits total number of checkpoints."})
    report_to: Optional[str] = field(default="none", metadata={"help": "Logging platform"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
peft_lora_targets_parsed = split_arg(script_args.peft_lora_targets)

# Step 1: Load the dataset
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
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

def prepare_dialogue(example, tokenizer):
    messages = example[script_args.messages_col_name]
    assert isinstance(messages, list) and len(messages) > 0
    if isinstance(messages[0], dict):
        assert "content" in messages[0] and "role" in messages[0]
    elif isinstance(messages[0], str):
        messages_new = []
        for i, message in enumerate(messages):
            messages_new.append({"content": message, "role": "user" if i % 2 == 0 else "assistant"})
    else:
        raise ValueError(f"Invalid type for messages: {type(messages[0])}")
    text = tokenizer.apply_chat_template(example[script_args.messages_col_name], tokenize=False)
    example["text"] = text
    return example

remove_columns = dataset.column_names['train']
dataset = dataset.map(partial(prepare_dialogue, tokenizer=tokenizer), num_proc=4, remove_columns=remove_columns)

# Step 2: Define the training arguments
# NOTE: Moving this to the front to make zero.Init() call before model instantiation
training_args = SFTConfig(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
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
torch_dtype = torch.float16 if script_args.fp16 else (torch.bfloat16 if script_args.bf16 else torch.float32)

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
)

# Step 4: Define the LoraConfig
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
else:
    peft_config = None


# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
)

trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)
