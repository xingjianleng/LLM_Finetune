from dataclasses import dataclass, field
from typing import Optional


templates = {
    "llama3": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
    "mixtral": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
}


def split_arg(arg):
    if isinstance(arg, str):
        return [item.strip() for item in arg.split(",")]
    return arg


def prepare_dialogue(example, tokenizer, script_args):
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


@dataclass
class SFTArguments:
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
    loraplus_lr_ratio: Optional[float] = field(default=None, metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."})
    loraplus_lr_embedding: Optional[float] = field(default=1e-6,
                                                   metadata={"help": "LoRA plus learning rate for lora embedding layers."})
    save_total_limit: Optional[int] = field(default=3, metadata={"help": "Limits total number of checkpoints."})
    report_to: Optional[str] = field(default="none", metadata={"help": "Logging platform"})
    load_from_cache_file: Optional[bool] = field(default=True,
                                                 metadata={"help": "Load from cache file"})
    attn_implementation: Optional[str] = field(default="flash_attention_2",
                                               metadata={"help": "Attention implementation"})
