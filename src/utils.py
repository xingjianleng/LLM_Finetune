from dataclasses import dataclass, field
from typing import Optional
from trl import SFTConfig, DPOConfig


templates = {
    "llama3": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
    "mistral": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
    "zephyr": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
    "qwen2": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
}
support_system_msg = {
    "llama3": True,
    "mistral": False,
    "zephyr": True,
    "qwen2": True
}
assistant_prefixs = {
    "llama3": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "mistral": "[/INST]",
    "zephyr": "<|assistant|>\n",
    "qwen2": "<|im_start|>assistant\n"
}


def split_arg(arg):
    if isinstance(arg, str):
        return [item.strip() for item in arg.split(",")]
    return arg


def prepare_dialogue(example, tokenizer, script_args):
    messages = example[script_args.messages_col_name]
    if type(messages) == list:
        assert len(messages) > 0, "Messages must not be empty"
        if isinstance(messages[0], dict):
            for i in range(len(messages)):
                assert "content" in messages[i] and "role" in messages[i]
        elif isinstance(messages[0], str):
            messages_new = []
            for i, message in enumerate(messages):
                messages_new.append({"content": message, "role": "user" if i % 2 == 0 else "assistant"})
            messages = messages_new
        else:
            raise ValueError(f"Invalid type for single message: {type(messages[0])}")

    elif type(messages) == str:
        # It must be a single-turn dialogue, where response is the chosen collumn, and the 
        promt_col = script_args.prompt_col_name
        if promt_col is None:
            raise ValueError("Prompt name not found in example")

        messages = [{"content": example[promt_col], "role": "user"}, {"content": messages, "role": "assistant"}]
    
    else:
        raise TypeError(f"Invalid type for messages: {type(messages)}")

    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"content": "", "role": "system"})
    # NOTE: In some models, the chat template doesn't support system messages, in this case, we combine with user prompt
    if not support_system_msg[script_args.template]:
        separator = "\n\n" if messages[0]["content"] else ""
        messages[1]["content"] = f'{messages[0]["content"]}{separator}{messages[1]["content"]}'
        messages.pop(0)

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    example["text"] = text
    return example


def prepare_dpo_dialogue(example, tokenizer, script_args):
    chosen_col = script_args.chosen_col_name
    rejected_col = script_args.rejected_col_name
    assert chosen_col in example and rejected_col in example and type(example[chosen_col]) == type(example[rejected_col]), \
        f"`chosen` and `rejected` must exist and be of the same type"

    if type(example[chosen_col]) == list:
        chosen_messages = example[chosen_col]
        rejected_messages = example[rejected_col]

        # Prepent empty system message if it's not there
        if chosen_messages[0]['role'] != 'system':
            chosen_messages.insert(0, {"content": "", "role": "system"})
            rejected_messages.insert(0, {"content": "", "role": "system"})
        
        # Combine the first user message with the system message if the template doesn't support system messages
        if not support_system_msg[script_args.template]:
            chosen_messages[1]["content"] = f'{chosen_messages[0]["content"]}\n\n{chosen_messages[1]["content"]}'
            rejected_messages[1]["content"] = f'{rejected_messages[0]["content"]}\n\n{rejected_messages[1]["content"]}'
            chosen_messages.pop(0)
            rejected_messages.pop(0)

        # Apply chat template
        chosen_text = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        rejected_text = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

    elif type(example[chosen_col]) == str:
        # This should only happen in our customized dataset, where we are using mistral template
        assert script_args.template == "mistral", "This should only happen in mistral template"

        prompt_col = script_args.prompt_col_name
        if prompt_col is None:
            raise ValueError("Prompt name not found in example")

        # Apply chat template
        chosen_messages = [{"content": example[prompt_col], "role": "user"}, {"content": example[chosen_col], "role": "assistant"}]
        rejected_messages = [{"content": example[prompt_col], "role": "user"}, {"content": example[rejected_col], "role": "assistant"}]
        chosen_text = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        rejected_text = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

    else:
        raise TypeError(f"Invalid type for chosen and rejected: {type(example[chosen_col])}")

    # Make sure the user query is the same
    assert chosen_text.split(assistant_prefixs[script_args.template])[0] == rejected_text.split(assistant_prefixs[script_args.template])[0], \
        "The chosen and rejected messages must have the same user query"
    # Assistant response should be the last part
    example['text_chosen'] = chosen_text.split(assistant_prefixs[script_args.template])[-1]
    example['text_rejected'] = rejected_text.split(assistant_prefixs[script_args.template])[-1]
    # Set the prompt
    example['text_prompt'] = chosen_text.split(assistant_prefixs[script_args.template])[0] + assistant_prefixs[script_args.template]

    return example


def rank0_print(msg, script_args):
    if script_args.local_rank == 0:
        print(msg)


@dataclass
class ScriptArguments(SFTConfig, DPOConfig):
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="HuggingFaceH4/ultrachat_200k", metadata={"help": "the dataset name"}
    )
    dataset_config_name: Optional[str] = field(
        default="default", metadata={"help": "the dataset config name"}
    )
    train_split: Optional[str] = field(default="train", metadata={"help": "the train split of the dataset"})
    test_split: Optional[str] = field(default=None, metadata={"help": "the test split of the dataset"})
    test_size: Optional[float] = field(default=None, metadata={"help": "the size of the test split, only used if test_split is None"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_targets: Optional[str] = field(default='q_proj,v_proj',
                                             metadata={"help": "the targets of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    peft_lora_bias: Optional[str] = field(default="none", metadata={"help": "the bias parameter of the LoRA adapters"})
    peft_lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the dropout parameter of the LoRA adapters"})
    peft_use_rslora: Optional[bool] = field(default=False, metadata={"help": "Enable RSLora"})
    peft_task_type: Optional[str] = field(default="CAUSAL_LM", metadata={"help": "the task type of the LoRA adapters"})
    template: Optional[str] = field(default=None, metadata={"help": "Chat template"})
    messages_col_name: Optional[str] = field(default="messages", metadata={"help": "Column name for messages"})
    chosen_col_name: Optional[str] = field(default="chosen", metadata={"help": "Column name for chosen message"})
    rejected_col_name: Optional[str] = field(default="rejected", metadata={"help": "Column name for rejected message"})
    prompt_col_name: Optional[str] = field(default="prompt", metadata={"help": "Column name for prompt"})
    loraplus_lr_ratio: Optional[float] = field(default=None, metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."})
    loraplus_lr_embedding: Optional[float] = field(default=1e-6,
                                                   metadata={"help": "LoRA plus learning rate for lora embedding layers."})
    load_from_cache_file: Optional[bool] = field(default=True,
                                                 metadata={"help": "Load from cache file for datasets library"})
    use_reentrant: Optional[bool] = field(default=False, metadata={"help": "Use reentrant lock for tokenization"})
    truncation_side: Optional[str] = field(default=None, metadata={"help": "Truncation side for tokenization"})
    attn_implementation: Optional[str] = field(default="flash_attention_2",
                                               metadata={"help": "Attention implementation"})
