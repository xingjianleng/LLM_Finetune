import argparse
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def main(args):
    with open(os.path.join(args.adapter_path, 'adapter_config.json')) as f:
        adapter_config = json.load(f)

    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=adapter_config["base_model_name_or_path"],
        return_dict=True,
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)

    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.merge_and_unload()

    output_path = args.adapter_path.rstrip('/') + '_merged'
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Merge LoRA weights to the original model')
    parser.add_argument('adapter_path', type=str, required=True, help='the base model name')
    args = parser.parse_args()
    main(args)
