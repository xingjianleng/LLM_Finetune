import argparse
import dotenv
from datasets import load_dataset, get_dataset_config_names


def main(args):
    configs = get_dataset_config_names(args.ds_name)
    for config in configs:
        # This will save to cache
        print(f"Downloading {args.ds_name} with config {config}")
        load_dataset(args.ds_name, config)


if __name__ == '__main__':
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description='Download dataset from hf')
    parser.add_argument('--ds_name', type=str, required=True, help='Dataset name')
    args = parser.parse_args()
    main(args)
