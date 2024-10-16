from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

import argparse

def configure_parser():
    parser = argparse.ArgumentParser(
        description="build model to use llmlingua"
    )
    parser.add_argument(
        "--model_name",
        help="Foundation Model",
        default="base model path",
    )
    parser.add_argument(
        "--pth_path",
        help="Trained model weights",
    )

    parser.add_argument(
        "--save_path",
        help="save path",
    )

    return parser

if __name__ == '__main__':
    parser = configure_parser()
    args = parser.parse_args()

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)

    saved_model_path = args.pth_path

    model.load_state_dict(torch.load(saved_model_path))

    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)