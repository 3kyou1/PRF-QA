import json
import os

from datasets import load_dataset
import argparse
def configure_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_dataset', type=str, default='../../../code/compress/data_collection/dataset/filter_stackexchange.json',
                        help='raw dataset path')
    parser.add_argument('--format_dataset', type=str, default='stackexchange.json',
                        help='format dataset path')
    return parser

parser = configure_parser()
args = parser.parse_args()

with open(args.raw_dataset, 'r') as f:
    dataset = json.load(f)
data = []
for idx, instance in enumerate(dataset):
    temp = {}
    temp["idx"] = idx
    temp["en"] = instance["en"]
    temp["zh"] = instance["zh"]
    data.append(temp)
os.makedirs("results/stackexchange/origin/", exist_ok=True)
json.dump(data,open(f"results/stackexchange/origin/{args.format_dataset}", "w"),indent=4,)
