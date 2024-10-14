

import argparse
from collections import defaultdict

import numpy as np
import torch

parser = argparse.ArgumentParser(description="compress any prompt.")
parser.add_argument(
    "--load_path",
    help="path to load data",
    default="../../../results/meetingbank/gpt-4-32k_comp/annotation_cs512_meetingbank_train_formated.pt",
)
parser.add_argument(
    "--save_path",
    help="path to save filtered data",
    default="../../../results/meetingbank/gpt-4-32k_comp/annotation_kept_cs512_meetingbank_train_formated.pt",
)
args = parser.parse_args()

res_pt = torch.load(args.load_path)

## filtering
variation_rate_list = res_pt["variation_rate"]
print(len(variation_rate_list))
threshold = np.percentile(variation_rate_list, 90)
kept, filtered = defaultdict(list), defaultdict(list)
for labels, origin, comp, retrieval, cr, vr, hr, mr, ag in zip(
    res_pt["labels"],
    res_pt["origin"],
    res_pt["comp"],
    res_pt["retrieval"],
    res_pt["comp_rate"],
    res_pt["variation_rate"],
    res_pt["hitting_rate"],
    res_pt["matching_rate"],
    res_pt["alignment_gap"],
):
    if vr >= threshold:
        filtered["labels"].append(labels)
        filtered["origin"].append(origin)
        filtered["comp"].append(comp)
        filtered["retrieval"].append(retrieval)
        filtered["comp_rate"].append(cr)
        filtered["variation_rate"].append(vr)
        filtered["hitting_rate"].append(hr)
        filtered["matching_rate"].append(mr)
        filtered["alignment_gap"].append(ag)
    else:
        kept["labels"].append(labels)
        kept["origin"].append(origin)
        kept["comp"].append(comp)
        kept["retrieval"].append(retrieval)
        kept["comp_rate"].append(cr)
        kept["variation_rate"].append(vr)
        kept["hitting_rate"].append(hr)
        kept["matching_rate"].append(mr)
        kept["alignment_gap"].append(ag)
alignment_gap_list = kept["alignment_gap"]
threshold = np.percentile(alignment_gap_list, 90)

torch.save(kept, args.save_path)

