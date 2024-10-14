import json
import os
from tqdm import tqdm
import argparse
from utils.evaluate import Relatedness,Correctness,Comprehensiveness
from utils.utils import config,Metrics_conversion,ensure_directory_exists

def evaluate(dataset_name,compress_model=""):
    if  'stackexchange' in compress_model:
        user_profile_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'profile',config['agent_model']+'_stackexchange',dataset_name)
    else:
        if "base" in compress_model:
            user_profile_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'profile',config['agent_model']+'_bert_base',dataset_name)
        elif "large" in compress_model:
            user_profile_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'profile',config['agent_model']+'_bert_large',dataset_name)
        elif 'Llama-2' in compress_model:
            user_profile_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'profile',config['agent_model']+'_llama2',dataset_name)
        elif 'phi-2' in compress_model:
            user_profile_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'profile',config['agent_model']+'_phi2',dataset_name)
        else:
            user_profile_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'profile',config['agent_model'],dataset_name)
        
    if  'stackexchange' in compress_model:
            answer_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'answer',config['agent_model'],'stackexchange',dataset_name)
    else:
        if "base" in compress_model:
            answer_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'answer',config['agent_model'],'bert_base',dataset_name)
        elif "large" in compress_model:
            answer_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'answer',config['agent_model'],'bert_large',dataset_name)
        elif 'phi-2' in compress_model:
            answer_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'answer',config['agent_model'],'phi-2',dataset_name)
        elif 'Llama-2' in compress_model:
            answer_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'answer',config['agent_model'],'llama2',dataset_name)
        else:
            answer_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'answer',config['agent_model'],'per-agent',dataset_name)
    answer_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'answer',config['agent_model'],'per-agent',dataset_name)

    with open(user_profile_datapath, 'r') as f:
        user_profile_dataset = json.load(f)
    with open(answer_datapath, 'r') as f:
        answer_dataset = json.load(f)
    
    eval_dataset = {}
    for id,qa in tqdm(answer_dataset.items(), desc="Evaluate Answer"):
        query = qa['query']
        answer = qa['answer']
        user_profile = user_profile_dataset[id]['user_profile']['global']
        relatedness = Relatedness(user_profile,query,answer)
        correctness = Correctness(query,answer)
        comprehensiveness = Comprehensiveness(query,answer)
        print([relatedness,correctness,comprehensiveness])
        relatedness = Metrics_conversion(relatedness)
        correctness = Metrics_conversion(correctness)
        comprehensiveness = Metrics_conversion(comprehensiveness)
        
        eval_dataset[id] = [relatedness,correctness,comprehensiveness]
        json_string = json.dumps(eval_dataset, indent=4)
        with open(dataset_name, "w") as json_file:
            json_file.write(json_string)

if __name__ == '__main__':

    dataset_list = ['shuishan.json','allenai.json','stackexchange.json']
    compress_model = 'xlm-roberta-large-stackexchange'
    for dataset in dataset_list:
        evaluate(dataset,compress_model)