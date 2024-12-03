from llmlingua import PromptCompressor
from tqdm import tqdm
import json
import os
from utils.utils import config
import jieba

def chinese_split(query):
    seg_list = jieba.cut(query,cut_all=False)
    seg_list = " ".join(seg_list)
    return seg_list

def calculate_jaccard_similarity(text1, text2):
    words1 = set(text1.split())
    words2 = set(text2.split())

    intersection = words1.intersection(words2)
    union = words1.union(words2)
    

    return len(intersection) / len(union)


def calculate_inclusion_coefficient(text1, text2):
    words1 = set(text1.split())
    words2 = set(text2.split())

    intersection = words1.intersection(words2)
    

    return len(intersection) / len(words1)

def eval(dataset_name,agent_model,model):
    cnt = 0
    datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'dataset',dataset_name)
    profile_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'profile',agent_model,dataset_name)
    sim_answer_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'answer',agent_model,'simple-answer',dataset_name)
    per_answer_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'answer',agent_model,model,dataset_name)
    large_answer_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'answer',agent_model,'per-agent',dataset_name)
    with open(datapath, 'r') as f:
        dataset = json.load(f)
    with open(profile_datapath, 'r') as f:
        profile = json.load(f)
    with open(sim_answer_datapath, 'r') as f:
        sim_answer_dataset = json.load(f)
    with open(per_answer_datapath, 'r') as f:
        per_answer_dataset = json.load(f)
    with open(large_answer_datapath, 'r') as f:
        large_answer_dataset = json.load(f)

    #将query与profile分成pool
    jaccard = {'jaccard-sim-query':0,'jaccard-sim-profile':0,'jaccard-per-query':0,'jaccard-per-profile':0}
    ic = {'ic-sim-query':0,'ic-sim-profile':0,'ic-per-query':0,'ic-per-profile':0}
    
    

    
    
    
    for id,queries in tqdm(dataset.items()):
        if id not in per_answer_dataset:
            continue
        user_profile = profile[id]["user_profile"]
        sim_answer = sim_answer_dataset[id]['answer']
        per_answer = per_answer_dataset[id]['answer']
        large_answer = large_answer_dataset[id]['answer']
        query_pool = ""
        profile_pool = ""
        if dataset_name == 'shuishan.json':
            for query in queries:
                query_pool = query_pool + chinese_split(query)
            for k,v in user_profile.items():
                profile_pool = profile_pool + chinese_split(v)
            sim_pool = chinese_split(sim_answer)
            per_pool = chinese_split(per_answer)
            large_pool = chinese_split(large_answer)
        else:
            for query in queries:
                query_pool = query_pool + query
            for k,v in user_profile.items():
                profile_pool = profile_pool + v
            sim_pool = sim_answer
            per_pool = per_answer
            large_pool = large_answer
        
        pool = {'query_pool':query_pool,'profile_pool':profile_pool,'sim_pool':sim_pool,'per_pool':per_pool}
        
        
        
        jaccard['jaccard-sim-query'] = jaccard['jaccard-sim-query'] + calculate_jaccard_similarity(pool['query_pool'],pool['sim_pool'])
        jaccard['jaccard-per-query'] = jaccard['jaccard-per-query'] + calculate_jaccard_similarity(pool['query_pool'],pool['per_pool'])
        jaccard['jaccard-sim-profile'] = jaccard['jaccard-sim-profile'] + calculate_jaccard_similarity(pool['profile_pool'],pool['sim_pool'])
        jaccard['jaccard-per-profile'] = jaccard['jaccard-per-profile'] + calculate_jaccard_similarity(pool['profile_pool'],pool['per_pool'])

        ic['ic-sim-query'] = ic['ic-sim-query'] + calculate_inclusion_coefficient(pool['sim_pool'],pool['query_pool'])
        ic['ic-per-query'] = ic['ic-per-query'] + calculate_inclusion_coefficient(pool['per_pool'],pool['query_pool'])
        ic['ic-sim-profile'] = ic['ic-sim-profile'] + calculate_inclusion_coefficient(pool['sim_pool'],pool['profile_pool'])
        ic['ic-per-profile'] = ic['ic-per-profile'] + calculate_inclusion_coefficient(pool['per_pool'],pool['profile_pool'])
    
    print(f"agent_model:{agent_model}")
    print(f"model:{model}")
    print(f"dataset:{dataset_name}")
    print(jaccard)
    print(ic)
    print(cnt)

if __name__ == '__main__':
    dataset_name = 'stackexchange.json'
    agent_model = 'qwen2:7b-instruct-fp16'
    agent_model = 'llama3.1:8b-instruct-fp16'
    model = 'bert_large'
    eval(dataset_name,agent_model,model)