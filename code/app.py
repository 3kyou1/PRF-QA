import json
import os
from tqdm import tqdm
from utils.utils import config,Metrics_conversion,ensure_directory_exists
from utils.user import User
from utils.answer_generation import answer_generation
from utils.query_domain_extract import query_domain_extract
from utils.domain_profile_extract import domain_profile_extract
from utils.domain_profile_synthesize import domain_profile_synthesize
from utils.evaluate import Relatedness,Correctness,Comprehensiveness
import argparse

def profile(dataset_name,language,compress_model="",device = "cuda:2"):
    datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'dataset',dataset_name)
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
        
    print(f"user_profile_datapath:{user_profile_datapath}")
    print(f"compress_model:{compress_model}")
    print(f"device:{device}")
    
    with open(datapath, 'r') as f:
        dataset = json.load(f)

    if os.path.exists(user_profile_datapath):
        with open(user_profile_datapath, 'r') as f:
            User_dataset = json.load(f)
    else:
        User_dataset = {}
    for id, querys in tqdm(dataset.items(), desc="User Profile Generation"):
        if id in User_dataset:
            print('-----------------')
            continue
        user = User(id,querys,language,compress_model,device)
        user.profile_process()
        user_dict = {
            "user_id": user.user_id,
            "user_profile": user.profile,
            "user_raw_tokens":user.raw_tokens,
            "user_com_tokens":user.com_tokens
        }
        User_dataset[id] = user_dict
        json_string = json.dumps(User_dataset, indent=4)
        with open(user_profile_datapath, "w") as json_file:
            json_file.write(json_string)

def answer(dataset_name,language,compress_model=""):
    datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'dataset',dataset_name)
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
    with open(user_profile_datapath, 'r') as f:
        user_profile_dataset = json.load(f)
    with open(datapath, 'r') as f:
        dataset = json.load(f)

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
        
    print(f"answer_datapath:{answer_datapath}")

    if os.path.exists(answer_datapath):
        with open(answer_datapath, 'r') as f:
            per_agent_answer = json.load(f)
    else:
        per_agent_answer = {}
    
    
    for id,queries in tqdm(dataset.items(), desc="Answer Generation"):
        if id in per_agent_answer:
            continue
        if id not in user_profile_dataset:
            continue
        user_profile = user_profile_dataset[id]
        qa={}
        query = queries[-1]
        domain = query_domain_extract(query,language)
        cur_domain_profile = domain_profile_extract(query,domain,language)
        if domain in user_profile['user_profile']:
            his_domain_profile = user_profile['user_profile'][domain]
            domain_profile = domain_profile_synthesize(cur_domain_profile,his_domain_profile,language)
        else:
            domain_profile = cur_domain_profile
        global_profile = user_profile['user_profile']['global']
        answer = answer_generation(query,domain,domain_profile,global_profile,language)
        qa['query'] = query
        qa['answer'] = answer
        per_agent_answer[id] = qa
        json_string = json.dumps(per_agent_answer, indent=4)
        with open(answer_datapath, "w") as json_file:
            json_file.write(json_string)
        
def evaluate():
    user_profile_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'profile',config['dataset'])
    qa_datapath = os.path.join(config['QA_datapath'],config['dataset'])
    with open(user_profile_datapath, 'r') as f:
        user_profile_dataset = json.load(f)
    with open(qa_datapath, 'r') as f:
        qa_set = json.load(f)
    eval_dataset = {}
    for id,qa in tqdm(qa_set.items(), desc="Evaluate Answer"):
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
        with open(config['dataset'], "w") as json_file:
            json_file.write(json_string)

def configure_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--compress_model', type=str, default='/home/hang_su/per-agent/model/xlm-roberta-large-stackexchange',
                        help='compress_model')
    parser.add_argument('--cuda_device', type=str, default='cuda:2',
                        help='GPU device')
    parser.add_argument('--dataset', type=str, default='allenai.json',
                        help='datset')
    return parser

if __name__ == '__main__':
    parser = configure_parser()
    args = parser.parse_args()

    dataset_list = ['cs101.json','wildchat.json','stackexchange.json']
    for dataset in dataset_list:
        language = ""
        if dataset =='shuishan.json':
            language = "zh"
        else:
            language = "en"
        profile(dataset,language, args.compress_model,args.cuda_device)
        answer(dataset,language, args.compress_model)
            
            
    