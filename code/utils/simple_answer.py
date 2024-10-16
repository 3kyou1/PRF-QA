
import os
import json
from tqdm import tqdm
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# Load config 
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)


def simple_answer(dataset):
    datapath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'dataset',dataset)

    answer_datapath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'answer',config['agent_model'],'simple-answer',dataset)
    
    print(answer_datapath)
    llm = Ollama(model=config['agent_model'],base_url = config["ollama_url"])
    prompt = ChatPromptTemplate.from_messages([
            ("user", "{input}")
            ])
    chain = prompt | llm
    with open(datapath, 'r') as f:
        data = json.load(f)
    simple_answer = {}
    for id,queries in tqdm(data.items(), desc="Processing dataset"):
        qa={}
        query  = queries[-1]
        response = chain.invoke({"input":query})
        qa['query'] = query
        qa['answer'] = response
        simple_answer[id] = qa
        json_string = json.dumps(simple_answer, indent=4)
        with open(answer_datapath, "w") as json_file:
            json_file.write(json_string)

if __name__ == '__main__':
    dataset_list = ['wildchat.json','cs101.json','stackexchange.json']
    dataset_list = ['stackexchange.json']
    for dataset in dataset_list:
        simple_answer(dataset)

    
    
    
    