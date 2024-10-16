import json
from tqdm import tqdm
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate



if __name__ == '__main__':
    per_datapath = '/home/shang/per-agent/answer/llama3:8b-instruct-fp16/per-agent/shuishan.json'
    sim_datapath = '/home/shang/per-agent/answer/llama3:8b-instruct-fp16/simple-answer/shuishan.json'
    profile_datapath = '/home/shang/per-agent/profile/llama3:8b-instruct-fp16/shuishan.json'
    extracted_path = '-'.join(per_datapath.split('/')[5:])
    with open(per_datapath, 'r') as f:
        per_answer = json.load(f)
    with open(sim_datapath, 'r') as f:
        sim_answer = json.load(f)
    with open(profile_datapath, 'r') as f:
        profile = json.load(f)
    llm = Ollama(model="qwen2:7b-instruct-fp16")
    # with open(extracted_path, 'r') as f:
    #     chinese = json.load(f)
    chinese = {}
    for id ,qa in tqdm(per_answer.items()):
        # f = open(f"/home/shang/per-agent/answer/translate.txt", "r")
        # chinese_prompt = f.read()
        # query_prompt = chinese_prompt.format(query = qa['query'])
        # sim_prompt = chinese_prompt.format(query = sim_answer[id]['answer'])
        # per_prompt = chinese_prompt.format(query = qa['answer'])
        # profile_prompt = chinese_prompt.format(query = profile[id]['user_profile']['global'])
        # prompt = ChatPromptTemplate.from_messages([
        #         ("user", "{input}")
        #         ])
        # chain = prompt | llm
        # query = qa['query']
        # sim = chain.invoke({"input":sim_prompt})
        # per = chain.invoke({"input":per_prompt})
        # pro = chain.invoke({"input":profile_prompt})
        query = qa['query']
        sim = sim_answer[id]['answer']
        per = qa['answer']
        pro = profile[id]['user_profile']['global']
        chinese[id] = {'profile':pro,'query':query,'sim':sim,'per':per}
        json_string = json.dumps(chinese, indent=4)
        with open(extracted_path, "w") as json_file:
            json_file.write(json_string)