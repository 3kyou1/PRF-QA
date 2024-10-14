import json
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from .log import generate_log
from .utils import config,profile_extract


def global_profile_generation(user_profile,language = 'en'):
    if language == 'en':
        f = open(f"{config['prompt_path']}/english/global_profile_generation.txt", "r")
    else:
        f = open(f"{config['prompt_path']}/chinese/global_profile_generation.txt", "r")
    global_profile_generation_prompt = f.read()
    global_profile_generation_prompt = global_profile_generation_prompt.format(user_profile=user_profile)
    global_profile_generation_prompt = global_profile_generation_prompt.replace("{","{{").replace("}","}}")
    prompt = ChatPromptTemplate.from_messages([
            ("user", "{input}")
            ])
    
    llm = Ollama(model=config['agent_model'],base_url = config["ollama_url"])
    chain = prompt | llm
    response = chain.invoke({"input":global_profile_generation_prompt})
    generate_log(response)
    res = profile_extract(response)
    if len(res) == 0:
        global_profile_generation(user_profile,language)
    generate_log(res)
    return res




if __name__ == '__main__':
    query1 = 'How do you pronounce the name "Elqosh"? It is apparently a Phoenician name.'
    domain = 'Linguistics'
    llm = Ollama(model="llama3:8b-instruct-fp16")
