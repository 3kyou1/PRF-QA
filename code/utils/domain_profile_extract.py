import json
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from .log import generate_log
from .utils import config,profile_extract


def domain_profile_extract(query,domain,language = 'en'):
    if language =='en':
        f = open(f"{config['prompt_path']}/english/domain_profile_extract.txt", "r")
    else:
        f = open(f"{config['prompt_path']}/chinese/domain_profile_extract.txt", "r")
    domain_profile_extract_prompt = f.read()
    domain_profile_extract_prompt = domain_profile_extract_prompt.format(query=query,domain=domain)
    domain_profile_extract_prompt = domain_profile_extract_prompt.replace("{","{{").replace("}","}}")
    prompt = ChatPromptTemplate.from_messages([
            ("user", "{input}")
            ])
    llm = Ollama(model=config['agent_model'],base_url = config["ollama_url"])
    chain = prompt | llm
    response = chain.invoke({"input":domain_profile_extract_prompt})
    generate_log(response)
    res = profile_extract(response)
    generate_log(res)
    return res



if __name__ == '__main__':
    query1 = 'How do you pronounce the name "Elqosh"? It is apparently a Phoenician name.'
    domain = 'Linguistics'
    llm = Ollama(model="llama3:8b-instruct-fp16")
    res = domain_profile_extract(query1,domain,llm)
    print(res)