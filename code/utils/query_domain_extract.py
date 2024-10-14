import json
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from .log import generate_log
from .utils import config,profile_extract


def query_domain_extract(query,language = 'en'):
    if language == 'en':
        f = open(f"{config['prompt_path']}/english/query_domain_extract.txt", "r")
    else:
        f = open(f"{config['prompt_path']}/chinese/query_domain_extract.txt", "r")
    query_domain_extract_prompt = f.read()
    query_domain_extract_prompt = query_domain_extract_prompt.format(query=query)
    query_domain_extract_prompt = query_domain_extract_prompt.replace("{","{{").replace("}","}}")
    prompt = ChatPromptTemplate.from_messages([
            ("user", "{input}")
            ])
    llm = Ollama(model=config['agent_model'],base_url = config["ollama_url"])
    chain = prompt | llm
    response = chain.invoke({"input":query_domain_extract_prompt})
    generate_log(response)
    res = profile_extract(response)
    generate_log(res)
    return res



if __name__ == '__main__':
    query1 = 'How do you pronounce the name "Elqosh"? It is apparently a Phoenician name.'
    llm = Ollama(model="llama3:8b-instruct-fp16")
    res = query_domain_extract(query1,llm)
    print(res)