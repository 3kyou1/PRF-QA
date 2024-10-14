import json
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from .log import generate_log
from .utils import config,profile_extract



def answer_generation(query,domain,user_domain_profile,user_global_profile,language = 'en'):
    if language == 'en':
        f = open(f"{config['prompt_path']}/english/answer_generation.txt", "r")
    else:
        f = open(f"{config['prompt_path']}/chinese/answer_generation.txt", "r")
    answer_generation_prompt = f.read()
    answer_generation_prompt = answer_generation_prompt.format(query=query,domain=domain,user_domain_profile=user_domain_profile,user_global_profile=user_global_profile)
    prompt = ChatPromptTemplate.from_messages([
            ("user", "{input}")
            ])
    llm = Ollama(model=config['agent_model'],base_url = config["ollama_url"])
    chain = prompt | llm
    response = chain.invoke({"input":answer_generation_prompt})
    generate_log(query)
    generate_log(response)
    return response




if __name__ == '__main__':
    query1 = 'How do you pronounce the name "Elqosh"? It is apparently a Phoenician name.'
    domain = 'Linguistics'
    llm = Ollama(model="llama3:8b-instruct-fp16")
