import json
import os
import sys
from tqdm import tqdm
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from .utils import config,profile_extract
from .log import generate_log

def Relatedness(user_profile,query,answer):
    f = open(f"{config['prompt_path']}/eval/Relatedness.txt", "r")    
    Relatedness_prompt = f.read()
    Relatedness_prompt = Relatedness_prompt.format(user_profile = user_profile,query=query,answer=answer).replace("{","{{").replace("}","}}")
    llm = Ollama(model=config['evaluate_model'],base_url = config["ollama_url"])
    prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}")
        ])
    chain = prompt | llm
    response = chain.invoke({"input":Relatedness_prompt})
    generate_log(response)
    res = profile_extract(response)
    return res

def Correctness(query,answer):
    f = open(f"{config['prompt_path']}/eval/Correctness.txt", "r")    
    Correctness_prompt = f.read()
    Correctness_prompt = Correctness_prompt.format(query=query,answer=answer).replace("{","{{").replace("}","}}")
    llm = Ollama(model=config['evaluate_model'],base_url = config["ollama_url"])
    prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}")
        ])
    chain = prompt | llm
    response = chain.invoke({"input":Correctness_prompt})
    generate_log(response)
    res = profile_extract(response)
    return res

def Comprehensiveness(query,answer):
    f = open(f"{config['prompt_path']}/eval/Comprehensiveness.txt", "r")
    Comprehensiveness_prompt = f.read()
    Comprehensiveness_prompt = Comprehensiveness_prompt.format(query=query,answer=answer).replace("{","{{").replace("}","}}")
    llm = Ollama(model=config['evaluate_model'],base_url = config["ollama_url"])
    prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}")
        ])
    chain = prompt | llm
    response = chain.invoke({"input":Comprehensiveness_prompt})
    generate_log(response)
    res = profile_extract(response)
    return res

