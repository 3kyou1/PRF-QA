import os
import json


# Load config 
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)


def extract_json(text):
    start = text.find("{")
    if start == -1:
        return text
    
    stack = []
    for i, char in enumerate(text[start:], start=start):
        if char == "{":
            stack.append(char)
        elif char == "}":
            stack.pop()
        
        if not stack:
            end = i + 1
            break
    else:
        return text
    
    return text[start:end]


def profile_extract(llm_response):
    llm_response = extract_json(llm_response)
    str = llm_response.replace('"',"")
    start = str.find(":")
    end = str.find(":",start+1)
    substring = str[start+2:end]
    end = substring.find("\n")
    profile = substring[:end-1]
    return profile

def Metrics_conversion(metrics):
    if 'disagree' in metrics:
        return 0
    elif 'neutral' in metrics:
        return 1
    else:
        return 2
    
def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)