from tqdm import tqdm
from transformers import AutoTokenizer
from .log import generate_log
from .query_domain_extract import query_domain_extract
from .domain_profile_extract import domain_profile_extract
from .domain_profile_synthesize import domain_profile_synthesize
from .global_profile_generation import global_profile_generation
from .promptcompressor import compress_prompt

tokenizer = AutoTokenizer.from_pretrained("/home/LLMs/Meta-Llama-3.1-8B-Instruct", trust_remote_code=True)


class User:
    def __init__(self, user_id,querys,language,compress_model,device):
        self.user_id = user_id
        self.querys = querys
        self.profile = {}
        self.last_qa = {}
        self.language = language
        self.compress_model = compress_model
        self.device = device
        self.raw_tokens = 0
        self.com_tokens = 0
    
    def profile_process(self):
        for i in tqdm(range(len(self.querys)-1), desc="Processing queries"):
            query = self.querys[i]
            generate_log({query})
            domain = query_domain_extract(query,self.language)
            self.domain_profile_generation(query,domain)
            self.user_global_profile_generation(domain)

    def domain_profile_generation(self,query,domain):
        domain_profile = domain_profile_extract(query,domain,self.language)
        self.raw_tokens = self.raw_tokens+len(tokenizer.encode(domain_profile))
        
        if "bert" in self.compress_model:
            domain_profile = compress_prompt(domain_profile,self.compress_model,self.device)
            self.com_tokens = self.com_tokens+len(tokenizer.encode(domain_profile))
        else :
            domain_profile = compress_prompt(domain_profile,self.compress_model,self.device)
        
        if domain in self.profile:
            domain_profile = domain_profile_synthesize(domain_profile,self.profile[domain],self.language)
            self.raw_tokens = self.raw_tokens+len(tokenizer.encode(domain_profile))
            if "bert" in self.compress_model:
                domain_profile = compress_prompt(domain_profile, self.compress_model, self.device)
                self.com_tokens = self.com_tokens+len(tokenizer.encode(domain_profile))
            else:
                domain_profile = compress_prompt(domain_profile,self.compress_model,self.device)
            self.profile[domain] = domain_profile
        else:
            self.profile[domain] = domain_profile
    
    def user_global_profile_generation(self,domain):
        if 'global' in self.profile:
            self.profile['global'] = global_profile_generation(self.profile,self.language)
        else:
            self.profile['global'] = global_profile_generation(self.profile,self.language)