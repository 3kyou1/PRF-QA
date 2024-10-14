from functools import lru_cache
from llmlingua import PromptCompressor
from .utils import config
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List
from .log import generate_log
import string

@lru_cache(maxsize=1)
def get_compressor(model_name,device_map):
    if 'stackexchange' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=2, ignore_mismatched_sizes=True
        )
        model.to(device_map)
        return tokenizer, model
    if 'bert' in model_name:
        return PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map = device_map
        )
    return PromptCompressor(
        model_name=model_name,
        device_map = device_map
    )
    



def split_text_into_chunks(text: str, chunk_size: int = 512) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(config['PromptCompressorModel'])
    encoded = tokenizer.encode(text)
    chunks = [encoded[i:i + chunk_size] for i in range(0, len(encoded), chunk_size)]
    text_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    return text_chunks

def compress_prompt(original_prompt,model_name,device_map, rate=0.6):
    if 'stackexchange' in model_name:
        device = torch.device(f"{device_map}" if torch.cuda.is_available() else "cpu")
        tokenizer,model = get_compressor(model_name,device_map)
        

        inputs = tokenizer(original_prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=-1)
        predictions = predictions.cpu()

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
        pred_labels = predictions[0].tolist()


        def is_punctuation(token):
            return token in string.punctuation

        selected_tokens = [token for token, label in zip(tokens, pred_labels) 
                       if (label == 1 or is_punctuation(token)) and token not in tokenizer.all_special_tokens]
        selected_text = tokenizer.convert_tokens_to_string(selected_tokens)
        
        selected_text = re.sub(r'\s+', ' ', selected_text).strip()
        generate_log(selected_text)
        return selected_text
    
    if 'bert' in model_name:
        compressor = get_compressor(model_name,device_map)
        results = compressor.compress_prompt_llmlingua2(
            original_prompt,
            rate=rate,
            force_tokens=['\n', '.', '!', '?', ','],
            chunk_end_tokens=['.', '\n'],
            return_word_label=True,
            drop_consecutive=True
        )
        generate_log(results['compressed_prompt'])
        return results['compressed_prompt']
    else:
        compressor = get_compressor(model_name,device_map)
        compressed_prompt = compressor.compress_prompt(original_prompt, instruction="", question="", rate = rate)
        return compressed_prompt['compressed_prompt']

