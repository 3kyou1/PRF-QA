import argparse
import copy
import json
import os
import time
from tqdm import tqdm
from transformers import AutoTokenizer

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

def compress(text,model_name,base_url,language = 'en'):
    if language =='en':
        f = open(f"../../../prompt/english/compression_instructions_llmlingua2.txt", "r")
    else:
        f = open(f"../../../prompt/chinese/compression_instructions_llmlingua2.txt", "r")
    compress_prompt = f.read()
    compress_prompt = compress_prompt.format(text_to_compress=text)
    
    prompt = ChatPromptTemplate.from_messages([
            ("user", "{input}")
            ])
    llm = Ollama(model=model_name,base_url = base_url)
    chain = prompt | llm
    response = chain.invoke({"input":compress_prompt})
    print(response)
    return response

def chunk_origin(origin_text):
    origin_list = []
    origin_token_ids = tokenizer.encode(origin_text)
    end_token_ids = set(tokenizer.encode(".") + tokenizer.encode("\n"))
    n = len(origin_token_ids)
    st = 0
    while st < n:
        if st + args.chunk_size > n - 1:
            chunk = tokenizer.decode(origin_token_ids[st:n])
            origin_list.append(chunk)
            break
        else:
            ed = st + args.chunk_size
            for j in range(0, ed - st):
                if origin_token_ids[ed - j] in end_token_ids:
                    ed = ed - j
                    break
            chunk = tokenizer.decode(origin_token_ids[st : ed + 1])
            origin_list.append(chunk)
            st = ed + 1
    return origin_list

def configure_parser():
    parser = argparse.ArgumentParser(description="compress any prompt.")

    parser.add_argument("--model_name", help="llm used to compress", default="qwen2:72b-instruct-fp16_compress_zh")
    
    parser.add_argument("--base_url", help="ollama deploy url", default="http://localhost:6666")
    
    parser.add_argument("--model_path", help="llm path", default="/home/LLMs/Qwen/Qwen2-72B-Instruct")
    
    parser.add_argument("--load_origin_from", help="dataset used to compress", required=True)

    parser.add_argument("--save_path", help="path to save results", required=True)

    parser.add_argument(
    "--load_prompt_from", help="", default="../../../code/compress/data_collection/compression_instructions.json"
    )

    parser.add_argument(
    "--language", help="", default="zh"
    )

    parser.add_argument("--chunk_size", type=int, default=-1)

    return parser

if __name__ == '__main__':
    parser = configure_parser()
    args = parser.parse_args()


    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    data = json.load(open(args.load_origin_from))

    print(f"num data: {len(data)}")

    results = {}
    results_list = []
    total_time = 0

    if os.path.exists(args.save_path):
        results = json.load(open(args.save_path))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    for sample in tqdm(data):
        idx = int(sample["idx"])
        origin = copy.deepcopy(sample[args.language])
        if origin is None:
            continue
        if idx in results or str(idx) in results:
            print(f"{idx}-th sample is processed")
            continue

        t = time.time()
        if isinstance(origin, list):
            if args.chunk_size > 0:
                chunk_list = []
                for j, document in enumerate(origin):
                    ori_list = chunk_origin(document)
                    chunk_list.extend(ori_list)
                origin = chunk_list
        else:
            origin = [origin]
            if args.chunk_size > 0:
                origin = chunk_origin(origin[0])
        print(f"num chunk: {len(origin)}")
        comp_list = []
        for j, chunk in enumerate(origin):
            comp = compress(chunk,args.model_name,args.base_url,args.language)
            comp_list.append(comp)
        assert len(origin) == len(comp_list)
        comp = "".join(comp_list)

        total_time += time.time() - t
        new_sample = copy.deepcopy(sample)
        new_sample[f"compress_{args.language}"] = comp
        assert len(origin) == len(comp_list)
        new_sample["prompt_list"] = origin[:]
        new_sample["compressed_prompt_list"] = comp_list[:]
        results[idx] = new_sample
        json.dump(
            results,
            open(args.save_path, "w", encoding="utf8"),
            indent=4,
            ensure_ascii=False,
        )
    print(args.save_path, total_time)
    json.dump(
        results, open(args.save_path, "w", encoding="utf8"), indent=4, ensure_ascii=False
    )


