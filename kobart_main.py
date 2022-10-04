import torch
import streamlit as st
import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
import re
from tqdm import tqdm
import requests
import json
from utils import *
import argparse
import os


def load_model():
    tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
    model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
    return model, tokenizer

model, tokenizer = load_model()


def summ_bart(text, beams, bound, min_length, ngram_size):
    temp_str = ""
    summ_str = ""
    split_lines = split_textline(text)
    for i in range(len(split_lines)):
        if i == len(split_lines)-1:
            temp_str += split_lines[i]
            raw_input_ids = tokenizer.encode(temp_str)
            input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
            if len(input_ids)>=1000:
                input_ids = input_ids[:999]
            summary_ids = model.generate(torch.tensor([input_ids]).to(device), num_beams=beams, max_length=512,min_length=min_length, no_repeat_ngram_size=ngram_size, eos_token_id=1)
            summ_str += tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        if len(temp_str)> bound:
            temp_str += split_lines[i]
            raw_input_ids = tokenizer.encode(temp_str)
            input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
            if len(input_ids)>=1000:
                input_ids = input_ids[:999]
            summary_ids = model.generate(torch.tensor([input_ids]).to(device), num_beams=beams, max_length=512, min_length=min_length, no_repeat_ngram_size=ngram_size,eos_token_id=1)
            summ_str += tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
            temp_str = ""
            continue
        temp_str += split_lines[i] 
    return summ_str

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_read_path', required=True, help='Enter json file path')
    parser.add_argument('--json_storage_path', required=True)
    
    args = parser.parse_args()
    print(args.json_read_path)
    print(args.json_storage_path)

    json_fn = args.json_read_path
    storage_fn = args.json_storage_path
    
    with open(json_fn, 'r', encoding='utf8') as json_file:
        refine_result = json.load(json_file)
    
    dir_name = '/'.join(storage_fn.split('/')[:-1])
    os.makedirs(dir_name, exist_ok=True)
    
    #After generating of summary datas, and store the data into storage path 
    
    for j in tqdm(range(len(refine_result))):
        for i in range(len(refine_result[j]['clue'])):
            pororo_txt = refine_result[j]['clue'][i]['sentence']
            origin_txt = refine_result[j]['refine_data'][i]
            pororo_str = ''.join(pororo_txt)
            processed_origin = remove_html(origin_txt)
            cnt = 0
            try:
                prefix = summ_bart(pororo_str, 6, min(len(pororo_str),300), 20,0)
                postfix = summ_bart(processed_origin, 6, min(len(processed_origin),300), 20,1)
                summ = prefix+postfix
                minimum_len = 10
                while (len(summ_bart(summ, 6, min(len(summ),300), minimum_len, 1))/len(origin_txt)) < 0.1 or calcul_simil(origin_txt, summ_bart(summ, 6,  min(len(summ),300),minimum_len,1)) > 0.8:
                    if cnt>10:
                        break
                    cnt += 1
                    minimum_len += 10

                summ = summ_bart(summ, 6, min(len(summ),300), minimum_len,1)
                similarity = calcul_simil(origin_txt, summ)
                length_ratio = len(summ)/len(origin_txt)
                summ_dict = {'summary':summ,'similarity':similarity, 'len_ratio':length_ratio}
                refine_result[j]['abs_summ'].append(summ_dict)
            except:
                print("fail!")
                continue
        with open(storage_fn, 'w', encoding='utf8') as json_file:
            json.dump(refine_result, json_file, ensure_ascii=False)
    
    