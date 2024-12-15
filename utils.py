import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_path: str, flash_attn: bool):
    if flash_attn:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            use_cache=False,
            use_flash_attention_2=flash_attn,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.config.pretraining_tp = 1
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            use_cache=False,
            device_map="auto",
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token     
    tokenizer.padding_side = "left"
    
    return model, tokenizer


def load_json_file(file_path: str) -> dict:
    with open(file_path) as file:
        data = json.load(file)
    print(f">>> JSON file lenghth: {len(data)}")
    print(f">>> an example from JSON file: \n{data[0]}\n")
    
    return data


def save_json_file(file_path: str, save_data) -> None:
    with open(file_path, 'w', encoding='utf-8') as outf:
        json.dump(save_data, outf, ensure_ascii=False, indent=1)
        

def get_probabilities(prompt, model, tokenizer):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    input_ids_list = [int(x) for x in input_ids[0]]
    
    output = model(input_ids)
    logits = output.logits         
    logits = logits.squeeze(dim=0) 
    
    logits_log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = [float(logits_log_softmax[i, input_ids_list[i+1]]) for i in range(len(input_ids_list)-1)]
    
    return all_prob

def get_ll(prompt, model, tokenizer):
    all_prob = get_probabilities(prompt, model, tokenizer)
    return -np.mean(all_prob[1:])

def get_ll_list(dataset, model, tokenizer, SAVE_DOT):
    ll_list = [round(get_ll(x, model, tokenizer), SAVE_DOT) for x in tqdm(dataset)]
    return ll_list

    
def tokenize_and_mask(text, span_length, pct, ceil_pct=True):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + 1 * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, max(len(tokens) - span_length, 1))
        end = start + span_length
        search_start = max(0, start - 1)
        search_end = min(len(tokens), end + 1)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text

def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]
    
def replace_masks(texts, mask_model, mask_tokenizer):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
    outputs = mask_model.generate(**tokens, max_length=500, do_sample=True, top_p=0.9, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

def extract_fills(texts):
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    pattern = re.compile(r"<extra_id_\d+>")
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills):
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    texts = [" ".join(x) for x in tokens]
    return texts

def perturb_texts_(texts, mask_model, mask_tokenizer, span_length, pct, ceil_pct=False):
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(masked_texts, mask_model, mask_tokenizer)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
    
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(masked_texts, mask_model, mask_tokenizer)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1
    return perturbed_texts

def perturb_texts(texts, mask_model, mask_tokenizer, span_length, pct, ceil_pct=False):
    chunk_size = 500

    outputs = []
    for i in range(0, len(texts), chunk_size):
        # outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
        outputs.extend(perturb_texts_([texts[i:i + chunk_size]], mask_model, mask_tokenizer, span_length, pct, ceil_pct=ceil_pct))
    return outputs

def get_aver_chunk_size(model_path: str, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    token_len_list = [len(tokenizer(x, return_tensors="pt", truncation=True).input_ids.cuda()[0]) for x in tqdm(dataset)]
    return int(np.mean(token_len_list))

def resize_prompt(model_path: str, dataset, chunk_token_size: int):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset_processed = [tokenizer.decode((tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda())[0][: chunk_token_size], skip_special_tokens=True) for prompt in dataset]
    return dataset_processed