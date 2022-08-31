from tqdm import tqdm
import torch


def encode_CODE_BERT_input(text_list, tokenizer, max_length):
    """ Use RobertaTokenizer to encode the input of CODE-BERT model"""

    all_input_ids = []
    for text in tqdm(text_list):
        input_ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt'
        )
        all_input_ids.append(input_ids)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids


def encode_CODE_T5_input(text_list, tokenizer, max_length):
    """ Use RobertaTokenizer to encode the input of CODE-T5 model"""
    all_input_ids = []
    all_attention_mask = []
    for text in tqdm(text_list):
        input_all = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt'
        )
        input_ids = input_all["input_ids"]
        attention_mask = input_all["attention_mask"]
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
    all_input_ids = torch.cat(all_input_ids, dim=0).unsqueeze(2)
    all_attention_mask = torch.cat(all_attention_mask, dim=0).unsqueeze(2)
    all_input = torch.cat((all_input_ids, all_attention_mask), dim=2)
    return all_input
