from tqdm import tqdm
import torch


def encode_BERT_input(text_list, tokenizer):

    """ Use BertTokenizer to encode the input of BERT model"""

    all_input_ids = []
    for text in tqdm(text_list):
        input_ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_tensors='pt'
        )
        all_input_ids.append(input_ids)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids


def encode_GPT2_input(text_list, tokenizer):

    """ Use GPT2Tokenizer to encode the input of GPT2 model"""

    i = 0
    all_input_ids = []
    all_attention_mask = []
    for text in tqdm(text_list):
        input_all = tokenizer(
            text,
            max_length=512,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_tensors='pt'
        )
        input_ids = input_all["input_ids"]
        attention_mask = input_all["attention_mask"]
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        i += 1
    all_input_ids = torch.cat(all_input_ids, dim=0).unsqueeze(2)
    all_attention_mask = torch.cat(all_attention_mask, dim=0).unsqueeze(2)
    all_input = torch.cat((all_input_ids, all_attention_mask), dim=2)
    return all_input


def encode_XLNet_input(text_list, tokenizer):

    """ Use XLNetTokenizer to encode the input of GPT2 model"""

    i = 0
    all_input_ids = []
    all_token_type_ids = []
    all_attention_mask = []
    for text in tqdm(text_list):
        input_all = tokenizer(
                        text,
                        add_special_tokens = True,
                        max_length = 512,
                        pad_to_max_length = True,
                        return_tensors = 'pt'
                   )
        input_ids = input_all["input_ids"]
        token_type_ids = input_all["token_type_ids"]
        attention_mask = input_all["attention_mask"]
        all_input_ids.append(input_ids)
        all_token_type_ids.append(token_type_ids)
        all_attention_mask.append(attention_mask)
        i += 1
    all_input_ids = torch.cat(all_input_ids, dim=0).unsqueeze(2)
    all_token_type_ids = torch.cat(all_token_type_ids, dim=0).unsqueeze(2)
    all_attention_mask = torch.cat(all_attention_mask, dim=0).unsqueeze(2)
    all_input = torch.cat((all_input_ids, all_token_type_ids, all_attention_mask), dim=2)
    return all_input