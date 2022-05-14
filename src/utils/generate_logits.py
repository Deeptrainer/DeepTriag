import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from utils.data_process import get_label, generate_bug_text
from utils.constant import TEST_BATCH_SIZE
from utils.model_util import test_BERT_model, test_GPT2_model, test_XLNet_model
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config
from transformers import XLNetForSequenceClassification, XLNetTokenizer
from utils.encoder import encode_BERT_input, encode_GPT2_input, encode_XLNet_input

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda')


def test_BERT_model_generate_logits(train_data_path, test_data_path, model_path):

    """ Test BERT model by test data and record the logits of test data """

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    train_label_value = train_df['product_component_pair'].values
    test_label_value = test_df['product_component_pair'].values

    test_texts = generate_bug_text(test_df)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    test_dataset_id = encode_BERT_input(test_texts, tokenizer)

    train_label, test_label, le_name_mapping, le_index_mapping = get_label(train_label_value, test_label_value)

    test_label_tensor = torch.tensor(test_label)

    test_dataset = TensorDataset(test_dataset_id, test_label_tensor)

    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=186, output_attentions=False,
                                                          output_hidden_states=False)
    model.load_state_dict(torch.load(model_path))

    model = model.cuda()

    model, avg_test_accuracy, avg_test_loss, test_logits = test_BERT_model(model=model, test_dataloader=test_dataloader)

    return test_logits


def test_GPT2_model_generate_logits(train_data_path, test_data_path, model_path):

    """ Test GPT2 model by test data and record the logits of test data """

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    train_label_value = train_df['product_component_pair'].values
    test_label_value = test_df['product_component_pair'].values

    test_texts = generate_bug_text(test_df)

    train_label, test_label, le_name_mapping, le_index_mapping = get_label(train_label_value, test_label_value)

    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path="gpt2", num_labels=186)
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path="gpt2")
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path="gpt2", config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)

    test_dataset_id = encode_GPT2_input(test_texts, tokenizer)
    test_label_tensor = torch.tensor(test_label)
    test_dataset = TensorDataset(test_dataset_id, test_label_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model, avg_test_accuracy, avg_test_loss, test_logits = test_GPT2_model(model=model, test_dataloader=test_dataloader)
    return test_logits


def test_XLNet_model_generate_logits(train_data_path, test_data_path, model_path):

    """ Test XLNet model by test data and record the logits of test data """

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    train_label_value = train_df['product_component_pair'].values
    test_label_value = test_df['product_component_pair'].values

    test_texts = generate_bug_text(test_df)
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    test_dataset_id = encode_XLNet_input(test_texts, tokenizer)

    train_label, test_label, le_name_mapping, le_index_mapping = get_label(train_label_value, test_label_value)
    test_label_tensor = torch.tensor(test_label)
    test_dataset = TensorDataset(test_dataset_id, test_label_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=186, output_attentions=False,
                                                           output_hidden_states=False)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model, avg_test_accuracy, avg_test_loss, test_logits = test_XLNet_model(model=model,
                                                                            test_dataloader=test_dataloader)

    return test_logits
