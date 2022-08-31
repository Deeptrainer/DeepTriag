import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from scripts.utils.constant import EPOCHS, MAX_LENGTH, MOZILLA_NUM_LABELS, CODE_BERT_TEST_BATCH_SIZE
from scripts.utils.data_process import generate_bug_text, get_label
from scripts.utils.encoder import encode_CODE_BERT_input
from scripts.utils.model_util import test_CODE_BERT_model
import numpy as np

device = torch.device('cuda')

if __name__ == '__main__':
    model_path = "../saved_model/CODE_BERT_model/{}_CODE_BERT_model".format(EPOCHS)
    # read train and test data from csv
    train_df = pd.read_csv("../data/mozilla_data/train_data.csv")
    test_df = pd.read_csv("../data/mozilla_data/test_data.csv")

    # get label according to product_component_pair of bugs
    train_label_value = train_df['product_component_pair'].values
    test_label_value = test_df['product_component_pair'].values

    # combine summary and description of bugs to generate input text
    test_texts = generate_bug_text(test_df)

    # RobertaTokenizer is used to tokenize the train data
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base', do_lower_case=True)

    # encode the train data by tokenizer
    test_dataset_id = encode_CODE_BERT_input(test_texts, tokenizer, max_length=MAX_LENGTH)

    # transfer label to tensor
    train_label, test_label, le_name_mapping, le_index_mapping = get_label(train_label_value, test_label_value)

    # use Dataloader to load data
    test_label_tensor = torch.tensor(test_label)
    test_dataset = TensorDataset(test_dataset_id, test_label_tensor)

    test_dataloader = DataLoader(test_dataset, batch_size=CODE_BERT_TEST_BATCH_SIZE)

    model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=MOZILLA_NUM_LABELS,
                                                             output_attentions=False, output_hidden_states=False)
    model.load_state_dict(torch.load(model_path))

    model = model.cuda()

    model, avg_test_accuracy, avg_test_loss, test_logits = test_CODE_BERT_model(model=model, test_dataloader=test_dataloader)

    np.save("../results/logits/mozilla_CODE_BERT_logits.npy", test_logits)