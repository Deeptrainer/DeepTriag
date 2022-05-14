import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from utils.data_process import generate_bug_text, get_train_label
from utils.encoder import encode_BERT_input
from utils.constant import BATCH_SIZE, LEARNING_RATE, EPOCHS
from utils.model_util import train_BERT_one_epoch
from tqdm import tqdm

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda')

if __name__ == '__main__':
    # read train data from csv
    train_df = pd.read_csv("./data/train_data.csv")

    # combine summary and description of bugs
    train_texts = generate_bug_text(train_df)

    # get label according to product_component_pair of bugs
    train_label_value = train_df['product_component_pair'].values

    # encode label to number
    train_label = get_train_label(train_label_value)

    # BertTokenizer is used to tokenize the train data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # encode the train data by tokenizer
    train_dataset_id = encode_BERT_input(train_texts, tokenizer)
    torch.save(train_dataset_id, "./data/BERT_train_torch.pt")

    # transfer label to tensor
    train_label_tensor = torch.tensor(train_label)

    # use Dataloader to read data
    train_dataset = TensorDataset(train_dataset_id, train_label_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # load the pretrained BERT Model: bert-base-uncased
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=186, output_attentions=False,
                                                          output_hidden_states=False)
    model = model.cuda()

    # set optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train BERT model by train data
    for epoch in tqdm(range(EPOCHS)):
        model, train_eval_accuracy, avg_train_loss = train_BERT_one_epoch(model, train_dataloader, optimizer, scheduler)

        # Save model of each epoch into BERT_model dir
        torch.save(model.state_dict(), "./saved_model/BERT_model/{}_BERT_model".format(epoch + 1))
