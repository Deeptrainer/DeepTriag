import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW, GPT2Config
from transformers import get_linear_schedule_with_warmup
from utils.constant import BATCH_SIZE, LEARNING_RATE, EPOCHS
from utils.data_process import generate_bug_text, get_train_label
from utils.encoder import encode_GPT2_input
from utils.model_util import train_GPT2_one_epoch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
from tqdm import tqdm

device = torch.device('cuda')

if __name__ == '__main__':
    # read train from csv
    train_df = pd.read_csv("./data/train_data.csv")

    # combine summary and description of bugs
    train_texts = generate_bug_text(train_df)

    # get label according to product_component_pair of bugs
    train_label_value = train_df['product_component_pair'].values

    # encode label to number
    train_label = get_train_label(train_label_value)

    # config the GPT2 tokenizer and load pretrained gpt2 model
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path="gpt2", num_labels=186)
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path="gpt2")
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path="gpt2", config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)

    # encode the train data by tokenizer
    train_dataset_id = encode_GPT2_input(train_texts, tokenizer)
    torch.save(train_dataset_id, "./data/GPT2_train_torch.pt")

    # transfer label to tensor
    train_label_tensor = torch.tensor(train_label)

    # use Dataloader to read data
    train_dataset = TensorDataset(train_dataset_id, train_label_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # set optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train GPT2 model by train data
    for epoch in tqdm(range(EPOCHS)):
        model, train_eval_accuracy, avg_train_loss = train_GPT2_one_epoch(model, train_dataloader, optimizer, scheduler)

        # save model of each epoch into GPT2_model dir
        torch.save(model.state_dict(), "./saved_model/GPT2_model/{}_GPT2_model".format(epoch+1))

