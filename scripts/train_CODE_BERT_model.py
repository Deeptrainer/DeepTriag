import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaForSequenceClassification
from scripts.utils.constant import CODE_BERT_BATCH_SIZE, CODE_BERT_LEARNING_RATE, EPOCHS, MAX_LENGTH, MOZILLA_NUM_LABELS
from scripts.utils.data_process import generate_bug_text, get_train_label
from scripts.utils.encoder import encode_CODE_BERT_input
from scripts.utils.model_util import train_CODE_BERT_one_epoch

device = torch.device('cuda')

if __name__ == '__main__':
    # read train data from csv
    train_df = pd.read_csv("../data/mozilla_data/train_data.csv")

    # combine summary and description of bugs
    train_texts = generate_bug_text(train_df)

    # get label according to product_component_pair of bugs
    train_label_value = train_df['product_component_pair'].values

    # encode label to number
    train_label = get_train_label(train_label_value)

    # RobertaTokenizer is used to tokenize the train data
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base', do_lower_case=True)

    # encode the train data by tokenizer
    train_dataset_id = encode_CODE_BERT_input(train_texts, tokenizer, max_length=MAX_LENGTH)

    # transfer label to tensor
    train_label_tensor = torch.tensor(train_label)

    # use Dataloader to load data
    train_dataset = TensorDataset(train_dataset_id, train_label_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=CODE_BERT_BATCH_SIZE, shuffle=True)

    # load the pretrained CODE_BERT Model: codebert-base
    model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=MOZILLA_NUM_LABELS,
                                                             output_attentions=False, output_hidden_states=False)
    model = model.cuda()

    # set optimizer and scheduler for model
    optimizer = AdamW(model.parameters(), lr=CODE_BERT_LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train CODE_BERT model by train data
    for epoch in range(EPOCHS):
        print("The epoch is {}: \n".format(epoch + 1))

        model, train_eval_accuracy, avg_train_loss = train_CODE_BERT_one_epoch(model, train_dataloader, optimizer, scheduler)

        # Save model of each epoch into CODE_BERT_model dir
        torch.save(model.state_dict(),
                   "../saved_model/CODE_BERT_model/{}_CODE_BERT_model".format(epoch + 1))
