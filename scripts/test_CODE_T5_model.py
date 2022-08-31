import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from scripts.utils.data_process import generate_bug_text, write_json_file
from scripts.utils.constant import EPOCHS, MAX_LENGTH, CODE_T5_TEST_BATCH_SIZE
from scripts.utils.model_util import test_CODE_T5_model
from scripts.utils.encoder import encode_CODE_T5_input
from transformers import T5ForConditionalGeneration, RobertaTokenizer

device = torch.device('cuda')

if __name__ == '__main__':
    model_path = "../saved_model/CODE_T5_model/{}_CODE_T5_model".format(EPOCHS)

    test_df = pd.read_csv("../data/mozilla_data/test_data.csv")

    test_texts = generate_bug_text(test_df)

    test_label_value = test_df['product_component_pair'].values

    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base', do_lower_case=True)

    test_dataset_id = encode_CODE_T5_input(test_texts, tokenizer, max_length=MAX_LENGTH)
    test_label_id = encode_CODE_T5_input(test_label_value, tokenizer, max_length=20)

    test_dataset = TensorDataset(test_dataset_id, test_label_id)
    test_dataloader = DataLoader(test_dataset, batch_size=CODE_T5_TEST_BATCH_SIZE, shuffle=False)

    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base', output_attentions=False,
                                                       output_hidden_states=False)

    model = model.cuda()

    model.load_state_dict(torch.load(model_path))

    model, avg_test_accuracy, score_matrix, output_list = test_CODE_T5_model(model=model,
                                                                             test_dataloader=test_dataloader,
                                                                             tokenizer=tokenizer, top_k=10)

    write_json_file(output_list, "../results/CODE_T5_results/mozilla_CODE_T5_results.json")
    np.save("../results/CODE_T5_scores/mozilla_CODE_T5_scores.npy", score_matrix)