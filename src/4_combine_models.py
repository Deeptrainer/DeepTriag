from utils.generate_logits import test_BERT_model_generate_logits, test_GPT2_model_generate_logits, \
    test_XLNet_model_generate_logits
from utils.constant import EPOCHS
from torch import nn
import torch
import numpy as np
from utils.model_util import flat_accuracy
from utils.data_process import get_label
import pandas as pd
import time

if __name__ == '__main__':
    start_time = time.time()
    train_data_path = "./data/train_data.csv"
    test_data_path = "./data/test_data.csv"
    # three Models of final epoch
    BERT_model_path = "./saved_model/BERT_model/{}_BERT_model".format(EPOCHS)
    GPT2_model_path = "./saved_model/GPT2_model/{}_GPT2_model".format(EPOCHS)
    XLNet_model_path = "./saved_model/XLNet_model/{}_XLNet_model".format(EPOCHS)
    # test three models and get logits
    BERT_logits = test_BERT_model_generate_logits(train_data_path, test_data_path, BERT_model_path)
    GPT2_logits = test_GPT2_model_generate_logits(train_data_path, test_data_path, GPT2_model_path)
    XLNet_logits = test_XLNet_model_generate_logits(train_data_path, test_data_path, XLNet_model_path)

    # put these logits into list
    class_logits_list = [BERT_logits, GPT2_logits, XLNet_logits]

    # use softmax to deal with logits
    softmax = nn.Softmax(dim=1)
    softmax_class_logits_list = []
    for i in range(len(class_logits_list)):
        torch_logits = torch.Tensor(class_logits_list[i])
        softmax_torch_logits = softmax(torch_logits)
        softmax_class_logits_list.append(softmax_torch_logits.detach().numpy())
        pred_flat = np.argmax(softmax_torch_logits, axis=1).flatten().detach().numpy()

    # get test label by train and test data
    train_df = pd.read_csv("./data/train_data.csv")
    test_df = pd.read_csv("./data/test_data.csv")
    train_label_value = train_df['product_component_pair'].values
    test_label_value = test_df['product_component_pair'].values
    train_label, test_label, le_name_mapping, le_index_mapping = get_label(train_label_value, test_label_value)

    # add up each softmax_logits to generate a combined logits to combine each model
    start_new_array = 0
    for i in range(len(softmax_class_logits_list)):
        start_new_array = start_new_array + softmax_class_logits_list[i]

    # get the top-1 accuracy of combination logits
    print("The accuracy of combined model")
    print(flat_accuracy(start_new_array, test_label))

    # save the logits
    np.save("./combination_result/final_logits.npy", start_new_array)

