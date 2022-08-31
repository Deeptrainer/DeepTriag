import numpy as np
import pandas as pd
import torch
from torch import nn
from scripts.utils.analysis_util import generate_T5_logits_matrix, generate_component_accuracy, \
    generate_T5_component_accuracy, get_list_by_index
from utils.data_process import get_label, read_json_file, write_json_file
from scripts.utils.constant import CODE_T5_WEIGHT

if __name__ == '__main__':
    # define all file paths
    train_data_path = "../data/mozilla_data/train_data.csv"
    test_data_path = "../data/mozilla_data/test_data.csv"

    tossed_index_list = np.load("../data/mozilla_data/tossed_index_list.npy")
    un_tossed_index_list = np.load("../data/mozilla_data/un_tossed_index_list.npy")

    CODE_BERT_logits_path = "../results/logits/mozilla_CODE_BERT_logits.npy"
    CODE_T5_score_matrix_path = "../results/CODE_T5_scores/mozilla_CODE_T5_scores.npy"
    CODE_T5_result_list_path = "../results/CODE_T5_results/mozilla_CODE_T5_results.json"

    # load all files from path
    CODE_BERT_logits = np.load(CODE_BERT_logits_path)
    CODE_T5_score_matrix = np.load(CODE_T5_score_matrix_path)
    CODE_T5_result_list = read_json_file(CODE_T5_result_list_path)
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    # get train and test label and encode label to number
    train_label_value = train_df['product_component_pair'].values
    test_label_value = test_df['product_component_pair'].values
    train_label, test_label, le_name_mapping, le_index_mapping = get_label(train_label_value, test_label_value)

    # Softmax CODE-BERT logits and CODE-T5 score matrix
    softmax = nn.Softmax(dim=1)
    score_torch_logits = torch.Tensor(CODE_T5_score_matrix)
    CODE_T5_softmax_score_torch_logits = CODE_T5_WEIGHT * softmax(score_torch_logits)
    CODE_BERT_logits = softmax(torch.Tensor(CODE_BERT_logits))

    # Transfer the CODE-T5 score matrix to same dimension as CODE-BERT logits
    CODE_T5_logits = generate_T5_logits_matrix(CODE_T5_softmax_score_torch_logits, le_name_mapping, CODE_T5_result_list)

    # Ensemble two models by adding CODE-T5 score matrix and CODE BERT logits
    Ensemble_logits = CODE_T5_logits + CODE_BERT_logits.numpy()

    # According to tossed bug index to get tossed bug data
    tossed_test_label = test_label[tossed_index_list]
    tossed_CODE_BERT_logits = CODE_BERT_logits[tossed_index_list, :]
    tossed_Ensemble_logits = Ensemble_logits[tossed_index_list, :]
    tossed_CODE_T5_result_list = get_list_by_index(CODE_T5_result_list, tossed_index_list)

    # According to un_tossed bug index to get un_tossed bug data
    non_tossed_test_label = test_label[un_tossed_index_list]
    non_tossed_CODE_BERT_logits = CODE_BERT_logits[un_tossed_index_list, :]
    non_tossed_Ensemble_logits = Ensemble_logits[un_tossed_index_list, :]
    non_tossed_CODE_T5_result_list = get_list_by_index(CODE_T5_result_list, un_tossed_index_list)

    # calculate component accuracy of Ensemble model
    ensemble_component_accuracy_dic = generate_component_accuracy(le_name_mapping, le_index_mapping, test_label,
                                                                  Ensemble_logits)
    tossed_ensemble_component_accuracy_dic = generate_component_accuracy(le_name_mapping, le_index_mapping,
                                                                         tossed_test_label, tossed_Ensemble_logits)
    non_tossed_ensemble_component_accuracy_dic = generate_component_accuracy(le_name_mapping, le_index_mapping,
                                                                             non_tossed_test_label,
                                                                             non_tossed_Ensemble_logits)

    # calculate component accuracy of CODE_BERT model
    CODE_BERT_component_accuracy_dic = generate_component_accuracy(le_name_mapping, le_index_mapping, test_label,
                                                                   CODE_BERT_logits.numpy())
    tossed_CODE_BERT_component_accuracy_dic = generate_component_accuracy(le_name_mapping, le_index_mapping,
                                                                          tossed_test_label,
                                                                          tossed_CODE_BERT_logits.numpy())
    non_tossed_CODE_BERT_component_accuracy_dic = generate_component_accuracy(le_name_mapping, le_index_mapping,
                                                                              non_tossed_test_label,
                                                                              non_tossed_CODE_BERT_logits.numpy())

    # calculate component accuracy of CODE_T5 model
    CODE_T5_component_accuracy_dic = generate_T5_component_accuracy(le_name_mapping, le_index_mapping, test_label,
                                                                    CODE_T5_result_list)
    tossed_CODE_T5_component_accuracy_dic = generate_T5_component_accuracy(le_name_mapping, le_index_mapping,
                                                                           tossed_test_label,
                                                                           tossed_CODE_T5_result_list)
    non_tossed_CODE_T5_component_accuracy_dic = generate_T5_component_accuracy(le_name_mapping, le_index_mapping,
                                                                               non_tossed_test_label,
                                                                               non_tossed_CODE_T5_result_list)

    # add these component accuracies to a dictionary
    component_accuracy_dic = {"Ensemble": ensemble_component_accuracy_dic,
                              "Tossed_Ensemble": tossed_ensemble_component_accuracy_dic,
                              "Non_Tossed_Ensemble": non_tossed_ensemble_component_accuracy_dic,
                              "CODE_BERT": CODE_BERT_component_accuracy_dic,
                              "Tossed_CODE_BERT": tossed_CODE_BERT_component_accuracy_dic,
                              "Non_Tossed_CODE_BERT": non_tossed_CODE_BERT_component_accuracy_dic,
                              "CODE_T5": CODE_T5_component_accuracy_dic,
                              "Tossed_CODE_T5": tossed_CODE_T5_component_accuracy_dic,
                              "Non_Tossed_CODE_T5": non_tossed_CODE_T5_component_accuracy_dic}

    # save the dictionary to json files
    write_json_file(component_accuracy_dic, "../results/Component_accuracy/mozilla_component_accuracy.json")
