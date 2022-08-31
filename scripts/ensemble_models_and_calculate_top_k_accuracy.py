import numpy as np
import pandas as pd
import torch
from torch import nn
from scripts.utils.analysis_util import generate_T5_logits_matrix, calculate_classify_top_k_accuracy_with_MRR, \
    get_list_by_index, generate_T5_top_k_accuracy_with_MRR
from utils.data_process import get_label, read_json_file
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
    tossed_CODE_T5_logits = CODE_T5_logits[tossed_index_list, :]
    tossed_Ensemble_logits = Ensemble_logits[tossed_index_list, :]
    tossed_CODE_T5_result_list = get_list_by_index(CODE_T5_result_list, tossed_index_list)

    # According to un_tossed bug index to get un_tossed bug data
    non_tossed_test_label = test_label[un_tossed_index_list]
    non_tossed_CODE_BERT_logits = CODE_BERT_logits[un_tossed_index_list, :]
    non_tossed_CODE_T5_logits = CODE_T5_logits[un_tossed_index_list, :]
    non_tossed_Ensemble_logits = Ensemble_logits[un_tossed_index_list, :]
    non_tossed_CODE_T5_result_list = get_list_by_index(CODE_T5_result_list, un_tossed_index_list)

    # calculate CODE-T5 tok-k accuracy and MRR.
    T5_top_k_accuracy = generate_T5_top_k_accuracy_with_MRR(CODE_T5_result_list, test_label, le_index_mapping,
                                                            test_label, CODE_T5_logits)
    tossed_T5_top_accuracy = generate_T5_top_k_accuracy_with_MRR(tossed_CODE_T5_result_list, tossed_test_label,
                                                                 le_index_mapping, test_label, tossed_CODE_T5_logits)
    non_tossed_T5_top_accuracy = generate_T5_top_k_accuracy_with_MRR(non_tossed_CODE_T5_result_list,
                                                                     non_tossed_test_label, le_index_mapping,
                                                                     test_label, non_tossed_CODE_T5_logits)

    # calculate CODE-BERT tok-k accuracy and MRR.
    classification_top_k_accuracy = calculate_classify_top_k_accuracy_with_MRR(test_label, CODE_BERT_logits, test_label)
    tossed_classification_top_k_accuracy = calculate_classify_top_k_accuracy_with_MRR(tossed_test_label,
                                                                                      tossed_CODE_BERT_logits,
                                                                                      test_label)
    non_tossed_classification_top_k_accuracy = calculate_classify_top_k_accuracy_with_MRR(non_tossed_test_label,
                                                                                          non_tossed_CODE_BERT_logits,
                                                                                          test_label)

    # calculate Ensemble models tok-k accuracy and MRR.
    ensemble_top_k_accuracy = calculate_classify_top_k_accuracy_with_MRR(test_label, Ensemble_logits, test_label)
    tossed_ensemble_top_k_accuracy = calculate_classify_top_k_accuracy_with_MRR(tossed_test_label,
                                                                                tossed_Ensemble_logits, test_label)
    non_tossed_ensemble_top_k_accuracy = calculate_classify_top_k_accuracy_with_MRR(non_tossed_test_label,
                                                                                    non_tossed_Ensemble_logits,
                                                                                    test_label)

    # add these accuracy and MRR to a dataframe
    top_k_accuracy_dic = {"ENSEMBLE": ensemble_top_k_accuracy,
                          "TOSSED_ENSEMBLE": tossed_ensemble_top_k_accuracy,
                          "NON_TOSSED_ENSEMBLE": non_tossed_ensemble_top_k_accuracy,
                          "CODE_BERT": classification_top_k_accuracy,
                          "TOSSED_CODE_BERT": tossed_classification_top_k_accuracy,
                          "NON_TOSSED_CODE_BERT": non_tossed_classification_top_k_accuracy,
                          "CODE_T5": T5_top_k_accuracy,
                          "TOSSED_CODE_T5": tossed_T5_top_accuracy,
                          "NON_TOSSED_CODE_T5": non_tossed_T5_top_accuracy}

    top_k_accuracy_df = pd.DataFrame(top_k_accuracy_dic)
    top_k_accuracy_df.index = ['TOP_1', 'TOP_3', 'TOP_5', 'TOP_10', "MRR"]
    top_k_accuracy_df = top_k_accuracy_df.T
    print(top_k_accuracy_df)

    # save the dataframe to xlsx file
    top_k_accuracy_df.to_excel("../results/Top_K_accuracy/mozilla_Top_k_accuracy.xlsx")