import numpy as np
import torch
from sklearn import metrics, preprocessing
from torch import nn


def generate_component_accuracy(le_name_mapping, le_index_mapping, test_label, logits):
    """ According to the logits to generate the accuracy of each component"""

    predictions = np.argmax(logits, axis=1).flatten()
    component_accuracy_df = {}
    for key in le_name_mapping.keys():
        component_accuracy_df[key] = {"component_num": 0, "correct_num": 0, "Wrong_components": {},
                                      "component_accuracy": 0}
    correct_num = 0
    for i in range(len(logits)):
        label = le_index_mapping[test_label[i]]
        prediction = le_index_mapping[predictions[i]]
        component_accuracy_df[label]["component_num"] += 1
        if prediction == label:
            component_accuracy_df[label]["correct_num"] += 1
            correct_num += 1
        else:
            if prediction not in component_accuracy_df[label]["Wrong_components"].keys():
                component_accuracy_df[label]["Wrong_components"][prediction] = 1
            else:
                component_accuracy_df[label]["Wrong_components"][prediction] += 1

    for key in component_accuracy_df.keys():
        component_details = component_accuracy_df[key]
        if component_details["component_num"] == 0:
            component_details["component_accuracy"] = 0
        else:
            component_details["component_accuracy"] = component_details["correct_num"] / component_details[
                "component_num"]

    print(correct_num / len(test_label))

    return component_accuracy_df


def generate_T5_component_accuracy(le_name_mapping, le_index_mapping, test_label, results_list):
    """ According to the results of CODE-T5 model (or T5 model) to generate the accuracy of each component"""

    predictions = []
    for i in range(len(results_list)):
        predictions.append(results_list[i][0])
    component_accuracy_df = {}
    for key in le_name_mapping.keys():
        component_accuracy_df[key] = {"component_num": 0, "correct_num": 0, "Wrong_components": {},
                                      "component_accuracy": 0}
    correct_num = 0
    for i in range(len(results_list)):
        label = le_index_mapping[test_label[i]]
        prediction = predictions[i]
        component_accuracy_df[label]["component_num"] += 1
        if prediction == label:
            component_accuracy_df[label]["correct_num"] += 1
            correct_num += 1
        else:
            if prediction not in component_accuracy_df[label]["Wrong_components"].keys():
                component_accuracy_df[label]["Wrong_components"][prediction] = 1
            else:
                component_accuracy_df[label]["Wrong_components"][prediction] += 1

    for key in component_accuracy_df.keys():
        component_details = component_accuracy_df[key]
        if component_details["component_num"] == 0:
            component_details["component_accuracy"] = 0
        else:
            component_details["component_accuracy"] = component_details["correct_num"] / component_details[
                "component_num"]
    print(correct_num / len(results_list))
    return component_accuracy_df


def generate_T5_top_k_accuracy_with_MRR(result_list, test_label, le_index_mapping, all_test_label, logits_matrix):
    """ According to the results of CODE-T5 model (or T5 model) to generate top-k accuracy and MRR"""
    accuracy_list = []
    for top_k in (1, 3, 5, 10):
        print("Top {} Accuracy: ".format(top_k))
        correct_num = 0
        for i in range(len(result_list)):
            result = result_list[i]
            top_k_result = result[0:top_k]
            if le_index_mapping[test_label[i]] in top_k_result:
                correct_num += 1
        test_accuracy = correct_num / len(test_label)
        print(test_accuracy)
        print()
        accuracy_list.append(np.round(test_accuracy, 3))

    encoder = preprocessing.OneHotEncoder()
    encoder.fit(all_test_label.reshape(-1, 1))
    one_hot_test_label = encoder.transform(test_label.reshape(-1, 1)).toarray()
    MRR_score = metrics.label_ranking_average_precision_score(one_hot_test_label, logits_matrix)
    print("MRR: {}".format(MRR_score))
    accuracy_list.append(round(MRR_score, 3))

    return accuracy_list


def softmax_matrix(input_matrix):
    """transfer logits by softmax"""

    softmax = nn.Softmax(dim=1)
    input_torch_matrix = torch.Tensor(input_matrix)
    output_softmax_matrix = softmax(input_torch_matrix).numpy()
    return output_softmax_matrix


def get_list_by_index(result_list, index_list):
    """Split the list by a index list"""
    filtered_result_list = []
    for i in range(len(index_list)):
        filtered_result_list.append(result_list[index_list[i]])
    return filtered_result_list


def calculate_classify_top_k_accuracy_with_MRR(test_labels, logits_matrix, all_test_label):
    """According to the logits of model to generate top-k accuracy and MRR"""
    accuracy_list = []
    for top_k in (1, 3, 5, 10):
        num_count = 0
        for i in range(len(logits_matrix)):
            logits_i = logits_matrix[i, :]
            index = np.argpartition(logits_i, -top_k)[-top_k:]
            if int(test_labels[i]) in index:
                num_count += 1
        top_k_accuracy = num_count / len(test_labels)
        print("TOP {} TEST ACCURACY: {}".format(top_k, top_k_accuracy, 3))
        accuracy_list.append(round(top_k_accuracy, 3))

    encoder = preprocessing.OneHotEncoder()

    encoder.fit(all_test_label.reshape(-1, 1))
    one_hot_test_label = encoder.transform(test_labels.reshape(-1, 1)).toarray()
    MRR_score = metrics.label_ranking_average_precision_score(one_hot_test_label, logits_matrix)
    print("MRR: {}".format(MRR_score))
    print()
    accuracy_list.append(round(MRR_score, 3))
    return accuracy_list


def generate_T5_logits_matrix(t5_score_matrix, le_name_mapping, t5_result_list):
    """Transfer score and results of CODE-T5 model to a matrix"""
    T5_logits_list = []
    for i in range(len(t5_score_matrix)):
        T5_vector = np.zeros(len(le_name_mapping))
        T5_score = t5_score_matrix[i, :]
        T5_result = t5_result_list[i]
        for j in range(len(T5_result)):
            prediction_component = T5_result[j]
            if prediction_component in le_name_mapping.keys():
                component_index = le_name_mapping[prediction_component]
                T5_vector[component_index] = T5_score[j]
        T5_logits_list.append(T5_vector)
    T5_logits = np.stack(T5_logits_list)
    return T5_logits
