from sklearn.preprocessing import LabelEncoder
import numpy as np


def generate_bug_text(data_df):

    """ combine the bug summary and bug description """

    data_df['bug_description'] = data_df['bug_description'].replace(np.nan, " ")
    bug_summary_description = data_df["bug_summary"] + "." + data_df["bug_description"]
    bug_text = bug_summary_description.values
    return bug_text

def get_train_label(train_component):

    """ Use LabelEncoder to encode the train label """

    le = LabelEncoder()
    encode_model = le.fit(train_component)
    train_label = encode_model.transform(train_component)
    return train_label


def get_label(train_component, test_component):

    """ Use LabelEncoder to encode the train and test label """

    le = LabelEncoder()
    encode_model = le.fit(train_component)
    train_label = encode_model.transform(train_component)
    test_label = encode_model.transform(test_component)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    le_index_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    return train_label, test_label, le_name_mapping, le_index_mapping
