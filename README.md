# DeepTriag

Dataset link: https://drive.google.com/file/d/1ZknWSOH_E2JotV6dZ2IiipdaPIQKBOZ3/view?usp=sharing

Directory prepare

1. Create the data directory under the script directory and put mozilla data and eclipse data into the data directory from the dataset link.
2. Create the saved_model directory under the script directory and create CODE_BERT_model directory  and CODE_T5_model under  saved_model directory.
3. Create the results directory under the script directory and create logits, CODE_T5_results, CODE_T5_scores, Top_K_accuracy directory, Component_accuracy under results directory.

Hyperparameter settings

1. Learning rate, batch size, epoch number can be set in the script/utils/constant.py 

Run steps

1. Run train_CODE_BERT_model.py to fine-tune CODE_BERT model. The model of each epoch is saved in the script/saved_model/CODE_BERT_model directory.
2. Run train_CODE_T5_model.py to fine-tune CODE_T5 model. The model of each epoch is saved in the script/saved_model/CODE_T5_model directory.
3. Run test_CODE_BERT_model.py to test the fine-tuned CODE_BERT model of final epoch and get logits. 
4. Run test_CODE_T5_model.py to test the fine-tuned CODE_T5 model of final epoch and get logits. 
5. Run ensemble_models_and_calculate_top_k_accuracy.py to ensemble fine-tuned CODE BERT and CODE T5 models and get calculate top-k accuracy.
6. Run ensemble_models_and_calculate_component_accuracy.py to ensemble fine-tuned CODE BERT and CODE T5 models and get calculate component accuracy.

Note

The above steps is used to fine-tune and test Mozilla data. If you want to fine-tune and test the Eclipse data, please change "mozilla_data" in the path to read files in the code to "eclipse_data".
