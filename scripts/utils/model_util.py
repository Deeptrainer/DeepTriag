import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch

device = torch.device('cuda')


def flat_accuracy(preds, labels):
    """ calculate accuracy scores by predictions result and labels """

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


def train_CODE_BERT_one_epoch(model, train_dataloader, optimizer, scheduler):
    """ The method is used to train the CODE_BERT model in one epoch """

    # make model to train status
    model.train()
    total_loss = 0
    logits_list = []
    label_list = []

    for step, batch in tqdm(enumerate(train_dataloader)):
        model.zero_grad()
        # input a batch of train data to model
        outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                        labels=batch[1].to(device=device, dtype=torch.long))

        # get loss and logits from outputs of model
        loss, train_logits = outputs[:2]
        total_loss += loss.item()

        train_logits = train_logits.detach().cpu().numpy()
        train_label_ids = batch[1].to('cpu').numpy()

        # do back propagation of model
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # add logits and label to list
        logits_list.append(train_logits)
        label_list.append(train_label_ids)

    # concatenate vector list to matrix
    logits_matrix = np.concatenate(logits_list)
    label_vector = np.concatenate(label_list)

    # calculate prediction accuracy and average loss
    train_eval_accuracy = flat_accuracy(logits_matrix, label_vector)
    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Train loss     : {avg_train_loss}')
    print(f'Train Accuracy     : {train_eval_accuracy:.4f}')

    return model, train_eval_accuracy, avg_train_loss


def test_CODE_BERT_model(model, test_dataloader):
    """ The method is used to test CODE_BERT model and return the accuracy, loss and logits """
    # make model in eval status
    model.eval()
    total_test_loss = 0
    total_eval_accuracy = 0
    logits_list = []
    print("test start")
    label_list = []
    for i, batch in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            # get loss and logits from outputs of model
            test_outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                                 labels=batch[1].to(device=device, dtype=torch.long))
            loss, logits = test_outputs[:2]
            total_test_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = batch[1].to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

            # add logits and label to list
            logits_list.append(logits)
            label_list.append(label_ids)

    # concatenate vector list to matrix
    logits_matrix = np.concatenate(logits_list)
    label_vector = np.concatenate(label_list)

    # calculate prediction accuracy and average loss
    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_accuracy = flat_accuracy(logits_matrix, label_vector)

    print(f'Test loss: {avg_test_loss}')
    print(f'Test Accuracy: {avg_test_accuracy:.4f}')
    print('\n')
    return model, avg_test_accuracy, avg_test_loss, logits_matrix


def train_CODE_T5_one_epoch(model, train_dataloader, optimizer, scheduler):
    """ The method is used to train the CODE_T5 model in one epoch """

    # make model to train status
    model.train()
    total_loss = 0
    train_eval_accuracy = 0
    for step, batch in tqdm(enumerate(train_dataloader)):
        model.zero_grad()
        # input a batch of train data to model
        outputs = model(input_ids=batch[0][:, :, 0].to(device), attention_mask=batch[0][:, :, 1].to(device),
                        labels=batch[1][:, :, 0].to(device))
        loss = outputs.loss
        total_loss += loss.item()

        # do back propagation of model
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    # calculate the average loss
    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Train loss     : {avg_train_loss}')
    return model, train_eval_accuracy, avg_train_loss


def test_CODE_T5_model(model, test_dataloader, tokenizer, top_k):
    """ The method is used to test CODE-T5 model and return the accuracy, loss , score and results """

    model.eval()
    total_correction = 0
    print("Test start")
    score_list = []
    output_list = []
    all_number = 0
    for i, batch in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            # use generate() in transformer to generate 10 results from fine-tuned CODE-T5 model
            generated_output = model.generate(input_ids=batch[0][:, :, 0].to(device), num_beams=top_k,
                                              num_return_sequences=top_k,
                                              output_scores=True, return_dict_in_generate=True)

            # if top k > 1 will generate sequences_scores
            if top_k != 1:
                s_score = generated_output.sequences_scores.to('cpu').numpy()
            # if top k = 1, the score is 1
            else:
                s_score = 1
            batch_size = batch[0].size()[0]
            # transfer a vector to a matrix with dimension((batch_size, top_k)
            s_score_matrix = s_score.reshape(batch_size, top_k)
            generated_output = generated_output.sequences.to('cpu')
            label = batch[1][:, :, 0].to('cpu')
            score_list.append(s_score_matrix)
            # decode the generated result from tokens to text string
            prediction_list = tokenizer.batch_decode(generated_output, skip_special_tokens=True)
            for k in range(len(label)):
                prediction_sub = prediction_list[k * top_k:(k+1) * top_k]
                output_list.append(prediction_sub)
                prediction = prediction_list[k * top_k]
                actual_label = tokenizer.decode(label[k], skip_special_tokens=True)
                if prediction == actual_label:
                    total_correction += 1
                all_number += 1

    score_matrix = np.concatenate(score_list)
    avg_test_accuracy = total_correction / all_number

    print(f'Test Accuracy: {avg_test_accuracy:.4f}')
    print('\n')
    return model, avg_test_accuracy, score_matrix, output_list

