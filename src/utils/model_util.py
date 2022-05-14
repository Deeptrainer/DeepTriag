import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device('cuda')


def flat_accuracy(preds, labels):

    """ calculate accuracy scores by predictions result and labels """

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


def train_BERT_one_epoch(model, train_dataloader, optimizer, scheduler):

    """ The method is used to train the BERT model in one epoch """

    model.train()
    total_loss = 0
    train_eval_accuracy = 0
    for step, batch in tqdm(enumerate(train_dataloader)):
        model.zero_grad()
        outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                        labels=batch[1].to(device=device, dtype=torch.long))
        loss, train_logits = outputs[:2]
        total_loss += loss.item()

        train_logits = train_logits.detach().cpu().numpy()
        train_label_ids = batch[1].to('cpu').numpy()
        train_eval_accuracy += flat_accuracy(train_logits, train_label_ids)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    train_eval_accuracy /= len(train_dataloader)
    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Train loss     : {avg_train_loss}')
    print(f'Train Accuracy     : {train_eval_accuracy:.4f}')

    return model, train_eval_accuracy, avg_train_loss


def test_BERT_model(model, test_dataloader):

    """ The method is used to test BERT model and return the accuracy, loss and logits """

    model.eval()
    total_test_loss = 0
    total_eval_accuracy = 0
    logits_list = []
    print("test start")
    for i, batch in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            test_outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                                 labels=batch[1].to(device=device, dtype=torch.long))
            loss, logits = test_outputs[:2]
            total_test_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = batch[1].to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            logits_list.append(logits)

    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_accuracy = total_eval_accuracy / len(test_dataloader)

    logits_matrix = np.concatenate(logits_list)
    # print(logits_matrix.shape)

    print(f'Test loss: {avg_test_loss}')
    print(f'Test Accuracy: {avg_test_accuracy:.4f}')
    print('\n')
    return model, avg_test_accuracy, avg_test_loss, logits_matrix


def train_GPT2_one_epoch(model, train_dataloader, optimizer, scheduler):

    """ The method is used to train the GPT2 model in one epoch """

    model.train()
    total_loss = 0
    train_eval_accuracy = 0
    for step, batch in tqdm(enumerate(train_dataloader)):
        model.zero_grad()
        outputs = model(input_ids=batch[0][:, :, 0].to(device), attention_mask=batch[0][:, :, 1].to(device),
                        labels=batch[1].to(device, dtype=torch.long))

        loss = outputs.loss
        train_logits = outputs.logits
        total_loss += loss.item()

        train_logits = train_logits.detach().cpu().numpy()
        train_label_ids = batch[1].to('cpu').numpy()
        train_eval_accuracy += flat_accuracy(train_logits, train_label_ids)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    train_eval_accuracy /= len(train_dataloader)
    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Train loss     : {avg_train_loss}')
    print(f'Train Accuracy     : {train_eval_accuracy:.4f}')
    return model, train_eval_accuracy, avg_train_loss


def test_GPT2_model(model, test_dataloader):

    """ The method is used to test GPT2 model and return the accuracy, loss and logits """

    model.eval()
    total_test_loss = 0
    total_eval_accuracy = 0
    logits_list = []
    print("test start")
    for i, batch in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            test_outputs = model(input_ids=batch[0][:, :, 0].to(device),
                                 attention_mask=batch[0][:, :, 1].to(device),
                                 labels=batch[1].to(device, dtype=torch.long))
            loss = test_outputs.loss
            logits = test_outputs.logits
            total_test_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = batch[1].to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            logits_list.append(logits)

    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_accuracy = total_eval_accuracy / len(test_dataloader)
    logits_matrix = np.concatenate(logits_list)

    print(f'Test loss: {avg_test_loss}')
    print(f'Test Accuracy: {avg_test_accuracy:.4f}')
    print('\n')
    return model, avg_test_accuracy, avg_test_loss, logits_matrix


def train_XLNet_one_epoch(model, train_dataloader, optimizer, scheduler):

    """ The method is used to train the XLNet model in one epoch """

    model.train()
    total_loss = 0
    train_eval_accuracy = 0
    for step, batch in tqdm(enumerate(train_dataloader)):
        model.zero_grad()
        outputs = model(input_ids=batch[0][:, :, 0].to(device), attention_mask=batch[0][:, :, 2].to(device),
                        token_type_ids=batch[0][:, :, 1].to(device), labels=batch[1].to(device, dtype=torch.long))
        loss = outputs.loss
        train_logits = outputs.logits
        total_loss += loss.item()

        train_logits = train_logits.detach().cpu().numpy()
        train_label_ids = batch[1].to('cpu').numpy()
        train_eval_accuracy += flat_accuracy(train_logits, train_label_ids)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    train_eval_accuracy /= len(train_dataloader)
    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Train loss     : {avg_train_loss}')
    print(f'Train Accuracy     : {train_eval_accuracy:.4f}')
    return model, train_eval_accuracy, avg_train_loss


def test_XLNet_model(model, test_dataloader):

    """ The method is used to test XLNet model and return the accuracy, loss and logits """

    model.eval()
    total_test_loss = 0
    total_eval_accuracy = 0
    logits_list = []
    print("test start")
    for i, batch in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            test_outputs = model(input_ids=batch[0][:, :, 0].to(device), attention_mask=batch[0][:, :, 2].to(device),
                                 token_type_ids=batch[0][:, :, 1].to(device),
                                 labels=batch[1].to(device, dtype=torch.long))
            loss = test_outputs.loss
            logits = test_outputs.logits
            total_test_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = batch[1].to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            logits_list.append(logits)

    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_accuracy = total_eval_accuracy / len(test_dataloader)
    logits_matrix = np.concatenate(logits_list)

    print(f'Test loss: {avg_test_loss}')
    print(f'Test Accuracy: {avg_test_accuracy:.4f}')
    print('\n')
    return model, avg_test_accuracy, avg_test_loss, logits_matrix

