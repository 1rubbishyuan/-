import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR
from model.model import LstmModel
from dataloader.dataloader import get_dataloader
import json
from typing import List


def train(
    model: nn.Module,
    dataloader: DataLoader,
    e_poch,
    device,
    lr: float = 0.001,
    early_stopping: int = 3,
):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.001)
    validation_set = get_dataloader("validation", 1, False, model.name)
    v_rightrate_list = []
    f_score_list = []
    loss_list = []
    best_rightrate = 0.5
    record = 0
    for i in range(e_poch):
        for j, data in tqdm(enumerate(dataloader)):
            input = data[0].to(device)
            target = data[1].to(device)

            optimizer.zero_grad()
            output = model(input)
            loss = F.cross_entropy(output, target)
            # loss_list.append(loss)
            loss.backward()
            loss_list.append(loss.item())
            optimizer.step()
            if j % 20 == 0:
                print(f"training loss: {loss}----- epoch:{i},batch:{j}")
        v_rightrate, f_score = validation(model, validation_set)
        # print(f"record: {record} {v_rightrate_list[-1]}")
        if len(v_rightrate_list) > 0 and v_rightrate <= best_rightrate:
            record += 1
        else:
            record = 0
        v_rightrate_list.append(v_rightrate)
        f_score_list.append(f_score)
        model.train()
        if v_rightrate > best_rightrate:
            best_rightrate = v_rightrate
            torch.save(model, f"{model.name}_{lr}_{dataloader.batch_size}.pth")
        if record >= early_stopping:
            break
    with open(f"../results/results_{model.name}.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "model": model.name,
                    "lr": lr,
                    "batchsize": dataloader.batch_size,
                    "v_rightrate_list": v_rightrate_list,
                    "v_fscore_list": f_score_list,
                    "loss_list": loss_list,
                }
            )
            + "\n"
        )


def train_lstm(
    model: LstmModel,
    dataloader: DataLoader,
    e_poch,
    device,
    lr: float = 0.001,
    early_stopping: int = 3,
):
    optimizer = Adam(model.parameters(), lr=lr)
    # scheduler = StepLR(optimizer, 2, 0.5)
    validation_set = get_dataloader("validation", 1, False, model.name)
    v_rightrate_list = []
    f_score_list = []
    loss_list = []
    best_rightrate = 0.5
    record = 0

    criterion = nn.BCELoss()
    for i in range(e_poch):
        hidden = model.initial_hc(dataloader.batch_size)
        for j, data in tqdm(enumerate(dataloader)):
            # print(hidden)
            hidden = tuple([e.data for e in hidden])
            # print(hidden)
            input = data[0].to(device)
            target = data[1].to(device)
            optimizer.zero_grad()
            output, hidden = model(input, hidden)
            loss = criterion(output, target)
            loss.backward()
            loss_list.append(loss.item())
            optimizer.step()
            if j % 20 == 0:
                print(f"training loss: {loss}----- epoch:{i},batch:{j}")
        # scheduler.step()
        v_rightrate, f_score = validation_lstm(model, validation_set)
        if len(v_rightrate_list) > 0 and v_rightrate <= best_rightrate:
            record += 1
        else:
            record = 0
        v_rightrate_list.append(v_rightrate)
        f_score_list.append(f_score)
        model.train()
        if v_rightrate > best_rightrate:
            best_rightrate = v_rightrate
            torch.save(model, f"{model.name}_{lr}_{dataloader.batch_size}.pth")
        if record >= early_stopping:
            break
    with open("../results/results_lstm.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "model": model.name,
                    "lr": lr,
                    "batchsize": dataloader.batch_size,
                    "v_rightrate_list": v_rightrate_list,
                    "v_fscore_list": f_score_list,
                    "loss_list": loss_list,
                }
            )
            + "\n"
        )


def validation(model: nn.Module, dataloader: DataLoader):
    model.eval()
    sum = 0
    right = 0
    tp = 0
    fp = 0
    p = 0
    for j, data in tqdm(enumerate(dataloader)):
        sum += 1
        input = data[0].to(model.device)
        target = data[1].squeeze(0)
        if torch.all(target == torch.tensor([1, 0])):
            p += 1
        output = model(input).squeeze(0)
        if output[0] > output[1]:
            output = torch.tensor([1, 0])
        else:
            output = torch.tensor([0, 1])
        if torch.all(output == target):
            right += 1
            if torch.all(output == torch.tensor([1, 0])):
                tp += 1
        else:
            if torch.all(output == torch.tensor([1, 0])):
                fp += 1

    print(f"validation_right: {right / sum}")
    print(f"f-score: {1 / ((tp + fp) / (tp + 0.0001) + p / (tp + 0.0001) + 0.0001)}")
    return right / sum, 1 / ((tp + fp) / (tp + 0.0001) + p / (tp + 0.0001) + 0.0001)


def validation_lstm(model: LstmModel, dataloader: DataLoader):
    model.eval()
    sum = 0
    right = 0
    tp = 0
    fp = 0
    p = 0
    for j, data in tqdm(enumerate(dataloader)):
        sum += 1
        hidden = model.initial_hc(1)
        input = data[0].to(model.device)
        target = data[1].squeeze(0)
        if torch.all(target == torch.tensor([0])):
            p += 1
        output, new_hidden = model(input, hidden)
        # print(output)
        if output[0] > 0.5:
            output = torch.tensor([1])
        else:
            output = torch.tensor([0])
        if torch.all(output == target):
            right += 1
            if torch.all(output == torch.tensor([0])):
                tp += 1
        else:
            if torch.all(output == torch.tensor([0])):
                fp += 1
    print(f"validation_right: {right / sum}")
    print(f"f-score: {1 / ((tp + fp) / (tp + 0.0001) + p / (tp + 0.0001) + 0.0001)}")
    return right / sum, 1 / ((tp + fp) / (tp + 0.0001) + p / (tp + 0.0001) + 0.0001)


def test_vote(model_list: List[nn.Module], dataloader: DataLoader):
    sum = 0
    right = 0
    tp = 0
    fp = 0
    p = 0
    for data in tqdm(dataloader):
        sum += 1
        pos = 0
        neg = 0
        for model in model_list:
            input = data[0].to(model.device)
            try:
                output = model(input).squeeze(0)
                if output[0] > output[1]:
                    pos += 1
                else:
                    neg += 1
            except:
                hidden = model.initial_hc(1)
                output, new_hidden = model(input, hidden)
                if output[0] < 0.5:
                    pos += 1
                else:
                    neg += 1
            print(output)
        target = data[1].squeeze(0)
        if torch.all(target == torch.tensor([1, 0])):
            p += 1
        if pos > neg:
            output = torch.tensor([1, 0])
        else:
            output = torch.tensor([0, 1])
        print(f"{output}:{target}")
        if torch.all(output == target):
            right += 1
            if torch.all(output == torch.tensor([1, 0])):
                tp += 1
        else:
            if torch.all(output == torch.tensor([1, 0])):
                fp += 1

    print(f"validation_right: {right / sum}")
    print(f"f-score: {1 / ((tp + fp) / (tp + 0.0001) + p / (tp + 0.0001) + 0.0001)}")
    return right / sum, 1 / ((tp + fp) / (tp + 0.0001) + p / (tp + 0.0001) + 0.0001)
