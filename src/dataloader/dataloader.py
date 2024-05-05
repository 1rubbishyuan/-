from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from word2vec import model, word2index
import os
import torch
import numpy as np

train_data_path = "../Dataset/train.txt"
validation_data_path = "../Dataset/validation.txt"
test_data_path = "../Dataset/test.txt"
root_data_path = "../Dataset"
max_length = 50


class SentenceDataset(Dataset):
    def __init__(self, type="train"):
        path = os.path.join(root_data_path, f"{type}.txt")
        self.dataset = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.dataset.append(line.split())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        feature = self.dataset[index][1:]
        label = int(self.dataset[index][0])
        return feature, label


def collate_fn(batch):
    features, labels = zip(*batch)
    # features_length = [len(feature) for feature in features]
    # max_length = max(features_length)
    final_labels = []
    for label in labels:
        if label == 1:
            final_labels.append([0, 1])
        elif label == 0:
            final_labels.append([1, 0])
    final_labels = torch.Tensor(final_labels)
    final_features = []
    for feature in features:
        tmp_features = []
        for word in feature:
            try:
                tmp_features.append(word2index[word])
            except:
                tmp_features.append(1)
        final_features.append(torch.tensor(tmp_features, dtype=torch.long))
    padded_feature = [
        torch.nn.functional.pad(feature, (0, max_length - len(feature)), "constant", 0)[
            :max_length
        ]
        for feature in final_features
    ]
    final_features = pad_sequence(padded_feature, batch_first=True, padding_value=0)
    return final_features, final_labels


def collate_fn_lstm(batch):
    features, labels = zip(*batch)
    # features_length = [len(feature) for feature in features]
    final_labels = torch.Tensor(labels)
    final_features = []
    for feature in features:
        tmp_features = []
        for word in feature:
            try:
                tmp_features.append(word2index[word])
            except:
                tmp_features.append(1)
        final_features.append(torch.tensor(tmp_features, dtype=torch.long))
    padded_feature = [
        torch.nn.functional.pad(feature, (0, max_length - len(feature)), "constant", 0)[
            :max_length
        ]
        for feature in final_features
    ]
    final_features = pad_sequence(padded_feature, batch_first=True, padding_value=0)
    return final_features, final_labels


def get_dataloader(type: str, batch_size: int, shuffle: bool, model_type: str = "lstm"):
    dataset = SentenceDataset(type)
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle,
        drop_last=True,
        collate_fn=collate_fn_lstm if model_type == "lstm" else collate_fn,
    )
    return dataloader


def init_model(old_model):
    vocab_size = len(word2index)
    initial_embdding = np.random.uniform(-1, 1, (vocab_size, 50))

    for word in word2index:
        if word in model:
            initial_embdding[word2index[word]] = model[word]

    old_model.embedding.weight.data.copy_(torch.from_numpy(initial_embdding))
    return old_model
