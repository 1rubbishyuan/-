import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from word2vec import word2index, model


class TextCnn(nn.Module):
    def __init__(
        self, word_vector_dim, num_filter, filter_sizes, device, vocabulary_size
    ):
        super(TextCnn, self).__init__()
        self.name = "textcnn"
        self.device = device
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=50)
        self.cnn_list = [
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filter,
                kernel_size=(size, word_vector_dim),
            )
            for size in filter_sizes
        ]
        self.fully_connect = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(
                in_features=num_filter * len(filter_sizes),
                out_features=2 * num_filter * len(filter_sizes),
            ),
            nn.ReLU(),
            nn.Linear(
                2 * num_filter * len(filter_sizes), num_filter * len(filter_sizes)
            ),
            nn.ReLU(),
            nn.Linear(num_filter * len(filter_sizes), 2),
        )

    def forward(self, x):
        """
        shape:[batchsize,max_sentence_length,word_vector_dim]
        """
        x = x.unsqueeze(1).to(self.device)
        x = self.embedding(x)
        # print()
        ans = [F.relu(cnn.to(self.device)(x)).squeeze(3) for cnn in self.cnn_list]
        ans = [F.max_pool1d(a, a.shape[2]).squeeze(2) for a in ans]
        ans = torch.cat(ans, dim=1).to(self.device)
        ans = self.fully_connect(ans)
        ans = F.softmax(ans, dim=1)
        # print(ans)
        return ans


class LstmModel(nn.Module):
    def __init__(
        self, word_vector_dim, hidden_layer_dim, layer_nums, device, vocabulary_size
    ):
        super(LstmModel, self).__init__()
        self.name = "lstm"
        self.hidden_layer_dim = hidden_layer_dim
        self.layer_nums = layer_nums
        self.device = device
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=50)
        self.lstm = nn.LSTM(
            word_vector_dim, hidden_layer_dim, layer_nums, batch_first=True, dropout=0.5
        )
        self.fully_connect = nn.Sequential(
            nn.Linear(hidden_layer_dim, 2 * hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2 * hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_layer_dim, 2),
        )

    def forward(self, x, hidden):
        x = self.embedding(x)
        ans, new_hidden = self.lstm(x, hidden)
        ans = ans.contiguous().view(-1, self.hidden_layer_dim)
        ans = self.fully_connect(ans)
        ans = F.softmax(ans, dim=1)
        ans = ans.view(x.shape[0], -1)
        ans = ans[:, -1]
        return ans, new_hidden

    def initial_hc(self, batchsize):
        h0 = torch.zeros(size=(self.layer_nums, batchsize, self.hidden_layer_dim)).to(
            self.device
        )
        c0 = torch.zeros(size=(self.layer_nums, batchsize, self.hidden_layer_dim)).to(
            self.device
        )
        return h0, c0


class MLP(nn.Module):

    def __init__(self, device, vocabulary_size, max_length):
        super(MLP, self).__init__()
        self.name = "MLP"
        self.device = device
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=50)
        self.max_length = max_length
        self.fully_connect = nn.Sequential(
            nn.Linear(max_length * 50, 32 * 50),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(32 * 50, 16 * 50),
            nn.ReLU(),
            nn.Linear(16 * 50, 8 * 50),
            nn.ReLU(),
            nn.Linear(8 * 50, 4 * 50),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(4 * 50, 2 * 50),
            nn.ReLU(),
            nn.Linear(2 * 50, 2),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(-1, self.max_length * 50)
        ans = self.fully_connect(x).to(self.device)
        ans = F.softmax(ans, dim=1)
        return ans
