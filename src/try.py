# import torch

# # model = torch.load("../Dataset/wiki_word2vec_50.bin")
# # model.eval()
# # print(model("开始"))

# # from gensim.models import KeyedVectors

# # # 加载词向量模型
# # model = KeyedVectors.load_word2vec_format(
# #     "../Dataset/wiki_word2vec_50.bin", binary=True
# # )
# from word2vec import model
# import torch
# from torch.nn.utils.rnn import pad_sequence
# from model.model import TextCnn, LstmModel
# from torch.cuda import is_available
# from word2vec import *

# device = "cuda:2" if is_available() else "cpu"
# device = torch.device(device)
# # 加载数据并进行填充
# # data = [
# #     torch.tensor([[1, 2, 3], [1, 1, 1]]),
# #     torch.tensor([[4, 5, 1]]),
# #     torch.tensor([[6, 7, 9]]),
# # ]
# # padded_data = pad_sequence(data, batch_first=True, padding_value=0)
# # print(padded_data)

# # 获取词向量
# # vector = model["幻亦真", "艾布拉姆斯", "开始", "0"]

# # print(torch.tensor(vector))

# # a = "1	死囚 爱 刽子手 女贼 爱 衙役 我们 爱 你们 难道 还有 别的 选择 没想到 胡军 除了 蓝宇 还有 东宫 西宫 我 个 去 阿兰 这样 真 他 nia 恶心 爱个 P 分明 只是 欲"
# # print(a.split())

# from dataloader.dataloader import get_dataloader, init_model

# # dataloader = get_dataloader("test", 2, True)
# # print(dataloader)
# # model = TextCnn(50, 1, [4, 3], device, vocabulary_size=len(word2index)).to(device)
# device = "cpu"
# model1 = LstmModel(50, 100, 2, device, vocabulary_size=len(word2index)).to(device)
# model1 = init_model(model1)
# print(model[0])
# print(word2index["我们"])
# # tensor = torch.tensor(7, dtype=torch.long)
# model1.eval()
# print(torch.tensor(word2index["我们"], dtype=torch.long))
# print(model1.embedding(torch.tensor(1, dtype=torch.long)))

# # for data in dataloader:
# #     # print(data)
# #     hidden = model.initial_hc(2)
# #     ans = model(data[0].to(device), hidden)
# #     break
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 模拟数据
lr_list = [0.1, 0.05, 0.001, 0.0005]
batchsize_list = [512, 256, 128, 64, 32, 16]
accuracy_list = np.random.rand(
    len(lr_list), len(batchsize_list)
)  # 这里用随机数代替正确率

# 创建网格
X, Y = np.meshgrid(batchsize_list, lr_list)

# 绘制三维图像
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, accuracy_list, cmap="viridis")

# 设置坐标轴标签
ax.set_xlabel("Batch Size")
ax.set_ylabel("Learning Rate")
ax.set_zlabel("Accuracy")

plt.savefig("3d_plot.png")
# 显示图像
# plt.show()
