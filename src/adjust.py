from train.train import train, train_lstm, validation, validation_lstm
from model.model import MLP, TextCnn, LstmModel
from dataloader.dataloader import init_model, get_dataloader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import torch

lr_list = [0.1, 0.05, 0.001, 0.0005]
batchsize_list = [512, 256, 128, 64, 32, 16]


def adjust(model_name: str, device, word2index):
    testset = get_dataloader("test", 1, False, model_name)
    z_rightrate = np.zeros(shape=(4, 6))
    z_f_score = np.zeros(shape=(4, 6))
    for i, lr in enumerate(lr_list):
        for j, batchsize in enumerate(batchsize_list):
            if model_name == "textcnn":
                kernels = [6, 5, 4, 3]
                model = TextCnn(50, 100, kernels, device, len(word2index)).to(device)
                model = init_model(model)
                dataloader = get_dataloader("train", batchsize, True, "textcnn")
                train(model, dataloader, 100, device, lr)
                rightrate, f_score = validation(model, testset)
                z_rightrate[i, j] = rightrate
                z_f_score[i, j] = f_score
            elif model_name == "lstm":
                model = LstmModel(50, 128, 2, device, len(word2index)).to(device)
                model = init_model(model)
                dataloader = get_dataloader("train", batchsize, True, "lstm")
                train_lstm(model, dataloader, 100, device, lr)
                rightrate, f_score = validation_lstm(model, testset)
                z_rightrate[i, j] = rightrate
                z_f_score[i, j] = f_score
            elif model_name == "MLP":
                model = MLP(device, len(word2index), 50).to(device)
                model = init_model(model)
                dataloader = get_dataloader("train", batchsize, True, "MLP")
                train(model, dataloader, 100, device, lr)
                rightrate, f_score = validation(model, testset)
                z_rightrate[i, j] = rightrate
                z_f_score[i, j] = f_score
    draw(z_rightrate, f"../results/draw/{model_name}_right.png")
    draw(z_f_score, f"../results/draw/{model_name}_f_score.png")


def draw(z_data, output_path):
    # 创建网格
    X, Y = np.meshgrid(batchsize_list, lr_list)

    # 绘制三维图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, z_data, cmap="viridis")

    # 找到最大值的索引
    max_idx = np.unravel_index(np.argmax(z_data, axis=None), z_data.shape)
    max_x, max_y = batchsize_list[max_idx[1]], lr_list[max_idx[0]]

    # 在最大值处添加标注
    ax.scatter(
        max_x,
        max_y,
        z_data[max_idx],
        color="red",
        s=100,
        label=f"Max Value : {z_data[max_idx]} , Param : {lr_list[max_idx[0]]}-{batchsize_list[max_idx[1]]}",
    )

    # 设置坐标轴标签和图例
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Learning Rate")
    ax.set_zlabel("Accuracy")
    ax.legend()

    # 保存图像
    plt.savefig(output_path)
