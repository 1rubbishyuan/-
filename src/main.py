from train.train import train, validation, train_lstm, validation_lstm, test_vote
from model.model import TextCnn, LstmModel, MLP
from dataloader.dataloader import get_dataloader, init_model
import torch
from argparse import ArgumentParser
from torch.cuda import is_available
from word2vec import word2index
from adjust import adjust, draw, lr_list, batchsize_list
import numpy as np
import os

argumentParser = ArgumentParser()
argumentParser.add_argument("--type", type=str, default="train")
argumentParser.add_argument("--model", type=str, default="lstm")
argumentParser.add_argument("--model_path", type=str, default="model_2.pth")
argumentParser.add_argument("--device", type=str, default="cuda:2")
args = argumentParser.parse_args()
device = args.device if is_available() else "cpu"
device = torch.device(device)
if __name__ == "__main__":
    if args.type == "train":
        adjust(args.model, device, word2index)
    elif args.type == "test":
        model = torch.load("lstm_0.001_64.pth")
        dataloader = get_dataloader("test", 1, True, args.model)
        if args.model != "lstm":
            validation(model, dataloader)
        else:
            validation_lstm(model, dataloader)
    elif args.type == "draw":
        model_names = ["textcnn", "lstm", "mlp"]
        for model_name in model_names:
            testset = get_dataloader("test", 1, False, model_name)
            z_rightrate = np.zeros(shape=(4, 6))
            z_f_score = np.zeros(shape=(4, 6))
            model_paths = os.listdir(f"./{model_name}_model")
            for path in model_paths:
                print(path)
                model = torch.load(os.path.join(f"./{model_name}_model", path))
                model = model.to(device)
                model.device = device
                if model_name != "lstm":
                    rightrate, f_score = validation(model, testset)
                else:
                    rightrate, f_score = validation_lstm(model, testset)
                lr = float(path.split("_")[1])
                batchsize = int(path.split("_")[2].split(".")[0])
                z_rightrate[lr_list.index(lr), batchsize_list.index(batchsize)] = (
                    rightrate
                )
                z_f_score[lr_list.index(lr), batchsize_list.index(batchsize)] = f_score
            draw(z_rightrate, f"../results/draw/{model_name}_right.png")
            draw(z_f_score, f"../results/draw/{model_name}_f_score.png")
    elif args.type == "vote":
        testset = get_dataloader("test", 1, False, "textcnn")
        model1 = torch.load("./lstm_model/lstm_0.001_256.pth")
        model2 = torch.load("./textcnn_model/textcnn_0.001_128.pth")
        model3 = torch.load("./mlp_model/MLP_0.001_64.pth")
        model_list = [
            model1.to(model1.device),
            model2.to(model2.device),
            model3.to(model3.device),
        ]
        test_vote(model_list, testset)
