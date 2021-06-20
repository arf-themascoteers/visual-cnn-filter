from simple_net import SimpleNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torch import utils

def visTensor(tensor):
    tensor = tensor.reshape(tensor.shape[0],tensor.shape[2],tensor.shape[3])
    for i in tensor:
        m = torch.mean(i)
        i[i < m] = 0
        i[i >= m] = 1
        plt.imshow(i.numpy())
        plt.show()


if __name__ == "__main__":
    model = SimpleNet()
    model.load_state_dict(torch.load("models/cnn.h5"))
    model.eval()
    layer = 1
    filter = model.net[0].weight.data.clone()
    visTensor(filter)