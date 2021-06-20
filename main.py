from torch.utils.data import DataLoader
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,10, (28,28)),
            nn.Flatten(),
            nn.Linear(10, 10)

        )

    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(x, dim=1)

def train():
    NUM_EPOCHS = 1000
    BATCH_SIZE = 1

    working_set = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True
    )

    dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=False)
    model = SimpleNet()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    data, y_true = next(iter(dataloader))
    data[data < 0.5] = 0
    data[data >= 0.5] = 1
    print(y_true)
    plt.imshow(data[0][0].numpy())
    plt.show()
    for epoch  in range(0, NUM_EPOCHS):
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.nll_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()
        print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    return model

def visTensor(tensor):
    tensor = tensor.reshape(tensor.shape[0],tensor.shape[2],tensor.shape[3])
    for i in tensor:
        m = torch.mean(i)
        i[i < m] = 0
        i[i >= m] = 1
        plt.imshow(i.numpy())
        plt.show()

model = train()
filter = model.net[0].weight.data.clone()
visTensor(filter)


