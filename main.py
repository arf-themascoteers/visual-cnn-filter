import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import cv2


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,5, (28,28)),
            nn.Flatten(),
            nn.Linear(5, 10)
        )

    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(x, dim=1)


def plot_tensor(tensor):
    mean = torch.mean(tensor)
    tensor = tensor.data.clone()
    tensor[tensor >= mean] = 255
    tensor[tensor < mean] = 0
    plt.imshow(tensor.numpy())
    plt.show()


def plot_filters(filters):
    filters = filters.clone()
    filters = filters.reshape(filters.shape[0], filters.shape[2], filters.shape[3])
    for tensor in filters:
        plot_tensor(tensor)


def train(model, data):
    NUM_EPOCHS = 5000
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch  in range(0, NUM_EPOCHS):
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.nll_loss(y_pred, torch.tensor([4]))
        loss.backward()
        optimizer.step()
        print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    return model


data = cv2.imread("4.png")
data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
data = torch.tensor(data, dtype=torch.float)
plot_tensor(data)
data = data.reshape(1,1,data.shape[0],data.shape[1])
model = SimpleNet()
filters = model.net[0].weight.data
plot_filters(filters)

train(model, data)

filters = model.net[0].weight.data
plot_filters(filters)




