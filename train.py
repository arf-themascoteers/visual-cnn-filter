from simple_net import SimpleNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def train():
    NUM_EPOCHS = 5000
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
    torch.save(model.state_dict(), 'models/cnn.h5')

train()


