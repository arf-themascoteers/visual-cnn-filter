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