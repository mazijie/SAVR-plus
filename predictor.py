import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(2048, 512, bias=False)
        self.bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 2048)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


def getPredictor():
    return Predictor()


if __name__ == '__main__':
    print(getPredictor())
