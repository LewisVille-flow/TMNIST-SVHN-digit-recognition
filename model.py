import torch.nn as nn

class RobustModel(nn.Module):
    def __init__(self):
        super(RobustModel, self).__init__()

        self.in_dim = 28 * 28 * 3
        self.out_dim = 10

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0),         # result = 32*26*26
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),        # result = 32*24*24
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=0),        # result = 32*10*10
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),        # result = 64*8*8
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),        # result = 64*6*6
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0),       # result = 128*1*1
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.fc7 = nn.Linear(128, 10)

        nn.init.xavier_uniform_(self.fc7.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.fc7(x.view(x.size(0), -1))
        #x = self.fc7(x)

        # return nn.softmax(x, dim=1) # softmax is included in nn.CrossEntropyLoss
        return x


