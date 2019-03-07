import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(3, 96, kernel_size=3),  # out 30x30
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3),  # out 28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # out 13x13 
        )
        self.layer2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(96, 192, kernel_size=3),  # out 11x11
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3),  # out 9x9
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # out 4x4
        )
        self.layer3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(192, 192, kernel_size=3),  # out 2x2
            nn.ReLU(),
            nn.Conv2d(192, 10, kernel_size=1),  # out 2x2
            nn.ReLU()
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(10*2*2, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        return out
