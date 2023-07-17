import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=9, padding=1),
            nn.ReLU(),
            nn.Conv2d(5, 5, kernel_size=9, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)) # 14x14x5
        self.layer2 = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=7, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=7, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(# 7x7x8
            nn.Conv2d(8, 16, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        
        self.fc = nn.Sequential(
            nn.Linear(16*18*2, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, 2)
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out