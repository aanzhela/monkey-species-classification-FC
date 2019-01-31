import torch.nn as nn


# Acc 0.6 on test
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        self.fc1 = nn.Linear(100*100*3, 1098)
        self.fc2 = nn.Linear(1098, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 100)
        self.fc5 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# Acc 0.59 on test
class FCDropout(nn.Module):
    def __init__(self):
        super(FCDropout, self).__init__()
        self.fc1 = nn.Linear(100*100*3, 1098)
        self.fc2 = nn.Linear(1098, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 100)
        self.fc5 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# test Acc=0.65
class FC4(nn.Module):
    def __init__(self):
        super(FC4, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 512)  # 3072, 1024
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x
