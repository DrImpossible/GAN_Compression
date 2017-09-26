import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        return out
