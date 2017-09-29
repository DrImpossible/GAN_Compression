import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.PReLU(),
            )
        self.isFake = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid())
        self.classifier = nn.Linear(64,10)


    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.model(x)
        isFakeout = self.isFake(x)
        isFakeout = isFakeout.view(isFakeout.size(0))
#        print(isFakeout.size())
        classifierout = self.classifier(x)
        return isFakeout, classifierout
