import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.3)
            )
        self.discr = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid())
        self.cfier = nn.Linear(64,10)


    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.model(x)
        discr_out = self.discr(x)
        discr_out = discr_out.view(discr_out.size(0))
        cfier_out = self.cfier(x)
        return discr_out , cfier_out
