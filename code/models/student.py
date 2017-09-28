import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


class Net(nn.Module):
#    def __init__(self):
#        super(Net,self).__init__()
#        # 3x32x32 ; 32x16x16 ; 64x8x8 ; 128x4x4 ; 256x2x2 ; 512x1x1
#        self.conv1 = nn.Sequential(
#                nn.Conv2d(3,32,kernel_size=5,padding=2),
#                nn.BatchNorm2d(32),
#                nn.PReLU(),
#                nn.MaxPool2d(kernel_size=3, stride=2))
#        self.conv2 = nn.Sequential(
#                nn.Conv2d(32,32,kernel_size=5,padding=2),
#                nn.BatchNorm2d(32),
#                nn.PReLU(),
#                nn.AvgPool2d(kernel_size=3, stride=2))
#        self.conv3 = nn.Sequential(
#                nn.Conv2d(32,64,kernel_size=5,padding=2),
#                nn.BatchNorm2d(64),
#                nn.PReLU(),
#                nn.AvgPool2d(kernel_size=3, stride=2))
#        self.conv4 = nn.Sequential(
#                nn.Conv2d(64,512,kernel_size=3,padding=0),
#                nn.BatchNorm2d(512),
#                nn.PReLU())
#        self.conv5 = nn.Sequential(
#                nn.Conv2d(512,10,kernel_size=1,padding=0))
#
#    def forward(self,x):
#        x = self.conv1(x)
#        x = self.conv2(x)
#        x = self.conv3(x)
#        #print(x.size())
#        x = self.conv4(x)
#        x = self.conv5(x)
#        x = x.view(x.size(0),-1)
#
#        return F.log_softmax(x)
#
    def __init__(self):
        super(Net,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(32*32*3,1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512,10))
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.net(x)
        return F.log_softmax(x)
