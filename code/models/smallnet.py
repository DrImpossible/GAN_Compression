import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet,self).__init__()
        # 3x32x32 ; 32x16x16 ; 64x8x8 ; 128x4x4 ; 256x2x2 ; 512x1x1
        self.conv1 = nn.Sequential(
                nn.Conv2d(3,32,kernel_size=3,padding=1),
                nn.BatchNorm2d(32),
                #nn.Dropout2d(0.5),
                nn.LeakyReLU(),
                nn.AvgPool2d(2))
        self.conv2 = nn.Sequential(
                nn.Conv2d(32,64,kernel_size=3,padding=1),
                nn.BatchNorm2d(64),
                #nn.Dropout2d(0.4),
                nn.LeakyReLU(),
                nn.AvgPool2d(2))
        self.conv3 = nn.Sequential(
                nn.Conv2d(64,128,kernel_size=3,padding=1),
                nn.BatchNorm2d(128),
                #nn.Dropout2d(0.3),
                nn.LeakyReLU(),
                nn.AvgPool2d(2))
        self.conv4 = nn.Sequential(
                nn.Conv2d(128,256,kernel_size=1),
                nn.BatchNorm2d(256),
                #nn.Dropout2d(0.3),
                nn.AvgPool2d(2),
                nn.LeakyReLU())
        self.conv5 = nn.Sequential(
                nn.Conv2d(256,512,kernel_size=1),
                nn.BatchNorm2d(512),
                #nn.Dropout2d(0.3),
                nn.AvgPool2d(2),
                nn.LeakyReLU())

    def forward(self,x):
        #print "0",x.size()
        x = self.conv1(x)
        #print "1",x.size()
        x = self.conv2(x)
        #print "2",x.size()
        x = self.conv3(x)
        #print "3",x.size()
        x = self.conv4(x)
        #print "4",x.size()
        x = self.conv5(x)
        #print "5",x.size()
        x = x.view(x.size(0),-1)
        return x
