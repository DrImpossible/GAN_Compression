import os

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.manual_seed(np.random.randint(1,1000))
torch.cuda.manual_seed(np.random.randint(50,100))

fake_sample_range = (0.0,0.3)
real_sample_range = (0.7,1.1)

print('Loading pretrained model')

checkpoint = torch.load('./checkpoint/ckpt.t7')
vgg = checkpoint['net']
vgg = vgg.cuda()
alpha=recons_loss_coeff_init  #coefficient for reconstruction loss
beta=adv_loss_coeff_init   #coefficient for adverserial loss
gen_acc = 0
# Make parameters of vgg16 non trainable
for p in vgg.parameters():
    p.requires_grad= False

classifier = Classifier()
classifier.fc1.weight = vgg.fc1.weight
classifier.fc1.bias = vgg.fc1.bias
classifier.fc2.weight = vgg.fc2.weight
classifier.fc2.bias = vgg.fc2.bias

criterion_adv = nn.BCELoss()
criterion_clfr = nn.CrossEntropyLoss(size_average=True)
criterion_rcns = nn.SmoothL1Loss()
criterion_cfierdiscr = nn.CrossEntropyLoss(size_average=True)

smallnet = torch.nn.DataParallel(smallnet, device_ids=range(torch.cuda.device_count()))
discr = torch.nn.DataParallel(discr, device_ids=range(torch.cuda.device_count()))
net = torch.nn.DataParallel(vgg, device_ids=range(torch.cuda.device_count()))
smallnet_opt = optim.Adam(smallnet.parameters(),lr=smallnet_lr_init)
discr_opt = optim.Adam(discr.parameters(),lr=discr_lr_init)
