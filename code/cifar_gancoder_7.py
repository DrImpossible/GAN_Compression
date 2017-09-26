#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 00:23:13 2017

@author: harish.kv
"""

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
from utils import progress_bar
import visdom

#hyper-parameters----------------------------------------------------

#only reconstruction phase
smallnet_lr_init = 0.008
discr_lr_init = 0.001  #very high so that it doesn't become too strong
adv_loss_coeff_init = 0.0000001  #more like a regularizer
recons_loss_coeff_init = 1
cfier_coeff_init = 0.006

#both adverserial and reconstruction losses
adv_loss_wakeup_epoch=45
smallnet_lr_later = 0.0007
discr_lr_later = 0.0001
adv_loss_coeff_later = 0.0005
recons_loss_coeff_later = 1
cfier_coeff_later = 0.05

#todo: other hacks
fake_sample_range = (0.0,0.3)
real_sample_range = (0.7,1.1)
label_reversal_freq = 27

#data-----------------------------------------------------------------------------------

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

torch.manual_seed(np.random.randint(1,1000))
torch.cuda.manual_seed(np.random.randint(50,100))


#---------------------------------------------------------------------------------------

print('Loading pretrained model')

checkpoint = torch.load('./checkpoint/ckpt.t7')
vgg = checkpoint['net']



##---------------------------------------------------------------------------------------

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

smallnet = SmallNet()

cifar_iterator = iter(trainloader)
images,labels = cifar_iterator.next()
images = Variable(images)
print("Smallnet: ",smallnet(images).size())

class Discr(nn.Module):
    def __init__(self):
        super(Discr,self).__init__()
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

discr = Discr()

#print "Discrnet  ",discrnet(smallnet(images)).size()


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

classifier = Classifier()
classifier.fc1.weight = vgg.fc1.weight
classifier.fc1.bias = vgg.fc1.bias
classifier.fc2.weight = vgg.fc2.weight
classifier.fc2.bias = vgg.fc2.bias


####define loss and parameters#####

criterion_adv = nn.BCELoss()
criterion_clfr = nn.CrossEntropyLoss(size_average=True)
criterion_rcns = nn.SmoothL1Loss()
criterion_cfierdiscr = nn.CrossEntropyLoss(size_average=True)
to_cuda = True
use_cuda = True

if to_cuda and torch.cuda.is_available():
    smallnet = smallnet.cuda()
    vgg = vgg.cuda()
    discr = discr.cuda()
    classifier = classifier.cuda()
    criterion_adv = criterion_adv.cuda()
    criterion_clfr = criterion_clfr.cuda()
    criterion_rcns = criterion_rcns.cuda()
    criterion_cfierdiscr = criterion_cfierdiscr.cuda()

    smallnet = torch.nn.DataParallel(smallnet, device_ids=range(torch.cuda.device_count()))
    discr = torch.nn.DataParallel(discr, device_ids=range(torch.cuda.device_count()))
    net = torch.nn.DataParallel(vgg, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

smallnet_opt = optim.Adam(smallnet.parameters(),lr=smallnet_lr_init)
discr_opt = optim.Adam(discr.parameters(),lr=discr_lr_init)

# Make parameters of vgg16 non trainable

for p in vgg.parameters():
    p.requires_grad= False

##setup visdom------------------------------
#start visdom server with the command ` python -m visdom.server `
#usually uses port 8097
#setup visdom
viz = visdom.Visdom()
cfier_plots = viz.line(X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1, 2)).cpu(),
                       opts=dict(
                               xlabel='Iteration',
                               ylabel='Accuracy',
                               title='Classifier Accuracy',
                               legend=['VGG', 'Gen']
                               ),
                       win='classifier'
                       )

nets_plots = viz.line(X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1,2)).cpu(),
                       opts=dict(
                               xlabel='Iteration',
                               ylabel='Loss',
                               title='Discriminator losses',
                               legend=['Discriminator loss','d_cfier loss']
                               ),
                       win='d_loss'
                       )
gen_plots = viz.line(X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1, 4)).cpu(),
                       opts=dict(
                               xlabel='Iteration',
                               ylabel='Generator Loss',
                               title='Generator losses',
                               legend=['Adv loss', 'Recons loss','d_cfier loss','Overall']
                               ),
                       win='g_loss'
                       )

#------------------------------------------------------------
####training####
alpha=recons_loss_coeff_init  #coefficient for reconstruction loss
beta=adv_loss_coeff_init   #coefficient for adverserial loss
gen_acc = 0
for epoch in range(500):
	print('*****Epoch ', epoch)

	if(epoch==adv_loss_wakeup_epoch):
	    print('Saving..')
	    state = {
	        'smallnet': smallnet.module if use_cuda else net,
	        'discr': discr.module if use_cuda else net,
	        'gen_acc': gen_acc,
	        'epoch': epoch,
	    }
	    if not os.path.isdir('checkpoint'):
	        os.mkdir('checkpoint')
	    torch.save(state, './checkpoint/gancoder1.t7')
	    # best_acc = acc
	    alpha=recons_loss_coeff_later
	    beta=adv_loss_coeff_later
	    smallnet_opt = optim.Adam(smallnet.parameters(),lr=smallnet_lr_later)
	    discr_opt = optim.Adam(discr.parameters(),lr=discr_lr_later)

	total_discr_loss = 0
	total_gen_loss = 0
	total_adv_loss = 0
	total_recons_loss = 0
	total_d_cfier_loss = 0
	total_g_cfier_loss = 0
	#train Descriminator

	discr.train()
	for batch_i, (imgs,lbls) in enumerate(trainloader):
		#Real samples
		images = Variable(imgs)
		lbls = Variable(lbls)
		real_or_fake = torch.FloatTensor(images.size(0))
		if epoch%label_reversal_freq == 0:
		     real_or_fake.fill_(0)
		else:
		     real_or_fake.fill_(1)
		labels = Variable(real_or_fake)
		if to_cuda:
		    images,labels,lbls = images.cuda(), labels.cuda(), lbls.cuda()
		discr_opt.zero_grad()
		vgg_feats, logits = vgg(images)
		if to_cuda:
			vgg_feats  = vgg_feats.cuda()
		outs_discr, outs_dcfier = discr(vgg_feats)
		if to_cuda:
		    outs_discr  = outs_discr.cuda()
		    outs_dcfier = outs_dcfier.cuda()
		disc_loss = criterion_adv(outs_discr,labels)
		#disc_loss.backward()
		discfier_loss = cfier_coeff_later* criterion_cfierdiscr(outs_dcfier,lbls)
		#discfier_loss.backward()
		total_loss = disc_loss + discfier_loss
		total_loss.backward()
		discr_opt.step()

		total_discr_loss += disc_loss.data[0]
		total_d_cfier_loss += discfier_loss[0]


		#Fake samples
		outs_smallnet = smallnet(images)
		real_or_fake = torch.FloatTensor(outs_smallnet.size(0))
		if epoch%label_reversal_freq == 0:
		     real_or_fake.fill_(1)
		else:
		     real_or_fake.fill_(0)
		labels = Variable(real_or_fake)
		if to_cuda:
		    feats,labels = outs_smallnet.cuda(), labels.cuda()

		discr_opt.zero_grad()
		#outs_discr = discr(feats)
		outs_discr, outs_dcfier = discr(feats.detach())  #To avoid computing gradients in Generator (smallnet)

		if to_cuda:
		    outs_discr  = outs_discr.cuda()
		    outs_dcfier = outs_dcfier.cuda()

		disc_loss = criterion_adv(outs_discr,labels)
		discfier_loss = cfier_coeff_later* criterion_cfierdiscr(outs_dcfier,lbls)
		total_loss = disc_loss + discfier_loss
		total_loss.backward()
		discr_opt.step()

		total_discr_loss += disc_loss.data[0]
		total_d_cfier_loss += discfier_loss[0]




		# Train Generator
		smallnet.train()

		smallnet_opt.zero_grad()
		# images = Variable(imgs)
		# labels = Variable(lbls)
		# if to_cuda:
		#     images,labels = images.cuda(), labels.cuda()
		feats = smallnet(images)
		if to_cuda:
		    feats = feats.cuda()

		out_discr, outs_dcfier = discr(feats)

		real_or_fake = torch.FloatTensor(out_discr.size(0))
		real_or_fake.fill_(1) #Clever stuff. Generator expects discriminator to classify its output as real
		labels = Variable(real_or_fake)

		if to_cuda:
		    out_discr,labels,outs_dcfier = out_discr.cuda(), labels.cuda(), outs_dcfier.cuda()

		# disc_loss = criterion_adv(out_discr,labels)
		adv_loss = criterion_adv(out_discr,labels)
		recons_loss = criterion_rcns(feats,vgg_feats)
		discfier_loss = cfier_coeff_later* criterion_cfierdiscr(outs_dcfier,lbls)

		gen_loss = alpha*recons_loss + beta*adv_loss + discfier_loss
		gen_loss.backward()
		smallnet_opt.step()

		total_gen_loss += gen_loss.data[0]
		total_adv_loss += adv_loss.data[0]
		total_recons_loss += recons_loss.data[0]
		total_g_cfier_loss += discfier_loss.data[0]

	print("Generator:", total_gen_loss / len(trainloader.dataset))
	print("Discriminator:", total_discr_loss / len(trainloader.dataset))

	#visdom disc,gen loss
	viz.line(X=torch.ones((1, 2)).cpu() * epoch,
	    Y=np.reshape(np.array([total_discr_loss,total_d_cfier_loss]),(1,2)),
	win=nets_plots,
	update='append'
	)

	viz.line(X=torch.ones((1, 4)).cpu() * epoch,
	    Y=np.reshape(np.array([beta*total_adv_loss,alpha*total_recons_loss,total_gen_loss,total_g_cfier_loss]),(1,4)),
	win=gen_plots,
	update='append'
	)

	#------Testing-----------
	smallnet.eval()
	classifier.eval()
	vgg.eval()
	total = 0
	vgg_correct = 0
	gen_correct = 0
	if (epoch+1)%5 == 0:
		print('Testing--------------------------------------')
		for batch_i, (imgs,lbls) in enumerate(testloader):
			images = imgs.cuda()
			labels = lbls.cuda()
			# labels = lbls
			images = Variable(images)
			vgg_feats, vgg_outputs = vgg(images)
			gen_feats = smallnet(images)

			vgg_outputs = classifier(vgg_feats)
			gen_outputs = classifier(gen_feats)

			_, vgg_predicted = torch.max(vgg_outputs.data, 1)
			_, gen_predicted = torch.max(gen_outputs.data, 1)
			total += labels.size(0)

			vgg_correct += (vgg_predicted == labels).sum()
			gen_correct += (gen_predicted == labels).sum()

		print('Test Accuracy of VGG : %.2f %%' % (100.0 * vgg_correct / total))
		print('Test Accuracy of Generator: %.2f %%' % (100.0 * gen_correct / total))
		gen_acc = (100.0 * gen_correct / total)

		#visdom accuracy
		viz.line(X=torch.ones((1, 2)).cpu() * epoch,
		        Y=np.reshape(np.array([(100.0 * vgg_correct / total),(100.0 * gen_correct / total)]),(1,2)),
		win=cfier_plots,
		update='append'
		)
