import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import models.smallnet as smallnet
import models.discriminator as discriminator
import models.classifier as classifier
import utils
import os
import shutil

def setup(model, opt, type):

    if type == "discriminator":
        criterion = nn.BCELoss().cuda()
    elif type == "teacher":
        criterion = nn.SmoothL1Loss().cuda()
    elif type == "student" or type == "classifier":
        criterion = nn.CrossEntropyLoss(size_average=True).cuda()

    if opt.optimType == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = opt.lr, momentum = opt.momentum, nesterov = opt.nesterov, weight_decay = opt.weightDecay)
    elif opt.optimType == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = opt.maxlr, weight_decay = opt.weightDecay)

    if opt.weight_init:
        utils.weights_init(model, opt)

    return model, criterion, optimizer

def save_checkpoint(opt, model, optimizer, best_acc, epoch):

    state = {
        'epoch': epoch + 1,
        'arch': opt.model_def,
        'state_dict': model.state_dict(),
        'best_prec1': best_acc,
        'optimizer' : optimizer.state_dict(),
    }
    filename = "savedmodels/" + opt.model_def + '_' + opt.name + '_' + "best.pth.tar"

    torch.save(state, filename)

def resumer(opt, model, optimizer):

    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))

        return model, optimizer, opt, best_prec1

def load_model(opt,type):

    if type == "teacher":
            print("=> using pre-trained model '{}'".format(opt.teacherarch))
            model = models.__dict__[opt.model_def](pretrained=True)
        return model
    else:
        if opt.pretrained_file != '':
            model = torch.load(opt.pretrained_filedir)
        else:
            if type == 'student':
                model = torch.load(opt.pretrained_filedir)
                if opt.cuda:
                    model = model.cuda()
            elif type == 'discriminator':
                model = discriminator.Net()
                if opt.cuda:
                    model = model.cuda()
            elif type == 'classifier':
                model = classifier.Net()
                if opt.cuda:
                    model = model.cuda()
