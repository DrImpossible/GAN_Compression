import numpy as np
import os
import argparse
import copy
import opts
import train
import utils
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as modelzoo
import models.__init__ as init
import mydatasets.__datainit__ as init_data
from tensorboard_logger import Logger

torch.manual_seed(np.random.randint(1,1000))
torch.cuda.manual_seed(np.random.randint(50,100))

#Incorporate it sometime
fake_sample_range = (0.0,0.3)
real_sample_range = (0.7,1.1)

parser = opts.myargparser()

def getOptim(opt, model, type):
    if type=='student' and opt.studentoptimType == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = opt.lr, momentum = opt.momentum, nesterov = opt.nesterov, weight_decay = opt.weightDecay)
    if type=='student' and opt.studentoptimType == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = opt.maxlr, weight_decay = opt.weightDecay)
    if type=='discriminator' and opt.discriminatoroptimType == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = opt.lr, momentum = opt.momentum, nesterov = opt.nesterov, weight_decay = opt.weightDecay)
    if type=='discriminator' and opt.discriminatoroptimType == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = opt.maxlr, weight_decay = opt.weightDecay)
    return optimizer

def main():
    global opt, best_studentprec1
    cudnn.benchmark = True

    opt = parser.parse_args()
    opt.logdir = opt.logdir+'/'+opt.name
    logger = Logger(opt.logdir)

    print(opt)
    best_studentprec1 = 0.0

    print('Loading models...')
    teacher = init.load_model(opt,'teacher')
    student = init.load_model(opt,'student')
    discriminator = init.load_model(opt,'discriminator')
    teacher = init.setup(teacher,opt,'teacher')
    student  = init.setup(student,opt,'student')
    discriminator  = init.setup(discriminator,opt,'discriminator')

    #Write the code to classify it in the 11th class
    print(teacher)
    print(student)
    print(discriminator)

    advCriterion = nn.BCELoss().cuda()
    similarityCriterion = nn.L1Loss().cuda()
    derivativeCriterion = nn.SmoothL1Loss().cuda()
    discclassifyCriterion = nn.CrossEntropyLoss(size_average=True).cuda()

    studOptim = getOptim(opt,student,'student')
    discrecOptim = getOptim(opt,discriminator,'discriminator')

    trainer = train.Trainer(student, teacher, discriminator,discclassifyCriterion, advCriterion, similarityCriterion, derivativeCriterion, studOptim, discrecOptim, opt, logger)
    validator = train.Validator(student, teacher, discriminator, opt, logger)

    #To update. Does not work as of now
    if opt.resume:
        if os.path.isfile(opt.resume):
            model, optimizer, opt, best_prec1 = init.resumer(opt, model, optimizer)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    dataloader = init_data.load_data(opt)
    train_loader = dataloader.train_loader
    val_loader = dataloader.val_loader

    for epoch in range(opt.start_epoch, opt.epochs):
        utils.adjust_learning_rate(opt, studOptim, epoch)
        print("Starting epoch number:",epoch+1,"Learning rate:", studOptim.param_groups[0]["lr"])

        if opt.testOnly == False:
            trainer.train(train_loader, epoch, opt)
        if opt.tensorboard:
            logger.scalar_summary('learning_rate', opt.lr, epoch)

        student_prec1 = validator.validate(val_loader, epoch, opt)
        best_studentprec1 = max(student_prec1, best_studentprec1)
        init.save_checkpoint(opt, teacher, student, discriminator, studOptim, discrecOptim, student_prec1, epoch)

        print('Best accuracy: [{0:.3f}]\t'.format(best_studentprec1))

if __name__ == '__main__':
    main()
