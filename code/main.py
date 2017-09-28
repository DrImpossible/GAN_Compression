import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as modelzoo
import opts
import train
import utils
import models.__init__ as init

import mydatasets.__datainit__ as init_data
from tensorboard_logger import Logger

torch.manual_seed(np.random.randint(1,1000))
torch.cuda.manual_seed(np.random.randint(50,100))

#Incorporate it sometime
fake_sample_range = (0.0,0.3)
real_sample_range = (0.7,1.1)

parser = opts.myargparser()

def main():
    global opt, best_studentprec1
    cudnn.benchmark = True

    opt = parser.parse_args()
    opt.logdir = opt.logdir+'/'+opt.name
    logger = Logger(opt.logdir)

    print(opt)
    best_studentprec1 = 0

    print('Loading models...')
    teacher = init.load_model(opt,'teacher')
    print(teacher)
    student = init.load_model(opt,'student')
    print(student)
    discriminator = init.load_model(opt,'discriminator')
    print(discriminator)
    #classifier = init.load_model(opt,'classifier')

    #Write the code to classify it in the 11th class
    teacher, derivativeCriterion, timepass = init.setup(teacher,opt,'teacher')
    student, classifyCriterion, studOptim = init.setup(student,opt,'student')
    discriminator, advCriterion, discOptim = init.setup(discriminator,opt,'discriminator')
    #classifier, classifycriterion, optimizer = init.setup(classifier,opt,'classifier')

    #Remove the last layer from the network and use the rest layers as is.

    # Make parameters of teacher network non-trainable
    for p in teacher.parameters():
        p.requires_grad= False

    #classifier.fc1.weight = teacher.fc1.weight
    #classifier.fc1.bias = teacher.fc1.bias
    #classifier.fc2.weight = teacher.fc2.weight
    #classifier.fc2.bias = teacher.fc2.bias

    trainer = train.Trainer(student, teacher, discriminator,classifyCriterion, advCriterion, derivativeCriterion, studOptim, discOptim, opt, logger)
    validator = train.Validator(student, teacher, discriminator, opt, logger)

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
        utils.adjust_learning_rate(opt, discOptim, epoch)
        print("Starting epoch number:",epoch+1,"Learning rate:", studOptim.param_groups[0]["lr"])

        if opt.testOnly == False:
            trainer.train(train_loader, epoch, opt)
        if opt.tensorboard:
            logger.scalar_summary('learning_rate', opt.lr, epoch)

        student_prec1 = validator.validate(val_loader, epoch, opt)
        best_studentprec1 = max(student_prec1, best_studentprec1)
        init.save_checkpoint(opt, teacher, student, discriminator, studOptim, discOptim, student_prec1, epoch)

        print('Best accuracy: [{0:.3f}]\t'.format(best_studentprec1))

if __name__ == '__main__':
    main()
