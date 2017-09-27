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
import datasets.__datainit__ as init_data
from tensorboard_logger import Logger

torch.manual_seed(np.random.randint(1,1000))
torch.cuda.manual_seed(np.random.randint(50,100))

fake_sample_range = (0.0,0.3)
real_sample_range = (0.7,1.1)

parser = opts.myargparser()

def main():
    global opt, best_prec1
    cudnn.benchmark = True

    opt = parser.parse_args()
    opt.logdir = opt.logdir+'/'+opt.name
    logger = Logger(opt.logdir)

    print(opt)
    best_error = 1e10
    best_acc = 0

    print('Loading pretrained model')
    teacher = init.load_model(opt,'teacher')
    classifier = init.load_model(opt,'classifier')
    student = init.load_model(opt,'student')
    discriminator = init.load_model(opt,'discriminator')

    teacher, criterion, optimizer = init.setup(teacher,opt,'teacher')
    student, classifycriterion, optimizer = init.setup(student,opt,'student')
    discriminator, classifycriterion, optimizer = init.setup(discriminator,opt,'discriminator')
    classifier, classifycriterion, optimizer = init.setup(classifier,opt,'classifier')

    teacher = teacher.features()
    # Make parameters of teacher network non-trainable
    for p in teacher.parameters():
        p.requires_grad= False

    classifier.fc1.weight = teacher.fc1.weight
    classifier.fc1.bias = teacher.fc1.bias
    classifier.fc2.weight = teacher.fc2.weight
    classifier.fc2.bias = teacher.fc2.bias

    print(model)

    trainer = train.Trainer(student, teacher, discriminator, classifier, classifycriterion, adversarialcriterion, softcriterion, studentoptimizer, discriminatoroptimizer, opt, logger)
    validator = train.Validator(student, teacher, discriminator, classifier, classifycriterion, adversarialcriterion, softcriterion, studentoptimizer, discriminatoroptimizer, opt, logger)

    if opt.resume:
        if os.path.isfile(opt.resume):
            model, optimizer, opt, best_prec1 = init.resumer(opt, model, optimizer)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    cudnn.benchmark = True
    dataloader = init_data.load_data(opt)
    train_loader = dataloader.train_loader
    val_loader = dataloader.val_loader

    for epoch in range(opt.start_epoch, opt.epochs):
        utils.adjust_learning_rate(opt, optimizer, epoch)
        print("Starting epoch number:",epoch+1,"Learning rate:", optimizer.param_groups[0]["lr"])

        if opt.testOnly == False:
            trainer.train(train_loader, epoch, opt)
        if opt.tensorboard:
            logger.scalar_summary('learning_rate', opt.lr, epoch)

        acc = validator.validate(val_loader, epoch, opt)
        best_acc = max(acc, best_acc)
        init.save_checkpoint(opt, student, teacher, discriminator, classifier, classifycriterion, adversarialcriterion, softcriterion, studentoptimizer, discriminatoroptimizer, opt, logger, best_acc, epoch)

        print('Best accuracy: [{0:.3f}]\t'.format(best_acc))

if __name__ == '__main__':
    main()
