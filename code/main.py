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

    checkpoint = torch.load('./checkpoint/ckpt.t7')
    vgg = checkpoint['net']
    vgg = vgg.cuda()

    # Make parameters of teacher network non-trainable
    for p in vgg.parameters():
        p.requires_grad= False

    classifier = Classifier()
    classifier.fc1.weight = vgg.fc1.weight
    classifier.fc1.bias = vgg.fc1.bias
    classifier.fc2.weight = vgg.fc2.weight
    classifier.fc2.bias = vgg.fc2.bias

    model = init.load_model(opt)
    model, criterion, optimizer = init.setup(model,opt)
    print(model)

    trainer = train.Trainer(model, criterion, optimizer, opt, logger)
    validator = train.Validator(model, criterion, opt, logger)

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
        init.save_checkpoint(opt, model, optimizer, best_acc, epoch)

        print('Best accuracy: [{0:.3f}]\t'.format(best_acc))

if __name__ == '__main__':
    main()


alpha=recons_loss_coeff_init  #coefficient for reconstruction loss
beta=adv_loss_coeff_init   #coefficient for adverserial loss
gen_acc = 0

criterion_adv = nn.BCELoss()
criterion_clfr = nn.CrossEntropyLoss(size_average=True)
criterion_rcns = nn.SmoothL1Loss()
criterion_cfierdiscr = nn.CrossEntropyLoss(size_average=True)

smallnet = torch.nn.DataParallel(smallnet, device_ids=range(torch.cuda.device_count()))
discr = torch.nn.DataParallel(discr, device_ids=range(torch.cuda.device_count()))
net = torch.nn.DataParallel(vgg, device_ids=range(torch.cuda.device_count()))
smallnet_opt = optim.Adam(smallnet.parameters(),lr=smallnet_lr_init)
discr_opt = optim.Adam(discr.parameters(),lr=discr_lr_init)
