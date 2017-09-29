import torch
import torch.nn as nn
import torch.optim as optim
import models.student as student
import models.teacher as teacher
import models.discriminator as discriminator
import utils
import os
import shutil

def setup(model, opt, type):
    if type == "discriminator":
        criterion = nn.BCELoss().cuda()
    elif type == "teacher":
        criterion = nn.L1Loss().cuda()
    elif type == "student": #or type == "classifier":
        criterion = nn.CrossEntropyLoss(size_average=True).cuda()

    if opt.optimType == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = opt.lr, momentum = opt.momentum, nesterov = opt.nesterov, weight_decay = opt.weightDecay)
    elif opt.optimType == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = opt.maxlr, weight_decay = opt.weightDecay)

    if opt.weight_init:
        utils.weights_init(model, opt)

    return model, criterion, optimizer

def save_checkpoint(opt, teacher, student, discriminator, studOptim, discOptim, best_acc, epoch):
    state = {
        'epoch': epoch + 1,
        'teacher_state_dict': teacher.state_dict(),
        'student_state_dict': student.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'best_studentprec1': best_acc,
        'studentoptimizer' : studOptim.state_dict(),
        'discriminatoroptimizer' : discOptim.state_dict(),
    }
    filename = "savedmodels/" + 'Checkpoint_' + opt.name + '_' + "best.pth.tar"
    torch.save(state, filename)

def resumer(opt, teacher, student, discriminator, studOptim, discOptim):
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint['epoch']
        best_studentprec1 = checkpoint['best_studentprec1']
        teacher.load_state_dict(checkpoint['state_dict'])
        student.load_state_dict(checkpoint['state_dict'])
        discriminator.load_state_dict(checkpoint['state_dict'])
        studOptim.load_state_dict(checkpoint['optimizer'])
        discOptim.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))

        return  teacher, student, discriminator, studOptim, discOptim, opt, best_studentprec1

def load_model(opt,type):
        if type == "teacher":
            checkpoint = torch.load(opt.teacher_filedir)
            model = teacher.Net()
            if opt.cuda:
                model = model.cuda()
            model.features = torch.nn.DataParallel(model.features)
            model.load_state_dict(checkpoint['state_dict'])
        elif type == "student":
            checkpoint = torch.load(opt.student_filedir)
            model = student.Net()
            if opt.cuda:
                model = model.cuda()
            #model.load_state_dict(checkpoint['state_dict'])
        elif type == 'discriminator':
            model = discriminator.Net()
            if opt.cuda:
                model = model.cuda()
        #elif type == 'classifier':
        #    model = classifier.Net()
        #    if opt.cuda:
        #        model = model.cuda()
        return model
