import torch
import torch.nn as nn
import torch.optim as optim
import models.student as student1
import models.student as student2
import models.student as student3
import models.student as student4
import models.teacher as teacher1
import models.teacher as teacher2
import models.teacher as teacher3
import models.teacher as teacher4
import models.teacher as teacher5
import models.discriminator as discriminator1
import models.discriminator as discriminator2
import models.discriminator as discriminator3
import utils
import os
import shutil

def setup(model, opt, type):
    if opt.weight_init:
        utils.weights_init(model, opt)

    return model

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
        if type == "teacher" and opt.teacherno == 1:
            checkpoint = torch.load(opt.teacher_filedir)
            model = teacher1.Net()
            if opt.cuda:
                model = model.cuda()
            model.features = torch.nn.DataParallel(model.features)
            model.load_state_dict(checkpoint['state_dict'])
        elif type == "student":
            #checkpoint = torch.load(opt.student_filedir)
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
