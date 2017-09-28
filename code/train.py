import torch
from torch.autograd import Variable
from utils import AverageMeter
from utils import precision
import torch.nn as nn

import utils
import torch.nn.functional as F
import math
import time

class Trainer():
    def __init__(self, student, teacher, discriminator, classifyCriterion, advCriterion, derivativeCriterion, studOptim, discOptim, opt, logger):
        self.opt = opt
        self.logger = logger
        self.discriminator = discriminator
        self.student = student
        self.teacher = teacher
        #self.classifier = classifier
        self.classifyCriterion = classifyCriterion
        self.advCriterion = advCriterion
        self.derivativeCriterion = derivativeCriterion
        self.studOptim = studOptim
        self.discOptim = discOptim

        self.teachertop1 = AverageMeter()
        self.studenttop1 = AverageMeter()
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.adversariallossLog = AverageMeter()
        self.crossentropylossLog = AverageMeter()
        self.reconstructionlossLog = AverageMeter()
        self.totaldisclossLog = AverageMeter()
        self.generatorlossLog = AverageMeter()

    def train(self, trainloader, epoch, opt):
        self.discriminator.train()
        self.student.train()
        self.teacher.train()
        self.adversariallossLog.reset()
        self.crossentropylossLog.reset()
        self.reconstructionlossLog.reset()
        self.totaldisclossLog.reset()
        self.generatorlossLog.reset()
        self.data_time.reset()
        self.batch_time.reset()
        self.teachertop1.reset()
        self.studenttop1.reset()

        end = time.time()
        for i, (input, target) in enumerate(trainloader, 0):
            #Generate fake samples
            isFakeTeacher = torch.ones(input.size(0))

            #if epoch%opt.revLabelFreq == 0:
            #    isFakeTeacher.fill_(0)

            if opt.cuda:
                input = input.cuda(async=True)
                target = target.cuda(async=True)
                labels = isFakeTeacher.cuda(async=True)

            input, target_var, labels = Variable(input), Variable(target), Variable(labels)
            self.data_time.update(time.time() - end)

            #Training the discriminator using Teacher
            self.discOptim.zero_grad()

            teacher_out = self.teacher(input)
            isReal, y_discriminator = self.discriminator(teacher_out)
            print(teacher_out.size(),target.size())
            print(target)
            teacherprec1, teacherprec5 = precision(teacher_out.data, target, topk=(1,5))

            adversarialLoss = self.advCriterion(isReal,labels)
            crossentropyLoss =  self.classifyCriterion(y_discriminator,target_var)
            totalDiscLoss = opt.weight_adversarial * adversarialLoss + opt.weight_classify * crossentropyLoss
            totalDiscLoss.backward()
            self.discOptim.step()

            self.adversariallossLog.update(adversarialLoss.data[0], input.size(0))
            self.crossentropylossLog.update(crossentropyLoss.data[0], input.size(0))
            self.totaldisclossLog.update(totalDiscLoss.data[0], input.size(0))

            #Training the discriminator using student
            self.discOptim.zero_grad()

            isFakeStudent = 1 - isFakeTeacher
            isFakeStudent = Variable(isFakeStudent)

            student_out = self.student(input)
            isReal, y_discriminator = self.discriminator(student_out.detach())  #To avoid computing gradients in Generator (smallnet)

            studentprec1, studentprec5 = precision(student_out.data, target, topk=(1,5))

            adversarialLoss = self.advCriterion(isReal,labels)
            crossentropyLoss =  self.classifyCriterion(y_discriminator,target_var)
            totalDiscLoss = opt.weight_adversarial * adversarialLoss + opt.weight_classify * crossentropyLoss
            totalDiscLoss.backward()
            self.discOptim.step()

            self.adversariallossLog.update(adversarialLoss.data[0], input.size(0))
            self.crossentropylossLog.update(crossentropyLoss.data[0], input.size(0))
            self.totaldisclossLog.update(totalDiscLoss.data[0], input.size(0))

            # Training the student network
            self.studOptim.zero_grad()
            student_out = self.student(input)

            isReal, y_discriminator = self.discriminator(student_out)  #To avoid computing gradients in Generator (smallnet)


            # disc_loss = criterion_adv(out_discr,labels)
            adversarialLoss = self.advCriterion(isReal,isFakeStudent)
            reconstructionLoss = self.derivativeCriterion(student_out,teacher_out)
            crossentropyLoss = self.classifyCriterion(y_discriminator,target)

            generatorLoss = opt.weight_reconstruction * reconstructionLoss + opt.weight_adversarial * adversarialLoss + opt.weight_classify * crossentropyLoss
            generatorLoss.backward()
            self.studOptim.step()

            self.adversariallossLog.update(adversarialLoss.data[0], input.size(0))
            self.crossentropylossLog.update(crossentropyLoss.data[0], input.size(0))
            self.reconstructionlossLog.update(reconstructionLoss.data[0], input.size(0))
            self.generatorlossLog.update(generatorLoss.data[0], input.size(0))
            self.teachertop1.update(teacherprec1[0], input.size(0))
            self.studenttop1.update(studentprec1[0], input.size(0))

            if opt.verbose == True and i % opt.printfreq == 0:
                print('Batch: [{0}][{1}/{2}]\t'
                'Time {batch_time.avg:.3f} ({batch_time.sum:.3f})\t'
                'Data {data_time.avg:.3f} ({data_time.sum:.3f})\t'
                'Adv Loss {loss.avg:.3f}\t'
                'CrossEnt Loss {loss.avg:.3f}\t'
                'Reconst Loss {loss.avg:.3f}\t'
                'TotalDisc Loss {loss.avg:.3f}\t'
                'Generator Loss {loss.avg:.3f}\t'
                'TeacherPrec@1 {top1.avg:.4f}\t'
                'StudentPrec@1 {top5.avg:.4f}'.format(
                epoch, i, len(trainloader), batch_time=self.batch_time,
                data_time=self.data_time, advloss=self.adversarialLoss,
                cntloss=self.crossentropyLoss,reconloss=self.reconstructionLoss,
                tdiscloss=self.totalDiscLoss,genloss=self.generatorLoss,
                top1=self.teachertop1, top5=self.studenttop1))

        # log to TensorBoard
        if opt.tensorboard:
            self.logger.scalar_summary('train_loss', self.losses.avg, epoch)
            self.logger.scalar_summary('train_acc', self.top1.avg, epoch)
            self.logger.scalar_summary('train_loss', self.losses.avg, epoch)
            self.logger.scalar_summary('train_acc', self.top1.avg, epoch)
            self.logger.scalar_summary('train_loss',  self.teachertop1.avg, epoch)
            self.logger.scalar_summary('train_acc', self.studenttop1.avg, epoch)

        print('Train: [{0}]\t'
        'Time {batch_time.sum:.3f}\t'
        'Data {data_time.sum:.3f}\t'
        'Adv Loss {loss.avg:.3f}\t'
        'CrossEnt Loss {loss.avg:.3f}\t'
        'Reconst Loss {loss.avg:.3f}\t'
        'TotalDisc Loss {loss.avg:.3f}\t'
        'Generator Loss {loss.avg:.3f}\t'
        'TeacherPrec@1 {top1.avg:.4f}\t'
        'StudentPrec@1 {top5.avg:.4f}\t'.format(
        epoch, batch_time=self.batch_time,
        data_time= self.data_time, loss=self.losses,
        top1=self.top1, top5=self.top5))

class Validator():
    def __init__(self, student, teacher, discriminator, opt, logger):
        self.opt = opt
        self.logger = logger
        self.discriminator = discriminator
        self.student = student
        self.teacher = teacher

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.teachertop1 = AverageMeter()
        self.studenttop1 = AverageMeter()

    def validate(self, valloader, epoch, opt):
        self.teacher.eval()
        self.student.eval()

        #self.classifier.eval()
        self.teachertop1.reset()
        self.studenttop1.reset()
        self.data_time.reset()
        self.batch_time.reset()
        end = time.time()

        for i, (input, target) in enumerate(valloader, 0):
            if opt.cuda:
                input = input.cuda(async=True)
                target = target.cuda(async=True)

            input, target_var = Variable(input, volatile=True), Variable(target, volatile=True)
            self.data_time.update(time.time() - end)

            teacher_out = self.teacher(input)
            student_out = self.student(input)

            #teacher_target = self.classifier(teacher_feats)
            #student_target = self.classifier(student_feats)

            teacherprec1, teacherprec5 = precision(teacher_out.data, target, topk=(1,5))
            studentprec1, studentprec5 = precision(student_out.data, target, topk=(1,5))

            self.teachertop1.update(teacherprec1[0], input.size(0))
            self.studenttop1.update(studentprec1[0], input.size(0))

        if opt.tensorboard:
            self.logger.scalar_summary('Teacher Accuracy', self.teachertop1.avg, epoch)
            self.logger.scalar_summary('Student Accuracy', self.studenttop1.avg, epoch)

        print('Val: [{0}]\t'
        'Time {batch_time.sum:.3f}\t'
        'Data {data_time.sum:.3f}\t'
        'TeacherPrec@1 {teachertop1.avg:.4f}\t'
        'StudentPrec@1 {studenttop1.avg:.4f}\t'.format(
        epoch, batch_time=self.batch_time,
        data_time= self.data_time,
        teachertop1=self.teachertop1, studenttop1=self.studenttop1))
