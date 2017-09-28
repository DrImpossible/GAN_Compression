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


        for i, (input, target) in enumerate(trainloader, 0):
            end = time.time()
            #Generate fake samples
            isFakeTeacher = torch.ones(input.size(0))

            #if epoch%opt.revLabelFreq == 0:
            #    isFakeTeacher.fill_(0)

            if opt.cuda:
                input = input.cuda(async=True)
                target = target.cuda(async=True)
                isFakeTeacher = isFakeTeacher.cuda(async=True)

            input, target_var, isFakeTeacher = Variable(input), Variable(target), Variable(isFakeTeacher)
            self.data_time.update(time.time() - end)

            #Training the discriminator using Teacher
            self.discOptim.zero_grad()

            teacher_out = self.teacher(input)
            isReal, y_discriminator = self.discriminator(teacher_out)
            #print(teacher_out.size(),target.size())
            #print(target)
            teacherprec1, teacherprec5 = precision(teacher_out.data, target, topk=(1,5))
            #print(isReal.size(),isFakeTeacher.size())
            adversarialLoss = self.advCriterion(isReal,isFakeTeacher)
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

            student_out = self.student(input)
            isReal, y_discriminator = self.discriminator(student_out.detach())  #To avoid computing gradients in Generator (smallnet)

            studentprec1, studentprec5 = precision(student_out.data, target, topk=(1,5))
            #print(isReal.size(),isFakeTeacher.size())
            adversarialLoss = self.advCriterion(isReal,isFakeTeacher)
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

            #print(isReal.size(),isFakeStudent.size())
            # disc_loss = criterion_adv(out_discr,isFakeTeacher)
            adversarialLoss = self.advCriterion(isReal,isFakeStudent)
            reconstructionLoss = self.derivativeCriterion(student_out,teacher_out)
            crossentropyLoss = self.classifyCriterion(y_discriminator,target_var)

            generatorLoss = opt.weight_reconstruction * reconstructionLoss + opt.weight_adversarial * adversarialLoss + opt.weight_classify * crossentropyLoss
            generatorLoss.backward()
            self.studOptim.step()

            self.batch_time.update(time.time() - end)

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
                'Adv Loss {advloss.avg:.3f}\t'
                'CrossEnt Loss {cntloss.avg:.3f}\t'
                'Reconst Loss {reconloss.avg:.3f}\t'
                'TotalDisc Loss {tdiscloss.avg:.3f}\t'
                'Generator Loss {genloss.avg:.3f}\t'
                'TeacherPrec@1 {teachertop1.avg:.4f}\t'
                'StudentPrec@1 {studenttop1.avg:.4f}'.format(
                epoch, i, len(trainloader), batch_time=self.batch_time,
                data_time=self.data_time, advloss=self.adversariallossLog,
                cntloss=self.crossentropylossLog,reconloss=self.reconstructionlossLog,
                tdiscloss=self.totaldisclossLog,genloss=self.generatorlossLog,
                teachertop1=self.teachertop1, studenttop1=self.studenttop1))

        # log to TensorBoard
        if opt.tensorboard:
            self.logger.scalar_summary('Adv Loss', self.adversariallossLog.avg, epoch)
            self.logger.scalar_summary('CrossEnt Loss', self.crossentropylossLog.avg, epoch)
            self.logger.scalar_summary('Reconst Loss', self.reconstructionlossLog.avg, epoch)
            self.logger.scalar_summary('TotalDisc Loss', self.totaldisclossLog.avg, epoch)
            self.logger.scalar_summary('Generator Loss',  self.generatorlossLog.avg, epoch)
            self.logger.scalar_summary('TeacherPrec@1', self.teachertop1.avg, epoch)
            self.logger.scalar_summary('StudentPrec@1', self.studenttop1.avg, epoch)

        print('Train: [{0}]\t'
        'Time {batch_time.sum:.3f}\t'
        'Data {data_time.sum:.3f}\t'
        'Adv Loss {advloss.avg:.3f}\t'
        'CrossEnt Loss {cntloss.avg:.3f}\t'
        'Reconst Loss {reconloss.avg:.3f}\t'
        'TotalDisc Loss {tdiscloss.avg:.3f}\t'
        'Generator Loss {genloss.avg:.3f}\t'
        'TeacherPrec@1 {teachertop1.avg:.4f}\t'
        'StudentPrec@1 {studenttop1.avg:.4f}\t'.format(
        epoch, batch_time=self.batch_time,
        data_time=self.data_time, advloss=self.adversariallossLog,
        cntloss=self.crossentropylossLog,reconloss=self.reconstructionlossLog,
        tdiscloss=self.totaldisclossLog,genloss=self.generatorlossLog,
        teachertop1=self.teachertop1, studenttop1=self.studenttop1))

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
            self.batch_time.update(time.time() - end)
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

        return self.studenttop1.avg
