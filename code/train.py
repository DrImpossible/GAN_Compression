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
    def __init__(self, student, teacher, discriminator, classifyCriterion, advCriterion, similarityCriterion, derivativeCriterion, studOptim, discOptim, opt, logger):
        self.opt, self.logger = opt, logger
        self.discriminator, self.student, self.teacher = discriminator, student, teacher
        #self.classifier = classifier
        self.classifyCriterion = classifyCriterion
        self.advCriterion = advCriterion
        self.similarityCriterion = similarityCriterion
        self.derivativeCriterion = derivativeCriterion
        self.studOptim, self.discOptim = studOptim, discOptim

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.teachertop1 = AverageMeter()
        self.studenttop1 = AverageMeter()
        self.disctop1 = AverageMeter()
        #self.discisreal = AverageMeter()

        self.discadversariallossLog = AverageMeter()
        self.disccrossentropylossLog = AverageMeter()
        self.disctotallossLog = AverageMeter()

        self.studadversariallossLog = AverageMeter()
        self.studcrossentropylossLog = AverageMeter()
        self.studreconstructionlossLog = AverageMeter()
        self.studderivativelossLog = AverageMeter()
        self.studtotallossLog = AverageMeter()

    def computenlogDisc(self, y_discriminator, target, out, isCorrect):
        discriminatortop1 = precision(y_discriminator.data, target)
        #discriminatorisreal = precision(out.data, isCorrect)

        discadversarialLoss = self.opt.wdiscAdv * self.advCriterion(out,Variable(isCorrect))
        disccrossentropyLoss =  self.opt.wdiscClassify * self.classifyCriterion(y_discriminator,Variable(target))
        disctotalLoss = discadversarialLoss + disccrossentropyLoss

        self.disctop1.update(discriminatortop1[0], target.size(0))
        #self.discisreal.update(discriminatorisreal[0], target.size(0))
        self.discadversariallossLog.update(discadversarialLoss.data[0], target.size(0))
        self.disccrossentropylossLog.update(disccrossentropyLoss.data[0], target.size(0))
        self.disctotallossLog.update(disctotalLoss.data[0], target.size(0))

        return disctotalLoss

    def computenlogStud(self, student_out, teacher_out, studentgrad_params, teachergrad_params, y_discriminator, target, out, isCorrect):
        teacherprec1 = precision(teacher_out.data, target)
        studentprec1 = precision(student_out.data, target)

        discadversarialLoss = self.opt.wdiscAdv * self.advCriterion(out,Variable(isCorrect))
        disccrossentropyLoss = self.opt.wdiscClassify * self.classifyCriterion(y_discriminator,Variable(target))
        studreconstructionLoss = self.opt.wstudSim * self.similarityCriterion(student_out,teacher_out.detach())
        studderivativeLoss = self.opt.wstudDeriv * self.derivativeCriterion(studentgrad_params,teachergrad_params.detach())

        studtotalLoss = studreconstructionLoss + studderivativeLoss + discadversarialLoss + disccrossentropyLoss

        self.teachertop1.update(teacherprec1[0], target.size(0))
        self.studenttop1.update(studentprec1[0], target.size(0))

        self.studadversariallossLog.update(discadversarialLoss.data[0], target.size(0))
        self.studcrossentropylossLog.update(disccrossentropyLoss.data[0], target.size(0))
        self.studreconstructionlossLog.update(studreconstructionLoss.data[0], target.size(0))
        self.studderivativelossLog.update(studderivativeLoss.data[0], target.size(0))
        self.studtotallossLog.update(studtotalLoss.data[0], target.size(0))

        return studtotalLoss

    def train(self, trainloader, epoch, opt):
        self.discriminator.train()
        self.student.train()
        self.teacher.train()

        self.discadversariallossLog.reset()
        self.disccrossentropylossLog.reset()
        self.disctotallossLog.reset()

        self.studadversariallossLog.reset()
        self.studcrossentropylossLog.reset()
        self.studreconstructionlossLog.reset()
        self.studderivativelossLog.reset()
        self.studtotallossLog.reset()

        self.data_time.reset()
        self.batch_time.reset()
        self.teachertop1.reset()
        self.studenttop1.reset()
        self.disctop1.reset()
        #self.discisreal.reset()

        self.discOptim.zero_grad()
        self.studOptim.zero_grad()

        for i, (input, target) in enumerate(trainloader, 0):
            end = time.time()

            #Generate fake samples
            isFakeTeacher = torch.ones(input.size(0))
            if epoch%opt.revLabelFreq == 0:
                isFakeTeacher.fill_(0)

            if opt.cuda:
                input, target, isFakeTeacher = input.cuda(async=True), target.cuda(async=True), isFakeTeacher.cuda(async=True)

            input = Variable(input)
            isFakeStudent = 1 - isFakeTeacher
            self.data_time.update(time.time() - end)

            #Forward-passing the Teacher and the Student
            teacher_out = self.teacher(input)
            student_out = self.student(input)
            teachercrossentropyLoss =  self.classifyCriterion(teacher_out,Variable(target))
            teachergrad_params = torch.autograd.grad(teachercrossentropyLoss, self.teacher.parameters(), create_graph=True)

            studcrossentropyLoss =  self.classifyCriterion(student_out,Variable(target))
            studentgrad_params = torch.autograd.grad(studcrossentropyLoss, self.student.parameters(), create_graph=True)
            teachergrad_params,studentgrad_params = teachergrad_params[-1],studentgrad_params[-1]

            #Training the discriminator using Teacher
            isReal, y_discriminator = self.discriminator(teacher_out.detach()) #To avoid computing gradients in Teacher
            disctotalLoss = self.computenlogDisc(y_discriminator, target, isReal, isFakeTeacher)
            disctotalLoss.backward()
            self.discOptim.step()
            self.discOptim.zero_grad()

            #Training the discriminator using student


            isReal, y_discriminator = self.discriminator(student_out.detach())  #To avoid computing gradients in Student
            disctotalLoss = self.computenlogDisc(y_discriminator, target, isReal, isFakeStudent)
            disctotalLoss.backward()
            self.discOptim.step()
            self.discOptim.zero_grad()

            # Training the student network
            isReal, y_discriminator = self.discriminator(student_out)
            studtotalLoss = self.computenlogStud(student_out, teacher_out, studentgrad_params, teachergrad_params, y_discriminator, target, isReal, isFakeTeacher)
            studtotalLoss.backward()
            self.studOptim.step()
            self.studOptim.zero_grad()

            self.batch_time.update(time.time() - end)

            if self.opt.verbose == True and i % self.opt.printfreq == 0:
                print('Batch: [{0}][{1}/{2}]\t'
                'Time {batch_time.avg:.3f} ({batch_time.sum:.3f})\t'
                'Data {data_time.avg:.3f} ({data_time.sum:.3f})\t'
                'DiscAdvLoss {discadvloss.avg:.3f}\t'
                'DiscCrossEntLoss {disccntloss.avg:.3f}\t'
                'DiscTotalLoss {disctotloss.avg:.3f}\t'
                'StudAdvLoss {studadvloss.avg:.3f}\t'
                'StudCrossEntLoss {studcntloss.avg:.3f}\t'
                'StudReconLoss {studreconloss.avg:.3f}\t'
                'StudDerivLoss {studderivloss.avg:.3f}\t'
                'StudTotalLoss {studtotloss.avg:.3f}\t'
                'DiscPrec@1 {discrtop1.avg:.4f}\t'
                #'DiscIsReal {discisreal.avg:.4f}'
                'TeacherPrec@1 {teachertop1.avg:.4f}\t'
                'StudentPrec@1 {studenttop1.avg:.4f}'.format(
                epoch, i, len(trainloader), batch_time=self.batch_time,
                data_time=self.data_time, discadvloss=self.discadversariallossLog,disccntloss=self.disccrossentropylossLog,
                disctotloss=self.disctotallossLog,studadvloss=self.studadversariallossLog,studcntloss=self.studcrossentropylossLog,
                studreconloss=self.studreconstructionlossLog,studderivloss=self.studderivativelossLog,studtotloss=self.studtotallossLog,
                discrtop1=self.disctop1, teachertop1=self.teachertop1, studenttop1=self.studenttop1))

        # log to TensorBoard
        if opt.tensorboard:
            self.logger.scalar_summary('DiscAdvLoss', self.discadversariallossLog.avg, epoch)
            self.logger.scalar_summary('DiscCrossEntLoss', self.disccrossentropylossLog.avg, epoch)
            self.logger.scalar_summary('DiscTotalLoss', self.disctotallossLog.avg, epoch)
            self.logger.scalar_summary('StudAdvLoss', self.studadversariallossLog.avg, epoch)
            self.logger.scalar_summary('StudCrossEntLoss', self.studcrossentropylossLog.avg, epoch)
            self.logger.scalar_summary('StudReconLoss', self.studreconstructionlossLog.avg, epoch)
            self.logger.scalar_summary('StudDerivLoss', self.studderivativelossLog.avg, epoch)
            self.logger.scalar_summary('StudTotalLoss', self.studtotallossLog.avg, epoch)
            self.logger.scalar_summary('DiscPrec1',  self.disctop1.avg.type(torch.FloatTensor)[0], epoch)
            #self.logger.scalar_summary('DiscIsReal',  self.discisreal.avg, epoch)
            self.logger.scalar_summary('TeacherPrec1', self.teachertop1.avg.type(torch.FloatTensor)[0], epoch)
            self.logger.scalar_summary('StudentPrec1', self.studenttop1.avg.type(torch.FloatTensor)[0], epoch)

        print('Train: [{0}]\t'
        'Time {batch_time.sum:.3f}\t'
        'Data {data_time.sum:.3f}\t'
        'DiscAdvLoss {discadvloss.avg:.3f}\t'
        'DiscCrossEntLoss {disccntloss.avg:.3f}\t'
        'DiscTotalLoss {disctotloss.avg:.3f}\t'
        'StudAdvLoss {studadvloss.avg:.3f}\t'
        'StudCrossEntLoss {studcntloss.avg:.3f}\t'
        'StudReconLoss {studreconloss.avg:.3f}\t'
        'StudDerivLoss {studderivloss.avg:.3f}\t'
        'StudTotalLoss {studtotloss.avg:.3f}\t'
        'DiscPrec@1 {discrtop1:.4f}\t'
        #'DiscIsReal {discisreal.avg:.4f}'
        'TeacherPrec@1 {teachertop1:.4f}\t'
        'StudentPrec@1 {studenttop1:.4f}'.format(
        epoch, batch_time=self.batch_time,
        data_time=self.data_time, discadvloss=self.discadversariallossLog,disccntloss=self.disccrossentropylossLog,
        disctotloss=self.disctotallossLog,studadvloss=self.studadversariallossLog,studcntloss=self.studcrossentropylossLog,
        studreconloss=self.studreconstructionlossLog,studderivloss=self.studderivativelossLog,studtotloss=self.studtotallossLog,
        discrtop1=self.disctop1.avg.type(torch.FloatTensor)[0], teachertop1=self.teachertop1.avg.type(torch.FloatTensor)[0], studenttop1=self.studenttop1.avg.type(torch.FloatTensor)[0]))

class Validator():
    def __init__(self, student, teacher, discriminator, opt, logger):
        self.opt, self.logger = opt, logger
        self.discriminator, self.student, self.teacher = discriminator, student, teacher

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


        for i, (input, target) in enumerate(valloader, 0):
            end = time.time()
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
