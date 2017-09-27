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
    def __init__(student, teacher, discriminator, classifier, classifyCriterion, advCriterion, softcriterion, studOptim, discOptim, opt, logger):
        self.opt = opt
        self.logger = logger

		self.discriminator = discriminator
        self.student = student
        self.teacher = teacher
        self.classifier = classifier
        self.classifyCriterion = classifyCriterion
        self.advCriterion = advCriterion
        self.softcriterion = softcriterion
        self.criterion = criterion
        self.studOptim = studOptim
        self.discOptim = discOptim

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

	def train(self, trainloader, epoch, opt):

        self.discriminator.train()
        self.student.train()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()
        self.data_time.reset()
        self.batch_time.reset()

        end = time.time()
        for i, (input, target) in enumerate(trainloader, 0):
            #Generate fake samples
            isFakeTeacher = torch.ones(input.size(0))

			if epoch%label_reversal_freq == 0:
			     isFakeTeacher.fill_(0)

			if opt.cuda:
                input = input.cuda(async=True)
                target = target.cuda(async=True)
                labels = isFakeTeacher.cuda(async=True)

            input, target_var, labels = Variable(input), Variable(target), Variable(labels)

			self.data_time.update(time.time() - end)

            #Training the discriminator using Teacher
			discOptim.zero_grad()

			teacher_feats, logits = teacher(input)
			isReal, y_discriminator = discriminator(teacher_feats)
			adversarialLoss = adversarialCriterion(isReal,labels)
			crosssentropyLoss = opt.weight_classify * classifyCriterion(y_discriminator,target_var)

			totalDiscLoss = adversarialLoss + crosssentropyLoss
			totalDiscLoss.backward()
			discOptim.step()

            self.adversarialLoss.update(adversarialLoss.data[0], input.size(0))
            self.crosssentropyLoss.update(crosssentropyLoss.data[0], input.size(0))

            #Training the discriminator using student
            discOptim.zero_grad()

            isFakeStudent = 1 - isFakeTeacher
    		isFakeStudent = Variable(isFakeStudent)
            student_feats = student(input)
    		isReal, y_discriminator = discriminator(student_feats.detach())  #To avoid computing gradients in Generator (smallnet)

    		adversarialLoss = adversarialCriterion(isReal,labels)
			crosssentropyLoss = opt.weight_classify * classifyCriterion(y_discriminator,target)
    		totalDiscLoss = adversarialLoss + crosssentropyLoss
			totalDiscLoss.backward()
			discOptim.step()

    	    self.adversarialLoss.update(adversarialLoss.data[0], input.size(0))
            self.crosssentropyLoss.update(crosssentropyLoss.data[0], input.size(0))

    		# Training the student network
    		studOptim.zero_grad()
    		student_feats = student(input)

    		isReal, y_discriminator = discriminator(student_feats)  #To avoid computing gradients in Generator (smallnet)

    		isFake = torch.ones(isReal.size(0))
    		isFake = Variable(isFake)

    		# disc_loss = criterion_adv(out_discr,labels)
    		adversarialLoss = adversarialCriterion(isReal,isFake)
    		reconstructionLoss = reconsructionCriterion(student_feats,teacher_feats)
    		crosssentropyLoss = opt.weight_classify * classifyCriterion(y_discriminator,target)

    		generatorLoss = opt.weight_reconstruction * reconstructionLoss + opt.weight_adversarial * adversarialLoss + crosssentropyLoss
    		generatorLoss.backward()
    		studOptim.step()

             self.adversarialLoss.update(adversarialLoss.data[0], input.size(0))
             self.crosssentropyLoss.update(crosssentropyLoss.data[0], input.size(0))
             self.reconstructionLoss.update(reconstructionLoss.data[0], input.size(0))
             self.generatorLoss.update(crosssentropyLoss.data[0], input.size(0))

    	    if opt.verbose == True and i % opt.printfreq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                'Time {batch_time.avg:.3f} ({batch_time.sum:.3f})\t'
                'Data {data_time.avg:.3f} ({data_time.sum:.3f})\t'
                'Loss {loss.avg:.3f}\t'
                'Prec@1 {top1.avg:.4f}\t'
                'Prec@5 {top5.avg:.4f}'.format(
                epoch, i, len(trainloader), batch_time=self.batch_time,
                data_time=self.data_time, loss=self.losses,
                top1=self.top1, top5=self.top5))

        # log to TensorBoard
        if opt.tensorboard:
            self.logger.scalar_summary('train_loss', self.losses.avg, epoch)
            self.logger.scalar_summary('train_acc', self.top1.avg, epoch)
            self.logger.scalar_summary('train_loss', self.losses.avg, epoch)
            self.logger.scalar_summary('train_acc', self.top1.avg, epoch)
            self.logger.scalar_summary('train_loss', self.losses.avg, epoch)
            self.logger.scalar_summary('train_acc', self.top1.avg, epoch)

        print('Train: [{0}]\t'
        'Time {batch_time.sum:.3f}\t'
        'Data {data_time.sum:.3f}\t'
        'Loss {loss.avg:.3f}\t'
        'Prec@1 {top1.avg:.4f}\t'
        'Prec@5 {top5.avg:.4f}\t'.format(
        epoch, batch_time=self.batch_time,
        data_time= self.data_time, loss=self.losses,
        top1=self.top1, top5=self.top5))

class Validator():
    def __init__(student, teacher, discriminator, classifier, classifycriterion, adversarialcriterion, softcriterion, studentoptimizer, discriminatoroptimizer, opt, logger):

        self.model = model
        self.criterion = criterion
        self.logger = logger
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.teachertop1 = AverageMeter()
        self.studenttop1 = AverageMeter()

    def validate(self, valloader, epoch, opt):

        self.teacher.eval()
        self.student.eval()
        self.classifier.eval()
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

			teacher_feats, teacher_outputs = self.teacher(input)
			student_feats, student_outputs = self.student(input)

			teacher_target = self.classifier(teacher_feats)
			student_target = self.classifier(student_feats)

            teacherprec1, teacherprec5 = precision(teacher_target.data, target, topk=(1,5))
            studentprec1, studentprec5 = precision(student_target.data, target, topk=(1,5))

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
