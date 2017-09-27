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
            isFake = torch.ones(input.size(0))

			if epoch%label_reversal_freq == 0:
			     isFake.fill_(0)

			if opt.cuda:
                input = input.cuda(async=True)
                target = target.cuda(async=True)
                labels = isFake.cuda(async=True)

            input, target_var, labels = Variable(input), Variable(target), Variable(labels)

			self.data_time.update(time.time() - end)

			discOptim.zero_grad()

			teacher_feats, logits = teacher(input)
			y_discriminator, y_dcfier = discriminator(teacher_feats)
			disc_loss = adversarialcriterion(y_discriminator,labels)
			discfier_loss = cfier_coeff_later* criterion_cfierdiscr(y_dcfier,target_var)

			total_loss = disc_loss + discfier_loss
			total_loss.backward()
			discOptim.step()

			total_discr_loss += disc_loss.data[0]
			total_d_cfier_loss += discfier_loss[0]

            #Fake samples
    		student_feats = student(input)
    		discOptim.zero_grad()
    		outs_discr, outs_dcfier = discriminator(student_feats.detach())  #To avoid computing gradients in Generator (smallnet)

    		disc_loss = adversarialcriterion(outs_discr,labels)
    		discfier_loss = cfier_coeff_later* criterion_cfierdiscr(outs_dcfier,target_var)
    		total_loss = disc_loss + discfier_loss
    		total_loss.backward()
    		discOptim.step()

    		total_discr_loss += disc_loss.data[0]
    		total_d_cfier_loss += discfier_loss[0]

    		# Train student network
    		studOptim.zero_grad()
    		student_feats = student(input)

    		out_discr, outs_dcfier = discriminator(student_feats)

    		isFake = torch.ones(out_discr.size(0))
    		labels = Variable(isFake)

    		#if to_cuda:
    		#    out_discr,labels,outs_dcfier = out_discr.cuda(), labels.cuda(), outs_dcfier.cuda()

    		# disc_loss = criterion_adv(out_discr,labels)
    		adv_loss = adversarialcriterion(out_discr,labels)
    		recons_loss = criterion_rcns(feats,vgg_feats)
    		discfier_loss = cfier_coeff_later* criterion_cfierdiscr(outs_dcfier,target_var)

    		gen_loss = opt.alpha*recons_loss + opt.beta*adv_loss + discfier_loss
    		gen_loss.backward()
    		studOptim.step()

    		total_gen_loss += gen_loss.data[0]
    		total_adv_loss += adv_loss.data[0]
    		total_recons_loss += recons_loss.data[0]
    		total_g_cfier_loss += discfier_loss.data[0]

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

        print('Train: [{0}]\t'
        'Time {batch_time.sum:.3f}\t'
        'Data {data_time.sum:.3f}\t'
        'Loss {loss.avg:.3f}\t'
        'Prec@1 {top1.avg:.4f}\t'
        'Prec@5 {top5.avg:.4f}\t'.format(
        epoch, batch_time=self.batch_time,
        data_time= self.data_time, loss=self.losses,
        top1=self.top1, top5=self.top5))

	#if(epoch==adv_loss_wakeup_epoch):
	#    print('Saving..')
	#    state = {
	#        'smallnet': smallnet.module if use_cuda else net,
	#        'discr': discr.module if use_cuda else net,
	#        'gen_acc': gen_acc,
	#        'epoch': epoch,
	#    }
	#    if not os.path.isdir('checkpoint'):
	#        os.mkdir('checkpoint')
	#    torch.save(state, './checkpoint/gancoder1.t7')
	#    # best_acc = acc
	#    alpha=recons_loss_coeff_later
	#    beta=adv_loss_coeff_later
	#    smallnet_opt = optim.Adam(smallnet.parameters(),lr=smallnet_lr_later)
	#    discr_opt = optim.Adam(discr.parameters(),lr=discr_lr_later)

	#total_discr_loss = 0
	#total_gen_loss = 0
	#total_adv_loss = 0
	#total_recons_loss = 0
	#total_d_cfier_loss = 0
	#total_g_cfier_loss = 0
	##train Descriminator

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

			vgg_feats, vgg_outputs = self.teacher(input)
			gen_feats = self.student(input)

			vgg_outputs = self.classifier(vgg_feats)
			gen_outputs = self.classifier(gen_feats)

            teacherprec1, teacherprec5 = precision(vgg_outputs.data, target, topk=(1,5))
            studentprec1, studentprec5 = precision(gen_outputs.data, target, topk=(1,5))

            self.teachertop1.update(prec1[0], input.size(0))
            self.studenttop1.update(prec5[0], input.size(0))

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
