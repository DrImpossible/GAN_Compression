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
    def __init__(self, model, criterion, optimizer, opt, logger):
		self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

	def train(self, trainloader, epoch, opt):

        self.model.train()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()
        self.data_time.reset()
        self.batch_time.reset()

        end = time.time()
        for i, (input, target) in enumerate(trainloader, 0):
			if opt.cuda:
                input = input.cuda(async=True)
                target = target.cuda(async=True)

            input, target_var = Variable(input), Variable(target)
			real_or_fake = torch.FloatTensor(input.size(0))
			if epoch%label_reversal_freq == 0:
			     real_or_fake.fill_(0)
			else:
			     real_or_fake.fill_(1)
			labels = Variable(real_or_fake.cuda(async=True))
			self.data_time.update(time.time() - end)

			discr_opt.zero_grad()
			vgg_feats, logits = vgg(images)
			outs_discr, outs_dcfier = discr(vgg_feats)
			disc_loss = criterion_adv(outs_discr,labels)
			discfier_loss = cfier_coeff_later* criterion_cfierdiscr(outs_dcfier,lbls)

			total_loss = disc_loss + discfier_loss
			total_loss.backward()
			discr_opt.step()

			total_discr_loss += disc_loss.data[0]
			total_d_cfier_loss += discfier_loss[0]

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

	#discr.train()
	for batch_i, (imgs,lbls) in enumerate(trainloader):
		discr_opt.zero_grad()
		vgg_feats, logits = vgg(images)
		if to_cuda:
			vgg_feats  = vgg_feats.cuda()
		outs_discr, outs_dcfier = discr(vgg_feats)
		if to_cuda:
		    outs_discr  = outs_discr.cuda()
		    outs_dcfier = outs_dcfier.cuda()
		disc_loss = criterion_adv(outs_discr,labels)
		#disc_loss.backward()
		discfier_loss = cfier_coeff_later* criterion_cfierdiscr(outs_dcfier,lbls)
		#discfier_loss.backward()
		total_loss = disc_loss + discfier_loss
		total_loss.backward()
		discr_opt.step()

		total_discr_loss += disc_loss.data[0]
		total_d_cfier_loss += discfier_loss[0]


		#Fake samples
		outs_smallnet = smallnet(images)
		real_or_fake = torch.FloatTensor(outs_smallnet.size(0))
		if epoch%label_reversal_freq == 0:
		     real_or_fake.fill_(1)
		else:
		     real_or_fake.fill_(0)
		labels = Variable(real_or_fake)
		if to_cuda:
		    feats,labels = outs_smallnet.cuda(), labels.cuda()

		discr_opt.zero_grad()
		#outs_discr = discr(feats)
		outs_discr, outs_dcfier = discr(feats.detach())  #To avoid computing gradients in Generator (smallnet)

		if to_cuda:
		    outs_discr  = outs_discr.cuda()
		    outs_dcfier = outs_dcfier.cuda()

		disc_loss = criterion_adv(outs_discr,labels)
		discfier_loss = cfier_coeff_later* criterion_cfierdiscr(outs_dcfier,lbls)
		total_loss = disc_loss + discfier_loss
		total_loss.backward()
		discr_opt.step()

		total_discr_loss += disc_loss.data[0]
		total_d_cfier_loss += discfier_loss[0]




		# Train Generator
		smallnet.train()

		smallnet_opt.zero_grad()
		# images = Variable(imgs)
		# labels = Variable(lbls)
		# if to_cuda:
		#     images,labels = images.cuda(), labels.cuda()
		feats = smallnet(images)
		if to_cuda:
		    feats = feats.cuda()

		out_discr, outs_dcfier = discr(feats)

		real_or_fake = torch.FloatTensor(out_discr.size(0))
		real_or_fake.fill_(1) #Clever stuff. Generator expects discriminator to classify its output as real
		labels = Variable(real_or_fake)

		if to_cuda:
		    out_discr,labels,outs_dcfier = out_discr.cuda(), labels.cuda(), outs_dcfier.cuda()

		# disc_loss = criterion_adv(out_discr,labels)
		adv_loss = criterion_adv(out_discr,labels)
		recons_loss = criterion_rcns(feats,vgg_feats)
		discfier_loss = cfier_coeff_later* criterion_cfierdiscr(outs_dcfier,lbls)

		gen_loss = alpha*recons_loss + beta*adv_loss + discfier_loss
		gen_loss.backward()
		smallnet_opt.step()

		total_gen_loss += gen_loss.data[0]
		total_adv_loss += adv_loss.data[0]
		total_recons_loss += recons_loss.data[0]
		total_g_cfier_loss += discfier_loss.data[0]

	print("Generator:", total_gen_loss / len(trainloader.dataset))
	print("Discriminator:", total_discr_loss / len(trainloader.dataset))

	#visdom disc,gen loss
	viz.line(X=torch.ones((1, 2)).cpu() * epoch,
	    Y=np.reshape(np.array([total_discr_loss,total_d_cfier_loss]),(1,2)),
	win=nets_plots,
	update='append'
	)

	viz.line(X=torch.ones((1, 4)).cpu() * epoch,
	    Y=np.reshape(np.array([beta*total_adv_loss,alpha*total_recons_loss,total_gen_loss,total_g_cfier_loss]),(1,4)),
	win=gen_plots,
	update='append'
	)

class Validator():
    def __init__(self, model, criterion, opt, logger):

        self.model = model
        self.criterion = criterion
        self.logger = logger
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

    def validate(self, valloader, epoch, opt):

        self.model.eval()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()
        self.data_time.reset()
        self.batch_time.reset()
        end = time.time()
		#------Testing-----------
		smallnet.eval()
		classifier.eval()
		vgg.eval()
		total = 0
		vgg_correct = 0
		gen_correct = 0
		if (epoch+1)%5 == 0:
			print('Testing--------------------------------------')
			for batch_i, (imgs,lbls) in enumerate(testloader):


        for i, (input, target) in enumerate(valloader, 0):
			if opt.cuda:
                input = input.cuda(async=True)
                target = target.cuda(async=True)

			input_var, target_var = Variable(input, volatile=True), Variable(target, volatile=True)
			self.data_time.update(time.time() - end)

			vgg_feats, vgg_outputs = vgg(input_var)
			gen_feats = smallnet(input_var)

			vgg_outputs = classifier(vgg_feats)
			gen_outputs = classifier(gen_feats)

			_, vgg_predicted = torch.max(vgg_outputs.data, 1)
			_, gen_predicted = torch.max(gen_outputs.data, 1)
			total += labels.size(0)

			vgg_correct += (vgg_predicted == labels).sum()
			gen_correct += (gen_predicted == labels).sum()

		print('Test Accuracy of VGG : %.2f %%' % (100.0 * vgg_correct / total))
		print('Test Accuracy of Generator: %.2f %%' % (100.0 * gen_correct / total))
		gen_acc = (100.0 * gen_correct / total)


		if opt.tensorboard:
            self.logger.scalar_summary('vgg_acc', (100.0 * vgg_correct / total), epoch)
            self.logger.scalar_summary('gen_acc', (100.0 * gen_correct / total), epoch)

		print('Val: [{0}]\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss {loss.avg:.3f}\t'
              'Prec@1 {top1.avg:.4f}\t'
              'Prec@5 {top5.avg:.4f}\t'.format(
               epoch, batch_time=self.batch_time,
               data_time= self.data_time, loss=self.losses,
               top1=self.top1, top5=self.top5))
