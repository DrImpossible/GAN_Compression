import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import utils

class LoadMNIST():
	'''
	Downloads and loads the MNIST dataset.
	Preprocessing -> Data is normalized in Transforms.
	'''
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			datasets.MNIST(opt.data_dir, train=True, download=True,
					transform=transforms.Compose([
						transforms.RandomCrop(28, padding=4),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,))
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			datasets.MNIST(opt.data_dir, train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
			  **kwargs)

class LoadCIFAR10():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10(opt.data_dir, train=True, download=True,
					transform=transforms.Compose([
						transforms.RandomCrop(32, padding=4),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
std=[x/255.0 for x in [63.0, 62.1, 66.7]])
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10(opt.data_dir, train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
std=[x/255.0 for x in [63.0, 62.1, 66.7]])
					   ])),
		  **kwargs)
		self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class LoadCIFAR100():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR100(opt.data_dir, train=True, download=True,
					transform=transforms.Compose([
						transforms.RandomCrop(32, padding=4),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize(mean=[x/255.0 for x in [129.3, 124.1, 112.4]],
std=[x/255.0 for x in [68.2, 65.4, 70.4]])
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			datasets.CIFAR100(opt.data_dir, train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[x/255.0 for x in [129.3, 124.1, 112.4]],
std=[x/255.0 for x in [68.2, 65.4, 70.4]])
					   ])),
		  **kwargs)

class LoadImagenet12():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		data_transforms = {
			'train': transforms.Compose([
				transforms.RandomSizedCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]),
			'val': transforms.Compose([
				#transforms.Scale(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
		}

		data_dir = opt.data_dir
		dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
		self.dsets = dsets
		self.train_loader = torch.utils.data.DataLoader(dsets["train"], **kwargs)
		self.val_loader = torch.utils.data.DataLoader(dsets["val"], **kwargs)
