import mydatasets.load_data as ld

def load_data(opt):
	'''
	Loads the dataset file from load_data.py file.
	'''
	if opt.dataset == "mnist":
		dataloader = ld.LoadMNIST(opt)

	elif opt.dataset == "cifar10":
		dataloader = ld.LoadCIFAR10(opt)

	elif opt.dataset == "cifar100":
		dataloader = ld.LoadCIFAR100(opt)

	elif opt.dataset == "imagenet12":
		dataloader = ld.LoadImagenet12(opt)

	return dataloader
