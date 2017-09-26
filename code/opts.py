import argparse

dset_choices = ['cifar10','cifar100','imagenet12']
reporttype_choices = ['acc']
criterion_choices = ['crossentropy']
optim_choices = ['sgd','adam']
model_def_choices = ['vgg16_bn']

def myargparser():
    parser = argparse.ArgumentParser(description='GAN Compression...')

    #data stuff
    parser.add_argument('--dataset', choices=dset_choices, help='chosen dataset'+'Options:'+str(dset_choices))
    parser.add_argument('--data_dir', required=True, help='Dataset directory')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (Default: 4)')
    parser.add_argument('--weight_init', action='store_true', help='Turns on weight inits')
    #other default stuff
    parser.add_argument('--epochs', required=True, type=int,help='number of total epochs to run')
    parser.add_argument('--expandConfig',  type=int, help='Configuration')
    parser.add_argument('--batch-size', required=True, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--nclasses', help='number of classes', default=0)
    parser.add_argument('--tenCrop', action='store_true', help='ten-crop testing')
    parser.add_argument('--printfreq', default=200, type=int, help='print frequency (default: 10)')
    parser.add_argument('--learningratescheduler', default='decayschedular', help='print frequency (default: 10)')

    #optimizer/criterion stuff
    parser.add_argument('--decayinterval', type=int, help='decays by a power of decay_var in these epochs')
    parser.add_argument('--decaylevel', type=int, help='decays by a power of decaylevel')
    parser.add_argument('--criterion', default="crossentropy", choices=criterion_choices, type=str, help='Criterion. Options:'+str(criterion_choices))
    parser.add_argument('--optimType', required=True, choices=optim_choices, type=str, help='Optimizers. Options:'+str(optim_choices))

    parser.add_argument('--maxlr', required=True, type=float, help='initial learning rate')
    parser.add_argument('--lr', type=float, help='initial learning rate')
    parser.add_argument('--minlr', required=True, type=float, help='initial learning rate')

    parser.add_argument('--nesterov', action='store_true', help='nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (Default: 0.9)')
    parser.add_argument('--weightDecay', default=0, type=float, help='weight decay (Default: 1e-4)')

    #extra model stuff
    parser.add_argument('--model_def', required=True, choices=model_def_choices, help='Architectures to be loaded. Options:'+str(model_def_choices))
    parser.add_argument('--name', required=True, type=str, help='name of experiment')
    #default
    parser.add_argument('--cachemode', default=True, help='if cachemode')
    parser.add_argument('--cuda', default=True, help='If cuda is available')
    parser.add_argument('--manualSeed', type=int, default=123, help='fixed seed for experiments')
    parser.add_argument('--ngpus', type=int, default=1, help='no. of gpus')
    parser.add_argument('--logdir', type=str, default='../logs', help='log directory')
    parser.add_argument('--tensorboard',help='Log progress to TensorBoard', default=True)
    parser.add_argument('--testOnly',  default=False, help='run on validation set only')
    parser.add_argument('--acc_type', default = "class")

    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--pretrained_file', default='')


    #Hyperparameters
    parser.add_argument('--smallnet_lr_init', default=100, type=int,
                    help='total number of layers (default: 100)')
    parser.add_argument('--discr_lr_init', default=2, type=int,
                    help='factor to compress by')
    parser.add_argument('--adv_loss_coeff_init', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
    parser.add_argument('--recons_loss_coeff_init', default=0, type=float,
                    help='dropout probability (default: 0.0)')
    parser.add_argument('--cfier_coeff_init', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
    parser.add_argument('--adv_loss_wakeup_epoch', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
    parser.add_argument('--smallnet_lr_later', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')

    parser.add_argument('--discr_lr_later', default=100, type=int,
                    help='total number of layers (default: 100)')
    parser.add_argument('--adv_loss_coeff_later', default=2, type=int,
                    help='factor to compress by')
    parser.add_argument('--recons_loss_coeff_later', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
    parser.add_argument('--cfier_coeff_later', default=0, type=float,
                    help='dropout probability (default: 0.0)')
    parser.add_argument('--label_reversal_freq', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')

    parser.add_argument('--from_modelzoo', action='store_true')
    #model stuff
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--store', default='', type=str, metavar='PATH',
                        help='path to storing checkpoints (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')


    return parser
