import argparse
import os
import torch


#import the utils folder
from utils import utils

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    
        #input dims
        self.parser.add_argument('--input_nc', type=int, default=22, help='# of input image channels')

        #output dims
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        #display step
        self.parser.add_argument('--display_freq', type=int, default=20, help='frequency of showing training results on screen')

        #learrning rate
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')

        #target shape
        self.parser.add_argument('--target_shape', type=int, default=256, help='target shape for training images')

        #device
        self.parser.add_argument('--device', type=str, default='cuda', help='device to use for training')

        #number of epochs
        self.parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')

        #save model
        self.parser.add_argument('--save_model', type=bool, default=True, help='save the model')

        #print_every
        self.parser.add_argument('--print_every', type=int, default=10, help='print every n steps')

        #losses and lambdas
        self.parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        self.parser.add_argument('--lambda_GAN', type=float, default=20.0, help='weight for GAN loss')
        self.parser.add_argument('--lambda_feat', type=float, default=2.0, help='weight for feature matching loss')
        self.parser.add_argument('--lambda_vgg', type=float, default=2.0, help='weight for vgg loss')
        self.parser.add_argument('--ssim', type=float, default=0.0, help='weight for ssim loss')

        #checkpoints path
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        #save the options to a file
        expr_dir = self.opt.checkpoints_dir
        utils.mkdirs(expr_dir)
        file_name =  os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(vars(self.opt).items()):
                opt_file.write('%s: %s \n' % (str(k), str(v)))

        return self.opt
            




