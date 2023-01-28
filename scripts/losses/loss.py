
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure
from .vgg import VGGLoss
import torch





class Loss(nn.Module):
    def __init__(self,opt):

        super().__init__()
        
        self.mse = nn.MSELoss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(opt.device)
        self.vgg = VGGLoss()
        self.L1 = nn.L1Loss()
        self.adv_loss = nn.BCEWithLogitsLoss()

        # self.parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        # self.parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')
        # self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        # self.parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')

        self.lambda_L1 = opt.lambda_L1
        self.lambda_GAN = opt.lambda_GAN
        self.lambda_feat = opt.lambda_feat
        self.lambda_vgg = opt.lambda_vgg
        self.ssim = opt.ssim

    def forward(self, real, condition, gen, disc):


        fake = gen(condition)
        disc_fake_hat = disc(fake, condition)    # imp step, detach the generator
        gen_adv_loss = self.adv_loss(disc_fake_hat, torch.ones_like(disc_fake_hat))
        gen_rec_loss = self.L1(real, fake)       #l1 loss

        # Structural similarity index (SSIM) between the true and predicted images
        ssim = 1 - self.ssim(fake, real)

        # Perceptual loss 
        # Extract features from the true and predicted images
        pl = self.vgg(fake, real)


        loss = self.lambda_L1 * gen_rec_loss +   \
               self.lambda_vgg * pl +  \
               self.lambda_feat * ssim + \
                self.lambda_GAN * gen_adv_loss

        all_losses = {'loss': loss,
            'gen_adv_loss': gen_adv_loss,
            'gen_rec_loss': gen_rec_loss,
            'pl': pl,
            'ssim': ssim
            }

        return loss, all_losses

