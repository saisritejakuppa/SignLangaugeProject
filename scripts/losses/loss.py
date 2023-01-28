
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

    def forward(self, y_true, y_pred):

        # Mean squared error between the true and predicted images
        mse = self.mse(y_true, y_pred)

        #l1 loss
        l1 = self.L1(y_true, y_pred)

        # Structural similarity index (SSIM) between the true and predicted images
        ssim = 1 - self.ssim(y_pred, y_true)

        # Perceptual loss 
        # Extract features from the true and predicted images
        pl = self.vgg(y_pred, y_true)

        adv_loss = self.adv_loss(y_pred,  torch.ones_like(y_pred))

        # Combine the MSE, SSIM and perceptual loss into a single loss value
        # loss = 0.4 * mse + 0.3 * ssim + 0.3 * pl + 1 * l1
        loss = self.lambda_L1 * l1 +   \
               self.lambda_vgg * pl +  \
               self.lambda_GAN * mse + \
               self.lambda_feat * ssim + \
                self.lambda_GAN * adv_loss

        return loss, [ssim, l1, pl, mse, adv_loss]

