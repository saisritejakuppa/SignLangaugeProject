from options.base_options import BaseOptions
from data.dataloader import CreateDataLoader
from models.create_model import GetModels
import torch
from models.networks import weights_init

opt = BaseOptions().parse()


# opt.image_paths =   './imgs'
# opt.heatmap_paths = './heatmaps'
print(opt)


# train_dataloader, val_dataloader, test_dataloader = CreateDataLoader(opt)
gen, disc = GetModels(opt)

gen_opt = torch.optim.Adam(gen.parameters(), lr=opt.lr)
disc_opt = torch.optim.Adam(disc.parameters(), lr=opt.lr)

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

# print(gen)
# print(disc)

#sample torch of size 1,23,256,256
# heatmap = torch.randn(4,23,256,256)
# sample_img = torch.randn(4,3,256,256)
# img = gen(heatmap)
# print('Output image is',img.shape)
# print('Sample image is',sample_img.shape)

# disc_out = disc(img, heatmap)
# print('Discriminator output is',disc_out.shape)


# from losses.loss import Loss
# loss = Loss(opt)
 
# total_loss


