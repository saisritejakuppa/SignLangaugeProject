from options.base_options import BaseOptions
from data.dataloader import CreateDataLoader
from models.create_model import GetModels
import torch
from models.networks import weights_init
from tqdm import tqdm
from torch import nn
from losses.loss import Loss

def train(gen, disc, dataloaders, opt):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    train_dataloader, val_dataloader, test_dataloader = dataloaders
    cur_step = 0

    for epoch in range(opt.n_epochs):
        for heatmaps, images in tqdm(train_dataloader):
            condition = nn.functional.interpolate(heatmaps, size=opt.target_shape)
            real = nn.functional.interpolate(images, size=opt.target_shape)

            cur_batch_size = len(condition)

            #to cuda
            condition = condition.to(opt.device)
            real = real.to(opt.device)

            #update discriminator
            disc_opt.zero_grad()
            with torch.no_grad():
                fake = gen(condition)
            
            disc_fake_hat = disc(fake.detach(), condition)    # imp step, detach the generator
            disc_fake_loss = nn.BCEWithLogitsLoss()(disc_fake_hat, torch.zeros_like(disc_fake_hat))

            disc_real_hat = disc(real, condition)
            disc_real_loss = nn.BCEWithLogitsLoss()(disc_real_hat, torch.ones_like(disc_real_hat))

            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            #update generator
            gen_opt.zero_grad()
            gen_loss, all_losses  = Loss(opt)(real, condition, gen, disc)
            gen_loss.backward()
            gen_opt.step()

            mean_generator_loss += gen_loss.item() / opt.print_every
            mean_discriminator_loss += disc_loss.item() / opt.print_every

            if cur_step % opt.print_every == 0:
                if cur_step > 0:
                    print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                else:
                    print("Pretrained initial state")
                mean_generator_loss = 0
                mean_discriminator_loss = 0

                #save the model
                if opt.save_model:
                        torch.save({'gen': gen.state_dict(),
                            'gen_opt': gen_opt.state_dict(),
                            'disc': disc.state_dict(),
                            'disc_opt': disc_opt.state_dict()
                        }, f"pix2pix_{cur_step}.pth")

            cur_step += 1
            

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

# loss = 
train(gen, disc, CreateDataLoader(opt), opt)











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


