
from scripts.models.models import UNet

from scripts.utils.dataloader import ImageHeatmapDataset, showbatch


import torch


from torch.utils.data import DataLoader

from glob import glob

image_paths = glob('/home/saiteja/extra/signgan/SignLangaugeRecognition/output/imgs/*.png')
heatmap_paths = glob('/home/saiteja/extra/signgan/SignLangaugeRecognition/output/heatmaps/*.npy')

# print('Number of images: ', len(image_paths))
# print('Number of heatmaps: ', len(heatmap_paths))

#sort them
image_paths.sort()
heatmap_paths.sort()

dataset = ImageHeatmapDataset(image_paths, heatmap_paths)

# Set the split lengths
train_length = int(len(dataset) * 0.8)
val_length = int(len(dataset) * 0.15)
test_length = len(dataset) - train_length - val_length

# Use random_split to split the dataset
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_length, val_length, test_length])




# dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# get a sample batch
# images, heatmaps = next(iter(dataloader))

# print('Images shape: ', images.shape)
# print('Heatmaps shape: ', heatmaps.shape)

# showbatch(images[2], heatmaps[2])

# print(kj)

#UNIT TEST
# test_unet = UNet(22, 3)
# print(test_unet)

# import torch
# print(test_unet(torch.randn(1, 22, 512, 512)).shape)
# print(nk)
# assert tuple(test_unet(torch.randn(1, 30, 256, 256)).shape) == (1, 3, 117, 117)
# print("Success!")

import torch.nn.functional as F


# criterion as L1 loss


n_epochs = 200
input_dim = 1
label_dim = 1
display_step = 20
batch_size = 4
lr = 0.0002
initial_shape = 512
target_shape = 373
device = 'cuda'


import torch
import torch.nn as nn


import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def forward(self, y_true, y_pred):
        # Mean squared error between the true and predicted images
        mse = self.mse(y_true, y_pred)

        # Structural similarity index (SSIM) between the true and predicted images
        ssim = 1 - self.ssim(y_pred, y_true)

        # Combine the MSE and SSIM into a single loss value
        loss = 0.5 * mse + 0.5 * ssim

        return loss, ssim, mse






from tqdm import tqdm


from torch.optim import Adam


num_epochs = 100
model = UNet(22, 3)

#convert to devicee

try:
    model = model.to(device)
except:
    print('No cuda')

loss_fn = Loss()
optimizer = Adam(model.parameters(), lr=1e-3)


dataloader = DataLoader(dataset, batch_size=1, shuffle=True)



import wandb

wandb.init(project='SignGAN', entity='saisritejak')


batch_log = 50


#get early stopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Initialize the scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

batch_size = 1

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def compute_val_loss(val_dataloader, model, loss_fn):
    with torch.no_grad():
        total_val_loss = 0
        for val_images, val_heatmaps in val_dataloader:
            val_images = val_images.to(torch.float).to(device)
            val_heatmaps = val_heatmaps.to(torch.float).to(device)
            val_generated_images = model(val_heatmaps)
        val_batch_loss,_,_ = loss_fn(val_images, val_generated_images)
        total_val_loss += val_batch_loss.item()
    return total_val_loss / len(val_dataloader)



# Initialize early stopping
class EarlyStopping:
    def __init__(self, patience=5, threshold=1e-4, threshold_mode='rel', metric_name='val_loss'):
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.metric_name = metric_name
        self.best = None
        self.num_bad_epochs = 0
        self.mode = 'min'

    def step(self, val_loss):
        current = val_loss
        if self.best is None:
            self.best = current
            return False
        if ((self.mode == 'min' and current < self.best - self.threshold) or
                (self.mode == 'max' and current > self.best + self.threshold)):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            return True

        return False


early_stopping = EarlyStopping(patience=5, metric_name='val_loss', threshold=0.01)



batch_log = 50


def train():
    # Iterate over the number of training epochs
    for epoch in range(num_epochs):
        # Iterate over the training data

        loss = 0
        cur_step = 0

        for images, heatmaps in tqdm(dataloader, desc="Epoch {}/{}".format(epoch+1, num_epochs)):
            # Move the data to the GPU (if available)
            images = images.to(torch.float).to(device)
            heatmaps = heatmaps.to(torch.float).to(device)

            # Forward pass
            generated_images = model(heatmaps)
            
            # Compute the loss
            batch_loss, ssim, mse = loss_fn(images, generated_images)

            #log the ssim and mse
            wandb.log({'ssim': ssim})
            wandb.log({'mse': mse})

            # Zero the gradients
            optimizer.zero_grad()

            # Backward pass
            batch_loss.backward()

            # Update the weights
            optimizer.step()

            # Add the loss to the total loss for the epoch
            loss += batch_loss.item()

            if cur_step % batch_log == 0:
                #log the loss
                wandb.log({'loss': batch_loss.item()})

                #log the images
                wandb.log({'images': [wandb.Image(images[0]), wandb.Image(generated_images[0])]})

            cur_step =  cur_step + 1

            # break
            
        # Print the current loss value
        print("Epoch: {}, Loss: {:.4f}".format(epoch+1, loss / len(dataloader)))

        #log the loss
        wandb.log({'loss': loss})

        #log the epoch
        wandb.log({'epoch': epoch})
        
        # Compute the validation loss
        val_loss = compute_val_loss(val_dataloader, model, loss_fn)
        wandb.log({'val_loss': val_loss})

        torch.save(model, f'models/unet_model_{epoch}.pt')



        #Check if the validation loss is less than the training loss
        # if early_stopping.step(val_loss):
        #     print("Early stopping at epoch {}".format(epoch+1))
        #     break
        




    # torch.save(model, 'unet_model.pt')



train()
torch.save(model, 'unet_model.pt')
