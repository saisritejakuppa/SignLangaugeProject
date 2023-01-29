import torch
from torch.utils.data import Dataset

#get transforms from torchvision
from torchvision import transforms
from PIL import Image

import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image



class ImageHeatmapDataset(Dataset):
    def __init__(self, image_paths, heatmap_paths):
        self.image_paths = image_paths
        self.heatmap_paths = heatmap_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])

        #crop the image the right half
        image = image[:, 640:, :]

        #reshape the image to 720*1280*3
        image = cv2.resize(image, (1280, 720))

        heatmap = np.load(self.image_paths[idx].replace('.png', '.npy').replace('img_frame', 'heatmaps').replace('imgs', 'heatmaps'))

        #center crop the image to 720*720*3
        image = image[0:720, 280:1000, :]
        heatmap = heatmap[:, 0:720, 280:1000]

        resized_heatmap = []
        for channel in range(heatmap.shape[0]):
            
            if channel < 3:  #head, body, shoulder
              heatmap = heatmap / 255.0

            if channel > 3:
                #normalize the heatmap
                heatmap[channel] = (heatmap[channel] - heatmap[channel].min()) / (heatmap[channel].max() - heatmap[channel].min())

            resized_heatmap.append(cv2.resize(heatmap[channel], (512, 512), interpolation = cv2.INTER_LINEAR))


        resized_heatmap = np.stack(resized_heatmap, axis = 0)

        # max_value = np.max(resized_heatmap)
        # min_value = np.min(resized_heatmap)
        # mean_value = np.mean(resized_heatmap)
        # std = np.std(resized_heatmap)
        # print(max_value, min_value, mean_value, std)


        resized_heatmap = (resized_heatmap - resized_heatmap.min()) / (resized_heatmap.max() - resized_heatmap.min())

        
        # resized_heatmap = resized_heatmap * 255


        resized_heatmap = resized_heatmap.astype(np.float32)

        # max_value = np.max(resized_heatmap)
        # min_value = np.min(resized_heatmap)
        # mean_value = np.mean(resized_heatmap)
        # std = np.std(resized_heatmap)
        # print(max_value, min_value, mean_value, std)


        #make a transform to convert to size of 720, 720
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            #normalize
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        ])

        #convert to a tensor
        image = transform(image)

        #make a transform to convert to size of 720, 720
        #add translation to minimum and rotation and gaussian noise and scaling
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            #add gaussian noise
            # transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)], p=0.5),
            #rotation
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),   
            #normalize
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])

        ])
        resized_heatmap = torch.stack([transform(heatmap) for heatmap in resized_heatmap])
        resized_heatmap = torch.as_tensor(resized_heatmap)
        resized_heatmap = resized_heatmap.reshape(22, 512, 512)


        return image, resized_heatmap
    
import os

def showbatch(images, heatmaps):
    #make the dir data if doesnot exist
    if not os.path.exists('output'):
        os.mkdir('output')

    # print('The shape of the heatmap,' , heatmaps.shape)



    # print('The shape of the image,' , images.shape)

    # #show the image
    # cv2.imshow('image', images)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #save the image
    #change the shape from c*w*h to cv2
    images = images.permute(1, 2, 0).numpy() * 255
    cv2.imwrite('output/finalimage.png', images)
    for no, heatmap in enumerate(heatmaps):
        #covnert to a numpy array from torch
        heatmap = heatmap.numpy()

        #get the max and min value inside the heatmap
        max_val = np.max(heatmap)
        min_val = np.min(heatmap)

        #normalize the heatmap  
        heatmap = (heatmap - min_val)/(max_val - min_val)

        #save a generated heatmap
        cv2.imwrite('output/heatmap_'+str(no)+'.png', heatmap*255)

    cv2.imwrite('output/image.png', images[0])


    
# from torch.utils.data import DataLoader

# from glob import glob

# image_paths = glob('/home/saiteja/extra/signgan/SignLangaugeRecognition/output/imgs/*.png')
# heatmap_paths = glob('/home/saiteja/extra/signgan/SignLangaugeRecognition/output/heatmaps/*.npy')

# print('Number of images: ', len(image_paths))
# print('Number of heatmaps: ', len(heatmap_paths))

# #sort them
# image_paths.sort()
# heatmap_paths.sort()

# dataset = ImageHeatmapDataset(image_paths, heatmap_paths)
# dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# #get a sample batch
# images, heatmaps = next(iter(dataloader))

# print('Images shape: ', images.shape)
# print('Heatmaps shape: ', heatmaps.shape)

# showbatch(images[2], heatmaps[2])





def CreateDataLoader(opt):

    dataset = ImageHeatmapDataset(opt.image_paths, opt.heatmap_paths)

    print('The length of the dataset is', len(dataset))

    # Set the split lengths
    train_length = int(len(dataset) * 0.8)
    val_length = int(len(dataset) * 0.15)
    test_length = len(dataset) - train_length - val_length

    # Use random_split to split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_length, val_length, test_length])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    return [train_dataloader, val_dataloader, test_dataloader]

