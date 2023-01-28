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
        #covnert to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        #crop the image the right half
        image = image[:, 640:, :]

        #reshape the image to 720*1280*3
        image = cv2.resize(image, (1280, 720))

        heatmap = np.load(self.image_paths[idx].replace('.png', '.npy').replace('img_frame', 'heatmaps').replace('imgs', 'heatmaps'))

        #center crop the image to 720*720*3
        image = image[0:720, 280:1000, :]
        heatmap = heatmap[:, 0:720, 280:1000]

        # print('heatmap shape', heatmap.shape)

        
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_grey = np.expand_dims(image_grey, axis = 0)
        # print('grey iamge', image_grey.shape)
        heatmap    = np.concatenate((heatmap, image_grey), axis =0) 

        # print('heatmap shape', heatmap.shape)





        resized_heatmap = []
        for channel in range(heatmap.shape[0]):
            resized_heatmap.append(cv2.resize(heatmap[channel], (512, 512), interpolation = cv2.INTER_LINEAR))
        resized_heatmap = np.stack(resized_heatmap, axis = 0)

        #make a transform to convert to size of 720, 720
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((373,373)),
            transforms.ToTensor()
        ])

        #convert to a tensor
        image = transform(image)

        #make a transform to convert to size of 720, 720
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        #convert to a tensor
        heatmap = transform(heatmap)

        return image, resized_heatmap
    
import os

def showbatch(images, heatmaps):
    #make the dir data if doesnot exist
    if not os.path.exists('data'):
        os.mkdir('data')

    # print('The shape of the heatmap,' , heatmaps.shape)



    # print('The shape of the image,' , images.shape)

    # #show the image
    # cv2.imshow('image', images)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #save the image
    #change the shape from c*w*h to cv2
    images = images.permute(1, 2, 0).numpy() * 255


    cv2.imwrite('data/finalimage.png', images)


    for no, heatmap in enumerate(heatmaps):
        #covnert to a numpy array from torch
        heatmap = heatmap.numpy()

        if no< 3:
            # print(heatmap.shape)
            
            cv2.imwrite('data/heatmap_'+str(no)+'.png', heatmap)
        
        else:
            #get the max and min value inside the heatmap
            max_val = np.max(heatmap)
            min_val = np.min(heatmap)

            #normalize the heatmap  
            heatmap = (heatmap - min_val)/(max_val - min_val)

            #save a generated heatmap
            cv2.imwrite('data/heatmap_'+str(no)+'.png', heatmap*255)

    cv2.imwrite('data/image.png', images[0])


    
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





import torch
from torch.utils.data import Dataset

#get transforms from torchvision
from torchvision import transforms
from PIL import Image

import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image



class ImageHeatmapDatasetV2(Dataset):
    def __init__(self, image_paths, heatmap_paths, gt_img_path = './'):
        self.image_paths = image_paths
        self.heatmap_paths = heatmap_paths
        self.gt_img_path = gt_img_path
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        #crop the image the right half
        image = image[:, 640:, :]

        #reshape the image to 720*1280*3
        image = cv2.resize(image, (1280, 720))


        heatmap = np.load(self.image_paths[idx].replace('.png', '.npy').replace('img_frame', 'heatmaps').replace('imgs', 'heatmaps'))

        #center crop the image to 720*720*3
        image = image[0:720, 280:1000, :]
        heatmap = heatmap[:, 0:720, 280:1000]

        # print('heatmap shape', heatmap.shape)

        gt_image = cv2.imread(self.gt_img_path)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        gt_image = gt_image[:, 640:, :]
        gt_image = cv2.resize(gt_image, (1280, 720))
        gt_image = gt_image[0:720, 280:1000, :]
        image_grey = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
        image_grey = np.expand_dims(image_grey, axis = 0)
        heatmap    = np.concatenate((heatmap, image_grey), axis =0) 

        # print('heatmap shape', heatmap.shape)





        resized_heatmap = []
        for channel in range(heatmap.shape[0]):
            resized_heatmap.append(cv2.resize(heatmap[channel], (512, 512), interpolation = cv2.INTER_LINEAR))
        resized_heatmap = np.stack(resized_heatmap, axis = 0)

        #make a transform to convert to size of 720, 720
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((373,373)),
            transforms.ToTensor()
        ])

        #convert to a tensor
        image = transform(image)

        #make a transform to convert to size of 720, 720
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        #convert to a tensor
        heatmap = transform(heatmap)

        return image, resized_heatmap
    
import os

def showbatch(images, heatmaps):
    #make the dir data if doesnot exist
    if not os.path.exists('data'):
        os.mkdir('data')

    # print('The shape of the heatmap,' , heatmaps.shape)



    # print('The shape of the image,' , images.shape)

    # #show the image
    # cv2.imshow('image', images)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #save the image
    #change the shape from c*w*h to cv2
    images = images.permute(1, 2, 0).numpy() * 255


    cv2.imwrite('data/finalimage.png', images)


    for no, heatmap in enumerate(heatmaps):
        #covnert to a numpy array from torch
        heatmap = heatmap.numpy()

        if no< 3:
            # print(heatmap.shape)
            
            cv2.imwrite('data/heatmap_'+str(no)+'.png', heatmap)
        
        else:
            #get the max and min value inside the heatmap
            max_val = np.max(heatmap)
            min_val = np.min(heatmap)

            #normalize the heatmap  
            heatmap = (heatmap - min_val)/(max_val - min_val)

            #save a generated heatmap
            cv2.imwrite('data/heatmap_'+str(no)+'.png', heatmap*255)

    cv2.imwrite('data/image.png', images[0])


    
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





