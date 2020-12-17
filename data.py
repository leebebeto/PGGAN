import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
class Celeb_HQ(Dataset):
    def __init__(self, train=True, rindex=0):
        # self.train_images = glob.glob('/home/nas1_userB/dataset/Celeb_HQ/train/**/*')
        self.train_images = glob.glob('/home/nas1_userB/dataset/Celeb_HQ/train/female/*')
        '''
        index2res
        0: 4 / 1: 8 / 2: 16 / 3: 32 / 4: 64 / 5: 128 / 6: 256 / 7: 512 / 8: 1024
        '''
        self.rindex = rindex
        # self.rindex2batch = {0: 16, 1: 16, 2: 16, 3: 16, 4: 16, 5: 16, 6: 8, 7: 4, 8: 3}
        self.rindex2batch = {0: 4, 1: 4, 2: 4, 3: 4, 4: 4, 5: 2, 6: 2, 7: 2, 8: 1}
        self.batch_size = self.rindex2batch[self.rindex]
        self.resolution = pow(2, (self.rindex+2))
        self.len = len(self.train_images)

    def renew(self, initial=False):
        if initial==False:
            self.rindex += 1

        self.batch_size = self.rindex2batch[self.rindex]
        self.resolution = pow(2, (self.rindex+2))
        self.transform =  transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.dataset = ImageFolder(root = '/home/nas1_userB/dataset/Celeb_HQ/train/',
                                   transform = self.transform)

        self.train_loader = DataLoader(dataset=self.dataset,
                                      batch_size = self.batch_size,
                                      shuffle = True,
                                      drop_last = True)


    # def __getitem__(self, index):
    #     image = Image.open(self.train_images[index]).convert('RGB')
    #     image = self.transform(image)
    #     return image

    def __len__(self):
        return self.len