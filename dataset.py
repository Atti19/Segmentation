import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
#from colormap import color_map

class PascalData(Dataset):
    def __init__(self, pathfle, pathlbl, pathimg,classes):
        self.pathlbl = pathlbl
        self.pathimg = pathimg
        self.pathfle = pathfle
        self.classes = classes
        self.people = []
        file = open(pathfle, 'r')
        lines = file.readlines()
        self.names = lines
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.names)
    def __getitem__(self, idx):
        toflip = np.random.rand(1)
        img_name = self.names[idx].strip() + '.jpg'
        img = Image.open(self.pathimg + img_name)
        #if toflip > 0.5:
            #img = transforms.RandomRotation(90)(img)
        img = self.tf(img)

        lbl_name = self.names[idx].strip() + '.png'
        lbl = Image.open(self.pathlbl + lbl_name)

        #if toflip > 0.5:
            #lbl = transforms.RandomRotation(90)(lbl)
        lbl = transforms.Resize((256, 256))(lbl)
        lbl = np.array(lbl)
        lbl[lbl == 255] = 0
        # lbl[lbl != 15] = 0
        # lbl[lbl == 15] = 1
        return img, lbl

    def getPerson(self):
        f = open("people.txt", "w")
        for name in self.names:
            lbl_name = name.strip() + '.png'
            lbl = Image.open(self.pathlbl + lbl_name)
            lbl = np.array(lbl)
            unique = np.unique(lbl)
            if 15 in unique:
                f.write(name)
        f.close()
