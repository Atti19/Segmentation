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
        file = open(pathfle, 'r')
        lines = file.readlines()
        self.names = lines
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.names)

    def encode(self, mask):
        #mask = mask.resize_(3, 246, 112
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(self.classes)), dtype=np.float32)
        for label_index, label in enumerate(self.classes):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        segmentation_mask = torch.Tensor(segmentation_mask)
        return segmentation_mask

    def __getitem__(self, idx):

        img_name = self.names[idx].strip() + '.jpg'
        img = Image.open(self.pathimg + img_name).resize((246,112))
        image = np.array(img)
        image = torch.Tensor(image).float()
        image = image.permute(2,0, 1)
        lbl_name = self.names[idx].strip() + '.png'
        lbl = Image.open(self.pathlbl + lbl_name).resize((246,112))
        lbl = np.array(lbl)
        lbl[lbl == 255] = 0
        return image, lbl


