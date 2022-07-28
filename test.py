from model import SegNet
import torch
import torchvision
import matplotlib.pyplot as plt
from dataset import PascalData
import numpy as np
from torch.utils.data import DataLoader
classes = np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ])
pathlbl = '/home/student/pascal/SegmentationClass/'
pathimg = '/home/student/pascal/JPEGImages/'
pathfle = '/home/student/pascal/train.txt'
pathfle2 = '/home/student/pascal/overfit.txt'
data = PascalData(pathfle, pathlbl, pathimg, classes)
datatest = PascalData(pathfle2, pathlbl, pathimg, classes)
train_dataloader = DataLoader(data, 32)
test_dataloader = DataLoader(datatest, 32)
def test():
    device = 'cpu'
    img, y = next(iter(train_dataloader))
    img = img[0]
    img = torch.unsqueeze(img,dim =0)
    img = img.cpu().to(device)
    img = img.permute(0, 1, 3, 2)
    model = SegNet()
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()
    model = model.to(device)
    inf = model(img)
    # inf = inf*12
    inf = torch.softmax(inf, dim=1)
    inf = torch.argmax(inf, dim=1)
    inf = torch.squeeze(inf)
    inf = inf.to(device)
    plt.imshow(inf)
    plt.show()
test()