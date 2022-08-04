import UNet
from model import SegNet
import torch
import torchvision
import matplotlib.pyplot as plt
from dataset import PascalData
import numpy as np
from torch.utils.data import DataLoader
import timeit
from torchsummary import summary
def accuracy(pred, truth):
    pred = torch.argmax(pred, dim=1)
    pixeli_tot = torch.count_nonzero(truth)
    correct = (pred == truth)
    pred = (pred != 0)
    correct_nr = torch.count_nonzero(correct & pred)
    return correct_nr, pixeli_tot


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
pathfle2 = '/home/student/pascal/val.txt'
pathpersonfile = '/home/student/PycharmProjects/segmentation/people.txt'
#pathfle2 = '/home/student/pascal/val.txt'
data = PascalData(pathfle2, pathlbl, pathimg, classes)

#datatest = PascalData(pathfle2, pathlbl, pathimg, classes)
device = 'cpu'
train_dataloader = DataLoader(data, 32)
model = UNet.UNet(3,21)
model.load_state_dict(torch.load('/media/student/HDD1/FullSetCheckpoints/model_weights1400.pth'))
model.eval()
model = model.to(device)
#test_dataloader = DataLoader(datatest, 32)
def test():

    img, y = next(iter(train_dataloader))
    img = img[0]
    img = torch.unsqueeze(img,dim =0)
    img = img.cpu().to(device)
    inf = model(img)
    #inf = inf*12
    #inf = torch.softmax(inf, dim=1)
    inf = torch.argmax(inf, dim=1)
    print(f"Accuracy:{accuracy(inf, y):.2f}%")
    inf = torch.squeeze(inf)
    inf = inf.to(device)
    plt.imshow(inf)
    plt.show()

def create_file():
    data.getPerson()
def summary():
    summary(model,(3,256,256))
summary()
#create_file()
#test()