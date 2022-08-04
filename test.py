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
from colormap import toColor
def accuracy(pred, truth):
    pred = torch.argmax(pred, dim=1)
    truth = torch.Tensor(truth)
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
model = UNet.UNet(3,21,bilinear=True)

model.load_state_dict(torch.load('/media/student/HDD1/FullSetCheckpointsUnet/model_weights130.pth'))
model.eval()
model = model.to(device)
#test_dataloader = DataLoader(datatest, 32)
def max_acc():
    accuracies = []
    #img, y = next(iter(train_dataloader))
    f = open("accuracies.txt","w")
    for i in range(1449):
        print(i)
        img, y = data.__getitem__(i)
    #img = img[2]
        img = torch.unsqueeze(img,dim =0)
        img = img.cpu().to(device)
        inf = model(img)
        inf = torch.argmax(inf, dim=1)
        cor, tot = accuracy(inf, y)
        accuracies.append(cor/tot*100)
        f.write(data.getName(i))
        f.write(str(accuracies[i]))
        f.write("\n")
    f.close()

def test():
    img, y = data.__getitem__(0)
    img = torch.unsqueeze(img, dim=0)
    img = img.cpu().to(device)
    inf = model(img)
    inf = torch.argmax(inf, dim=1)
    inf = toColor(inf)
    inf = torch.Tensor(inf)
    inf = torch.permute(inf,(1,2,0))
    inf = inf.to(device)
    plt.imshow(inf)
    plt.show()
def create_file():
    data.getPerson()

#summary(model, (16,3,256,256))

#create_file()
#max_acc()
test()