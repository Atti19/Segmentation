from model import SegNet
import torch
import torchvision
import matplotlib.pyplot as plt
from dataset import PascalData
import numpy as np
from torch.utils.data import DataLoader
import timeit
def accuracy(pred, truth):

    pixeli_tot = torch.count_nonzero(truth)
    correct = (pred == truth)
    pred = pred != 0
    correct = torch.count_nonzero(correct == pred)
    return (correct/pixeli_tot) * 100


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
pathfle = '/home/student/pascal/test_p.txt'
#pathfle2 = '/home/student/pascal/val.txt'
data = PascalData(pathfle, pathlbl, pathimg, classes)
#datatest = PascalData(pathfle2, pathlbl, pathimg, classes)
train_dataloader = DataLoader(data, 32)
#test_dataloader = DataLoader(datatest, 32)
def test():
    device = 'cpu'
    img, y = next(iter(train_dataloader))
    img = img[0]
    img = torch.unsqueeze(img,dim =0)
    img = img.cpu().to(device)
    #img = img.permute(0, 1, 2, 3)
    model = SegNet()
    model.load_state_dict(torch.load('model_weights_p.pth'))
    model.eval()
    model = model.to(device)
    inf = model(img)
    # inf = inf*12
    #inf = torch.softmax(inf, dim=1)
    inf = torch.argmax(inf, dim=1)
    print(f"Accuracy:{accuracy(inf, y):.2f}%")
    inf = torch.squeeze(inf)
    inf = inf.to(device)
    plt.imshow(inf)
    plt.show()
test()