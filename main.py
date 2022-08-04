import torchvision.models
import visdom
from torch.optim.lr_scheduler import CosineAnnealingLR
import VisdomUtil
from dataset import PascalData
from torch.utils.data import DataLoader
from model import SegNet
from PIL import Image
import PIL
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
import timeit
from UNet import UNet
from VisdomUtil import VisdomLinePlotter
pathlbl = '/home/student/pascal/SegmentationClass/'
pathimg = '/home/student/pascal/JPEGImages/'
pathfle = '/home/student/pascal/train.txt'
pathfle2 = '/home/student/pascal/val.txt'
pathpersonfile = '/home/student/PycharmProjects/segmentation/people.txt'
def imagesize(pathfle, pathimg, pathlbl):
    file = open(pathfle, 'r')
    lines = file.readlines()
    width = [int] * len(lines)
    height = [int] * len(lines)
    for i in range(len(lines)):
        line = lines[i].strip()
        line = line + '.jpg'
        im = Image.open(pathimg+line)
        width[i], height[i] = im.size
    return min(width), min(height)

#dataload
start = timeit.default_timer()
#classes = np.load('/home/student/Downloads/classes.npy')
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
batch_size = 16
#classes = torch.Tensor(classes)
data = PascalData(pathfle, pathlbl, pathimg, classes)
datatest = PascalData(pathfle2, pathlbl, pathimg, classes)
train_dataloader = DataLoader(data, batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(datatest, batch_size)


#params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model = SegNet()
model = UNet(3, 21)

#model.initialize_weights()
#model.load_state_dict(torch.load('model_weights.pth'))
#model.eval()
model.to(device)
learning_rate = 0.1
epochs = 600
#class_weights = np.ones(21)
#class_weights = class_weights*4
#class_weights[0] = 0.01
class_weights = np.ones(2)
# class_weights[0] = 0.000000000000000000000000000000000001
# class_weights[1] = 1000
# class_weights = torch.Tensor(class_weights).to(device)
#loss&optim
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

def accuracy(pred, truth):
    pred = torch.argmax(pred, dim=1)
    pixeli_tot = torch.count_nonzero(truth)
    correct = (pred == truth)
    pred = (pred != 0)
    correct_nr = torch.count_nonzero(correct & pred)
    return correct_nr, pixeli_tot

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    ep_loss = 0
    ep_cor = 0
    ep_tot = 0
    nr = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y = y.long()
        #X = X.float()
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if batch % 10 == 0:
        loss = loss.item()
            #print("loss: ", loss)
            #print(f"train_acc:{accuracy(pred, y):.2f}% ")
        ep_loss += loss
        nr +=1
        cor, tot = accuracy(pred, y)
        ep_cor += cor
        ep_tot += tot
    return ep_loss/nr, (ep_cor / ep_tot) *100#, dice_coef2(y, pred)


def val_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    ep_cor, ep_tot = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.long()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            cor, tot = accuracy(pred, y)
            ep_cor += cor
            ep_tot += tot

    test_loss /= num_batches
    #print(f"test_loss: {test_loss:>8f} \n")

    return test_loss, (ep_cor / ep_tot) *100#, dice_coef2(y, pred)

def train():
    x = VisdomUtil.VisdomLinePlotter(env_name="Unet")
    scheduler = CosineAnnealingLR(optimizer, eta_min = 0.001, T_max = 10000)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss, acc = train_loop(train_dataloader, model, loss_fn, optimizer)
        vallosses, accval = val_loop(test_dataloader, model, loss_fn)
        scheduler.step()
        if t%5 == 0:
            torch.save(model.state_dict(), f'/media/student/HDD1/FullSetCheckpointsUnet/model_weights{t}.pth')
        #torch.save(model, 'model.pth')
        print("Loss:", loss)
        print("Accuracy:", acc)
        x.plot("trainloss","loss","trainloss", t, loss)
        x.plot("val loss","loss", "valloss", t, vallosses)
        x.plot("trainaccuracy",'accuracy','trainacc', t, acc.cpu())
        x.plot("valaccuracy",'accuracy','valacc',t, accval.cpu())
        x.plot("learning", 'lr', 'learningrate', t, optimizer.param_groups[0]['lr'])
     #   x.plot("Dice train", 'traindiceloss', 'Dice loss train', t, dicetrain)
     #   x.plot("Dice test", 'testdiceloss', 'Dice loss test', t, dicetest)
    print("Done!")

train()

stop = timeit.default_timer()
print(stop-start)










