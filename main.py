import visdom
from torch.optim.lr_scheduler import StepLR
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
from VisdomUtil import VisdomLinePlotter
pathlbl = '/home/student/pascal/SegmentationClass/'
pathimg = '/home/student/pascal/JPEGImages/'
pathfle = '/home/student/pascal/train.txt'
pathfle2 = '/home/student/pascal/val.txt'
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
batch_size = 32
#classes = torch.Tensor(classes)
data = PascalData(pathfle, pathlbl, pathimg, classes)
datatest = PascalData(pathfle2, pathlbl, pathimg, classes)
train_dataloader = DataLoader(data, batch_size, shuffle=True)
test_dataloader = DataLoader(datatest, batch_size)


#params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SegNet()
model.initialize_weights()
#model.load_state_dict(torch.load('model_weights.pth'))
#model.eval()
model.to(device)
learning_rate = 0.1
epochs = 50000
class_weights = np.ones(21)
class_weights = class_weights*20
class_weights[0] = 1
class_weights = torch.Tensor(class_weights).to(device)
#loss&optim
loss_fn = nn.CrossEntropyLoss(weight = class_weights)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

def mean_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (n_samples, height, width, n_channels)
    batch_size = y_true.shape[0]
    channel_num = y_true.shape[-1]
    mean_dice_channel = 0.
    for i in range(batch_size):
        for j in range(channel_num):
            channel_dice = single_dice_coef(y_true[i, :, :, j], y_pred_bin[i, :, :, j])
            mean_dice_channel += channel_dice/(channel_num*batch_size)
    return mean_dice_channel

def accuracy(pred, truth):
    pred = torch.argmax(pred, dim=1)
    pixeli_tot = torch.count_nonzero(truth)
    correct = (pred == truth)
    pred = (pred != 0)
    correct = torch.count_nonzero(correct == pred)
    return (correct/pixeli_tot) * 100
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
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
        return loss, accuracy(pred, y)#, mean_dice_coef(y, pred)


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.long()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    #print(f"test_loss: {test_loss:>8f} \n")

    return test_loss#, mean_dice_coef(y, pred)

def train():
    x = VisdomUtil.VisdomLinePlotter(env_name="experimentscheduler_flip")
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss, acc = train_loop(train_dataloader, model, loss_fn, optimizer)
        testlosses = test_loop(test_dataloader, model, loss_fn)
        scheduler.step()
        if t%5 == 0:
            torch.save(model.state_dict(), f'/media/student/HDD1/CheckpointsSegmentare/model_weights{t}.pth')
        #torch.save(model, 'model.pth')
        x.plot("trainloss","loss","trainloss", t, loss)
        x.plot("test loss","loss", "testloss", t, testlosses)
        x.plot("trainaccuracy",'accuracy','trainacc', t, acc.cpu())
    print("Done!")

train()

stop = timeit.default_timer()
print(stop-start)










