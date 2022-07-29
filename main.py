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
pathlbl = '/home/student/pascal/SegmentationClass/'
pathimg = '/home/student/pascal/JPEGImages/'
pathfle = '/home/student/pascal/test_p.txt'
pathfle2 = '/home/student/pascal/overfit.txt'
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
#classes = torch.Tensor(classes)
data = PascalData(pathfle, pathlbl, pathimg, classes)
datatest = PascalData(pathfle2, pathlbl, pathimg, classes)
train_dataloader = DataLoader(data, 32)
test_dataloader = DataLoader(datatest, 32)


#params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SegNet()

model.to(device)
learning_rate = 0.001
epochs = 1000

#loss&optim
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
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

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print("loss: ", loss)
        return loss#, mean_dice_coef(y, pred)

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
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss#, mean_dice_coef(y, pred)

def train():
    losslist = []
    testlosslist = []
    #dice = []
    #dicetest = []
    figure, axis = plt.subplots(2, 2)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        losslist.append(loss)
        testlosses = test_loop(test_dataloader, model, loss_fn)
        testlosslist.append(testlosses)
        #dice.append(diceloss)
        #dicetest.append(testdice)
        torch.save(model.state_dict(), 'model_weights.pth')
        torch.save(model, 'model.pth')
    axis[0, 0].plot((list(range(epochs))), losslist)
    axis[0, 0].set_title("Train Loss")
    axis[0, 1].plot((list(range(epochs))),testlosslist)
    #axis[0, 1].set_title("Test Loss")
    #axis[1, 0].plot((list(range(epochs))), dice)
    #axis[1, 0].set_title("Dice Train Loss")
    #axis[1, 1].plot((list(range(epochs))), dicetest)
    #axis[1, 1].set_title("Dice Test Loss")
    plt.show()
    plt.savefig("loss.png")
    print("Done!")

train()












