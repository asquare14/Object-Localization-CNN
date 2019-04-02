%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import time
import json
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
import torch.nn as nn
import torch.nn.functional as F
import tempfile
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
from neural_net import *
from parameter import *
from custom_data_loader import *
from intersection import *
import csv
torch.set_default_tensor_type('torch.DoubleTensor')


def intersection(b1, b2):
    A = b1.size(0)
    B = b2.size(0)
    max_xy = torch.min(b1[:, 2:].unsqueeze(1).expand(A, B, 2),
                       b2[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(b1[:, :2].unsqueeze(1).expand(A, B, 2),
                       b2[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter =inter[:, :, 0] * inter[:, :, 1]
    area_a = ((b1[:, 2]-b1[:, 0]) *
              (b1[:, 3]-b1[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((b2[:, 2]-b2[:, 0]) *
              (b2[:, 3]-b2[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return ((inter / union)*100/float(A*A)).sum()

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

path = "../input/trainingtest" #path to test.csv and train.csv
path1 = "../input/imagedata/images" # path to where image folder is stored

criterion = nn.MSELoss()
net = Net().cuda()  #works only when gpu is on


optimizer = optim.Adam(net.parameters(), lr=learning_rate)

df = pd.read_csv(os.path.join(path, "training.csv"))
train, test = train_test_split(df, test_size=testing_size)

train_data = dataset(df = train , root_dir = os.path.join(path1, "images"))
train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=0)

test_data = dataset(df = test , root_dir = os.path.join(path1, "images"))
test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=True, num_workers=0)

for epoch in range(n_epochs):
    train_loss = 0
    count = 0

    net.train(True)

    for data in train_dataloader:
        inputs = data["image"].cuda()
        target = data["box"].type(torch.DoubleTensor).cuda()

        inputs = Variable(inputs)
        target = Variable(target)

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, target)

        loss.backward()

        optimizer.step()

        train_loss += loss.data.item()

        count+=1

        print(loss.data.item())

    average_training_loss = train_loss/count

    if for_testing:
        with torch.no_grad():
            valid_loss = 0
            count_test=0
            for data in test_dataloader:
                inputs = data["image"].cuda()

                target = data["box"].type(torch.DoubleTensor).cuda()
                inputs = Variable(inputs)
                target = Variable(target)

                outputs = net(inputs)

                valid_loss += intersection(outputs,target).item()
                count_test+=1
            average_testing_acc = float(valid_loss)/count_test
            print(average_testing_acc)

if run_prediction:
    predict_set = pd.read_csv(os.path.join(path , "test.csv"))
    f = open(path + "output.csv", "w")
    writer = csv.writer(f)
    row = []
    row.append(["image_name","x1", "x2", "y1", "y2"])
    with torch.no_grad():
        for index, data in predict_set.iterrows():
                print(index)
                image = cv2.imread(os.path.join(path1, "images/"+data[0]))
                image = image.reshape(1,3,640,480)
                image = torch.from_numpy(image)
                inputs = Variable(image.type(torch.DoubleTensor).cuda())
                outputs = net(inputs)
                row.append([data["image_name"], outputs[0][0].item(), outputs[0][2].item(), outputs[0][1].item(), outputs[0][3].item()])
    writer.writerows(row)
    f.close()
