import os
import sys
sys.path.append("..")
import time
import numpy as np
import librosa
import scipy.io.wavfile as wav
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
import torchvision
from torch.optim.lr_scheduler import StepLR,MultiStepLR,LambdaLR,ExponentialLR
from data import SpeechCommandsDataset,SpeechCommandsDataset_npy
from utils import generate_random_silence_files
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F
dtype = torch.float
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(0) 
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

train_data_dir = "data/train_data.npy"
train_label_dir = "data/train_label.npy"
valid_data_dir = "data/valid_data.npy"
valid_label_dir = "data/valid_label.npy"
test_data_dir = "data/test_data.npy"
test_label_dir = "data/test_label.npy"



# generate the 2 labels
testing_words =[ 'up', 'down']
"""
testing_words = [x for x in testing_words if os.path.isdir(os.path.join(train_data_root,x))]
testing_words = [x for x in testing_words if os.path.isdir(os.path.join(train_data_root,x)) 
                 if x[0] != "_"]
"""
print("{} testing words:".format(len(testing_words)))
print(testing_words)



sr = 16000
size = 16000


n_fft = int(30e-3*sr)
hop_length = int(10e-3*sr)
n_mels = 40
fmax = 4000
fmin = 20
delta_order = 2
stack = True



transform = torchvision.transforms.ToTensor()




batch_size = 200

train_dataset = SpeechCommandsDataset_npy(train_data_dir, train_label_dir, transform = transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = SpeechCommandsDataset_npy(valid_data_dir, valid_label_dir, transform = transform)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

test_dataset = SpeechCommandsDataset_npy(test_data_dir, test_label_dir, transform = transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)


#####################################################################################################################3
# create network
# - Import the computational modules and combinators required for the networl
from rockpool.nn.modules import LIFTorch  # , LIFSlayer
from rockpool.nn.networks.wavesense import WaveSenseNet
from rockpool.parameters import Constant

dilations = [2, 32]
n_out_neurons = 4
n_inp_neurons = 16
n_neurons = 16
n_hidden = [40,40,40]
kernel_size = 2
alpha = torch.ones(40)*0.7
beta = torch.ones(40)*0.7
tau_lp = 0.01
threshold = 1.0
dt = 0.01

# - Use a GPU if available
# device = "gpu" if torch.cuda.is_available() else "cpu"
from rockpool.nn.modules import LinearTorch, InstantTorch,LIFBitshiftTorch
from rockpool.nn.combinators import Sequential


model =  Sequential(
        LinearTorch((n_inp_neurons, n_hidden[0])),
        LIFTorch(
        (n_hidden[0],),
        leak_mode = 'decays',
        alpha=alpha,
        beta=beta,

        dt=dt,

        ),
        LinearTorch((n_hidden[0], n_hidden[1])),
        LIFTorch(
        (n_hidden[1],),
        leak_mode = 'decays',
        alpha=alpha,
        beta=beta,
        dt=dt,

        ),
        LinearTorch((n_hidden[1], n_hidden[2])),
        LIFTorch(
        (n_hidden[2],),
        leak_mode = 'decays',
        alpha=alpha,
        beta=beta,
        dt=dt,

        ),
        LinearTorch((n_hidden[2], n_out_neurons)),
    ).to(device)
#model.save()
print(model.parameters().astorch())
criterion = nn.CrossEntropyLoss()
print("device:",device)

def test(data_loader,is_show = 0):
    test_acc = 0.
    sum_sample = 0.
    fr_ = []
    for i, (images, labels) in enumerate(data_loader):
        model.reset_state()
        images = images.float().to(device)
        labels = labels.view((-1)).long().to(device)
        output, state, recordings = model(images)
        output = output.sum(dim=1)


        # pass the last timestep of the output and the target through CE
        loss = criterion(output, labels)
        _, predicted = torch.max(output.data, 1)
        labels = labels.cpu()
        predicted = predicted.cpu().t()

        test_acc += (predicted ==labels).sum()
        sum_sample+=predicted.numel()

    return test_acc.data.cpu().numpy()/sum_sample


def train(epochs,criterion,optimizer,scheduler=None):
    acc_list = []
    best_acc = 0

    path = 'model/xylo_test_v1' 
    for epoch in range(epochs):
        train_acc = 0
        train_loss = 0
        sum_sample = 0
        train_loss_sum = 0
        for i, (images, labels) in enumerate(train_dataloader):
            # reset states and gradients
            model.reset_state()
            optimizer.zero_grad()

            images = images.float().to(device)
            labels = labels.view((-1)).long().to(device)
            output, state, recordings = model(images)
            output = output.sum(dim=1)


            # pass the last timestep of the output and the target through CE
            loss = criterion(output, labels)
            _, predicted = torch.max(output.data, 1)
            # backward
            loss.backward()
            train_loss += loss.item()
            # apply gradients
            optimizer.step()
            labels = labels.cpu()
            predicted = predicted.cpu().t()

            train_acc += (predicted ==labels).sum()
            sum_sample+=predicted.numel()
        if scheduler:
            scheduler.step()
        train_acc = train_acc.data.cpu().numpy()/sum_sample
        valid_acc = test(test_dataloader,1)
        train_loss_sum+= train_loss

        acc_list.append(train_acc)
        print('lr: ',optimizer.param_groups[0]["lr"])
        if valid_acc>best_acc and train_acc>0.25:
            best_acc = valid_acc
            torch.save(model.state_dict(),'model.pt')
  
        print('epoch: {:3d}, Train Loss: {:.4f}, Train Acc: {:.4f},Valid Acc: {:.4f}'.format(epoch,
                                                                           train_loss_sum/len(train_dataloader),
                                                                           train_acc,valid_acc), flush=True)
    return acc_list


learning_rate = 1e-3


test_acc = test(test_dataloader)
print(test_acc)
optimizer = torch.optim.Adam(model.parameters().astorch(), lr=1e-3)

scheduler = StepLR(optimizer, step_size=50, gamma=.5) # 20
# epoch=0
epochs =150

acc_list = train(epochs,criterion,optimizer,scheduler)

test_acc = test(test_dataloader)
print(test_acc)