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
from data import SpeechCommandsDataset,Pad, MelSpectrogram, Rescale,Normalize,AFE
from utils import generate_random_silence_files
dtype = torch.float
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(0) 
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

train_data_root = "/data/speech_commands"
test_data_root = "/data/speech_commands"

training_words = os.listdir(train_data_root)
training_words = [x for x in training_words if os.path.isdir(os.path.join(train_data_root,x))]
training_words = [x for x in training_words if os.path.isdir(os.path.join(train_data_root,x)) if x[0] != "_" ]
print("{} training words:".format(len(training_words)))
print(training_words)

# generate the 12 labels
testing_words =['up', 'down']
"""
testing_words = [x for x in testing_words if os.path.isdir(os.path.join(train_data_root,x))]
testing_words = [x for x in testing_words if os.path.isdir(os.path.join(train_data_root,x)) 
                 if x[0] != "_"]
"""
print("{} testing words:".format(len(testing_words)))
print(testing_words)

label_dct = {k:i for i,k in enumerate(testing_words+ ["_silence_", "_unknown_"])}

for w in training_words:
    label = label_dct.get(w)
    if label is None:
        label_dct[w] = label_dct["_unknown_"]

print("label_dct:")
print(label_dct)

sr = 16000
size = 16000
'''
noise_path = os.path.join(train_data_root, "_background_noise_")
noise_files = []
for f in os.listdir(noise_path):
    if f.endswith(".wav"):
        full_name = os.path.join(noise_path, f)
        noise_files.append(full_name)
print("noise files:")
print(noise_files)

# generate silence training and validation data

silence_folder = os.path.join(train_data_root, "_silence_")
if not os.path.exists(silence_folder):
    os.makedirs(silence_folder)
    # 260 validation / 2300 training
    generate_random_silence_files(2560, noise_files, size, os.path.join(silence_folder, "rd_silence"))

    # save 260 files for validation
    silence_files = [fname for fname in os.listdir(silence_folder)]
    with open(os.path.join(train_data_root, "silence_validation_list.txt"),"w") as f:
        f.writelines("_silence_/"+ fname + "\n" for fname in silence_files[:260])
'''
n_fft = int(30e-3*sr)
hop_length = int(10e-3*sr)
n_mels = 40
fmax = 4000
fmin = 20
delta_order = 2
stack = True

melspec = MelSpectrogram(sr, n_fft, hop_length, n_mels, fmin, fmax, delta_order, stack=stack)
pad = Pad(size)
rescale = Rescale()
normalize = Normalize()

transform = torchvision.transforms.Compose([pad])
from rockpool.devices.xylo.syns61201 import AFESim
from rockpool.timeseries import TSContinuous
def trans(wave):


    num_filters = 16  # number of filters (on the AFE hardware, this is fixed to 16)
    Q = 5  # Q (Quality) factor for the filters
    fc1 = 100.0  # center frequency of the first filter in Hz
    f_factor = (
        1.325  # scaling to determine the center frequencies of the subsequent filters
    )

    fs = 16000  # sampling frequency for the input data, in Hz
    LNA_gain = 0.0  # gain of the Low-Noise amplifier in dB

    thr_up = 0.8  # threshold of the neurons used for signal-to-event conversion
    leakage = 5.0  # leakage of the neurons used for event conversion
    digital_counter = 8  # digital counter for event conversion. This is a factor to decrease the event rate by

    manual_scaling = None  # if `None`, input is normalised automatically prior to the LNA
    add_noise = True  # add simulated noise on each stage of the filtering
    seed = None  # the AFE is subject to random mismatch, the mismatch can be seeded

    # - Initialize the AFE simulation, and convert it to a high-level `TimedModule`

    afe = AFESim(
        shape=num_filters,
        Q=Q,
        fc1=fc1,
        f_factor=f_factor,
        thr_up=thr_up,
        leakage=leakage,
        LNA_gain=LNA_gain,
        fs=fs,
        digital_counter=digital_counter,
        manual_scaling=manual_scaling,
        add_noise=add_noise,
        seed=seed,
    ).timed()

    dt = 1/sr
    inp_ts = TSContinuous.from_clocked(wave, dt=dt, periodic=False, name="Chirp input")
    filt_spikes, state, rec = afe(inp_ts, record=False)  
    del afe
    del inp_ts
    return filt_spikes.raster(dt=10e-3, add_events=True)
def collate_fn(data):
    X_batch = [torch.tensor(d[0]) for d in data]
    #std = X_batch.std(axis=(0,2), keepdims=True)
    X_batch = torch.stack(X_batch)
    y_batch = torch.tensor([d[1] for d in data])
    
    return X_batch, y_batch 

batch_size = 1
train_dataset = SpeechCommandsDataset(train_data_root, label_dct, transform = transform, mode="train", max_nb_per_class=None)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
valid_dataset = SpeechCommandsDataset(train_data_root, label_dct, transform = transform, mode="valid", max_nb_per_class=None)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
test_dataset = SpeechCommandsDataset(test_data_root, label_dct, transform=transform, mode="test")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
transform = AFE()  
print(time.time())
spike_trains_recode = {0:[],1:[],2:[],3:[]}
spike_recode = []
label_recode = []
for i, (images, labels) in enumerate(train_dataloader):
    if (len(spike_trains_recode[labels[0].item()]) <= 1800): 
        inp = images[0].numpy()
  
        spike_trains = trans(inp)
        spike_trains_recode[labels[0].item()].append(spike_trains)
        spike_recode.append(spike_trains)
        label_recode.append(labels[0].numpy())

np.save("train_data.npy",np.array(spike_recode))
np.save("train_label.npy",np.array(label_recode))
print("training done")
spike_trains_recode = {0:[],1:[],2:[],3:[]}
spike_recode = []
label_recode = []
for i, (images, labels) in enumerate(valid_dataloader):
    if (len(spike_trains_recode[labels[0].item()]) <= 300): 
        inp = images[0].numpy()
  
        spike_trains = trans(inp)
        spike_trains_recode[labels[0].item()].append(spike_trains)
        spike_recode.append(spike_trains)
        label_recode.append(labels[0].numpy())

np.save("valid_data.npy",np.array(spike_recode))
np.save("valid_label.npy",np.array(label_recode))
print("valid done")
spike_trains_recode = {0:[],1:[],2:[],3:[]}
spike_recode = []
label_recode = []
for i, (images, labels) in enumerate(test_dataloader):
    if (len(spike_trains_recode[labels[0].item()]) <= 270): 
        inp = images[0].numpy()
  
        spike_trains = trans(inp)
        spike_trains_recode[labels[0].item()].append(spike_trains)
        spike_recode.append(spike_trains)
        label_recode.append(labels[0].numpy())

np.save("test_data.npy",np.array(spike_recode))
np.save("test_label.npy",np.array(label_recode))
print("test done")
print(time.time())

