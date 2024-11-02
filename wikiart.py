import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm


class WikiArtImage:
    def __init__(self, imgdir, label, filename):
        self.imgdir = imgdir
        self.label = label
        self.filename = filename
        self.image = None
        self.loaded = False

    def get(self):
        if not self.loaded:
            self.image = read_image(os.path.join(self.imgdir, self.label, self.filename)).float()
            self.loaded = True

        return self.image

class WikiArtDataset(Dataset):
    def __init__(self, imgdir, device="cpu"):
        walking = os.walk(imgdir)
        filedict = {}
        indices = []
        classes = set()
        print("Gathering files for {}".format(imgdir))
        for item in walking:
            sys.stdout.write('.')
            arttype = os.path.basename(item[0])
            artfiles = item[2]
            for art in artfiles:
                filedict[art] = WikiArtImage(imgdir, arttype, art)
                indices.append(art)
                classes.add(arttype)
        print("...finished")
        self.filedict = filedict
        self.imgdir = imgdir
        self.indices = indices
        self.classes = list(classes)
        self.device = device
        
    def __len__(self):
        return len(self.filedict)

    def __getitem__(self, idx):
        imgname = self.indices[idx]
        imgobj = self.filedict[imgname]
        ilabel = self.classes.index(imgobj.label)
        image = imgobj.get().to(self.device)

        return image, ilabel

class WikiArtModel(nn.Module):
    def __init__(self, num_classes=27):
        super(WikiArtModel, self).__init__()
        self.conv2d_1 = nn.Conv2d(3, 32, kernel_size=4, padding=2)  # Output: (32, 416, 416)
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=2, padding=1)   # Output: (32, 209, 209)
        
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=4, padding=2) # Output: (64, 209, 209)
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=2, padding=1)   # Output: (64, 106, 106) (rounded up from 105.5)

        self.flatten = nn.Flatten()
        self.batchnorm1d = nn.BatchNorm1d(64 * 106 * 106)
        self.linear1 = nn.Linear(64 * 106 * 106, 300)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(300, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, image):
        output = self.conv2d_1(image)
        output = self.relu(output)
        output = self.maxpool2d_1(output)

        output = self.conv2d_2(output)
        output = self.relu(output)
        output = self.maxpool2d_2(output)

        output = self.flatten(output)
        output = self.batchnorm1d(output)
        output = self.linear1(output)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.linear2(output)
        return self.softmax(output)
