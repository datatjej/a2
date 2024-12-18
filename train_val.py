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
from wikiart import WikiArtDataset, WikiArtModel
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")

args = parser.parse_args()

config = json.load(open(args.config))

trainingdir = config["trainingdir"]
validationdir = config["validationdir"]

device = config["device"]

print("Running...")

traindataset = WikiArtDataset(trainingdir, device)
validationset = WikiArtDataset(validationdir, device)

def train(epochs=3, batch_size=32, modelfile=None, device="cpu"):
    train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validationset, batch_size=batch_size, shuffle=False)	

    model = WikiArtModel().to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss().to(device)

    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(train_loader)):
            X, y = batch
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            accumulate_loss += loss
            optimizer.step()

        print("In epoch {}, loss = {}".format(epoch, accumulate_loss))

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_id, batch in enumerate(tqdm.tqdm(val_loader)):
                X, y = batch
                y = y.to(device)
                output = model(X)
                loss = criterion(output, y)
                val_loss += loss.item()
        
        print("In epoch {}, validation loss = {}".format(epoch, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if modelfile:
                torch.save(model.state_dict(), modelfile)

    return model


model = train(config["epochs"], config["batch_size"], modelfile=config["modelfile"], device=device)    
