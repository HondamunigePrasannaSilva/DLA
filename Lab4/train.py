

import torch
import wandb
import numpy as np
from classifiers import *
from dataset import *
import torchvision.transforms as T
from tqdm import tqdm 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

"""
    In this file you can train classifier to use for adversarial attack
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']

hyperparameters = {
    'epochs' :1, 
    'lr' : 0.001, 
    'batch_size' : 256
}


def model_pipeline():

    with wandb.init(project="DLA-LAB4", config=hyperparameters, mode="disabled"):
        #access all HPs through wandb.config, so logging matches executing
        config = wandb.config

        #make the model, data and optimization problem
        model, criterion, optimizer, trainloader, testloader, validationloader = create(config)

        #train the model
        model_trained = train(model, trainloader, criterion, optimizer, validationloader,testloader, config)

        #test the model
        print("Accuracy test: ",test(model, testloader))

        # OOD Pipeline

    return

def create(config):
    
    #Create a model

    model = classifiers['resnet18'].to(device)
    
    #Create the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    trainloader,testloader,validationloader = getDataCifar(batch_size=config.batch_size)

    return model, criterion, optimizer,trainloader, testloader, validationloader


def train(model, trainloader, criterion, optimizer, validationloader,testloader, config):
    
    #telling wand to watch
    if wandb.run is not None:
        wandb.watch(model, optimizer, log="all", log_freq=1)

    model.train()
    losses, valacc = [], []  

    for epoch in range(config.epochs):
        
        progress_bar = tqdm(total=len(trainloader),unit='step', desc=f'epoch {epoch}', leave=True)
        
        for batch, (images, labels) in enumerate(trainloader):
            loss = train_batch(images, labels,model, optimizer, criterion)

            # Progress bar stuff
            progress_bar.update(1)
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            losses.append(loss.item())

        wandb.log({"epoch":epoch, "loss":np.mean(losses)})
        
        if epoch%5==0:
            val = test(model, validationloader)
            acc = test(model, testloader)
            wandb.log({"validation_accuracy":val, "test_accuracy":acc})
            torch.save(model.state_dict(), "cifar10.pt")


    return model

def train_batch(images, labels, model, optimizer, criterion):

    #insert data into cuda if available
    images,labels = images.to(device), labels.to(device)
    
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    #backward pass
    loss.backward()

    #step with optimizer
    optimizer.step()

    return loss

def test(model, test_loader):
    model.eval()

    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            oututs = model(images)
            _, predicated = torch.max(oututs.data, 1)
            total += labels.size(0)

            correct += (predicated == labels).sum().item()

    return correct/total
