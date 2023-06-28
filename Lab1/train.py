

import torch
import wandb
import numpy as np
from mlp import *
from cnnet import *
from dataset import *

import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hyperparameters = {
    'epochs' : 50, 
    'lr' : 0.001, 
    'batch_size' : 256, 
    'input_size' : 32*32, 
    'width' : 16, 
    'depth' : 5, 
}



def model_pipeline():

    with wandb.init(project="DLA-LAB1", config=hyperparameters):
        #access all HPs through wandb.config, so logging matches executing
        config = wandb.config

        #make the model, data and optimization problem
        model, criterion, optimizer, trainloader, testloader, validationloader = create(config)

        #train the model
        train(model, trainloader, criterion, optimizer, validationloader, config)

        #test the model
        print("Accuracy test: ",test(model, testloader))
        
    return model

def create(config):
    
    #Create a model
    #model = MLP(block, hidden_size=config.width, num_hidden_layers=config.depth, output_size=10, image_size=config.input_size).to(device)

    model = CNNNet_().to(device)
    model.load_state_dict(torch.load("cifar10.pt"))
    nparameters = sum(p.numel() for p in model.parameters())
    print(nparameters)
    #Create the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    trainloader,testloader,validationloader = getDataCifar(batch_size=config.batch_size)

    return model, criterion, optimizer,trainloader, testloader, validationloader

# Function to train a model.
def train(model, trainloader, criterion, optimizer, validationloader, config):
    
    #telling wand to watch
    if wandb.run is not None:
        wandb.watch(model, optimizer, log="all", log_freq=1)

    model.train()
    losses, valacc = [], []  
    #progress_bar = tqdm(range(config.epochs), leave=True)
    for epoch in range(config.epochs):
        
       # progress_bar = tqdm(trainloader, desc=f'Training epoch {epoch}', leave=True)
        
        for batch, (images, labels) in enumerate(trainloader):
        
            loss = train_batch(images, labels,model, optimizer, criterion)
            
            
        acc = test(model, validationloader)
        #progress_bar.update(1)
        
        logs = {"loss": loss.detach().item(), "val_acc": acc}
        #progress_bar.set_postfix(**logs)
        print(logs)
        losses.append(loss.item())
        valacc.append(acc)

        wandb.log({"epoch":epoch, "loss":np.mean(losses), "val_acc":np.mean(valacc)}, step=epoch)
        torch.save(model.state_dict(), "cifar10.pt")

    return np.mean(losses)

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



if __name__ == "__main__":
    
    model_pipeline()