

import torch
import wandb
import numpy as np
from mlp import *
from cnnet import *
from dataset import *
import torchvision.transforms as T
from tqdm import tqdm 
import matplotlib.pyplot as plt
import cv2
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

    with wandb.init(project="DLA-LAB1", config=hyperparameters, mode="disabled"):
        #access all HPs through wandb.config, so logging matches executing
        config = wandb.config

        #make the model, data and optimization problem
        model, criterion, optimizer, trainloader, testloader, validationloader = create(config)

        #train the model
        train(model, trainloader, criterion, optimizer, validationloader,testloader, config)

        #test the model
        print("Accuracy test: ",test(model, testloader))
        
    return model

def create(config):
    
    #Create a model
    #model = MLP(block, hidden_size=config.width, num_hidden_layers=config.depth, output_size=10, image_size=config.input_size).to(device)

    model = CNNNet_().to(device)
    nparameters = sum(p.numel() for p in model.parameters())
    #print(nparameters)
    #Create the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    trainloader,testloader,validationloader = getSTL(batch_size=config.batch_size)

    return model, criterion, optimizer,trainloader, testloader, validationloader

# Function to train a model.
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
            progress_bar.update(1)
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            losses.append(loss.item())

        wandb.log({"epoch":epoch, "loss":np.mean(losses)})
        

        if epoch%1==0:
            val = test(model, validationloader)
            acc = test(model, testloader)
            wandb.log({"validation_accuracy":val, "test_accuracy":acc})
            torch.save(model.state_dict(), "SLT.pt")

    return #np.mean(losses)

def train_batch(images, labels, model, optimizer, criterion):

    #insert data into cuda if available
    images,labels = images.to(device), labels.to(device)
    
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward pass
    outputs, _, _ = model(images)
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
            oututs, _, _ = model(images)
            _, predicated = torch.max(oututs.data, 1)
            total += labels.size(0)

            correct += (predicated == labels).sum().item()

    cam_test(model, test_loader)
    return correct/total
    
import torchvision.utils as vutils

def cam_test(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)
            oututs, b_gap, a_gap = model(images)
            
            _, predicated = torch.max(oututs.data, 1)
            total += labels.size(0)
            correct += (predicated == labels).sum().item()
            k = 6
            vutils.save_image(images[k], "image.jpg")
            c = torch.sum(b_gap[k]*a_gap[k][:,None, None], dim = 0)
            # c  = cam 
            # Normalize
            c = (c-torch.min(c))/(torch.max(c)-torch.min(c))
            
            cam_img = np.uint8(255 * c.cpu().numpy())

            hm = cv2.applyColorMap(cv2.resize(cam_img, (96, 96)), cv2.COLORMAP_JET)
            re = hm*0.3+(images[k].permute(1,2,0).cpu().numpy()*255 )*0.5

            cv2.imwrite('CAM.jpg', re)
            """transform = T.Resize(size = (96,96))
            c_ = transform(c[1, None, None])
            vutils.save_image(c_, "image_1.jpg")
            vutils.save_image(c, "image_2.jpg")
            fig = plt.figure(figsize =(10, 10))
            plt.imshow(c_.permute(1, 2, 0).cpu())
            fig.savefig("im.png")
"""


            break

    return 

if __name__ == "__main__":
    
    model_pipeline()