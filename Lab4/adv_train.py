

import torch
import wandb
import numpy as np
from classifiers import *
from dataset import *
import torchvision.transforms as T
from torchvision.utils import *
from tqdm import tqdm 
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.metrics import RocCurveDisplay
from attack import *
from OOD_detection_pipeline import *
from torch.utils.data import DataLoader , TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']

"""
    In this file you can do adversarial training and check if a 
    more robust model on perturbated images.
    The attack used is FGSM!
"""

hyperparameters = {
    'epochs' :5, 
    'lr' : 0.01, 
    'batch_size' : 256,
    'log':'disabled',
    'model':'classifier_a',
    'adv_train': False,
    'eps':0.3
}


def model_pipeline():

    with wandb.init(project="DLA-LAB4", config=hyperparameters, mode=hyperparameters['log']):

        #access all HPs through wandb.config, so logging matches executing
        config = wandb.config

        #make the model, data and optimization problem
        model, criterion, optimizer, trainloader, testloader, validationloader = create(config)

        #train the model
        model_trained = adv_train(model, trainloader, criterion, optimizer, validationloader,testloader, config)

        #test the model
        print("Accuracy test: ",test(model_trained, testloader, criterion))
        # Test on the perturbated images
        print("Accuracy test on perturbated images: ",test(model_trained, testloader, criterion, attack = True))

        # OOD pipeline
        dl_fake = createAdversarialImages(model, testloader,criterion, config.eps)
        OOD_pipeline(model, testloader, dl_fake , datasetname = 'MNSIT')
    return

def create(config):
    
    #Create a model
    model = classifiers[config.model].to(device)
    
    #Create the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    trainloader,testloader,validationloader = getDataMnist(batch_size=config.batch_size)

    return model, criterion, optimizer,trainloader, testloader, validationloader#, scheduler

def adv_train(model, trainloader, criterion, optimizer, validationloader,testloader, config):
    
    #telling wand to watch
    if wandb.run is not None:
        wandb.watch(model, optimizer, log="all", log_freq=1)

    model.train()
    losses, valacc = [], []  

    for epoch in range(config.epochs):
        
        progress_bar = tqdm(total=len(trainloader),unit='step', desc=f'epoch {epoch}', leave=True)
        
        for _, (images, labels) in enumerate(trainloader):

            images,labels = images.to(device), labels.to(device)

            if config.adv_train == True:
                # Create adv images
                adv_images, _ = fgsm_attack(model, criterion, images,labels, eps = config.eps, target = False)
                adv_images = adv_images.to(device)

                # Create a batch made of [Original Images, Adv Images]
                labels = torch.cat([labels, labels], dim= 0).to(device)
                images = torch.cat([images, adv_images], dim= 0).to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            #backward pass
            loss.backward()

            #step with optimizer
            optimizer.step()

            # Progress bar stuff
            progress_bar.update(1)
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            losses.append(loss.item())

        wandb.log({"epoch":epoch, "loss":np.mean(losses)})
        
        if epoch%1==0:
            val = test(model, validationloader, criterion)
            acc = test(model, testloader, criterion)
            acc_adv = test(model, testloader, criterion, True)
            wandb.log({"validation_accuracy":val, "test_accuracy":acc,  "test_accuracy_adv":acc_adv})

            if config.log == 'disabled':
                print(f"validation_accuracy:{val}, test_accuracy:{acc},  test_accuracy_adv:{acc_adv}")

            torch.save(model.state_dict(), "./Lab4/Models/advtrainMnist.pt")
        

    return model

def test(model, test_loader,criterion, attack = False):
    model.eval()

   
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        if attack == True:
            images, _ = fgsm_attack(model, criterion, images,labels, eps = hyperparameters['eps'], target = False)

        with torch.no_grad():
            oututs = model(images)

        _, predicated = torch.max(oututs.data, 1)
        total += labels.size(0)
        correct += (predicated == labels).sum().item()

    return correct/total

def createAdversarialImages(model, dataloader,criterion, eps):
        
    for i, (images, label) in enumerate(dataloader):
        images, label = images.to(device), label.to(device)
        if i == 0:    
            adv_images, _ = fgsm_attack(model, criterion, images,label, eps = eps, target = False)
            l = label
        else:
            
            adv_images = torch.cat([adv_images,fgsm_attack(model, criterion, images,label, eps = eps, target = False)[0]], dim = 0)
            l = torch.cat([l, label], dim = 0)
    
    # Create a dataloader with fake images and true labels
    dataset_ = TensorDataset(adv_images , l)
    advtestloader = DataLoader(dataset_, batch_size = 32)

    return advtestloader
    
def testAdversarialImages(model, eps):
    
    _,testloader,_ = getDataCifar(batch_size=10)
    images,labels = next(iter(testloader))
    
    criterion = torch.nn.CrossEntropyLoss()
    
    adv_images, acc = fgsm_attack(model, criterion, images,labels, eps = eps, target = False)
    
    save_image(images[0],'./Lab4/imgs/no_adv.png')
    save_image(adv_images[0],'./Lab4/imgs/adv.png')



if __name__ == "__main__":
    
    model_pipeline()
    
    #model = classifiers['resnet18'].to(device)
    #model.load_state_dict(torch.load("/home/hsilva/DLA/cifar10.pt"))

    #testAdversarialImages(model, eps=0.03)


