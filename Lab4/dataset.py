
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torch.utils.data import Subset
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2

def getDataMnist(batch_size = 32):

    # Standard MNIST transform.
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST train and test.
    ds_train = MNIST(root='./data', train=True, download=True, transform=transform)
    ds_test = MNIST(root='./data', train=False, download=True, transform=transform)

    # Split train into train and validation.
    val_size = 5000
    I = np.random.permutation(len(ds_train))
    ds_val = Subset(ds_train, I[:val_size])
    ds_train = Subset(ds_train, I[val_size:])

    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True ,  num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size,  num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size,  num_workers=8)

    return train_dataloader, test_dataloader, val_dataloader



def getDataCifar(batch_size = 32):
    
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    trainset = CIFAR10(root='./data', train=True,download=True, transform=transform_train)
    testset = CIFAR10(root='./data', train=False,download=True, transform=transform_test)

    # Split train into train and validation.
    val_size = 5000
    I = np.random.permutation(len(trainset))
    ds_val = Subset(trainset, I[:val_size])
    ds_train = Subset(trainset, I[val_size:])

    trainloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size,shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size,  num_workers=8)


    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    return trainloader, testloader,val_dataloader

def getDataCifar100(batch_size = 32):
    transform = transforms.Compose(
    [transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
     ])


    trainset = CIFAR100(root='./data', train=True,download=True, transform=transform)
    testset = CIFAR100(root='./data', train=False,download=True, transform=transform)

    # Split train into train and validation.
    val_size = 5000
    I = np.random.permutation(len(trainset))
    ds_val = Subset(trainset, I[:val_size])
    ds_train = Subset(trainset, I[val_size:])

    trainloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size,shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size,  num_workers=8)


    return trainloader, testloader,val_dataloader