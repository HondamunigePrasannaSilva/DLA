

import torch
import wandb
import numpy as np
from classifiers import *
from dataset import *
import torchvision.transforms as T
from tqdm import tqdm 
import matplotlib.pyplot as plt
import cv2
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']

hyperparameters = {
    'epochs' :601, 
    'lr' : 0.01, 
    'batch_size' : 512
}


def model_pipeline():

    with wandb.init(project="DLA-LAB4", config=hyperparameters, mode="disabled"):
        #access all HPs through wandb.config, so logging matches executing
        config = wandb.config

        #make the model, data and optimization problem
        model, criterion, optimizer, trainloader, testloader, validationloader, valloaderOOD, scheduler = create(config)

        #train the model
        model_trained = train(model, trainloader, criterion, optimizer, validationloader,testloader, config, scheduler)

        #test the model
        print("Accuracy test: ",test(model, testloader))

        # OOD Pipeline
        OOD_pipeline(model_trained, validationloader, valloaderOOD)

    return

def create(config):
    
    #Create a model

    model = classifiers['resnet18'].to(device)
    
    #Create the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    trainloader,testloader,validationloader = getDataCifar(batch_size=config.batch_size)
    _,_,valloaderOOD = getDataCifar100(batch_size=config.batch_size)

    return model, criterion, optimizer,trainloader, testloader, validationloader, valloaderOOD, scheduler

def train(model, trainloader, criterion, optimizer, validationloader,testloader, config, scheduler):
    
    #telling wand to watch
    if wandb.run is not None:
        wandb.watch(model, optimizer, log="all", log_freq=1)

    model.train()
    losses, valacc = [], []  

    for epoch in range(config.epochs):
        
        progress_bar = tqdm(total=len(trainloader),unit='step', desc=f'epoch {epoch}', leave=False)
        
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

            if wandb.run is not None:
                wandb.log({"validation_accuracy":val, "test_accuracy":acc})
            
            print(f"Validation accuracy:{val}, Test Accuracy:{acc}")    
            
            torch.save(model.state_dict(), "cifar10_2.pt")
        
        scheduler.step()


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
    
    model.train()
    return correct/total

def OOD_pipeline(model, dl_test, dl_fake , datasetname = 'CIFAR10'):

    # Function to collect all logits from the model on entire dataset.
    def collect_logits(model, dl):
        logits = []
        with torch.no_grad():
            for (Xs, _) in dl:
                Xs = Xs.to(device) 
                logits.append(model(Xs.to(device)).cpu().numpy())
        return np.vstack(logits)

    logits_ID = collect_logits(model, dl_test)
    logits_OOD = collect_logits(model, dl_fake)
    

    y = np.concatenate((np.ones(len(dl_fake.dataset)), np.zeros(len(dl_test.dataset))))

    # with variance
    list_ID_ = [ logits_ID[i].std()   for i in range(len(logits_ID)  )]
    list_OOD_ = [ logits_OOD[i].std() for i in range(len(logits_OOD) )]
    
    #list_ID_ = [ logits_ID[i].max()   for i in range(len(logits_ID)  )]
    #list_OOD_ = [ logits_OOD[i].max() for i in range(len(logits_OOD) )]

    scores = list_ID_ + list_OOD_

    RocCurveDisplay.from_predictions( y, scores, name="", color="darkorange",)
    img_path = []
    
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"OOD Detetection on {datasetname}")
    plt.legend()
    plt.savefig("./Lab4/img/AUROC_1.png")
    img_path.append(f"./Lab4/img/AUROC_1.png")
    plt.clf()


    precision, recall, _ = precision_recall_curve(y, scores)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"OOD Detetection on {datasetname}")
    plt.savefig("./Lab4/img/AUPR_1.png")
    img_path.append(f"./Lab4/img/AUPR_1.png")
    plt.clf()


    list_ID_ = [ logits_ID[i].max()   for i in range(len(logits_ID)  )]
    list_OOD_ = [ logits_OOD[i].max() for i in range(len(logits_OOD) )]

    scores = list_ID_ + list_OOD_

    RocCurveDisplay.from_predictions( y, scores, name="", color="darkorange",)

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"OOD Detetection on {datasetname}")
    plt.legend()
    plt.savefig("./Lab4/img/AUROC_2.png")
    img_path.append(f"./Lab4/img/AUROC_2.png")
    plt.clf()


    precision, recall, _ = precision_recall_curve(y, scores)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"OOD Detetection on {datasetname}")
    
    plt.savefig("./Lab4/img/AUPR_2.png")
    img_path.append(f"./Lab4/img/AUPR_2.png")

    plt.clf()

    eps_ = ['AUROC_var', 'AUPR_var','AUROC_max', 'AUPR_max']

    
    
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  


    for i, image_path in enumerate(img_path):
        img = mpimg.imread(image_path)  
        axs[i].imshow(img)  
        axs[i].axis('off')
        axs[i].set_title(eps_[i], fontsize=15)

    plt.tight_layout()  
    plt.figtext(0.5, 0.05, "OOD detection using variance and max logits to determine if OOD", ha='center', fontsize=20)
    plt.savefig(f"Lab4/img/all_img_3.png")  
    plt.close()




if __name__ == "__main__":
    
    #model_pipeline()
    
    model = classifiers['resnet18'].to(device)
    model.load_state_dict(torch.load("/home/hsilva/DLA/cifar10_3.pt"))

    _,_,validationloader = getDataCifar(batch_size=256)
    _,_,valloaderOOD = getDataCifar100(batch_size=256)

    OOD_pipeline(model, validationloader, valloaderOOD)


