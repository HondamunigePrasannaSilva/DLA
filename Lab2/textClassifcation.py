
import torch
from transformers import GPT2Tokenizer, GPT2Model
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader , TensorDataset, Dataset
import numpy as np
from tqdm import tqdm 
import wandb
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function that get the dataset from hugging face
def get_rotten_tomatoes_dataset():

    dataset = load_dataset('rotten_tomatoes', split=['train', 'test'])
    return dataset[0], dataset[1] # train_dataset, test_dataset

config = { 'epochs':10,'lr':0.001,'batch_size':128,'log':'disabled','linear':False }

def main():
        
    parser = argparse.ArgumentParser(description='DiffDefence: main module!')
    
    parser.add_argument("--epochs",             type=int, default=100, help='num epochs')
    parser.add_argument("--lr",                 type=float, default=0.003, help='lr of the linear classifier')
    parser.add_argument("--batch_size",         type=int, default=128, help='batch size')
    parser.add_argument("--log",                type=str, default="disabled", help='to log using ')
    parser.add_argument("--linear",             type=bool, default=True, help='classifier on top GPT2 ()')
    
    args = parser.parse_args()

    # Load the configuration on the dict!
    config['epochs']            = args.epochs
    config['lr']                = args.lr    
    config['batch_size']        = args.batch_size
    config['log']               = args.log
    config['linear']            = args.linear



    model_pipeline(config)



# Define MLP to setup on top of GPT2

class TextMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,  linear = False):
        super(TextMLP, self).__init__()

        if linear == False: 
            self.seq = nn.Sequential(
                nn.Linear(input_dim, hidden_dim*2),
                nn.BatchNorm1d(hidden_dim*2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.seq = nn.Sequential(
                nn.Linear(input_dim, output_dim, bias=False),
            )

    def forward(self, x):
        return self.seq(x)
    

def model_pipeline(config):

    with wandb.init(project="DLA-LAB2", config=config, mode=config['log']):
        #access all HPs through wandb.config, so logging matches executing
        config = wandb.config

        #make the model, data and optimization problem
        train_data,test_data, gpt2_model, model, criterion, optimizer = create(config)

        #train the model
        model = train(train_data,test_data, gpt2_model, model, criterion, optimizer, config)

        #test the model
        print("Accuracy test: ",test(test_data, gpt2_model, model))

        
    return model

def createDataloader(encoder_dataset):
    
    labels = torch.tensor(encoder_dataset['label'])

    # Trasform input_ids and attention mask in tensor to batch it
    tensor_ids = torch.tensor(encoder_dataset['input_ids'])
    tensor_att = torch.tensor(encoder_dataset['attention_mask'])
    
    # Calculate the minimum padding neccessary
    minimum_padding = tensor_att.squeeze().sum(dim=1).max()+1
    
    #concatenating 
    data = torch.cat([tensor_ids, tensor_att], dim = 1).to(torch.long)


    return data,labels, minimum_padding
    
def create(config):

    train_data, test_data = get_rotten_tomatoes_dataset()

    # Load GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # so i can use the last hidden state
    tokenizer.padding_side='left'

    # Encode the dataset using tokenizer and padding on left, to take the last hidden state
    encoded_dataset_train = train_data.map(lambda data: tokenizer(data['text'], truncation=True, padding='max_length', return_tensors="pt"))
    encoded_dataset_test = test_data.map(lambda data: tokenizer(data['text'], truncation=True, padding='max_length', return_tensors="pt"))

    # Create the data loader 
    trainset,label1,  minimum_padding_1 = createDataloader(encoded_dataset_train)
    testset,label2,  minimum_padding_2 = createDataloader(encoded_dataset_test)
    
    trainset = TensorDataset(trainset[:,:,-max(minimum_padding_1, minimum_padding_2):] , label1)
    testset = TensorDataset(testset[:,:,-max(minimum_padding_1, minimum_padding_2):] , label2)

    trainloader = DataLoader(trainset, batch_size = config['batch_size'], shuffle=True, num_workers=8, drop_last=True)
    testloader = DataLoader(testset, batch_size = config['batch_size'])
    
    # Load the GPT-2 model and Freeze GPT-2 parameters
    gpt2_model = GPT2Model.from_pretrained("gpt2").to(device)

    # Define MLP model 
    model = TextMLP(input_dim=max(minimum_padding_1, minimum_padding_2)*768, hidden_dim = 1024, output_dim = 2, linear = False).to(device)

    #Create the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    return trainloader,testloader, gpt2_model, model, criterion, optimizer


def train(train_data,test_data, gpt2_model, model, criterion, optimizer, config):
    
    if wandb.run is not None:
        wandb.watch(model, optimizer, log="all", log_freq=1)


    for param in gpt2_model.parameters():
        param.requires_grad = False

    loss_plot , accuracy_test = [], []
    for epoch in range(config['epochs']):
        
        progress_bar = tqdm(total=len(train_data),unit='step', desc=f'epoch {epoch}', leave=True)
        losses = []
        for data, label in train_data:
            data = data.to(device)
            label = label.to(device)

            # Get feature from GPT2 model
            with torch.no_grad():
                output = gpt2_model(input_ids = data[:,0,:],attention_mask = data[:,1,:])
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            last_hidden_state = output.last_hidden_state
            outputs = model(last_hidden_state.view(last_hidden_state.shape[0],last_hidden_state.shape[1]*last_hidden_state.shape[2]))

            loss = criterion(outputs, label)
            
            #backward pass
            loss.backward()

            #step with optimizer
            optimizer.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            
            losses.append(loss.item())
            

        acc = test(test_data, gpt2_model, model)
        accuracy_test.append(acc)

        wandb.log({"epoch":epoch, "loss":np.mean(losses), "test_accuracy":acc})
        
        torch.save(model.state_dict(), "Lab2/finetune.pt")
        
        loss_plot.append(np.mean(losses))

        torch.save(torch.tensor(loss_plot), "Lab2/loss_plot.pt")
        torch.save(torch.tensor(accuracy_test), "Lab2/acc_plot.pt")

        if config['log'] == 'disabled':
            print(f"Loss: {np.mean(losses)} Test accuracy: {acc}")

    return model

def test(test_data, gpt2_model, model):
        
    for param in gpt2_model.parameters():
        param.requires_grad = False

    model.eval()

    total, correct = 0, 0
    with torch.no_grad():

        for data, label in test_data:
            data = data.to(device)
            label = label.to(device)

            # Get feature from GPT2 model        
            output = gpt2_model(input_ids = data[:,0,:],attention_mask = data[:,1,:])
            
            # forward pass
            last_hidden_state = output.last_hidden_state
            outputs = model(last_hidden_state.view(last_hidden_state.shape[0],last_hidden_state.shape[1]*last_hidden_state.shape[2]))

            _, predicated = torch.max(outputs.data, 1)
            total += label.size(0)

            correct += (predicated == label).sum().item()
    
    model.train()
    
    return correct/total


if __name__ == '__main__':
    
    main()        