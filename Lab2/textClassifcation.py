
import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel, AdamW
from torch import nn
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from torch.utils.data import DataLoader , TensorDataset, Dataset
import numpy as np
from tqdm import tqdm 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define MLP to setup on top of GPT2

class TextMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,  linear = False):
        super(TextMLP, self).__init__()

        if linear != True: 
            self.seq = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.seq = nn.Sequential(
                nn.Linear(input_dim, output_dim),
            )

    def forward(self, x):
        return self.seq(x)



def get_rotten_tomatoes_dataset():

    dataset = load_dataset('rotten_tomatoes', split=['train', 'test'])
    return dataset[0], dataset[1] # train_dataset, test_dataset

def createDataloader(encoder_dataset, batch_size, shuffle):
    
    labels = torch.tensor(encoder_dataset['label'])

    # Trasform input_ids and attention mask in tensor to batch it
    tensor_ids = torch.tensor(encoder_dataset['input_ids'])
    tensor_att = torch.tensor(encoder_dataset['attention_mask'])
    
    # Calculate the minimum padding neccessary
    minimum_padding = tensor_att.squeeze().sum(dim=1).max()+1
    
    # clip the unwanted data to make it less heavy    
    tensor_att = tensor_att.squeeze()[:, -minimum_padding:][:, None, :]
    tensor_ids = tensor_ids.squeeze()[:, -minimum_padding:][:, None, :]

    #concatenating 
    data = torch.cat([tensor_ids, tensor_att], dim = 1).to(torch.long)

    dataset_ = TensorDataset(data , labels)

    # Create a dataloader [batch_size, 2, minimum_padding]
    return DataLoader(dataset_, batch_size = batch_size, shuffle=shuffle), minimum_padding

def create():

    train_data, test_data = get_rotten_tomatoes_dataset()

    # Load GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='left'

    # Encode the dataset using tokenizer and padding on left, to take the last hidden state
    encoded_dataset_train = train_data.map(lambda data: tokenizer(data['text'], truncation=True, padding='max_length', return_tensors="pt"))
    encoded_dataset_test = test_data.map(lambda data: tokenizer(data['text'], truncation=True, padding='max_length', return_tensors="pt"))
    
    # Create the data loader 
    trainloader, minimum_padding = createDataloader(encoded_dataset_train, 32, True)
    testloader, _ = createDataloader(encoded_dataset_test, 32, False)

    # Load the GPT-2 model and Freeze GPT-2 parameters
    gpt2_model = GPT2Model.from_pretrained("gpt2").to(device)

    # Define MLP model 
    model = TextMLP(input_dim=minimum_padding*768, hidden_dim = 1024, output_dim = 2, linear = False).to(device)

    #Create the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return trainloader,testloader, gpt2_model, model, criterion, optimizer


def train():
    
    train_data,test_data, gpt2_model, model, criterion, optimizer = create()

    for param in gpt2_model.parameters():
        param.requires_grad = False


    for epoch in tqdm(range(10)):
        
        progress_bar = tqdm(total=len(train_data),unit='step', desc=f'epoch {epoch}', leave=True)
        
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
            
    return 0

train()

    
