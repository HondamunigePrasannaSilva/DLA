
import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel, AdamW
from torch import nn
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from torch.utils.data import DataLoader , TensorDataset, Dataset
import numpy as np

"""class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, data):
        self.labels = [data[label] for label in df['label']]

        self.texts = [tokenizer(data['text'], truncation=True, padding='max_length') for text in df['text']]
        
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, idx):
        # Get a batch of labels
        return np.array(self.labels[idx])
    
    def get_batch_texts(self, idx):
        # Get a batch of inputs
        return self.texts[idx]
    
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y
"""

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



def get_imdb_dataset():

    dataset = load_dataset('imdb', split=['train', 'test'])
    return dataset[0], dataset[1] # train_dataset, test_dataset


def create():

    train_data, test_data = get_imdb_dataset()

    # Load GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Encode the dataset
    encoded_dataset_train = train_data.map(lambda data: tokenizer(data['text'], truncation=True, padding='max_length'), batched=True)
    #encoded_dataset_test = test_data.map(lambda data: tokenizer(data['text'], truncation=True, padding='max_length'), batched=True)

    train_dataloader = DataLoader(encoded_dataset_train, batch_size=8, shuffle=True)

    # Load the GPT-2 model and Freeze GPT-2 parameters
    gpt2_model = GPT2Model.from_pretrained("gpt2")
    
    for param in gpt2_model.parameters():
        param.requires_grad = False

    model = TextMLP(input_dim=6*768, hidden_dim = 1024, output_dim = 2)

    #Create the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    #return encoded_dataset_train,encoded_dataset_test, gpt2_model
    return train_data,test_data, gpt2_model, model, criterion, optimizer

create()

"""
def train():
    
    train_data,test_data, gpt2_model, model, criterion, optimizer = create()

    for epoch in range (10):

        for idx in range(len(train_data)):


    return 0
"""

    
