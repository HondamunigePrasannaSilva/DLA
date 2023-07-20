
import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel, AdamW
from torch import nn
from sklearn.metrics import accuracy_score
from datasets import load_dataset


def get_imdb_dataset():
    dataset = load_dataset('imdb', split=['train', 'test'])
    train_dataset = dataset[0]
    test_dataset = dataset[1]
    return train_dataset, test_dataset


train_data, test_data = get_imdb_dataset()

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Encode the dataset
encoded_dataset = train_data.map(lambda example: tokenizer(example['text'], truncation=True, padding='max_length'), batched=True)
