import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel, AdamW
from torch import nn
from sklearn.metrics import accuracy_score
from datasets import load_dataset

# Load IMDb dataset
dataset = load_dataset('imdb', split='train')

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode the dataset
encoded_dataset = dataset.map(lambda example: tokenizer(example['text'], truncation=True, padding='max_length'), batched=True)

# Define MLP classifier on top of GPT-2
class GPT2Classifier(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GPT2Classifier, self).__init__(config)
        self.gpt2 = GPT2Model(config)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

# Set random seed for reproducibility
torch.manual_seed(42)

# Prepare data for fine-tuning
input_ids = torch.tensor(encoded_dataset['input_ids'])
attention_mask = torch.tensor(encoded_dataset['attention_mask'])
labels = torch.tensor(encoded_dataset['label'])

# Instantiate GPT-2 classifier model
gpt2_model = GPT2Classifier.from_pretrained('gpt2', num_labels=2)

# Define optimizer and loss function
optimizer = AdamW(gpt2_model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# Fine-tuning loop
for epoch in range(3):  # Number of epochs
    running_loss = 0.0
    predictions = []
    true_labels = []

    for i in range(len(input_ids)):
        optimizer.zero_grad()

        # Forward pass
        logits = gpt2_model(input_ids[i].unsqueeze(0), attention_mask[i].unsqueeze(0))
        loss = loss_fn(logits, labels[i].unsqueeze(0))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted_label = torch.argmax(logits)
        predictions.append(predicted_label.item())
        true_labels.append(labels[i].item())

    epoch_loss = running_loss / len(input_ids)
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
