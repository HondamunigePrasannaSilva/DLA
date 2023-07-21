import torch.nn as nn
import torch
"""
    This class implements a Multi Layer Perceptron with residual connection
"""

class block_(nn.Module):
    def __init__(self, hidden_size, residual = None) :
        super(block_, self).__init__()
        
        self.residual = residual
        self.linear = nn.Linear(hidden_size, hidden_size, bias = False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x
        
        x = self.linear(x)
        x = self.relu(x)

        if self.residual is not None:
            x = x + identity

        return x
    
class MLP(nn.Module):

    def __init__(self,block, num_hidden_layers, hidden_size, output_size, image_size):
        super(MLP, self).__init__()

        self.firstLayer = nn.Linear(in_features=image_size, out_features=hidden_size)
        self.relu = nn.ReLU()
    
        self.hiddenLayer = self._make_layer(block, num_block=num_hidden_layers, hidden_size=hidden_size)

        self.lastLayer = nn.Linear(hidden_size, output_size) 
        self.relu_2 = torch.nn.ReLU()
    
    def forward(self, x):

        x = self.firstLayer(x.flatten(1))
        x = self.relu(x)

        x = self.hiddenLayer(x)

        x = self.lastLayer(x)
        
        return x
    
    def _make_layer(self, block, num_block, hidden_size):
        layers = []

        for i in range(num_block):
            layers.append(block(hidden_size = hidden_size, residual = None))

        return nn.Sequential(*layers)