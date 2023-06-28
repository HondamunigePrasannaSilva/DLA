import torch.nn as nn
import torch

# mettere una
# Implementing resnet18  
class block(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1):
        super(block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = x+identity
        
        return x
    
class CNNNet(nn.Module):
    def __init__(self, block, image_channels, num_classes, num_hidden_block):
        super(CNNNet, self).__init__()
        

        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, num_hidden_block)
        

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
    
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


    def _make_layer(self, block, num_block):
        layers = []

        for i in range(1, num_block):
            layers.append(block(self.in_channels,self.in_channels))

        return nn.Sequential(*layers)

   

def CNNNet_(img_channels=3, num_classes = 10):
    return CNNNet(block, image_channels=img_channels, num_classes=num_classes, num_hidden_block = 5)