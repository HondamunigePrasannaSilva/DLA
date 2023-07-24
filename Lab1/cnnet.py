import torch.nn as nn
import torch

# mettere una
# Implementing resnet18  
class block(nn.Module):
    def __init__(self, in_channels, out_channels,res, stride=1):
        super(block, self).__init__()
        
        self.res = res

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),

        )

        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x

        x = self.seq(x)
        
        if self.res == True:
            x = x+identity
            x = self.relu(x)
        
        return x
    
class CNNNet(nn.Module):
    def __init__(self, block, image_channels, num_classes, num_hidden_block, residual):
        super(CNNNet, self).__init__()
        

        self.in_channels = 128
        self.conv1 = nn.Conv2d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, num_hidden_block, residual)
        

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, num_classes)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        output = x  #[256, 64, 8, 8]

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)   #[128]
        output_gap = x
        x = self.fc(x)

        return x, output, output_gap


    def _make_layer(self, block, num_block, res):
        layers = []

        for i in range(1, num_block):
            layers.append(block(self.in_channels,self.in_channels, res))

        return nn.Sequential(*layers)

   

def CNNNet_(img_channels=3, num_classes = 10, num_bloc = 5, res = False):
    return CNNNet(block, image_channels=img_channels, num_classes=num_classes, num_hidden_block = num_bloc, residual = res)