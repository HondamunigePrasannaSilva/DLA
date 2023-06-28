import torch.nn as nn
import torch

from cnnet import *
    
class CAM(nn.Module):
    def __init__(self, image_channels, num_classes):
        super(CAM, self).__init__()
        
        self.CNNet = CNNNet_(img_channels=image_channels, num_classes = num_classes)
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=2, stride=1, padding=3), # [10, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=2, stride=1, padding=3), # [10, 18, 18]
            nn.ReLU(),
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=2, stride=1, padding=3), # [10, 23, 23]
            nn.ReLU(),
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=2, stride=1, padding=3), # [10, 23, 23]
            nn.ReLU(),
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=2, stride=1, padding=3), # [10, 28, 28]
            nn.ReLU(),
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1, stride=1, padding=2), # [10, 32, 32]

        )
        

    
    def forward(self,x):

        x, output = self.CNNet()
        o = self.seq(output)
      
   