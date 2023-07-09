import torch.nn as nn

class classifier_a(nn.Module):
    def __init__(self, inchannels):
        super(classifier_a, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=inchannels,out_channels=64, kernel_size=[5,5],stride=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=[5,5],stride=2) #10x10x64
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(out_features=128, in_features=6400) 
        self.fc2 = nn.Linear(out_features=10, in_features=128)

    def forward(self, x):
        #x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout1(x)
        
        x = x.view(-1, 6400) # 6400 = 10x10x64

        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

       
        return x


class classifier_b(nn.Module):
    def __init__(self, inchannels):
        super(classifier_b, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=inchannels,out_channels=64, kernel_size=[8,8],stride=2) #11x11x64
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128, kernel_size=[6,6],stride=2) #3x3x128
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=128, kernel_size=[5,5],stride=1, padding=1)#1x1x128

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(out_features=10, in_features=128)

    def forward(self, x):

        x = self.dropout1(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.dropout2(x)

        x = x.view(-1, 128)
        
        x = self.fc1(x)

        return x


class block_resnet2d(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None,stride=1):
        super(block_resnet2d, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.identity_downsample = identity_downsample
        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)

        return x
    
class ResNet(nn.Module):
    # Resnet 18 [2, 2, 2, 2]
    def __init__(self, block, image_channels, num_classes):
        super(ResNet, self).__init__()
        # for resnet18
        layers = [2, 2, 2, 2]
        self.expansion = 1

        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, layers[0], 64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc = nn.Linear(512*self.expansion, num_classes)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x



    def _make_layer(self, block, num_residual_block, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, 
                                                out_channels*self.expansion,
                                                kernel_size=1,
                                                stride=stride,
                                                bias=False),
                                                nn.BatchNorm2d(out_channels*self.expansion),
                                                )
        layers.append(
            block(self.in_channels,out_channels, identity_downsample, stride)
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, num_residual_block):
            layers.append(block(self.in_channels,out_channels ))

        return nn.Sequential(*layers)



def CreateResNet2D(img_channels=3, num_classes = 10):
    return ResNet(block_resnet2d, image_channels=img_channels, num_classes=num_classes)

# dict used to get classifiers 
classifiers = {
    'classifier_a': classifier_a(inchannels=1),'classifier_b': classifier_b(inchannels=1),
    'resnet18':ResNet(block_resnet2d, image_channels=3, num_classes=10)
}

def getClassifier(classifierName):
    return classifiers[classifierName]