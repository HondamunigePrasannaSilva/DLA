
import torch
from cnnet import *
from dataset import *
import cv2
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import torchvision
import torchvision.transforms as transforms

# Transform the PyTorch tensors to PIL images
transform = transforms.ToPILImage()

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


model = CNNNet_().to(device)
model.load_state_dict(torch.load("cifar10.pt"))

model._modules.get("layer1").register_forward_hook(hook_feature)

trainloader,testloader,validationloader = getDataCifar(batch_size=1)
im, l = next(iter(testloader))
im = im[0].cpu()
plt.imshow(transform(im))
plt.savefig("prova.png")


params = list(model.parameters())
weights = np.squeeze(params[-2].data.cpu().numpy())

"""
def return_CAM(feature_conv, weight, class_idx):

    size_upsample = (224, 224)
    
    # we only consider one input image at a time, therefore in the case of 
    # VGG16, the shape is (1, 512, 7, 7)
    bz, nc, h, w = feature_conv.shape 
    output_cam = []
    for idx in class_idx:
        beforeDot =  feature_conv.reshape((nc, h*w))# -> (512, 49)
        cam = np.matmul(weight[idx], beforeDot) # -> (1, 512) x (512, 49) = (1, 49)
        cam = cam.reshape(h, w) # -> (7 ,7)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

"""