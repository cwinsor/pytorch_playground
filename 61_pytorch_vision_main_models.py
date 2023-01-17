'''
PyTorch models and pre-trained weights
from https://pytorch.org/vision/main/models.html#models
'''

# The torchvision.models subpackage contains definitions of models for
# addressing different tasks, including: 
# image classification, 
# pixelwise semantic segmentation, 
# object detection, 
# instance segmentation, 
# person keypoint detection, 
# video classification, and 
# optical flow.

# TorchVision offers pre-trained weights for every provided architecture, 
# using the PyTorch torch.hub. Instancing a pre-trained model will download 
# its weights to a cache directory. This directory can be set using 
# the TORCH_HOME environment variable.

import torch
from torchvision.models import resnet50, ResNet50_Weights

# Old weights with accuracy 76.130%
resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# New weights with accuracy 80.858%
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Best available weights (currently alias for IMAGENET1K_V2)
# Note that these weights may change across versions
resnet50(weights=ResNet50_Weights.DEFAULT)

# Strings are also supported
y = resnet50(weights="IMAGENET1K_V2")

# No weights - random initialization
resnet50(weights=None)
print(y)


# Before using the pre-trained models, one must preprocess the 
# image (resize with right resolution/interpolation, apply inference transforms, 
# rescale the values etc). There is no standard way to do this as it depends 
# on how a given model was trained. It can vary across model families, 
# variants or even weight versions. Using the correct preprocessing method 
# is critical and failing to do so may lead to decreased accuracy or incorrect outputs.

# All the necessary information for the inference transforms of each pre-trained 
# model is provided on its weights documentation. To simplify inference, TorchVision 
# bundles the necessary preprocessing transforms into each model weight. These are 
# accessible via the weight.transforms attribute:

# Initialize the Weight Transforms
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

img = torch.rand(10, 3,300,100)

# Apply it to the input image
img_transformed = preprocess(img)
print("here")