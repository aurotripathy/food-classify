""" 

"""
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import models
from pudb import set_trace

# three channels (or feature maps)
# 4D matrix, one dimension represents batch-size
input = torch.zeros([1, 3, 224, 224], dtype=torch.float32)


print('Input shape:\n', input.shape)

class SliceBranch(torch.nn.Module):
    """ 2D convolution with three inputs, two outputs """ 
    def __init__(self, input_size, output_size):
        super(SliceBranch, self).__init__()
        kernel_size = (224, 5)  # 
        self.conv = torch.nn.Conv2d(input_size,
                                    output_size,
                                    kernel_size,
                                    stride=1,
                                    padding=0,
                                    bias=True)
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.maxpool = torch.nn.MaxPool2d((1, 5))
        
    def forward(self, x):
        out1 = F.relu(self.bn(self.conv(x)))
        print('out1', out1.shape)

        out2 = self.maxpool(out1)
        print('out2', out2.shape)

        out3 = self.maxpool(out2)
        print('out3', out3.shape)
        
        out4 = self.maxpool(out3)
        print('out4', out4.shape)
        
        return out4



class TwoInputsNet(torch.nn.Module):
  def __init__(self, nb_classes):
    super(TwoInputsNet, self).__init__()
    self.slice_branch = SliceBranch(3,320)
    self.res50_model = models.resnet101(pretrained=True)
    self.res50_features = torch.nn.Sequential(*list(self.res50_model.children())[:-1])
    print('Trucncated Resnet\n', self.res50_features)

    self.fc1 = torch.nn.Linear(2368, 2048)  
    self.fc2 = torch.nn.Linear(2048, nb_classes)  

  def forward(self, x):
    s_b = self.slice_branch(x)
    print('s_b', s_b.shape)
    resnet50 = self.res50_features(x)
    print('resnet50', resnet50.shape)
    out = torch.cat([s_b, resnet50], dim=1)    
    print('concat out\n', out.shape)
    out = torch.flatten(out, 1)
    out = self.fc1(out)
    print('fc1 out\n', out.shape)
    out = self.fc2(out)
    print('fc2 out\n', out.shape)
    return out


# set_trace()
model = TwoInputsNet(101)
reult = model(input)

