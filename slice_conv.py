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
        kernel_size = (224, 5)  # 3 by 3
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
  def __init__(self):
    super(TwoInputsNet, self).__init__()
    self.slice_branch = SliceBranch(3,320)
    self.res50_model = models.resnet50(pretrained=True)
    self.res50_features = torch.nn.Sequential(*list(self.res50_model.children())[:-1])

    # self.fc1 = torch.nn.Linear(640 + 320, 2048)  # set up first FC layer
    # self.fc2 = torch.nn.Linear(2048, 2048)  # set up the other FC layer

  def forward(self, x):
    s_b = self.slice_branch(x)
    resnet50 = self.res50_features(x)
    out = torch.cat(s_b, resnet50)    

    return out


# set_trace()
model = TwoInputsNet()


