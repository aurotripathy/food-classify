from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
TODO
https://pytorch.org/hub/pytorch_vision_wide_resnet/
"""

drop_prob = 0.5

def get_model(model_name, nb_classes):
        if model_name == 'resnet101':
                # pretrained=True will download its weights
                model = models.resnet101(pretrained=True)
                num_in_features_last = model.fc.in_features
                # Newly constructed module has requires_grad=True by default
                model.fc = nn.Linear(num_in_features_last, nb_classes)
                return model
        elif model_name == 'resnet_plus_slice':
                return Resnet101PlusSlice(nb_classes, drop_prob)
        else:
                print('Error in model selection')
                exit(2)

class SliceBranch(torch.nn.Module):
    """ Describe slice branch from the paper """ 
    def __init__(self, input_size, output_size):
        super(SliceBranch, self).__init__()
        kernel_size = (224, 5)  # 
        self.wide_conv = torch.nn.Conv2d(input_size,
                                    output_size,
                                    kernel_size,
                                    stride=1,
                                    padding=0,
                                    bias=True)
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.maxpool = torch.nn.MaxPool2d((1, 5))
        
    def forward(self, x):
        out1 = F.relu(self.bn(self.wide_conv(x)))
        out2 = self.maxpool(out1)
        out3 = self.maxpool(out2)
        out4 = self.maxpool(out3)
        
        return out4



class Resnet101PlusSlice(torch.nn.Module):
  def __init__(self, nb_classes, drop_prob):
    super(ResnetPlusSlice, self).__init__()
    self.slice_branch = SliceBranch(3, 320)
    self.res101_pretrained = models.resnet101(pretrained=True)
    self.res101_branch = torch.nn.Sequential(*list(self.res101_pretrained.children())[:-1])

    self.fc1 = torch.nn.Linear(2368, 2048)
    self.dropout = nn.Dropout(p=drop_prob)
    self.fc2 = torch.nn.Linear(2048, nb_classes)  

  def forward(self, x):
    s_b = self.slice_branch(x)
    r_b = self.res101_branch(x)
    out = torch.cat([s_b, r_b], dim=1)    
    out = torch.flatten(out, 1)
    out = self.fc1(out)
    out = self.dropout(out)
    out = self.fc2(out)
    return out

