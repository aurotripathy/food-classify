from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_model(model_name, nb_classes):
        if model_name == 'resnet_std':
                model = models.resnet101(pretrained=True)  # pretrained=True will download its weights
                num_in_features_last = model.fc.in_features
                # Newly constructed module has requires_grad=True by default
                model.fc = nn.Linear(num_in_features_last, nb_classes)
                return model
        elif model_name == 'resnet_ext':
                return TwoInputsNet(nb_classes)
        else:
                print('Error in model selection')
                exit(2)

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
        out2 = self.maxpool(out1)
        out3 = self.maxpool(out2)
        out4 = self.maxpool(out3)
        
        return out4



class TwoInputsNet(torch.nn.Module):
  def __init__(self, nb_classes):
    super(TwoInputsNet, self).__init__()
    self.slice_branch = SliceBranch(3,320)
    self.res50_model = models.resnet50(pretrained=True)
    self.res50_features = torch.nn.Sequential(*list(self.res50_model.children())[:-1])

    self.fc1 = torch.nn.Linear(2368, 2048)  
    self.fc2 = torch.nn.Linear(2048, nb_classes)  

  def forward(self, x):
    s_b = self.slice_branch(x)
    resnet50 = self.res50_features(x)
    out = torch.cat([s_b, resnet50], dim=1)    
    out = torch.flatten(out, 1)
    out = self.fc1(out)
    out = self.fc2(out)
    return out

