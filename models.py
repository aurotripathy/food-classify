from torchvision import models

def get_model():
        return models.resnet101(pretrained=True)  # pretrained=True will download its weights
