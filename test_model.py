import torch

def test_model(model, dataloaders):

    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)  # predictions == argmax
            running_corrects += torch.sum(predictions == labels.data)
            acc = running_corrects.double() / dataset_sizes['test']
        logging.info('Test accuracy: {:.4f}'.format(acc))

def test_model_tencrop(model, dataloaders):
    """
    https://pytorch.org/docs/master/torchvision/transforms.html
    use torchvision.transforms.TenCrop
    """
    pass
