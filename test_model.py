import torch
import argparse
from models import get_model
from utils import get_data_loaders
import logging
from os.path import join
from pudb import set_trace

def test_model(model, dataloaders, device):

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

def test_model_tencrop(model, dataloaders, device):
    """
    https://pytorch.org/docs/master/torchvision/transforms.html
    use torchvision.transforms.TenCrop
    """
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data" , type=str, required=True, 
                        help="Path to the training data (in PyTorch ImageFolder format)")
    parser.add_argument("--batch-size" , type=int, required=False, default=64,
                        help="Batch size (will be split among devices used by this invocation)")
    parser.add_argument("--model-file" , type=str, required=True, 
                        help="Full file path to the model")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logs_folder = '.'
    log_file = 'test_results.log'
    logging.basicConfig(filename=join(logs_folder, log_file),
                    level=logging.INFO,
                    filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    
    dataloaders, dataset_sizes, class_names = get_data_loaders(args.train_data, args.batch_size)
    model = get_model('resnet_plus_slice', len(class_names))
    model = torch.nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_file))
    print(device)
    print("Test size: {} images".format(dataset_sizes['test'],))
    set_trace()
    test_model(model, dataloaders, device)
