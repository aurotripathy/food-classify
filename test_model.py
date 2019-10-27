import torch
import argparse
from models import get_model
from utils import get_data_loaders, get_tencrop_data_loader, get_logfilename_with_datetime
import logging
from os.path import join, exists
from pudb import set_trace

def test_model(model, dataloaders, device):

    model.eval()
    running_corrects = 0
    with torch.no_grad():
        set_trace()
        for _, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)  # predictions == argmax
            running_corrects += torch.sum(predictions == labels.data)
            acc = running_corrects.double() / dataset_sizes['test']
        logging.info('Test accuracy: {:.4f}'.format(acc))

def test_model_tencrop(model, dataloader, device):
    """
    https://pytorch.org/docs/master/torchvision/transforms.html
    uses torchvision.transforms.TenCrop in the dataloader
    """
    model.eval()
    running_corrects = 0
    with torch.no_grad():
        set_trace()
        for _, (inputs, labels) in enumerate(dataloader['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            bs, ncrops, c, h, w = inputs.size()
            result = model(inputs.view(-1, c, h, w)) # fuse batch size and ncrops
            result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops            
            _, predictions = torch.max(result_avg, 1)  # predictions == argmax
            running_corrects += torch.sum(predictions == labels.data)
            acc = running_corrects.double() / dataset_sizes['test']
        logging.info('Test accuracy: {:.4f}'.format(acc))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data" , type=str, required=True, 
                        help="Path to the train/val/test folder (in PyTorch ImageFolder format). Uses only the test folder")
    parser.add_argument("--batch-size" , type=int, required=False, default=64,
                        help="Batch size (will be split among devices used by this invocation)")
    parser.add_argument("--model-file" , type=str, required=True, 
                        help="Full file path to the model")
    parser.add_argument("--logs-folder", type=str, required=False, default='logs',
                        help="Location of logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if not exists(args.logs_folder):
        os.makedirs(args.logs_folder)
    log_file = get_logfilename_with_datetime('test-log')
    logging.basicConfig(filename=join(args.logs_folder, log_file),
                    level=logging.INFO,
                    filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    
    dataloader, dataset_sizes, class_names = get_tencrop_data_loader(args.test_data, args.batch_size)
    # dataloaders, dataset_sizes, class_names = get_data_loaders(args.train_data, args.batch_size)
    model = get_model('wide_resnet_plus_slice', len(class_names))
    model = torch.nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_file))
    print(device)
    print("Test size: {} images".format(dataset_sizes['test'],))
    test_model_tencrop(model, dataloader, device)
    # test_model(model, dataloaders, device)
