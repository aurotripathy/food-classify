"""
Fine-tune from ResNet101 ImageNet weights:
(1) learn the fully-connected layer from scratch
(2) optionally fine-tune the rest of the layers
"""
from __future__ import print_function, division

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import copy
from utils import show_batch, get_data_loaders, display_losses
from utils import get_logfilename_with_datetime
import argparse
import logging
import numpy as np
import os
from os.path import join
from models import get_model
from test_model import test_model

def train_val_model(model, criterion, optimizer, scheduler, num_epochs=15):

    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    train_epoch_losses = []
    val_epoch_losses = []
    for epoch in range(num_epochs):
        logging.info('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Learning rate {}'.format(scheduler.get_lr()))
        
        for phase in ['train', 'val']:  # each epoch does train and validate
            model.train() if phase == 'train' else model.eval()
            
            running_losses = 0.0
            running_corrects = 0

            # Iterate over mini-batches
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # zero out gradients

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # forward pass
                    _, predictions = torch.max(outputs, 1)  # predictions == argmax
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_losses += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)

            epoch_loss = running_losses / dataset_sizes[phase]
            epoch_accuracy = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                train_epoch_losses.append(epoch_loss)
            else:
                val_epoch_losses.append(epoch_loss)
            logging.info('\t{} loss: {:.4f}, {} accuracy: {:.4f}'.format(
                phase, epoch_loss, phase, epoch_accuracy))

            if phase == 'val' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(best_model_weights, join(args.trained_models_folder, 'checkpoint.pth'))

            if phase == 'train':
                scheduler.step()  # called every epoch

    logging.info('Best validation accuracy: {:4f}'.format(best_accuracy))
    model.load_state_dict(best_model_weights)  # retain best weights
    return model, train_epoch_losses, val_epoch_losses



def configure_run_model():

    criterion = nn.CrossEntropyLoss()
    model = get_model('wide_resnet_plus_slice', nb_classes)

    # Optimize all paramters
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    if multi_gpu:
        logging.info("Using {} GPUs.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    # Decay learning rate by a factor every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer,
                                           step_size=args.lr_step_size,
                                           gamma=args.lr_step_gamma)

    model = train_val_model(model, criterion, optimizer, exp_lr_scheduler, args.epochs)
    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data" , type=str, required=True, 
                        help="Path to the training data (in PyTorch ImageFolder format)")
    parser.add_argument("--batch-size" , type=int, required=False, default=64,
                        help="Batch size (will be split among devices used by this invocation)")
    parser.add_argument("--epochs", type=int, required=False, default=100,
                        help="Epochs")
    parser.add_argument("--lr-step-size" , type=int, required=False, default=30,
                        help="Number of epochs after which the learning rate will decay")
    parser.add_argument("--lr-step-gamma" , type=float, required=False, default=0.1,
                        help="Factor of learn rate decay")    
    parser.add_argument("--logs-folder", type=str, required=False, default='logs',
                        help="Location of logs")
    parser.add_argument("--plots-folder", type=str, required=False, default='plots',
                        help="Location of logs")
    parser.add_argument("--trained-models-folder", type=str, required=False,
                        default="trained_models",
                        help="Location of models")
    
    return parser.parse_args()

args = get_args()
if not os.path.exists(args.logs_folder):
    os.makedirs(args.logs_folder)
if not os.path.exists(args.plots_folder):
    os.makedirs(args.plots_folder)
if not os.path.exists(args.trained_models_folder):
    os.makedirs(args.trained_models_folder)
log_file = get_logfilename_with_datetime('train-log')
logging.basicConfig(filename=join(args.logs_folder, log_file),
                    level=logging.INFO,
                    filemode='w',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s - %(levelname)s - %(message)s')


print(args)
logging.info(args)

dataloaders, dataset_sizes, class_names = get_data_loaders(args.train_data, args.batch_size)
logging.info("Train size {}, Val size {}, Test size {}".format(dataset_sizes['train'],
                                                               dataset_sizes['val'],
                                                               dataset_sizes['test']))
logging.info('Class names:{}'.format(class_names))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    total_gpus = torch.cuda.device_count()
    logging.info('Total number of GPUs:{}'.format(total_gpus))
    if total_gpus == 1:
        multi_gpu = False
    elif total_gpus > 1:
        multi_gpu = True

else:
    print("No GPUs, Cannot proceed. This training regime needs GPUs.")
    exit(1)

nb_classes = len(class_names)
# Get a batch of training data and show it
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
show_batch(out, title=[class_names[x] for x in classes])

model, train_losses, val_losses = configure_run_model()
display_losses(train_losses, val_losses, 'Train-Val Loss')
test_model(model, dataloaders, device)

