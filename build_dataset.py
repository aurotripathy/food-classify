"""
Setup the train/validate/test data to be directly used by the class,
torchvision.datasets.ImageFolder
"""
import cv2
import os
import glob
import random
from shutil import copyfile, rmtree
import sys
from sys import exit
from os.path import join, basename, dirname, exists
from pudb import set_trace
import argparse

def copy_files(source_filenames, dest_path):
    for source_file in source_filenames:
        dest_dirname = basename(dirname(source_file))
        actual_file = basename(source_file)
        dest_dir = join(dest_path, dest_dirname)
        if not exists(dest_dir):
            os.makedirs(dest_dir)
        dest_file = join(dest_dir, actual_file)

        
        try:
            copyfile(source_file, dest_file)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(2)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(2)


def create_train_val_test(folder):
    image_files = []
    for image_file in glob.glob(folder + '/*/*.jpg'):
        image_files.append(image_file)
    random.shuffle(image_files)
    print('Total images files: {}'.format(len(image_files)))
    
    split_1 = int(0.7 * len(image_files))
    split_2 = int(0.85 * len(image_files))
    train_filenames = image_files[:split_1]
    print('Total train files', len(train_filenames))
    val_filenames = image_files[split_1:split_2]
    print('Total val files', len(val_filenames))
    test_filenames = image_files[split_2:]
    print('Total test files', len(test_filenames))
    
    train_path = join(train_val_test_folder, 'train')
    val_path = join(train_val_test_folder, 'val')
    test_path = join(train_val_test_folder, 'test')

    for filenames, path in zip([train_filenames, val_filenames, test_filenames],
                               [train_path, val_path, test_path]):
        if not exists(path):
            os.makedirs(path)
            copy_files(filenames, path)

set_trace()
parser = argparse.ArgumentParser()
parser.add_argument("--root-folder" , type=str, required=True, 
                    help="Path to the downloaded images")
args = parser.parse_args()

# root folder '../../data/food-101/food-101/'
root_folder = args.root_folder
images_folder = join(root_folder, 'images')
train_val_test_folder = join(root_folder, 'train_val_test')
# Create a new output each time
if os.path.exists(train_val_test_folder):
    rmtree(train_val_test_folder)
os.makedirs(train_val_test_folder)
        
create_train_val_test(images_folder)
