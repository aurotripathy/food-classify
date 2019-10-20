# food-classify

This repo is an attempt to reproduce the results of the paper [Wide-Slice Residual Networks for Food Recognition](https://arxiv.org/pdf/1612.06543.pdf) using the PyTorch framework.

#### Download the Dataset
As in the paper, we use the [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) Dataset. You can trigger the download directly from [here](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz).  Untar it once download is complete.

#### Build the train/validate/test partitions from the dataset
<code>
  python build_dataset.py --root-folder path/to/untarred/location
</code>

#### Run a training regime

Use the script [run_train.sh](https://github.com/aurotripathy/food-classify/blob/master/run_train.sh) to train the model. 

#### Run a test regime
Use the script [run_test.sh](https://github.com/aurotripathy/food-classify/blob/master/run_test.sh) to test the model. 

#### Inital results

https://github.com/aurotripathy/food-classify/blob/master/plots/Train-Val%20Loss.png
