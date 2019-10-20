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

### Inital results

##### Validation Accuray (0.8465)

[train log](https://github.com/aurotripathy/food-classify/blob/master/logs/train-log-2019-10-19%2023:43.log)

##### Ten-crop Test Accuracy (0.8471)

[test log](https://github.com/aurotripathy/food-classify/blob/master/logs/test-log-2019-10-20%2011:41.log)

##### Loss Plot
![Loss Plot](https://github.com/aurotripathy/food-classify/blob/master/plots/Train-Val%20Loss.png)
