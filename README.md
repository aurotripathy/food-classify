# food-classify


sudo docker run -it --network=host --shm-size=20G --device=/dev/kfd --device=/dev/dri --group-add video -v $HOME/food-101:/food-101 computecqe/pytorch:rocm27-RC4-22-ub1604-py3.6_food

# First, build the dataset
python build_dataset.py --root-folder /media/auro/RAID\ 5/food-101/food-101
