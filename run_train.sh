# for wide resent101, large model BS is much smaller, otherwise 1024
python3 train_validate.py --batch-size 128 \
       --epochs 100 \
       --train-data ./../../data/food-101/food-101/train_val_test/

