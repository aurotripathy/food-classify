# for wide resnet101, due to the model being large the BS is much smaller; 
# for smaller models BS can be 1024 for 32GB GPU machines
python3 train_validate.py --batch-size 128 \
       --epochs 100 \
       --train-data ./../../data/food-101/food-101/train_val_test/

