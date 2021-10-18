#!/bin/bash

File=pseudo_ogbn_mag_red.py
Data=ogbn-mag
Aggre=lstm

# mkdir red_range
# mkdir red_random
# cd red_range
# mkdir mag_lstm_pseudo_log/
# cd ../red_random
# mkdir mag_lstm_pseudo_log/
# cd ..

# reddit dataset 
# batch_size=(157393 78697 39349 19675 9838 4949)
batch_size=( 39349 19675 9838 4949)

for i in ${batch_size[@]};do
  python $File \
  --dataset $Data \
  --aggre $Aggre \
  --selection-method range \
  --batch-size $i \
  --num-epochs 1 \
  --eval-every 5 > red_range/mag_lstm_pseudo_log/bs_${i}_1_epoch.log
# done
# for i in ${batch_size[@]};do
  python $File \
  --dataset $Data \
  --aggre $Aggre \
  --selection-method random \
  --batch-size $i \
  --num-epochs 1 \
  --eval-every 5 > red_random/mag_lstm_pseudo_log/bs_${i}_1_epoch.log
done



# for i in ${batch_size[@]};do
#   # python products_pseudo_pure_.py --dataset ogbn-products --aggre mean --selection-method range --batch-size $i --num-epochs 6 --eval-every 5 > random/products_mean_pseudo_log/bs_${i}_6_epoch.log
#   python products_pseudo_pure_.py --dataset ogbn-products --aggre lstm --selection-method range --batch-size $i --num-epochs 6 --eval-every 5 > range/products_lstm_pseudo_log/bs_${i}_6_epoch.log
#   python products_pseudo_pure_.py --dataset ogbn-products --aggre lstm --selection-method random --batch-size $i --num-epochs 6 --eval-every 5 > random/products_lstm_pseudo_log/bs_${i}_6_epoch.log
# done
