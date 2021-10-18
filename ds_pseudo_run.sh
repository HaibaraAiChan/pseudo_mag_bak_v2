#!/bin/bash

####################### ogbn-mag dataset
File=pseudo_ogbn_mag.py
Data=ogbn-mag
Aggre=lstm
batch_size=(157393 78697 39349 19675 9838 4949)


####################### reddit dataset 
# File=pseudo_ogbn_mag.py
# Data=reddit
# Aggre=mean
# batch_size=(153431 76716 38358 19179 9590 4795 2400 1200)

lr=003
runs=2
epochs=6

mkdir range_${lr}
mkdir random_${lr}
cd range_${lr} 
mkdir ${Data}_${Aggre}_pseudo_log/
cd ../random_${lr}
mkdir ${Data}_${Aggre}_pseudo_log/
cd ..




for i in ${batch_size[@]};do
  python $File \
  --dataset $Data \
  --aggre $Aggre \
  --selection-method range \
  --batch-size $i \
  --num-runs $runs \
  --num-epochs $epochs \
  --eval-every 5 > range_${lr}/${Data}_${Aggre}_pseudo_log/bs_${i}_6_epoch.log
# done
# for i in ${batch_size[@]};do
  python $File \
  --dataset $Data \
  --aggre $Aggre \
  --selection-method random \
  --batch-size $i \
  --num-runs $runs \
  --num-epochs $epochs \
  --eval-every 5 > random_${lr}/${Data}_${Aggre}_pseudo_log/bs_${i}_6_epoch.log
done



# for i in ${batch_size[@]};do
#   # python products_pseudo_pure_.py --dataset ogbn-products --aggre mean --selection-method range --batch-size $i --num-epochs 6 --eval-every 5 > random/products_mean_pseudo_log/bs_${i}_6_epoch.log
#   python products_pseudo_pure_.py --dataset ogbn-products --aggre lstm --selection-method range --batch-size $i --num-epochs 6 --eval-every 5 > range/products_lstm_pseudo_log/bs_${i}_6_epoch.log
#   python products_pseudo_pure_.py --dataset ogbn-products --aggre lstm --selection-method random --batch-size $i --num-epochs 6 --eval-every 5 > random/products_lstm_pseudo_log/bs_${i}_6_epoch.log
# done
