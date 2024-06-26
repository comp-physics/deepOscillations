#!/usr/bin/env bash

samples=1000
exponent_truth=13
exponent_approx=4
epochs=10000

func_str='Levin1'
save_dir='...'
seed_start=1
seed_end=1

n_array=(2 3 4 5) 
b_array=(2 3 4 )
s_array=(1000)

for neurons in ${n_array[@]}; do
for b_layers in ${b_array[@]}; do
for samples in ${s_array[@]}; do
for exponent_approx in {1..12}; do
    for ((seed=$seed_start;seed<=$seed_end;seed++)); do
        python3 ./main.py $seed $samples $exponent_truth $exponent_approx $epochs $b_layers $neurons $func_str $save_dir &
   done
   wait
done
wait
done
wait
done
wait
done
wait
