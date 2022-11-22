samples=1000
exponent_truth=13
exponent_approx=4
epochs=10000
b_layers=3
neurons=125
func_str='Levin1'
save_dir='/scratch/epickeri/Oscillations/results/'
seed_start=1
seed_end=1

func_strs="Levin2 Levin1"
func_strs="EvansWebster1 EvansWebster2 EvansWebster3 EvansWebster6"

n_array=(125)
b_array=(3)
s_array=(1000 5000 10000 20000)

for func_str in $func_strs;
do
for neurons in ${n_array[@]}
do
for b_layers in ${b_array[@]}
do
for samples in ${s_array[@]}
do
for exponent_approx in {4..11}
do
   for ((seed=$seed_start;seed<=$seed_end;seed++))
   do
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
done
wait
