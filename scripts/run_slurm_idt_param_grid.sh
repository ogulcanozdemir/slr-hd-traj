#!/bin/sh

clusters=(16)
traj_length=(10 15 20)
temp_stride=(3 6 9)

# clusters loop
for k in "${clusters[@]}"
do
	# trajectory length loop
    for tl in "${traj_length[@]}"
    do
		# temporal stride loop
		for nt in "${temp_stride[@]}"
		do
			echo "cluster=$k, trajectory_length=$tl, tempora_stride=$nt"
			sbatch slurm_idt_cpu.sh\
				-dp=/raid/users/oozdemir/data/ToyDataset \
				-trp=/raid/users/oozdemir/code/tm-shd-slr/data/splits/train.txt \
				-tsp=/raid/users/oozdemir/code/tm-shd-slr/data/splits/test.txt \
				-ep=/raid/users/oozdemir/code/tm-shd-slr/experiments \
				-kf=0 \
				-tl=$tl \
				-nt=$nt \
				-k=$k
		done
	done
done
