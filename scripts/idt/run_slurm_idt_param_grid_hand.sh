#!/bin/sh

hand_radius=(45)
clusters=(64)
traj_length=(20)
temp_stride=(3)

# clusters loop
for k in "${clusters[@]}"
do
	# trajectory length loop
    for tl in "${traj_length[@]}"
    do
		# temporal stride loop
		for nt in "${temp_stride[@]}"
		do
		    # hand radius
		    for hr in "${hand_radius[@]}"
		    do
                echo "cluster=$k, trajectory_length=$tl, tempora_stride=$nt"
                sbatch slurm_idt_hand_cpu.sh\
                    -kf=0 \
                    -tl=$tl \
                    -nt=$nt \
                    -k=$k \
                    -hr=$hr \
                    -dp=/raid/users/oozdemir/data/BosphorusSign/General \
                    -trp=/raid/users/oozdemir/code/tm-shd-slr/data/splits/general/train-copy.txt \
                    -tsp=/raid/users/oozdemir/code/tm-shd-slr/data/splits/general/test-copy.txt \
                    -cip=/raid/users/oozdemir/code/tm-shd-slr/data/splits/general/class_indices-copy.txt \
                    -ep=/raid/users/oozdemir/code/tm-shd-slr/experiments_2/idt_hand_general_variance_inconsistent
            done
		done
	done
done
