#!/bin/sh

clusters=(8 16 32 64)
traj_length=(10 15 20)
temp_stride=(3 6 9)

# temporal stride loop
for nt in "${temp_stride[@]}"
do
    # trajectory length loop
    for tl in "${traj_length[@]}"
    do
        # clusters loop
        for k in "${clusters[@]}"
        do
			echo "cluster=$k, trajectory_length=$tl, tempora_stride=$nt"
			sbatch slurm_idt_cpu.sh\
				-dp=/raid/users/oozdemir/data/BosphorusSign/ToyDataset_features \
				-trp=/raid/users/oozdemir/code/tm-shd-slr/data/splits/toydata/train.txt \
				-tsp=/raid/users/oozdemir/code/tm-shd-slr/data/splits/toydata/test.txt \
				-cip=/raid/users/oozdemir/code/tm-shd-slr/data/splits/toydata/class_indices.txt \
				-ep=/raid/users/oozdemir/code/tm-shd-slr/experiments_2/idt_baseline_toydata/ \
				-kf=0 \
				-tl=$tl \
				-nt=$nt \
				-k=$k
		done
	done
done
