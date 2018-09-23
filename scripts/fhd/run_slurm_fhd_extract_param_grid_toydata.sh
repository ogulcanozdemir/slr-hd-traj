#!/bin/sh

crop_size=(240)
cell_size=(60)
block_size=(4)
bins=(8)
clusters=(8 16 32 64)

for k in "${clusters[@]}"
do
    for cp in "${crop_size[@]}"
    do
        for cs in "${cell_size[@]}"
        do
            for bs in "${block_size[@]}"
            do
                for bn in "${bins[@]}"
                do
                echo "cluster=$k, crop_size=$cp, cell_size=$cs, block_size=$bs, n_bins=$bn"
                sbatch slurm_fhd_cpu_extract.sh\
                    -dp=/raid/users/oozdemir/data/BosphorusSign/ToyDataset_fhd_240 \
                    -trp=/raid/users/oozdemir/code/tm-shd-slr/data/splits/toydata/train.txt \
                    -tsp=/raid/users/oozdemir/code/tm-shd-slr/data/splits/toydata/test.txt \
                    -cip=/raid/users/oozdemir/code/tm-shd-slr/data/splits/toydata/class_indices.txt \
                    -descp=/raid/users/oozdemir/data/BosphorusSign/ToyDataset_fhd_240 \
                    -ep=/raid/users/oozdemir/code/tm-shd-slr/experiments_2/fhd_240 \
                    -crop-size=$cp \
                    -nbins=$bn \
                    -ncell=$cs \
                    -nblock=$bs \
                    -k=$k
                done
            done
        done
    done
done