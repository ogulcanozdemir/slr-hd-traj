#!/bin/sh

crop_size=(80)
cell_size=(40)
block_size=(2)
bins=(8)
clusters=(64)
temporal_stride=(3 5 7 9)

for nt in "${temporal_stride[@]}"
do
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
                    echo "temporal_stride=$nt, cluster=$k, crop_size=$cp, cell_size=$cs, block_size=$bs, n_bins=$bn"
                    sbatch slurm_fhd_cpu_extract.sh\
                        -dp=/raid/users/oozdemir/data/BosphorusSign/General_fhd \
                        -trp=/raid/users/oozdemir/code/tm-shd-slr/data/splits/general/train-copy.txt \
                        -tsp=/raid/users/oozdemir/code/tm-shd-slr/data/splits/general/test-copy.txt \
                        -cip=/raid/users/oozdemir/code/tm-shd-slr/data/splits/general/class_indices-copy.txt \
                        -descp=/raid/users/oozdemir/data/BosphorusSign/General_fhd \
                        -ep=/raid/users/oozdemir/code/tm-shd-slr/experiments_2/fhd_general_inconsistent \
                        -crop-size=$cp \
                        -nbins=$bn \
                        -ncell=$cs \
                        -nblock=$bs \
                        -nt=$nt \
                        -k=$k
                    done
                done
            done
        done
    done
done