#!/bin/sh

learning_rate=(1e-5)
epochs=(1000)
batch_size=(32 64 128)
num_hid_lstm=(100)
drop_lstm=(0)


for e in "${epochs[@]}"
do
    for lr in "${learning_rate[@]}"
    do
        for b in "${batch_size[@]}"
        do
            for nhlstm in "${num_hid_lstm[@]}"
            do
                for dlstm in "${drop_lstm[@]}"
                do
                    echo "epochs=$e, learning_rate=$lr, batch_size=$b, num_hid_lstm=$nhlstm, drop_lstm=$dlstm"
                    sbatch slurm_fhd_gpu.sh\
                        -dp=/raid/users/oozdemir/data/BosphorusSign/ToyDataset_fhd_scaled \
                        -trp=/raid/users/oozdemir/code/tm-shd-slr/data/splits/toydata/train.txt \
                        -tsp=/raid/users/oozdemir/code/tm-shd-slr/data/splits/toydata/test.txt \
                        -cip=/raid/users/oozdemir/code/tm-shd-slr/data/splits/toydata/class_indices.txt \
                        -descp=/raid/users/oozdemir/data/BosphorusSign/ToyDataset_fhd_scaled \
                        -ep=/raid/users/oozdemir/code/tm-shd-slr/experiments_2/fhd_toydata_temporal_lstm \
                        -kf=0 \
                        -lr=$lr \
                        -e=$e \
                        -b=$b \
                        -nhlstm=$nhlstm \
                        -dlstm=0 \
                        -crop-size=80 \
                        -nbins=8 \
                        -ncell=40 \
                        -nblock=2
                done
            done
        done
    done
done