#!/bin/sh

learning_rate=(1e-5)
epochs=(50000)
batch_size=(128)
num_hid_lstm=(200)
drop_lstm=(0.25)


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
                        -dp=/raid/users/oozdemir/data/BosphorusSign/General_fhd \
                        -trp=/raid/users/oozdemir/code/tm-shd-slr/data/splits/general/train.txt \
                        -tsp=/raid/users/oozdemir/code/tm-shd-slr/data/splits/general/test.txt \
                        -cip=/raid/users/oozdemir/code/tm-shd-slr/data/splits/general/class_indices.txt \
                        -descp=/raid/users/oozdemir/data/BosphorusSign/General_fhd \
                        -ep=/raid/users/oozdemir/code/tm-shd-slr/experiments_2/fhd_general_temporal_lstm \
                        -kf=0 \
                        -lr=$lr \
                        -e=$e \
                        -b=$b \
                        -nhlstm=$nhlstm \
                        -dlstm=$dlstm \
                        -crop-size=80 \
                        -nbins=8 \
                        -ncell=40 \
                        -nblock=2
                done
            done
        done
    done
done
