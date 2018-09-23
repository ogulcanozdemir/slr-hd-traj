#!/bin/sh

data_path='/raid/users/oozdemir/data/BosphorusSign/ToyDataset'

for i in $(find $data_path -name 'color.mp4' ); do 
	filename="$i"
	dest="${i::-4}_scaled_g.mp4"
	echo "$i"
	/raid/users/oozdemir/tools/ffmpeg/ffmpeg -loglevel panic -i $i -vf scale=-1:360 -c:v libx264 -crf 0 -preset veryslow -c:a copy $dest
done
