#!/bin/bash

size=4096x2160
framerate=30
start=1
frames=1000

if (($# > 0)); then
	size=$1
fi
if (($# > 1)); then
	framerate=$2
fi
if (($# > 2)); then
	start=$3
fi
if (($# > 3)); then
	frames=$4
fi

echo "--------------------------------------Concat--------------------------------------"
echo
echo "                       Size : $size"
echo "                  Framerate : $framerate"
echo "                      Start : $start"
echo "                     Frames : $frames"
echo
echo "----------------------------------------------------------------------------------"

ffmpeg -framerate $framerate\
	-start_number $start\
	-i '.img/img%05d.png'\
	-pix_fmt yuv420p\
	-frames $frames\
	-s $size\
	-r $framerate\
	TrainingDataset_"$size"_"$framerate".yuv

echo "Concat Done"
