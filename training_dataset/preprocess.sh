#!/bin/bash

FILE=RAISE_1k_id.csv

from=1
to=1000

if (($# > 0)); then
	from=$1
fi
if (($# > 1)); then
	to=$2
fi

echo "--------------------------------------Preprocess--------------------------------------"
echo
echo "                       From : $from"
echo "                         To : $to"
echo
echo "--------------------------------------------------------------------------------------"

i=0
awk -F ',' '{ print($1) }' $FILE | while read name
do
	let i++;
	if (($i > $to)); then
		echo "Preprocess Done"
		exit 0
	fi
	if (($i >= $from)); then
		name=${name%.*}
		echo $name
		width=$(identify -format \"%w\" $name.tif | sed 's/\"//g')
		height=$(identify -format \"%h\" $name.tif | sed 's/\"//g')
		if (($width < $height)); then
			convert -delete 1--1 ".img/$name.tif" -rotate 270 -gravity Center -crop 4096x2160+0+0 ".img/$name.png"
		else
			convert -delete 1--1 ".img/$name.tif" -gravity Center -crop 4096x2160+0+0 ".img/$name.png"
		fi
	fi
done
