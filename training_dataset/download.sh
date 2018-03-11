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

echo "--------------------------------------Download--------------------------------------"
echo
echo "                       From : $from"
echo "                         To : $to"
echo
echo "------------------------------------------------------------------------------------"

i=0
awk -F ',' '{ print($1, $4) }' $FILE | while read name url 
do
	let i++;
	if(($i > $to)); then
		echo "Download Done"
		exit 0
	fi
	if(($i >= $from)); then
		wget -nc -O ".img/$name.tif" $url
	fi
done
