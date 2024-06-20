#!/bin/bash -e

## Script for generating the instances
## With density: 			30, 50, 70
## Range: 					1...5, 1...20
## Numer of departments:  	4, 8, 12, 16, 20

density=(30 50 70)
range=(5 20)
departments=(4 8 12 16 20)

for i in ${departments[@]}
do 
	for j in ${density[@]}
	do
		for k in ${range[@]}
		do
			echo "python3 python/instancegenerator.py -n $i -d $j -r $k" 
			python3 python/instancegenerator.py -n $i -d $j -r $k
		done
	done
done
