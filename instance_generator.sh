#!/bin/bash -e

departments=(8 12 16 20)
dist=(30 50 70)
range=(5 20)

for (( i=0; i<${#departments[@]}; i++ ))
do
	for (( j=0; j<${#dist[@]}; j++ ))
	do
		for (( k=0; k<${#range[@]}; k++ ))
		do
			echo "python3 python/instancegenerator.py -n ${departments[$i]} -d ${dist[$j]} -r ${range[$k]}"
			python3 python/instancegenerator.py -n ${departments[$i]} -d ${dist[$j]} -r ${range[$k]}
			python3 python/instancegenerator.py -n ${departments[$i]} -d ${dist[$j]} -r ${range[$k]}
		done
	done
done


