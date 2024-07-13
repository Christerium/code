#!/bin/bash -e

# Skript für die Berechnung der pmed Instanzen
# Variable zum einlesen des filenamen

inputdir="./"
#outputdir="./pmed_output/2_5_y_liftmax"
outputdir="./"
outputfilename="output.csv"

for file in "$inputdir"/*.txt
do
	filename=$(basename "$file")
	echo "Bearbeite File:  $filename"
	if [ -f "$outputdir"/"$outputfilename" ]
	then
		grep "STAT" "$file" >> "$outputdir"/output.csv || { echo "No STAT line found"; }
		echo "Line has been appended"
	else
		grep "STAH" "$file" > "$outputdir"/output.csv || { echo "No STAH line found"; }
		grep "STAT" "$file" >> "$outputdir"/output.csv || { echo "No STAT line found"; }
		echo "New File"
	fi	
	#if [ "$filename" != "pmed13.txt" ]
	#	then
	#		echo "Bearbeite File:  $filename"
	#		./nested_pcenter -f $file --startH 2 --endH 5 --instanceformat 1 --xyModel 0 --lifting 1 &> ./"$outputdir"/"${filename%.*}".log
	#fi
done


