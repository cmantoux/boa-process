#!/bin/bash
for M in 0 1 2 3
do
	for N in 0 1 2
	do
		for T in 0 1
		do
			for p in 0 1 2 3
			do
				for H in 0 1 2 3
				do
					for s in 0
					do
						for eps in 0
						do
							sbatch run_E8_slurm.sh $M $N $T $p $H $s $eps
						done
					done
				done
			done
		done	
	done
done
