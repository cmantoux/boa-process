#!/bin/bash
for N in 0 1 2
do
	for T in 0 1
	do
		for p in 0 1 2 3 2 3 4 5 6 7 8 9 10
		do
			for H in 0 1 2 3
			do
				for s in 0 1
				do
					for eps in 0
					do
						sbatch run_E42_slurm.sh $N $T $p $H $s $eps
					done
				done
			done
		done
	done	
done
