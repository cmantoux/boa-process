#!/bin/bash
for N in 0 1 2
do
	for T in 0 1 2
	do
		for p in 0 1 2 3
		do
			for H in 0 1 2 3 4
			do
				for s in 0 1
				do
					for eps in 0
					do
						sbatch run_E3_slurm.sh $N $T $p $H $s $eps
					done
				done
			done
		done
	done	
done
