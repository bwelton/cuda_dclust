#!/bin/ksh
#PBS -N cuda_dbscan
#PBS -j oe
#PBS -l nodes=1
#PBS -l walltime=00:2:00

source /opt/modules/default/init/ksh
module swap boost/1.57.0 boost/1.60.0
module unload gcc/4.9.0
module load gcc/4.9.0
module load cudatoolkit/7.0.28-1.0502.10280.4.1
cd $BINDIR 

aprun -n 1 -N 1 $BINDIR/cuda_dbscan 


