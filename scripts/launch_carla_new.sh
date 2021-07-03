#!/bin/bash
((j = 0))
for (( i=0; i<$1; i++ ))
do
    port=$((2*j+i+$2))
    fuser $port/tcp -k
    fuser $((port + 1))/tcp -k
    fuser $((port + 2))/tcp -k
    /mnt/ssd/varun/CARLA_0.9.10.1/CarlaUE4.sh -world-port=$port -opengl -world-port=$port &
    ((j+=1))
    #echo $j
    #echo $port
done
wait
