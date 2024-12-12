#!/bin/bash
NCLIENTS=$1
NROUNDS=$2
FILEPATH="/results"
DATASET=$3
MODEL=$4
ISENC=$5
SERVERIP=$7
PART=$6
filename="/results/xp.txt"
filepre="xp_101_"
fileext=".txt"

#if docker on the same machine
# script to find and replace server IP and outputs number of the current client
if [ $# -eq 5 ]; then
   PART=$(python ./FL/docker_compose_cl.py $ISENC)
else
  python ./FL/docker_cl.py $SERVERIP $ISENC
fi

echo "Starting client ${PART}"

current_date_time="`date +%Y%m%d-%H%M%S` "
f="${FILEPATH}/${filepre}${NROUNDS}_cl${PART}${fileext}"
#echo -n $current_date_time > $f
#sleep 60
if [[ "$ISENC" == "true" ]]; then
    python ./FL/fl_client_enc.py --nclients=${NCLIENTS} --partition=${PART} --filepath=${FILEPATH} --dataset=${DATASET} --model=${MODEL} #| awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $f
else
    python ./FL/fl_client.py --nclients=${NCLIENTS} --partition=${PART} --filepath=${FILEPATH} --dataset=${DATASET} --model=${MODEL} #| awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $f
fi
pid=($!)
wait $pid
echo -e "" >> $f

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
echo -e "" >> $filename
# Wait for all background processes to complete
wait
