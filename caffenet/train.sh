#!/usr/bin/env bash
set ff =unix
GPU_ID=1
CAFFE_DIR=/home/lc/caffe-1.0
SCRIPT_DIR="$(dirname "$0")"
LOG_DIR=${SCRIPT_DIR}/log
SNAPSHOT_DIR=${SCRIPT_DIR}/snapshots
#WEIGHTS=${SCRIPT_DIR}/snapshots/gender_iter_8000.caffemodel
LOG=${LOG_DIR}/train-"$(date +%Y-%m-%d-%H-%M-%S)".log


for dir in $LOG_DIR $SNAPSHOT_DIR
do
    if [ ! -d $dir ]
    then
       mkdir "$dir"
    fi
done

GLOG_logtostderr=1 \
$CAFFE_DIR/build/tools/caffe train \
    -solver ${SCRIPT_DIR}/solver.prototxt \
	-gpu 2 \
    2>&1 | tee $LOG

