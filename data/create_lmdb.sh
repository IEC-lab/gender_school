#########################################################################
# File Name: create_lmdb.sh
# Created Time: 2016年07月05日 星期二 16时42分43秒
#########################################################################
#!/bin/bash

echo "Create lmdb dataset "
SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
DATA_DIR=/home/lc/data/gender_200w/align_school_3
CAFFE_DIR=/home/lc/caffe-drf
TOOLS_DIR=$CAFFE_DIR/build/tools

# set RESIZE to resize images to 224x224.
RESIZE=false
if $RESIZE; then
	RESIZE_HEIGHT=200
	RESIZE_WIDTH=200
else
	RESIZE_HEIGHT=0
	RESIZE_WIDTH=0
fi

GLOG_logtostderr=1 $TOOLS_DIR/convert_imageset \
	--shuffle \
	$DATA_DIR/ \
	$DATA_DIR/train.txt \
	$SCRIPT_DIR/train.lmdb

GLOG_logtostderr=1 $TOOLS_DIR/convert_imageset \
	--shuffle \
	$DATA_DIR/ \
	$DATA_DIR/val.txt \
	$SCRIPT_DIR/val.lmdb

echo "Done."
