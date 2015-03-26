#!/usr/bin/env sh
# partition a database into several shards
# N.B. set the path to the imagenet train + val data dirs

TOOLS=build/tools

if [ $# -lt 4 ]; then
    echo "usage: convert_imageset.sh [leveldb|lmdb|etc.] src_db_path dest_dir count"
    exit -1;
fi

# leveldb|lmdb
FORMAT=$1

# source database file/dir, NO '/' on tail
SRC_DB=$2 # /data/ilsvrc12_train_leveldb
DEST_PATH=$3 # /data/

# partition count
COUNT=$4 # 4

if [ ! -d "$SRC_DB" ]; then
  echo "Error: SRC_DB is not a path to a directory: $SRC_DB"
  echo "Set the SRC_DB variable in partition_imageset.sh to the path" \
       "where the source database is stored."
  exit 1
fi

if [ ! -d "$DEST_PATH" ]; then
  echo "Error: DEST_PATH is not a path to a directory: $DEST_PATH"
  echo "Set the DEST_PATH variable in partition_imageset.sh to the path" \
       "where the partitioned shards are stored."
  exit 1
fi

echo "Partition started:"

GLOG_logtostderr=1 $TOOLS/partition_imageset \
    --src_path=$SRC_DB \
    --dest_dir=$DEST_PATH \
    --format=$FORMAT \
    --count=$COUNT

echo "Done."
