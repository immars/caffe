// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;
DEFINE_string(format, "",
        "REQUIRED. The backend {lmdb, leveldb, etc.} for data source");
DEFINE_string(dest_format, "",
        "OPTIONAL. The backend for output. default=format");
DEFINE_string(src_path, "",
        "REQUIRED. data source to be partitioned");
DEFINE_string(dest_dir, "",
        "REQUIRED. dir to which partition result is stored");
DEFINE_int32(count, 0, "partition count. count or size, not both.");

DEFINE_int32(size, 0, "partition size in record. count or size, not both.");

#define DEFAULT_COUNT 2
#define DEFAULT_SIZE 250000

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("partition datum images db into several shards.\n"
        "Usage:\n"
        "    partition_imageset --src_format={} --src_path={} "
        "--dest_dir={} --count={}|--size={}\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_src_path == "" || FLAGS_format == ""
      || FLAGS_dest_dir == ""
      || (FLAGS_count == 0 && FLAGS_size == 0)) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/partition_imageset");
    return 1;
  }
  if (FLAGS_dest_format == "") {
    // default to src format
    FLAGS_dest_format = FLAGS_format;
  }
  if (FLAGS_src_path[FLAGS_src_path.size()-1] == '/') {
    // remove trailing '/'
    FLAGS_src_path = FLAGS_src_path.substr(0, FLAGS_src_path.size() - 1);
  }

  string base_name = FLAGS_src_path.substr(FLAGS_src_path.rfind("/") + 1);
  LOG(ERROR) << "Original db name:" << base_name;
  // Create new DB
  scoped_ptr<db::DB> src(db::GetDB(FLAGS_format));
  src->Open(FLAGS_src_path, db::READ);

  if (FLAGS_size == 0) {
    LOG(ERROR) << "Determine partition size for count:" << FLAGS_count;
    if (FLAGS_count == 1) {
      LOG(ERROR) << "Only 1 shard, size unlimited";
      FLAGS_size = INT32_MAX;
      if (FLAGS_format == FLAGS_dest_format) {
        LOG(ERROR) << "Output and input will be same! "
            << "Supply different dest_format, or supply count>1. ";
        exit(-1);
      }
    } else {
      // scan DB once to get record count, to calc size
      scoped_ptr<db::Cursor> cursor(src->NewCursor());
      int total = 0;
      while (true) {
        cursor->Next();
        if (!cursor->valid()) {
          break;
        }
        total++;
      }
      FLAGS_size = total / FLAGS_count;
      if (total % FLAGS_count != 0) {
        FLAGS_size++;
      }
    }
  }
  LOG(ERROR) << "Partition size:" << FLAGS_size;
  scoped_ptr<db::Cursor> cursor(src->NewCursor());
  for (int partition_id = 0; cursor->valid(); partition_id++) {
    int count = 0;
    std::ostringstream buf;
    buf << FLAGS_dest_dir << "/" << base_name << "_" << partition_id;
    string partition_path = buf.str();
    LOG(ERROR) << "Start partition: " << partition_path;
    scoped_ptr<db::DB> dest(db::GetDB(FLAGS_dest_format));
    dest->Open(partition_path, db::NEW);
    scoped_ptr<db::Transaction> txn(dest->NewTransaction());
    for (int data_id = 0; data_id < FLAGS_size; data_id ++) {
      cursor->Next();
      if (!cursor->valid()) {
        break;
      }
      count++;
      txn->Put(cursor->key(), cursor->value());
      if (count % 1000 == 0) {
        txn->Commit();
        txn.reset(dest->NewTransaction());
        LOG(ERROR) << "Processed " << count << " files.";
      }
    }
    if (count % 1000 != 0) {
      txn->Commit();
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  return 0;
}
