#pragma once

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "base/context/eventlog.h"
#include "base/eventing/file/eventlog.pb.h"

namespace vertexai {
namespace eventing {
namespace file {

class EventLog final : public context::EventLog {
 public:
  explicit EventLog(const proto::EventLog& config);
  ~EventLog();

  void LogEvent(context::proto::Event event) override;

  void FlushAndClose() override;

 private:
  void LogRecordLocked(proto::Record record);

  // The client configuration.
  proto::EventLog config_;

  std::mutex mu_;

  // The output stream chain.  Note that for portability, we use a OstreamOutputStream; if this becomes an issue,
  // FileOutputStream is slightly faster.
  std::ofstream std_file_out_;
  std::unique_ptr<google::protobuf::io::OstreamOutputStream> ostr_out_;
  std::unique_ptr<google::protobuf::io::GzipOutputStream> gzip_out_;
  std::unique_ptr<google::protobuf::io::CodedOutputStream> coded_out_;

  // Whether the log's been closed.
  bool closed_ = false;
};

class Reader final {
 public:
  explicit Reader(const std::string& filename);

  bool Read(context::proto::Event* event);

 private:
  std::mutex mu_;
  std::ifstream std_file_in_;
  std::unique_ptr<google::protobuf::io::IstreamInputStream> ostr_in_;
  std::unique_ptr<google::protobuf::io::GzipInputStream> gzip_in_;
  std::unique_ptr<google::protobuf::io::CodedInputStream> coded_in_;
  int idx = 0;
  proto::Record record;
};

}  // namespace file
}  // namespace eventing
}  // namespace vertexai
