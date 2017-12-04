#include "base/eventing/file/eventlog.h"

#include <memory>
#include <mutex>
#include <utility>

#include "base/util/compat.h"
#include "base/util/logging.h"
#include "base/util/uuid.h"

namespace gpi = google::protobuf::io;

namespace vertexai {
namespace eventing {
namespace file {

EventLog::EventLog(const proto::EventLog& config)
    : config_{config},
      std_file_out_{config.filename(), std::ios::binary},
      ostr_out_{compat::make_unique<gpi::OstreamOutputStream>(&std_file_out_)},
      gzip_out_{compat::make_unique<gpi::GzipOutputStream>(ostr_out_.get(), gpi::GzipOutputStream::Options())},
      coded_out_{compat::make_unique<gpi::CodedOutputStream>(gzip_out_.get())} {
  if (!std_file_out_) {
    throw std::runtime_error(std::string("unable to open \"") + config.filename() + "\" for writing");
  }
  LOG(INFO) << "Writing event log to " << config.filename();
  proto::Record record;
  record.mutable_magic()->set_value(proto::Magic::Eventlog);
  LogRecordLocked(std::move(record));
}

EventLog::~EventLog() { FlushAndClose(); }

void EventLog::LogEvent(context::proto::Event event) {
  std::lock_guard<std::mutex> lock{mu_};
  if (closed_) {
    return;
  }
  if (!wrote_uuid_) {
    event.mutable_activity_id()->set_stream_uuid(ToByteString(stream_uuid()));
    wrote_uuid_ = true;
  }
  proto::Record record;
  *record.add_event() = std::move(event);
  LogRecordLocked(std::move(record));
}

void EventLog::FlushAndClose() {
  std::lock_guard<std::mutex> lock{mu_};
  if (closed_) {
    return;
  }
  closed_ = true;
  coded_out_.reset();
  gzip_out_.reset();
  ostr_out_.reset();
  std_file_out_.close();
}

void EventLog::LogRecordLocked(proto::Record record) {
  coded_out_->WriteVarint32(record.ByteSize());
  record.SerializeToCodedStream(coded_out_.get());
}

Reader::Reader(const std::string& filename)
    : std_file_in_{filename, std::ios::binary},
      ostr_in_{compat::make_unique<gpi::IstreamInputStream>(&std_file_in_)},
      gzip_in_{compat::make_unique<gpi::GzipInputStream>(ostr_in_.get())},
      coded_in_{compat::make_unique<gpi::CodedInputStream>(gzip_in_.get())} {}

bool Reader::Read(context::proto::Event* event) {
  std::lock_guard<std::mutex> lock{mu_};

  for (;;) {
    if (idx < record.event_size()) {
      event->Clear();
      event->Swap(record.mutable_event(idx));
      idx++;
      return true;
    }

    // Read another block of events.
    idx = 0;
    record.Clear();
    auto limit = coded_in_->ReadLengthAndPushLimit();
    if (!record.ParseFromCodedStream(coded_in_.get())) {
      coded_in_->PopLimit(limit);
      return false;
    }
    auto complete = coded_in_->CheckEntireMessageConsumedAndPopLimit(limit);
    if (!complete) {
      return false;
    }

    if (record.has_magic() && record.magic().value() != proto::Magic::Eventlog) {
      return false;
    }

    if (!record.has_magic() && !record.event_size()) {
      return false;
    }
  }
}

}  // namespace file
}  // namespace eventing
}  // namespace vertexai
