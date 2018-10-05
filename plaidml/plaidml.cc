// Copyright 2018 Intel Corporation.

#include "plaidml/plaidml.h"

#include <google/protobuf/descriptor.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>
#include <zip.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>

#include <boost/filesystem.hpp>

#include "base/config/config.h"
#include "base/util/any_factory_map.h"
#include "base/util/compat.h"
#include "base/util/env.h"
#include "base/util/error.h"
#include "base/util/logging.h"
#include "base/util/runfiles_db.h"
#include "base/util/sync.h"
#include "base/util/type_url.h"
#include "base/util/zipfile.h"
#include "plaidml/base/base_cpp.h"
#include "plaidml/base/context.h"
#include "plaidml/base/status.h"
#include "plaidml/base/status_strings.h"
#include "plaidml/plaidml.pb.h"
#include "tile/base/buffer.h"
#include "tile/base/lru_cache.h"
#include "tile/base/program_cache.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/parser.h"
#include "tile/lang/symbolic.h"
#include "tile/proto/metadata.pb.h"
#include "tile/proto/support.h"
#include "tile/proto/tile.pb.h"
#include "tile/stripe/stripe.h"

namespace {
constexpr std::size_t kApplierForShapeCacheSize = 8;
constexpr std::size_t kRuninfoCacheSize = 8;
const char* PLAIDML_EXPERIMENTAL = "PLAIDML_EXPERIMENTAL";
const char* PLAIDML_DEFAULT_CONFIG = "PLAIDML_DEFAULT_CONFIG";
const char* PLAIDML_EXPERIMENTAL_CONFIG = "PLAIDML_EXPERIMENTAL_CONFIG";
const char* PLAIDML_DEVICE_IDS = "PLAIDML_DEVICE_IDS";
}  // namespace

namespace context = vertexai::context;
namespace plaidml = vertexai::plaidml;
namespace status_strings = vertexai::status_strings;
namespace tile = vertexai::tile;
namespace gp = google::protobuf;
namespace gpi = google::protobuf::io;
namespace gpu = google::protobuf::util;

using tile::lang::BoundFunction;
using tile::lang::FConstValue;
using tile::lang::FunctionApplication;
using tile::lang::Gradient;
using tile::lang::IConstValue;
using tile::lang::PlaceholderValue;
using tile::lang::RunInfo;
using tile::lang::TensorValue;
using tile::lang::Value;

struct plaidml_devconf {
  std::shared_ptr<tile::Platform> platform;
  tile::proto::Device device;
};

namespace {

void FillPropString(const std::string& str, void* output_buffer, size_t output_buffer_size,
                    size_t* output_buffer_size_required) noexcept {
  auto buf = static_cast<char*>(output_buffer);
  if (output_buffer_size_required) {
    *output_buffer_size_required = str.length() + 1;
  }
  if (buf && output_buffer_size) {
    auto copied = str.copy(buf, output_buffer_size - 1);
    std::memset(buf + copied, '\0', output_buffer_size - copied);
  }
}

}  // namespace

extern const char* PLAIDML_VERSION;
extern "C" const char* plaidml_get_version() { return PLAIDML_VERSION; }

extern "C" bool plaidml_query_devconf(vai_ctx* ctx, plaidml_devconf* devconf, plaidml_device_property property,
                                      void* output_buffer, size_t output_buffer_size,
                                      size_t* output_buffer_size_required) {
  if (!ctx) {
    vertexai::SetLastStatus(VAI_STATUS_CANCELLED, status_strings::kCancelled);
    return false;
  }

  try {
    context::Activity activity{ctx->activity.ctx(), "vertexai::QueryDevConf"};
    if (!devconf) {
      FillPropString("", output_buffer, output_buffer_size, output_buffer_size_required);
      vertexai::SetLastOOM();
      return false;
    }
    switch (property) {
      case PLAIDML_DEVICE_ID:
        FillPropString(devconf->device.dev_id(), output_buffer, output_buffer_size, output_buffer_size_required);
        return true;
      case PLAIDML_DEVICE_DESCRIPTION:
        FillPropString(devconf->device.description(), output_buffer, output_buffer_size, output_buffer_size_required);
        return true;
      case PLAIDML_DEVICE_DETAILS:
        FillPropString(devconf->device.details(), output_buffer, output_buffer_size, output_buffer_size_required);
        return true;
      case PLAIDML_DEVICE_CONFIG:
        FillPropString(devconf->device.config(), output_buffer, output_buffer_size, output_buffer_size_required);
        return true;
      default:
        FillPropString("", output_buffer, output_buffer_size, output_buffer_size_required);
        vertexai::SetLastStatus(VAI_STATUS_NOT_FOUND, "The requested property is not available");
        return false;
    }
  } catch (...) {
    vertexai::SetLastOOM();
    return false;
  }
}

// plaidml_device

class Evaluator final {
 public:
  explicit Evaluator(plaidml_devconf* devconf)
      : platform_{devconf->platform},
        id_{devconf->device.dev_id()},
        program_cache_{std::make_shared<tile::ProgramCache>(platform_, 20 /* TODO: Make this configurable */)} {}

  const std::shared_ptr<tile::Platform>& get_platform() const { return platform_; }
  const std::string& get_id() const { return id_; }
  const std::shared_ptr<tile::ProgramCache>& get_program_cache() const { return program_cache_; }

  std::shared_ptr<tile::Program> MakeProgram(const context::Context& ctx, const tile::proto::Program& prog) {
    std::shared_ptr<tile::Program> compiled;
    std::tie(std::ignore, compiled) = program_cache_->GetProgram(ctx, "sdk", prog);
    return compiled;
  }

 private:
  std::shared_ptr<tile::Platform> platform_;
  std::string id_;
  std::shared_ptr<tile::ProgramCache> program_cache_;
};

struct plaidml_device {
  std::shared_ptr<Evaluator> evaluator;
};

extern "C" plaidml_device* plaidml_open_device(vai_ctx* ctx, plaidml_devconf* devconf) {
  if (!ctx) {
    vertexai::SetLastStatus(VAI_STATUS_CANCELLED, status_strings::kCancelled);
    return nullptr;
  }

  try {
    context::Activity activity{ctx->activity.ctx(), "vertexai::OpenDevices"};
    std::unique_ptr<plaidml_device_enumerator> enumerator;
    if (!devconf) {
      enumerator.reset(plaidml_alloc_device_enumerator(ctx, nullptr, nullptr));
      if (!enumerator) {
        return nullptr;
      }
      devconf = plaidml_get_devconf(ctx, enumerator.get(), 0);
      if (!devconf) {
        vertexai::SetLastStatus(VAI_STATUS_NOT_FOUND, vertexai::status_strings::kNoDevices);
        return nullptr;
      }
    }
    LOG(INFO) << "Opening device \"" << devconf->device.dev_id() << "\"";
    return new plaidml_device{std::make_shared<Evaluator>(devconf)};
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return nullptr;
  }
}

extern "C" void plaidml_close_device(plaidml_device* device) { delete device; }

// plaidml_device_enumerator

struct plaidml_device_enumerator {
  std::string config_source;
  std::shared_ptr<tile::Platform> platform;
  std::vector<plaidml_devconf> devices;
  std::vector<plaidml_devconf> unmatched_devices;
};

plaidml_device_enumerator* _plaidml_alloc_device_enumerator(
    vai_ctx* ctx, const char* configuration, const std::string& config_source,
    void (*callback)(void* arg, plaidml_device_enumerator* device_enumerator), void* arg) {
  if (!callback) {
    vertexai::Sync<plaidml_device_enumerator*> sync;
    _plaidml_alloc_device_enumerator(ctx, configuration, config_source, sync.callback(), sync.arg());
    return sync.WaitForResult();
  }

  if (!ctx) {
    vertexai::SetLastStatus(VAI_STATUS_CANCELLED, status_strings::kCancelled);
    callback(arg, nullptr);
    return nullptr;
  }

  plaidml_device_enumerator* result = nullptr;

  std::set<std::string> device_ids;
  std::stringstream devids(vertexai::env::Get(PLAIDML_DEVICE_IDS));
  std::copy(std::istream_iterator<std::string>(devids), std::istream_iterator<std::string>(),
            std::inserter(device_ids, device_ids.end()));

  try {
    context::Activity activity{ctx->activity.ctx(), "vertexai::EnumerateDevices"};
    auto enumerator = vertexai::compat::make_unique<plaidml_device_enumerator>();
    enumerator->config_source = config_source;
    plaidml::proto::Config config;
    try {
      config = vertexai::ParseConfig<plaidml::proto::Config>(configuration);
    } catch (...) {
      vertexai::SetLastException(std::current_exception());
      callback(arg, nullptr);
      return nullptr;
    }
    enumerator->platform =
        vertexai::AnyFactoryMap<tile::Platform>::Instance()->MakeInstance(activity.ctx(), config.platform());
    tile::proto::ListDevicesRequest req;
    tile::proto::ListDevicesResponse resp;
    enumerator->platform->ListDevices(activity.ctx(), req, &resp);
    for (const auto& dev : resp.devices()) {
      plaidml_devconf devconf = {enumerator->platform, dev};
      if (!device_ids.empty() && device_ids.find(devconf.device.dev_id()) == device_ids.end()) {
        enumerator->unmatched_devices.emplace_back(devconf);
        continue;
      }
      enumerator->devices.emplace_back(devconf);
    }
    for (const auto& dev : resp.unmatched_devices()) {
      plaidml_devconf devconf = {enumerator->platform, dev};
      enumerator->unmatched_devices.emplace_back(devconf);
    }
    result = enumerator.release();
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
  }

  // N.B. We're careful to invoke the callback exactly once.
  callback(arg, result);

  return nullptr;
}

extern "C" plaidml_device_enumerator* plaidml_alloc_device_enumerator(
    vai_ctx* ctx, void (*callback)(void* arg, plaidml_device_enumerator* device_enumerator), void* arg) {
  static vertexai::RunfilesDB runfiles_db{"com_intel_plaidml"};

  std::string config_file;
  std::string exp = vertexai::env::Get(PLAIDML_EXPERIMENTAL);
  if (!exp.empty() && exp != "0") {
    config_file = vertexai::env::Get(PLAIDML_EXPERIMENTAL_CONFIG);
  } else {
    config_file = vertexai::env::Get(PLAIDML_DEFAULT_CONFIG);
  }
  std::string translated = runfiles_db[config_file.c_str()];
  std::ifstream cfs(runfiles_db[config_file.c_str()]);
  std::string config;
  config.assign(std::istreambuf_iterator<char>(cfs), std::istreambuf_iterator<char>());
  return _plaidml_alloc_device_enumerator(ctx, config.c_str(), config_file, callback, arg);
}

extern "C" plaidml_device_enumerator* plaidml_alloc_device_enumerator_with_config(
    vai_ctx* ctx, const char* configuration, void (*callback)(void* arg, plaidml_device_enumerator* device_enumerator),
    void* arg) {
  return _plaidml_alloc_device_enumerator(ctx, configuration, "CUSTOM", callback, arg);
}

extern "C" void plaidml_free_device_enumerator(plaidml_device_enumerator* device_enumerator) {
  delete device_enumerator;
}

// Gets the configuration file that was used to initialize devices
extern "C" const char* plaidml_get_enumerator_config_source(plaidml_device_enumerator* enumerator) {
  return enumerator->config_source.c_str();
}

extern "C" size_t plaidml_get_devconf_count(vai_ctx* ctx, plaidml_device_enumerator* enumerator, bool valid_devices) {
  if (!enumerator) {
    vertexai::SetLastOOM();
    return 0;
  }
  if (valid_devices) {
    return enumerator->devices.size();
  } else {
    return enumerator->unmatched_devices.size();
  }
}

extern "C" plaidml_devconf* plaidml_get_devconf(vai_ctx* ctx, plaidml_device_enumerator* enumerator, size_t index) {
  if (!enumerator) {
    vertexai::SetLastOOM();
    return nullptr;
  }
  if (enumerator->devices.size() <= index) {
    vertexai::SetLastStatus(VAI_STATUS_OUT_OF_RANGE, "Requested device index is out of range");
    return nullptr;
  }
  return &enumerator->devices.at(index);
}

extern "C" plaidml_devconf* plaidml_get_invalid_devconf(vai_ctx* ctx, plaidml_device_enumerator* enumerator,
                                                        size_t index) {
  if (!enumerator) {
    vertexai::SetLastOOM();
    return nullptr;
  }
  if (enumerator->unmatched_devices.size() <= index) {
    vertexai::SetLastStatus(VAI_STATUS_OUT_OF_RANGE, "Requested valdevice index is out of range");
    return nullptr;
  }
  return &enumerator->unmatched_devices.at(index);
}
// plaidml_buffer and plaidml_mapping
namespace {

class BufferState : public tile::lang::BufferBase {
 public:
  // cppcheck-suppress passedByValue  // NOLINT
  BufferState(std::shared_ptr<tile::Buffer> buffer, std::shared_ptr<Evaluator> evaluator)
      : buffer_{std::move(buffer)}, evaluator_{std::move(evaluator)} {}

  const std::shared_ptr<tile::Buffer>& buffer() const { return buffer_; }
  const std::shared_ptr<Evaluator>& get_evaluator() const { return evaluator_; }

 private:
  std::shared_ptr<tile::Buffer> buffer_;
  std::shared_ptr<Evaluator> evaluator_;
};

}  // namespace

struct plaidml_buffer {
  context::Activity activity;
  std::shared_ptr<BufferState> state;
  plaidml_buffer() {}
  plaidml_buffer(context::Activity activity_, std::shared_ptr<BufferState> state_)
      : activity{std::move(activity_)}, state{state_} {}
};

struct plaidml_mapping {
  std::unique_ptr<tile::View> view;
  context::Context ctx;
};

namespace {

class MapCompletion final {
 public:
  MapCompletion(void (*callback)(void* arg, plaidml_mapping* mapping), void* arg) : callback_{callback}, arg_{arg} {}

  void OnException(std::exception_ptr ep) noexcept {
    vertexai::SetLastException(ep);
    InvokeCallback(nullptr);
  }

  void OnComplete(const context::Context& ctx, boost::future<std::unique_ptr<tile::View>> f) {
    auto mapping = vertexai::compat::make_unique<plaidml_mapping>(plaidml_mapping{f.get(), ctx});
    if (InvokeCallback(mapping.get())) {
      mapping.release();
    }
  }

  context::Rundown* rundown() { return &rundown_; }

 private:
  void OnCancel() noexcept {
    vertexai::SetLastStatus(VAI_STATUS_CANCELLED, status_strings::kCancelled);
    InvokeCallback(nullptr);
  }

  bool InvokeCallback(plaidml_mapping* mapping) noexcept {
    {
      std::lock_guard<std::mutex> lock{mu_};
      if (invoked_callback_) {
        return false;
      }
      invoked_callback_ = true;
    }
    callback_(arg_, mapping);
    return true;
  }

  void (*callback_)(void* arg, plaidml_mapping* mapping);
  void* arg_;
  std::mutex mu_;
  bool invoked_callback_ = false;

  // N.B. This rundown should be the last member, so that it's the first destroyed; that way, if a cancellation
  // callback
  // arrives during destruction, the rest of the MapCompletion will still be in a valid state to handle it.  Also note
  // that once the rundown is destroyed, subsequent callbacks cannot occur.
  context::Rundown rundown_{[this]() { OnCancel(); }};
};

}  // namespace

extern "C" plaidml_buffer* plaidml_alloc_buffer(vai_ctx* ctx, plaidml_device* device, uint64_t size) {
  if (!device) {
    IVLOG(1, "Called plaidml_alloc_buffer on invalid device; thus out of memory.");
    vertexai::SetLastOOM();
    return nullptr;
  }

  if (!ctx) {
    vertexai::SetLastStatus(VAI_STATUS_CANCELLED, status_strings::kCancelled);
    return nullptr;
  }

  size = std::max(size, uint64_t(1));

  try {
    context::Activity activity{ctx->activity.ctx(), "vertexai::AllocBuffer"};
    return new plaidml_buffer{std::move(activity),
                              std::make_shared<BufferState>(device->evaluator->get_platform()->MakeBuffer(
                                                                ctx->activity.ctx(), device->evaluator->get_id(), size),
                                                            device->evaluator)};
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return nullptr;
  }
}

extern "C" void plaidml_free_buffer(plaidml_buffer* buffer) { delete buffer; }

extern "C" plaidml_mapping* plaidml_map_buffer_current(plaidml_buffer* buffer,
                                                       void (*callback)(void* arg, plaidml_mapping* mapping),
                                                       void* arg) {
  if (!callback) {
    vertexai::Sync<plaidml_mapping*> sync;
    plaidml_map_buffer_current(buffer, sync.callback(), sync.arg());
    return sync.WaitForResult();
  }
  if (!buffer) {
    vertexai::SetLastOOM();
    callback(arg, nullptr);
    return nullptr;
  }

  std::shared_ptr<MapCompletion> completion;
  try {
    completion = std::make_shared<MapCompletion>(callback, arg);
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    callback(arg, nullptr);
    return nullptr;
  }

  try {
    completion->rundown()->TryEnterGate(buffer->activity.ctx().gate());
    context::Activity activity{buffer->activity.ctx(), "tile::MapCurrent"};
    auto fut = buffer->state->buffer()->MapCurrent(activity.ctx());
    fut.then([ completion, buffer_state = buffer->state,
               activity = std::move(activity) ](boost::future<std::unique_ptr<tile::View>> f) noexcept {
      try {
        completion->OnComplete(activity.ctx(), std::move(f));
      } catch (...) {
        completion->OnException(std::current_exception());
      }
    });
  } catch (...) {
    completion->OnException(std::current_exception());
  }
  return nullptr;
}

extern "C" plaidml_mapping* plaidml_map_buffer_discard(vai_ctx* ctx, plaidml_buffer* buffer) {
  if (!buffer) {
    vertexai::SetLastOOM();
    return nullptr;
  }

  if (!ctx) {
    vertexai::SetLastStatus(VAI_STATUS_CANCELLED, status_strings::kCancelled);
    return nullptr;
  }

  std::unique_ptr<plaidml_mapping> mapping;
  try {
    context::Activity activity(ctx->activity.ctx(), "vertexai::DiscardCurrent");
    auto view = buffer->state->buffer()->MapDiscard(activity.ctx());
    mapping = vertexai::compat::make_unique<plaidml_mapping>(plaidml_mapping{std::move(view), activity.ctx()});
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return nullptr;
  }
  return mapping.release();
}

extern "C" char* plaidml_get_mapping_base(vai_ctx* ctx, plaidml_mapping* mapping) {
  if (!ctx) {
    vertexai::SetLastStatus(VAI_STATUS_CANCELLED, status_strings::kCancelled);
    return nullptr;
  }
  if (!mapping) {
    return nullptr;
  }
  try {
    context::Activity activity(ctx->activity.ctx(), "vertexai::GetMapBase");
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return nullptr;
  }
  return mapping->view->data();
}

extern "C" size_t plaidml_get_mapping_size(vai_ctx* ctx, plaidml_mapping* mapping) {
  if (!ctx) {
    vertexai::SetLastStatus(VAI_STATUS_CANCELLED, status_strings::kCancelled);
    return 0;
  }
  if (!mapping) {
    return 0;
  }
  try {
    context::Activity activity(ctx->activity.ctx(), "vertexai::GetMapSize");
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return 0;
  }
  return mapping->view->size();
}

extern "C" bool plaidml_writeback_mapping(vai_ctx* ctx, plaidml_mapping* mapping) {
  if (!ctx) {
    vertexai::SetLastStatus(VAI_STATUS_CANCELLED, status_strings::kCancelled);
    return false;
  }

  if (!mapping) {
    vertexai::SetLastOOM();
    return false;
  }
  try {
    context::Activity activity(ctx->activity.ctx(), "vertexai::WriteBackMap");
    mapping->view->WriteBack(mapping->ctx);
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return false;
  }

  return true;
}

extern "C" void plaidml_free_mapping(plaidml_mapping* mapping) { delete mapping; }

namespace {

tile::DataType MakeTileDataType(plaidml_datatype datatype) {
  switch (datatype) {
    case PLAIDML_DATA_BOOLEAN:
      return tile::DataType::BOOLEAN;
    case PLAIDML_DATA_INT8:
      return tile::DataType::INT8;
    case PLAIDML_DATA_INT16:
      return tile::DataType::INT16;
    case PLAIDML_DATA_INT32:
      return tile::DataType::INT32;
    case PLAIDML_DATA_INT64:
      return tile::DataType::INT64;
    case PLAIDML_DATA_UINT8:
      return tile::DataType::UINT8;
    case PLAIDML_DATA_UINT16:
      return tile::DataType::UINT16;
    case PLAIDML_DATA_UINT32:
      return tile::DataType::UINT32;
    case PLAIDML_DATA_UINT64:
      return tile::DataType::UINT64;
    case PLAIDML_DATA_FLOAT16:
      return tile::DataType::FLOAT16;
    case PLAIDML_DATA_FLOAT32:
      return tile::DataType::FLOAT32;
    case PLAIDML_DATA_FLOAT64:
      return tile::DataType::FLOAT64;
    default:
      return tile::DataType::INVALID;
  }
}

}  // namespace

extern "C" void plaidml_set_floatx(plaidml_datatype datatype) {
  tile::DataType dt = MakeTileDataType(datatype);
  if (dt == tile::DataType::INVALID) {
    vertexai::SetLastStatus(VAI_STATUS_INVALID_ARGUMENT, status_strings::kInvalidArgument);
    return;
  }
  tile::lang::SetFloatX(dt);
}

// plaidml_shape

struct plaidml_shape {
  tile::TensorShape shape;
  size_t offset_in_elements = 0;
  bool valid = true;
};

extern "C" plaidml_shape* plaidml_alloc_shape(vai_ctx* ctx, plaidml_datatype datatype) {
  if (!ctx) {
    vertexai::SetLastStatus(VAI_STATUS_CANCELLED, status_strings::kCancelled);
    return nullptr;
  }

  tile::DataType dt = MakeTileDataType(datatype);
  if (dt == tile::DataType::INVALID) {
    vertexai::SetLastStatus(VAI_STATUS_INVALID_ARGUMENT, status_strings::kInvalidArgument);
    return nullptr;
  }

  try {
    context::Activity activity(ctx->activity.ctx(), "vertexai::AllocShape");
    auto shp = vertexai::compat::make_unique<plaidml_shape>();
    shp->shape.type = dt;
    return shp.release();
  } catch (...) {
    return nullptr;
  }
}

extern "C" void plaidml_free_shape(plaidml_shape* shape) { delete shape; }

extern "C" bool plaidml_set_shape_offset(vai_ctx* ctx, plaidml_shape* shape, uint64_t offset_in_elements) {
  if (!shape) {
    vertexai::SetLastOOM();
    return false;
  }
  shape->offset_in_elements = offset_in_elements;
  // TODO: Use offset_in_elements as part of the tensor shape passed to devices.
  return true;
}

extern "C" bool plaidml_add_dimension(vai_ctx* ctx, plaidml_shape* shape, uint64_t size_in_elements,
                                      int64_t stride_in_elements) {
  if (!shape) {
    vertexai::SetLastOOM();
    return false;
  }
  try {
    shape->shape.dims.emplace_back(stride_in_elements, size_in_elements);
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    shape->valid = false;
    return false;
  }
  return true;
}

extern "C" plaidml_datatype plaidml_get_shape_type(plaidml_shape* shape) {
  if (!shape) {
    vertexai::SetLastOOM();
    return PLAIDML_DATA_INVALID;
  }
  switch (shape->shape.type) {
    case tile::DataType::BOOLEAN:
      return PLAIDML_DATA_BOOLEAN;
    case tile::DataType::INT8:
      return PLAIDML_DATA_INT8;
    case tile::DataType::INT16:
      return PLAIDML_DATA_INT16;
    case tile::DataType::INT32:
      return PLAIDML_DATA_INT32;
    case tile::DataType::INT64:
      return PLAIDML_DATA_INT64;
    case tile::DataType::UINT8:
      return PLAIDML_DATA_UINT8;
    case tile::DataType::UINT16:
      return PLAIDML_DATA_UINT16;
    case tile::DataType::UINT32:
      return PLAIDML_DATA_UINT32;
    case tile::DataType::UINT64:
      return PLAIDML_DATA_UINT64;
    case tile::DataType::FLOAT16:
      return PLAIDML_DATA_FLOAT16;
    case tile::DataType::FLOAT32:
      return PLAIDML_DATA_FLOAT32;
    case tile::DataType::FLOAT64:
      return PLAIDML_DATA_FLOAT64;
    default:
      return PLAIDML_DATA_INVALID;
  }
}

extern "C" uint64_t plaidml_get_shape_offset(plaidml_shape* shape) {
  if (!shape) {
    vertexai::SetLastOOM();
    return 0;
  }
  return shape->offset_in_elements;
}

size_t plaidml_get_shape_dimension_count(plaidml_shape* shape) {
  if (!shape) {
    vertexai::SetLastOOM();
    return 0;
  }
  return shape->shape.dims.size();
}

uint64_t plaidml_get_shape_dimension_size(plaidml_shape* shape, size_t dim) {
  if (!shape) {
    vertexai::SetLastOOM();
    return 0;
  }
  if (dim >= shape->shape.dims.size()) {
    vertexai::SetLastStatus(VAI_STATUS_OUT_OF_RANGE, "Dimension input out of range");
    return 0;
  }
  return shape->shape.dims[dim].size;
}

int64_t plaidml_get_shape_dimension_stride(plaidml_shape* shape, size_t dim) {
  if (!shape) {
    vertexai::SetLastOOM();
    return 0;
  }
  if (dim >= shape->shape.dims.size()) {
    vertexai::SetLastStatus(VAI_STATUS_OUT_OF_RANGE, "Dimension input out of range");
    return 0;
  }
  return shape->shape.dims[dim].stride;
}

uint64_t plaidml_get_shape_buffer_size(plaidml_shape* shape) {
  if (!shape) {
    vertexai::SetLastOOM();
    return 0;
  }
  return shape->shape.byte_size();
}

uint64_t plaidml_get_shape_element_count(plaidml_shape* shape) {
  if (!shape) {
    vertexai::SetLastOOM();
    return 0;
  }
  return shape->shape.elem_size();
}

// plaidml_function

struct plaidml_function {
  std::shared_ptr<BoundFunction> func;
};

extern "C" plaidml_function* plaidml_build_coded_function(const char* code, const char* id) {
  try {
    std::string sid;
    if (id != NULL) {
      sid = std::string(id);
    }
    return new plaidml_function{std::make_shared<BoundFunction>(code, sid)};
  } catch (...) {
    vertexai::SetLastOOM();
    return nullptr;
  }
}

extern "C" void plaidml_free_function(plaidml_function* function) { delete function; }

namespace {

//  V0 format:
//  0..7  : shape size
//  8..ss : shape
//  ...   : tensor data
void WriteTensor(zipFile f, const std::string& name, const TensorValue& tensor) {
  std::vector<size_t> rdims;
  const auto& tdims = tensor.shape().dims;
  for (size_t i = 0; i < tdims.size(); i++) {
    rdims.push_back(tdims[i].size);
  }

  std::unique_ptr<vai_ctx> ctx{vai_alloc_ctx()};
  if (!ctx) {
    throw std::runtime_error("Unable to allocate context while writing tensor");
  }

  std::shared_ptr<BufferState> bs = std::static_pointer_cast<BufferState>(tensor.buffer());
  context::Activity activity(ctx->activity.ctx(), "vertexai::WriteTensor");
  plaidml_buffer tb{std::move(activity), bs};
  std::unique_ptr<plaidml_mapping> tm{plaidml_map_buffer_current(&tb, nullptr, nullptr)};
  if (!tm) {
    throw std::runtime_error("Unable to map tensor in order to write tensor data");
  }
  if (zipOpenNewFileInZip64(f, name.c_str(), NULL, NULL, 0, NULL, 0, NULL, Z_NO_COMPRESSION, 0, 1) != ZIP_OK) {
    throw std::runtime_error("Could not write file into zip file");
  }
  std::string shape_buf;
  IntoProto(tensor.shape()).SerializeToString(&shape_buf);
  uint64_t shape_sz = shape_buf.size();
  zipWriteInFileInZip(f, &shape_sz, sizeof(shape_sz));
  zipWriteInFileInZip(f, &shape_buf[0], shape_sz);
  if (zipWriteInFileInZip(f, plaidml_get_mapping_base(ctx.get(), tm.get()),
                          plaidml_get_mapping_size(ctx.get(), tm.get())) != ZIP_OK) {
    throw std::runtime_error("Could not write tensor into zipfile");
  }
  zipCloseFileInZip(f);
}

void WriteString(zipFile f, const std::string& name, const std::string& value) {
  if (zipOpenNewFileInZip64(f, name.c_str(), NULL, NULL, 0, NULL, 0, NULL, Z_DEFLATED, Z_DEFAULT_COMPRESSION, 1) !=
      ZIP_OK) {
    throw std::runtime_error("Could not open new file in zip file");
  }
  if (zipWriteInFileInZip(f, &value[0], value.size()) != ZIP_OK) {
    throw std::runtime_error("Could not write into zip");
  }
  zipCloseFileInZip(f);
}

void WriteVersion(zipFile f) { WriteString(f, "version", "0"); }

void WriteFunction(zipFile f, const BoundFunction& func) {
  if (func.out_bound().size() > 0) {
    throw std::runtime_error("Can't save a function that has bound outputs");
  }
  if (func.out_bound().size() > 0) {
    throw std::runtime_error("Can't save a function that has bound outputs");
  }
  std::string xo = to_string(Xify(func.prog()));
  WriteString(f, "code", xo);
  for (const auto& kvp : func.in_bound()) {
    WriteTensor(f, "data_" + kvp.first, *kvp.second);
  }
}

void WriteMetadata(zipFile f, const BoundFunction& func, const std::map<std::string, std::shared_ptr<Value>>& inputs) {
  tile::metadata::proto::Metadata md;

  for (std::size_t idx = 0; idx < func.num_inputs(); ++idx) {
    auto it = inputs.find(func.input_name(idx));
    if (it == inputs.end()) {
      throw std::runtime_error{"Unbound invoker input: " + func.input_name(idx)};
    }
    TensorValue* tv = dynamic_cast<TensorValue*>(it->second.get());
    if (!tv) {
      continue;
    }
    (*md.mutable_inputs())[func.input_name(idx)] = tile::IntoProto(tv->shape());
  }

  gpu::JsonPrintOptions options;
  options.add_whitespace = true;
  auto resolver = gpu::NewTypeResolverForDescriptorPool(vertexai::kTypeVertexAI, gp::DescriptorPool::generated_pool());
  std::string serialized;
  gpu::BinaryToJsonString(resolver, vertexai::kTypeVertexAIPrefix + md.GetDescriptor()->full_name(),
                          md.SerializeAsString(), &serialized, options);
  WriteString(f, "metadata", serialized);
}

std::shared_ptr<TensorValue> ReadTensor(vai_ctx* ctx, vertexai::UnZipArchive* zip_file,
                                        const std::shared_ptr<Evaluator>& evaluator, const std::string& name) {
  auto tensor_file = zip_file->OpenFile(name);
  context::Activity activity(ctx->activity.ctx(), "vertexai::ReadTensor");

  uint64_t shape_size;
  tensor_file.ReadInto(&shape_size, sizeof(shape_size));

  std::string proto_buf(shape_size, '\0');
  tensor_file.ReadInto(&proto_buf[0], proto_buf.size());

  tile::proto::TensorShape ts_proto;
  ts_proto.ParseFromString(proto_buf);
  auto ts = tile::FromProto(ts_proto);
  std::shared_ptr<BufferState> bs = std::make_shared<BufferState>(
      evaluator->get_platform()->MakeBuffer(ctx->activity.ctx(), evaluator->get_id(), ts.byte_size()), evaluator);
  plaidml_buffer tb{std::move(activity), bs};
  std::unique_ptr<plaidml_mapping> tm{plaidml_map_buffer_discard(ctx, &tb)};
  if (!tm) {
    throw std::runtime_error("Unable to map tensor in read_tensor");
  }

  tensor_file.ReadInto(plaidml_get_mapping_base(ctx, tm.get()), plaidml_get_mapping_size(ctx, tm.get()));
  plaidml_writeback_mapping(ctx, tm.get());
  return tile::lang::TensorValue::make(bs, ts);
}

}  // namespace

extern "C" bool plaidml_save_function(plaidml_function* function, const char* filename) {
  std::unique_ptr<vai_ctx> ctx{vai_alloc_ctx()};
  try {
    zipFile out_file = zipOpen64(filename, 0);
    WriteVersion(out_file);
    WriteFunction(out_file, *function->func);
    zipClose(out_file, nullptr);
    return true;
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return false;
  }
}

extern "C" plaidml_function* plaidml_load_function(vai_ctx* ctx, plaidml_device* platform, const char* filename) {
  if (!platform) {
    vertexai::SetLastOOM();
    return 0;
  }
  try {
    vertexai::UnZipArchive zip_file(filename);
    auto code = zip_file.OpenFile("code").ReadString();
    tile::lang::Parser parser;
    tile::lang::Program p = DeXify(parser.Parse(code));
    // Unfortunately, we don't serialize the number of temps (which is needed to do inlining)
    // So we recompute that here, based on the fact that all temps start with _T (otherwise reserved)
    for (const tile::lang::Op& op : p.ops) {
      if (op.output.size() >= 2 && op.output[0] == '_' && op.output[1] == 'T') {
        p.next_tmp = std::max(p.next_tmp,
                              static_cast<uint64_t>(std::atoi(op.output.substr(2, op.output.size() - 2).c_str()) + 1));
      }
    }
    std::vector<std::shared_ptr<TensorValue>> inputs;
    for (const auto& in : p.inputs) {
      if (in.name[0] == '_') {
        inputs.push_back(ReadTensor(ctx, &zip_file, platform->evaluator, "data_" + in.name));
      }
    }
    return new plaidml_function{std::make_shared<BoundFunction>(p, inputs)};
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return NULL;
  }
}

size_t plaidml_get_function_input_count(plaidml_function* function) {
  if (function == NULL) return 0;
  return function->func->num_inputs();
}

const char* plaidml_get_function_input(plaidml_function* function, size_t i) {
  if (function == NULL || i > function->func->num_inputs()) return NULL;
  return function->func->input_name(i).c_str();
}

size_t plaidml_get_function_output_count(plaidml_function* function) {
  if (function == NULL) return 0;
  return function->func->num_outputs();
}

const char* plaidml_get_function_output(plaidml_function* function, size_t i) {
  if (function == NULL || i > function->func->num_outputs()) return NULL;
  return function->func->output_name(i).c_str();
}

// plaidml_var

struct plaidml_var {
  std::shared_ptr<Value> value;
};

extern "C" void plaidml_free_var(plaidml_var* var) { delete var; }

extern "C" plaidml_var* plaidml_alloc_placeholder(size_t num_dimensions) {
  try {
    return new plaidml_var{std::make_shared<PlaceholderValue>(num_dimensions)};
  } catch (...) {
    vertexai::SetLastOOM();
    return nullptr;
  }
}

extern "C" plaidml_var* plaidml_alloc_int64(int64_t value) {
  try {
    return new plaidml_var{IConstValue::make(value)};
  } catch (...) {
    vertexai::SetLastOOM();
    return nullptr;
  }
}

extern "C" plaidml_var* plaidml_alloc_real(double value) {
  try {
    return new plaidml_var{FConstValue::make(value)};
  } catch (...) {
    vertexai::SetLastOOM();
    return nullptr;
  }
}

extern "C" plaidml_var* plaidml_alloc_tensor(vai_ctx* ctx, plaidml_buffer* buffer, plaidml_shape* shape) {
  if (buffer == NULL || shape == NULL) {
    vertexai::SetLastOOM();
    return nullptr;
  }

  if (!ctx) {
    vertexai::SetLastStatus(VAI_STATUS_CANCELLED, status_strings::kCancelled);
    return nullptr;
  }

  try {
    return new plaidml_var{TensorValue::make(buffer->state, shape->shape)};
  } catch (...) {
    vertexai::SetLastOOM();
    return nullptr;
  }
}

// plaidml_composer

struct plaidml_composer {
  std::shared_ptr<BoundFunction> func;
};

// Predeclare applier
struct plaidml_applier {
  std::shared_ptr<FunctionApplication> apply;  // The actual application
};

extern "C" plaidml_composer* plaidml_alloc_composer() {
  try {
    return new plaidml_composer{std::make_shared<BoundFunction>()};
  } catch (...) {
    vertexai::SetLastOOM();
    return nullptr;
  }
}

extern "C" bool plaidml_add_composer_input(plaidml_composer* composer, const char* name, plaidml_var* var) {
  if (composer == NULL || name == NULL || var == NULL) {
    vertexai::SetLastOOM();
    return false;
  }
  try {
    auto vptr = std::dynamic_pointer_cast<PlaceholderValue>(var->value);
    if (!vptr) {
      throw vertexai::error::InvalidArgument{"Composer input must be a placeholder"};
    }
    composer->func->AddInput(name, vptr);
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return false;
  }
  return true;
}

extern "C" bool plaidml_add_composer_output(plaidml_composer* composer, const char* name, plaidml_var* var) {
  if (composer == NULL || name == NULL || var == NULL) {
    vertexai::SetLastOOM();
    return false;
  }
  try {
    composer->func->AddOutput(name, var->value);
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return false;
  }
  return true;
}

extern "C" bool plaidml_add_composer_dependency(plaidml_composer* composer, plaidml_applier* must_run_before) {
  if (composer == NULL || must_run_before == NULL) {
    vertexai::SetLastOOM();
    return false;
  }
  try {
    must_run_before->apply->SetDone();
    composer->func->AddDependency(*must_run_before->apply);
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return false;
  }
  return true;
}

extern "C" bool plaidml_add_composer_update(plaidml_composer* composer, plaidml_var* dest_tensor,
                                            plaidml_var* src_tensor) {
  if (composer == NULL || dest_tensor == NULL || src_tensor == NULL) {
    vertexai::SetLastOOM();
    return false;
  }
  try {
    auto tptr = std::dynamic_pointer_cast<TensorValue>(dest_tensor->value);
    if (!tptr) {
      throw vertexai::error::InvalidArgument("Composer update dest must be a tensor");
    }
    composer->func->AddUpdate(tptr, src_tensor->value);
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return false;
  }
  return true;
}

extern "C" plaidml_function* plaidml_build_composed_function(plaidml_composer* composer) {
  if (composer == NULL) {
    vertexai::SetLastOOM();
    return nullptr;
  }
  try {
    composer->func->Done();
    return new plaidml_function{composer->func};
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return nullptr;
  }
}

extern "C" void plaidml_free_composer(plaidml_composer* composer) { delete composer; }

// plaidml_applier

// Allocates an applier describing the application of the given function to some
// number of inputs, or returns NULL if the library cannot allocate sufficient memory.
extern "C" plaidml_applier* plaidml_alloc_applier(plaidml_function* function) {
  if (function == NULL) {
    vertexai::SetLastOOM();
    return nullptr;
  }
  try {
    return new plaidml_applier{std::make_shared<FunctionApplication>(function->func)};
  } catch (...) {
    vertexai::SetLastOOM();
    return nullptr;
  }
}

extern "C" bool plaidml_apply_add_dependency(plaidml_applier* applier, plaidml_applier* must_run_before) {
  if (applier == NULL || must_run_before == NULL) {
    vertexai::SetLastOOM();
    return false;
  }
  try {
    must_run_before->apply->SetDone();
    applier->apply->AddDependency(*must_run_before->apply);
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return false;
  }
  return true;
}

extern "C" bool plaidml_apply_add_input(plaidml_applier* applier, const char* name, plaidml_var* var) {
  if (applier == NULL || name == NULL || var == NULL) {
    vertexai::SetLastOOM();
    return false;
  }
  try {
    applier->apply->SetInput(name, var->value);
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return false;
  }
  return true;
}

extern "C" plaidml_var* plaidml_apply_alloc_output(plaidml_applier* applier, const char* name) {
  if (applier == NULL || name == NULL) {
    vertexai::SetLastOOM();
    return nullptr;
  }
  try {
    return new plaidml_var{applier->apply->GetOutput(name)};
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return nullptr;
  }
}

extern "C" void plaidml_free_applier(plaidml_applier* applier) { delete applier; }

// plaidml_invoker

namespace {

class NamedBuffer : public tile::lang::BufferBase {
 public:
  // cppcheck-suppress passedByValue  // NOLINT
  explicit NamedBuffer(std::string name) : name_{std::move(name)} {}

  const std::string& name() const { return name_; }

 private:
  std::string name_;
};

template <class V>
std::map<std::string, std::shared_ptr<tile::Buffer>> BindBuffers(
    const std::map<std::string, std::shared_ptr<tile::lang::BufferBase>>& uses,
    const std::map<std::string, std::shared_ptr<V>>& bindings, std::shared_ptr<Evaluator>* evaluator) {
  std::map<std::string, std::shared_ptr<tile::Buffer>> result;
  for (const auto& kvp : uses) {
    std::shared_ptr<BufferState> bs = std::dynamic_pointer_cast<BufferState>(kvp.second);
    if (!bs) {
      std::shared_ptr<NamedBuffer> nb = std::dynamic_pointer_cast<NamedBuffer>(kvp.second);
      if (!nb) {
        throw std::runtime_error("Internal error");
      }
      auto value = bindings.at(nb->name());
      if (value->type() != Value::TENSOR) {
        throw std::runtime_error("Tensor parameter not bound to tensor value");
      }
      bs = std::dynamic_pointer_cast<BufferState>(std::dynamic_pointer_cast<TensorValue>(value)->buffer());
      if (!bs) {
        throw std::runtime_error("Internal error");
      }
    }
    if (!*evaluator) {
      *evaluator = bs->get_evaluator();
    } else {
      if (*evaluator != bs->get_evaluator()) {
        throw std::runtime_error("Cross device functions not supported");
      }
    }
    result[kvp.first] = bs->buffer();
  }
  return result;
}

struct ApplierParameterShape {
  ApplierParameterShape() = default;

  template <class V>
  explicit ApplierParameterShape(const std::shared_ptr<V>& value) : type{value->type()} {
    switch (type) {
      case Value::TENSOR:
        shape = std::dynamic_pointer_cast<TensorValue>(value)->shape();
        break;
      case Value::FCONST:
        fconst = std::dynamic_pointer_cast<FConstValue>(value)->value();
        break;
      case Value::ICONST:
        iconst = std::dynamic_pointer_cast<IConstValue>(value)->value();
        break;
      default:
        throw std::runtime_error{"Corrupted input found in function application key construction"};
    }
  }

  Value::Type type;
  tile::TensorShape shape;
  std::int64_t iconst = 0;
  double fconst = 0.0;
};

inline bool operator<(const ApplierParameterShape& lhs, const ApplierParameterShape& rhs) {
  if (lhs.type < rhs.type) {
    return true;
  }
  if (rhs.type < lhs.type) {
    return false;
  }
  switch (lhs.type) {
    case Value::TENSOR:
      return lhs.shape < rhs.shape;
    case Value::FCONST:
      return lhs.fconst < rhs.fconst;
    case Value::ICONST:
      return lhs.iconst < rhs.iconst;
    default:
      throw std::runtime_error{"Corrupted type in parameter shape"};
  }
}

template <class V>
std::map<std::string, ApplierParameterShape> ToApplierParameterShapes(
    const std::map<std::string, std::shared_ptr<V>>& bindings) {
  std::map<std::string, ApplierParameterShape> result;
  for (const auto& kvp : bindings) {
    result[kvp.first] = ApplierParameterShape{kvp.second};
  }
  return result;
}

}  // namespace

struct plaidml_invoker {
  std::shared_ptr<BoundFunction> func;
  std::map<std::string, std::shared_ptr<Value>> inputs;
  std::map<std::string, std::shared_ptr<TensorValue>> outputs;

  tile::LruCache<std::map<std::string, ApplierParameterShape>, std::shared_ptr<FunctionApplication>>
      applier_for_output_shape_cache{kApplierForShapeCacheSize};
  std::shared_ptr<FunctionApplication> applier_for_output_shape;

  tile::LruCache<std::pair<std::map<std::string, ApplierParameterShape>, std::map<std::string, ApplierParameterShape>>,
                 std::shared_ptr<RunInfo>>
      runinfo_cache{kRuninfoCacheSize};

  std::shared_ptr<RunInfo> runinfo;
};

namespace {

void BuildInvokerRunInfo(plaidml_invoker* invoker) {
  if (invoker->runinfo) {
    return;
  }
  invoker->runinfo = invoker->runinfo_cache.Lookup(
      std::make_pair(ToApplierParameterShapes(invoker->inputs), ToApplierParameterShapes(invoker->outputs)),
      [invoker]() {
        auto applier = std::make_shared<FunctionApplication>(invoker->func);
        for (const auto& it : invoker->inputs) {
          if (it.second->type() == Value::TENSOR) {
            applier->SetInput(
                it.first, std::make_shared<TensorValue>(std::make_shared<NamedBuffer>(it.first),
                                                        std::dynamic_pointer_cast<TensorValue>(it.second)->shape()));
          } else {
            applier->SetInput(it.first, it.second);
          }
        }
        applier->SetDone();
        auto composer = vertexai::compat::make_unique<BoundFunction>();
        composer->AddDependency(*applier);
        for (const auto& it : invoker->outputs) {
          composer->AddUpdate(std::make_shared<TensorValue>(std::make_shared<NamedBuffer>(it.first),
                                                            std::dynamic_pointer_cast<TensorValue>(it.second)->shape()),
                              applier->GetOutput(it.first));
        }
        composer->Done();
        return std::make_shared<RunInfo>(composer->PrepareToRun());
      });
}

}  // namespace

extern "C" plaidml_invoker* plaidml_alloc_invoker(vai_ctx* ctx, plaidml_function* function) {
  if (!ctx || !function) {
    vertexai::SetLastOOM();
    return nullptr;
  }
  try {
    auto invoker = vertexai::compat::make_unique<plaidml_invoker>();
    invoker->func = function->func;
    return invoker.release();
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return nullptr;
  }
}

extern "C" void plaidml_free_invoker(plaidml_invoker* invoker) { delete invoker; }

extern "C" bool plaidml_set_invoker_input(plaidml_invoker* invoker, const char* name, plaidml_var* var) {
  if (!invoker || !name) {
    vertexai::SetLastOOM();
    return false;
  }
  try {
    if (var) {
      switch (var->value->type()) {
        case Value::TENSOR:
        case Value::FCONST:
        case Value::ICONST:
          break;
        default:
          throw vertexai::error::InvalidArgument{"Invocation inputs must be tensors or constants"};
      }
      invoker->inputs[name] = var->value;
    } else {
      invoker->inputs.erase(name);
    }
    invoker->applier_for_output_shape.reset();
    invoker->runinfo.reset();
    return true;
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return false;
  }
}

extern "C" plaidml_shape* plaidml_alloc_invoker_output_shape(plaidml_invoker* invoker, const char* name) {
  if (!invoker || !name) {
    vertexai::SetLastOOM();
    return nullptr;
  }
  try {
    if (!invoker->applier_for_output_shape) {
      invoker->applier_for_output_shape =
          invoker->applier_for_output_shape_cache.Lookup(ToApplierParameterShapes(invoker->inputs), [invoker]() {
            auto applier = std::make_shared<FunctionApplication>(invoker->func);
            for (const auto& it : invoker->inputs) {
              if (it.second->type() == Value::TENSOR) {
                applier->SetInput(it.first, std::make_shared<TensorValue>(
                                                std::make_shared<NamedBuffer>(it.first),
                                                std::dynamic_pointer_cast<TensorValue>(it.second)->shape()));
              } else {
                applier->SetInput(it.first, it.second);
              }
            }
            return applier;
          });
    }

    auto shape = vertexai::compat::make_unique<plaidml_shape>();
    shape->shape = invoker->applier_for_output_shape->GetOutputShape(name);
    return shape.release();
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return nullptr;
  }
}

extern "C" bool plaidml_set_invoker_output(plaidml_invoker* invoker, const char* name, plaidml_var* var) {
  if (!invoker || !name) {
    vertexai::SetLastOOM();
    return false;
  }
  try {
    if (var) {
      if (var->value->type() != Value::TENSOR) {
        throw vertexai::error::InvalidArgument{"Invocation outputs must be tensors"};
      }
      invoker->outputs[name] = std::dynamic_pointer_cast<TensorValue>(var->value);
    } else {
      invoker->outputs.erase(name);
    }
    invoker->runinfo.reset();
    return true;
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return false;
  }
}

extern "C" bool plaidml_save_invoker(plaidml_invoker* invoker, const char* filename, plaidml_file_format format) {
  if (!invoker || !filename || !format) {
    vertexai::SetLastOOM();
    return false;
  }

  try {
    auto path = boost::filesystem::path(filename);
    if (!boost::filesystem::exists(path.parent_path())) {
      boost::filesystem::create_directory(path.parent_path());
    }

    switch (format) {
      case PLAIDML_FILE_FORMAT_TILE: {
        zipFile out_file = zipOpen64(filename, 0);
        WriteVersion(out_file);
        WriteFunction(out_file, *invoker->func);
        WriteMetadata(out_file, *invoker->func, invoker->inputs);
        zipClose(out_file, nullptr);
        return true;
      }

      case PLAIDML_FILE_FORMAT_STRIPE_HUMAN:
      case PLAIDML_FILE_FORMAT_STRIPE_PROTOTXT:
      case PLAIDML_FILE_FORMAT_STRIPE_BINARY:
        // We'll handle the Stripe file formats after the switch().
        break;

      default:
        throw std::runtime_error{"Unsupported save file format"};
    }

    // At this point, we're saving a Stripe file format.
    BuildInvokerRunInfo(invoker);
    invoker->runinfo->program_name = path.stem().string();
    auto program = GenerateStripe(*invoker->runinfo);

    std::ofstream file{path.string()};

    switch (format) {
      case PLAIDML_FILE_FORMAT_STRIPE_HUMAN:
        file << *program;
        break;

      case PLAIDML_FILE_FORMAT_STRIPE_PROTOTXT: {
        auto pb_program = tile::stripe::IntoProto(*program);
        gpi::OstreamOutputStream out{&file};
        gp::TextFormat::Print(pb_program, &out);
      } break;

      case PLAIDML_FILE_FORMAT_STRIPE_BINARY: {
        auto pb_program = tile::stripe::IntoProto(*program);
        pb_program.SerializeToOstream(&file);
      } break;

      default:
        break;
    }

    return true;
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return false;
  }
}

// plaidml_invocation
//
// Currently, the actual invocation structure is a placeholder; it
// represents a particular invocation of a Plaid function, but it
// doesn't have any useful data.  The intention is that it gives us a
// place to stand to query information about the invocation, attach
// callbacks, &c.

struct plaidml_invocation {};

extern "C" plaidml_invocation* plaidml_schedule_invocation(vai_ctx* ctx, plaidml_invoker* invoker) {
  if (!ctx || !invoker) {
    vertexai::SetLastOOM();
    return nullptr;
  }
  context::Activity activity{ctx->activity.ctx(), "plaidml::invoker::ScheduleInvocation"};
  try {
    auto invocation = vertexai::compat::make_unique<plaidml_invocation>();
    auto rundown = std::make_shared<context::Rundown>();
    rundown->TryEnterGate(activity.ctx().gate());
    BuildInvokerRunInfo(invoker);

    // Gather up the appropriate buffers
    std::shared_ptr<Evaluator> evaluator;

    auto in_buffers = BindBuffers(invoker->runinfo->input_buffers, invoker->inputs, &evaluator);
    auto out_buffers = BindBuffers(invoker->runinfo->output_buffers, invoker->outputs, &evaluator);

    std::unordered_set<const tile::Buffer*> output_set;
    for (const auto& kv : out_buffers) {
      output_set.insert(kv.second.get());
    }

    if (!evaluator) {
      throw vertexai::error::FailedPrecondition{"Function has neither inputs nor outputs"};
    }

    tile::proto::Program prog;
    prog.set_dev_id(evaluator->get_id());
    prog.set_code(invoker->runinfo->code);
    for (const auto& kv : invoker->runinfo->input_shapes) {
      auto& input = (*prog.mutable_inputs())[kv.first];
      *input.mutable_shape() = tile::IntoProto(kv.second);
      if (output_set.count(in_buffers[kv.first].get())) {
        input.set_consumed(true);
      }
    }
    for (const auto& kv : invoker->runinfo->output_shapes) {
      *(*prog.mutable_outputs())[kv.first].mutable_shape() = tile::IntoProto(kv.second);
    }

    size_t max_trials = 1;
    auto env_trials = vertexai::env::Get("PLAIDML_KERNEL_TRIALS");
    if (env_trials.length()) {
      auto env_value = std::atoi(env_trials.c_str());
      if (env_value) {
        max_trials = env_value;
      }
    }

    size_t max_trial_runs = 1;
    auto env_runs = vertexai::env::Get("PLAIDML_KERNEL_TRIAL_RUNS");
    if (env_runs.length()) {
      auto env_value = std::atoi(env_runs.c_str());
      if (env_value) {
        max_trial_runs = env_value;
      }
    }

    auto* params = prog.mutable_tile_scanning_params();
    params->set_max_trials(max_trials);
    params->set_max_trial_runs(max_trial_runs);

    auto program = evaluator->MakeProgram(activity.ctx(), prog);

    // Run the program
    auto result = program->Run(activity.ctx(), in_buffers, out_buffers);
    result.then(boost::launch::async, [rundown = std::move(rundown)](decltype(result) fut) {
      try {
        fut.get();
      } catch (const std::exception& ex) {
        // TODO: We need a better way to notify users if the asynchronous results
        // of an invocation are valid, perhaps by allowing a callback to be specified.
        LOG(ERROR) << ex.what();
      }
    });

    return invocation.release();
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return nullptr;
  }
}

extern "C" void plaidml_free_invocation(plaidml_invocation* invocation) { delete invocation; }

// plaidml_gradient

struct plaidml_gradient {
  std::shared_ptr<Gradient> grad;
};

extern "C" plaidml_gradient* plaidml_alloc_gradient(plaidml_var* var) {
  if (var == NULL || !var->value) {
    vertexai::SetLastOOM();
    return nullptr;
  }
  try {
    auto ptr = std::make_shared<Gradient>(var->value);
    return new plaidml_gradient{ptr};
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return nullptr;
  }
}

extern "C" void plaidml_free_gradient(plaidml_gradient* grad) { delete grad; }

plaidml_var* plaidml_compute_grad_wrt(plaidml_gradient* grad, plaidml_var* wrt) {
  if (grad == NULL || wrt == NULL || !grad->grad || !wrt->value) {
    vertexai::SetLastOOM();
    return nullptr;
  }
  try {
    auto ptr = (*(grad->grad))(wrt->value);
    return new plaidml_var{ptr};
  } catch (...) {
    vertexai::SetLastException(std::current_exception());
    return nullptr;
  }
}
