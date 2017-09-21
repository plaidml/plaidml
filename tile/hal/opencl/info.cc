// Copyright 2017, Vertex.AI. CONFIDENTIAL

#include "tile/hal/opencl/info.h"

#include <boost/regex.hpp>
#include <google/protobuf/text_format.h>

#include <algorithm>

#include "base/util/logging.h"
#include "tile/hal/opencl/opencl.pb.h"
#include "tile/proto/hal.pb.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

namespace gp = ::google::protobuf;

hal::proto::HardwareInfo GetHardwareInfo(const proto::DeviceInfo& info) {
  hal::proto::HardwareInfo result;
  result.set_type(info.type());
  result.set_name(info.name());
  result.set_vendor(info.vendor());
  result.set_vendor_id(info.vendor_id());

  hal::proto::HardwareSettings* settings = result.mutable_settings();

  // Threads to use per work group.
  settings->mutable_threads()->set_value(256);

  // Vector size.
  settings->mutable_vec_size()->set_value(
      1);  // TODO(T404) re-enable when possible - info.preferred_vector_width_float())

  settings->mutable_use_global()->set_value(false);  // Use shared memory on most platforms.

// Memory width
#ifdef __APPLE__
  settings->mutable_mem_width()->set_value(32);
#else
  settings->mutable_mem_width()->set_value(info.global_mem_cacheline_size());
#endif

  // Maximum local memory
  settings->mutable_max_mem()->set_value(info.local_mem_size());

  // Maximum register size
  settings->mutable_max_regs()->set_value(16 * 1024);

  // Minimum number of work groups to get full utilization, 4 * CU's is an estimate
  settings->mutable_goal_groups()->set_value(info.max_compute_units() * 4);

  // Minimum roof, this is a total guess
  settings->mutable_goal_flops_per_byte()->set_value(50);

  // Workgroup dimension sizes
  for (auto size : info.work_item_dimension_size()) {
    settings->add_dim_sizes(size);
  }

  // N.B. We never enable half-width computation by default; the user
  // can set it as an override if desired.

  return result;
}

void LogInfo(const std::string& prefix, const gp::Message& info) {
  if (!VLOG_IS_ON(3)) {
    return;
  }

  VLOG(3) << prefix << ':';
  gp::TextFormat::Printer printer;
  printer.SetUseShortRepeatedPrimitives(true);
  std::string str;
  printer.PrintToString(info, &str);
  boost::regex re{R"([^\n]+)"};
  std::for_each(boost::sregex_iterator{str.begin(), str.end(), re}, boost::sregex_iterator(),
                [&prefix](const boost::smatch& match) { VLOG(3) << prefix << '.' << match.str(); });
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
