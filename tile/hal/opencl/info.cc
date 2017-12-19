// Copyright 2017, Vertex.AI.

#include "tile/hal/opencl/info.h"

#include <google/protobuf/text_format.h>

#include <algorithm>
#include <boost/regex.hpp>

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
  result.set_platform(info.platform_name());
  result.mutable_info()->PackFrom(info);

  hal::proto::HardwareSettings* settings = result.mutable_settings();

  // Threads to use per work group.
  settings->set_threads(1);

  // Vector size.
  // TODO(T404) re-enable when possible - info.preferred_vector_width_float())
  settings->set_vec_size(1);

  // Use shared memory by default since most platforms support it.
  settings->set_use_global(false);

// Memory width
#ifdef __APPLE__
  settings->set_mem_width(32);
#else
  settings->set_mem_width(info.global_mem_cacheline_size());
#endif

  // Maximum local memory
  settings->set_max_mem(info.local_mem_size());

  // Maximum register size
  settings->set_max_regs(16 * 1024);

  // Minimum number of work groups to get full utilization, 4 * CU's is an estimate
  settings->set_goal_groups(info.max_compute_units() * 4);

  // Minimum roof, this is a total guess
  settings->set_goal_flops_per_byte(50);

  // Workgroup dimension sizes
  for (auto size : info.work_item_dimension_size()) {
    settings->add_dim_sizes(size);
  }

  // Enable out-of-order execution if the hardware supports it.
  settings->set_is_synchronous(false);

  // Enable the use of mad() calls by default. Users may disable this as an override.
  settings->set_disable_mad(false);

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
