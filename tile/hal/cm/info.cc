// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/info.h"

#include <google/protobuf/text_format.h>

#include <algorithm>
#include <boost/format.hpp>
#include <boost/regex.hpp>

#include "base/util/logging.h"
#include "tile/hal/cm/cm.pb.h"
#include "tile/proto/hal.pb.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

namespace gp = ::google::protobuf;

hal::proto::HardwareInfo GetHardwareInfo(const proto::DeviceInfo& info) {
  hal::proto::HardwareInfo result;
  result.set_type(info.type());
  auto vendor = info.vendor();
  if (vendor.find("NVIDIA") != std::string::npos) {
    vendor = "NVIDIA";
  } else if (vendor.find("Intel") != std::string::npos) {
    vendor = "Intel";
  } else if (vendor.find("Advanced Micro Devices") != std::string::npos) {
    vendor = "AMD";
  }
  if (info.name() == "CPU") {
    result.set_name(std::string("Not Supported"));
  } else {
    auto name = info.name();
    if (name.find("Intel(R) ") == 0) {
      name = name.substr(9);
    } else if (name.find("AMD ") == 0) {
      name = name.substr(4);
    }
    if (vendor.size() > 0) {
      result.set_name(str(boost::format("CM %1% %2%") % vendor % name));
    } else {
      result.set_name(str(boost::format("CM %1%") % name));
    }
  }
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
  settings->set_mem_width(info.global_mem_cacheline_size());

  // Maximum local memory
  settings->set_max_mem(info.local_mem_size());

  // Maximum register size
  settings->set_max_regs(16 * 1024);

  // Minimum number of work groups to get full utilization, 4 * CU's is an
  // estimate
  settings->set_goal_groups(info.max_compute_units() * 4);

  // Minimum roof, this is a total guess
  settings->set_goal_flops_per_byte(50);

  // Workgroup dimension sizes
  for (auto size : info.work_item_dimension_size()) {
    settings->add_dim_sizes(size);
  }

  // Enable out-of-order execution if the hardware supports it.
  settings->set_is_synchronous(false);

  // Enable the use of mad() calls by default. Users may disable this as an
  // override.
  settings->set_disable_mad(false);

  // Enable input/output buffer aliasing by default.  This may be overridden.
  settings->set_disable_io_aliasing(false);

  return result;
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
