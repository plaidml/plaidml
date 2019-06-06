// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/library.h"

#include <utility>

#include "base/util/error.h"
#include "tile/hal/opencl/ocl.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

Library* Library::Downcast(hal::Library* library, const std::shared_ptr<DeviceState>& device_state) {
  Library* exe = dynamic_cast<Library*>(library);
  if (!exe || exe->device_state() != device_state) {
    throw error::InvalidArgument{"Incompatible library for Tile device"};
  }
  return exe;
}

Library::Library(const std::shared_ptr<DeviceState>& device_state, const std::map<std::string, CLObj<cl_program>>& program,
                 const std::vector<lang::KernelInfo>& kernel_info, std::vector<context::proto::ActivityID> kernel_ids)
    : device_state_{device_state},
      program_{std::move(program)},
      kernel_info_{kernel_info},
      kernel_ids_{std::move(kernel_ids)} {}

std::map<std::string, std::string> Library::Serialize() {
  std::map<std::string, std::string> result;
  for (auto& prog_it : program_) {
    std::size_t size;
    Err::Check(ocl::GetProgramInfo(prog_it.second.get(), CL_PROGRAM_BINARY_SIZES, sizeof(size), &size, nullptr),
               "Unable to compute binary size for " + prog_it.first);
    std::string prog_str;
    prog_str.resize(size);
    const char* datum = prog_str.data();
    Err::Check(ocl::GetProgramInfo(prog_it.second.get(), CL_PROGRAM_BINARIES, sizeof(datum), &datum, nullptr),
               "Unable to serialize binary for " + prog_it.first);
    result.emplace(prog_it.first, prog_str);
  }
  return result;
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
