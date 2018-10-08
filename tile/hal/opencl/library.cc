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

Library::Library(const std::shared_ptr<DeviceState>& device_state, CLObj<cl_program> program,
                 const std::vector<lang::KernelInfo>& kernel_info, std::vector<context::proto::ActivityID> kernel_ids)
    : device_state_{device_state},
      program_{std::move(program)},
      kernel_info_{kernel_info},
      kernel_ids_{std::move(kernel_ids)} {}

std::string Library::Serialize() {
  std::size_t size;
  Err::Check(clGetProgramInfo(program_.get(), CL_PROGRAM_BINARY_SIZES, sizeof(size), &size, nullptr),
             "Unable to compute binary size");
  std::string result;
  result.resize(size);
  const char* datum = result.data();
  Err::Check(clGetProgramInfo(program_.get(), CL_PROGRAM_BINARIES, sizeof(datum), &datum, nullptr),
             "Unable to serialize binary");
  return result;
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
