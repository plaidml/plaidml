// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/opencl/ocl.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

// Represents a chunk of OpenCL memory.
class Buffer : public hal::Buffer {
 public:
  // Casts a hal::Buffer to a Buffer, throwing an exception if the supplied hal::Buffer isn't an OpenCL buffer, or if
  // it's a buffer for a different context.
  static std::shared_ptr<Buffer> Downcast(const std::shared_ptr<hal::Buffer>& buffer, const CLObj<cl_context>& cl_ctx);
  static Buffer* Downcast(hal::Buffer* buffer, const CLObj<cl_context>& cl_ctx);

  virtual void SetKernelArg(const CLObj<cl_kernel>& kernel, std::size_t index) = 0;

  virtual void* base() const { return nullptr; }
  virtual cl_mem mem() const { return nullptr; }
  std::uint64_t size() const { return size_; }

 protected:
  explicit Buffer(const CLObj<cl_context>& cl_ctx, std::uint64_t size);

 private:
  CLObj<cl_context> cl_ctx_;
  const std::uint64_t size_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
