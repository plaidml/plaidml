// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/buffer.h"

#include <memory>

#include "base/util/error.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

std::shared_ptr<Buffer> Buffer::Downcast(const std::shared_ptr<hal::Buffer>& buffer, const CLObj<cl_context>& cl_ctx) {
  std::shared_ptr<Buffer> buf = std::dynamic_pointer_cast<Buffer>(buffer);
  if (!buf || buf->cl_ctx_ != cl_ctx) {
    throw error::InvalidArgument{"Incompatible buffer for Tile device"};
  }
  return buf;
}

Buffer* Buffer::Downcast(hal::Buffer* buffer, const CLObj<cl_context>& cl_ctx) {
  Buffer* buf = dynamic_cast<Buffer*>(buffer);
  if (!buf || buf->cl_ctx_ != cl_ctx) {
    throw error::InvalidArgument{"Incompatible buffer for Tile device"};
  }
  return buf;
}

Buffer::Buffer(const CLObj<cl_context>& cl_ctx, std::uint64_t size) : cl_ctx_{cl_ctx}, size_{size} {}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
