// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>

#include "tile/base/buffer.h"
#include "tile/platform/local_machine/mem_deps.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// An interface for manipulating a particular chunk of memory.
class MemChunk : public Buffer {
 public:
  // Gets the chunk's memory dependency.
  virtual std::shared_ptr<MemDeps> deps() = 0;

  // Gets the chunk's underlying HAL buffer.
  virtual std::shared_ptr<hal::Buffer> hal_buffer() = 0;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
