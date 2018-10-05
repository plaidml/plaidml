// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "tile/base/hal.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class Arena : public hal::Arena, public std::enable_shared_from_this<Arena> {
 public:
  explicit Arena(std::uint64_t size);

  std::shared_ptr<hal::Buffer> MakeBuffer(std::uint64_t offset, std::uint64_t size) final;

 private:
  std::vector<char> mem_;
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
