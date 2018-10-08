// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <chrono>

#include "tile/base/hal.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class Result final : public hal::Result {
 public:
  Result(const context::Context& ctx, const char* verb, std::chrono::high_resolution_clock::time_point start,
         std::chrono::high_resolution_clock::time_point end);

  std::chrono::high_resolution_clock::duration GetDuration() const final;
  void LogStatistics() const final;

 private:
  context::Context ctx_;
  const char* verb_;
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::high_resolution_clock::time_point end_;
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
