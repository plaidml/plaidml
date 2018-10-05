// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cpu/result.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

namespace pb = google::protobuf;

namespace {
const context::Clock kSystemClock;
}  // namespace

Result::Result(const context::Context& ctx, const char* verb, std::chrono::high_resolution_clock::time_point start,
               std::chrono::high_resolution_clock::time_point end)
    : ctx_{ctx}, verb_{verb}, start_{start}, end_{end} {}

std::chrono::high_resolution_clock::duration Result::GetDuration() const { return end_ - start_; }

void Result::LogStatistics() const {
  pb::Duration start;
  pb::Duration end;
  context::StdDurationToProto(&start, start_.time_since_epoch());
  context::StdDurationToProto(&end, end_.time_since_epoch());
  kSystemClock.LogActivity(ctx_, verb_, start, end);
}

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
