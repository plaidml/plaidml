// Copyright 2017, Vertex.AI.

#include "tile/hal/opencl/result.h"

#include <google/protobuf/util/time_util.h>

#include <utility>

#include "base/util/error.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

namespace gpu = google::protobuf::util;

Result::Result(const context::Context& ctx, std::shared_ptr<DeviceState> device_state, CLObj<cl_event> event)
    : ctx_{ctx}, device_state_{std::move(device_state)}, event_{std::move(event)} {}

std::chrono::high_resolution_clock::duration Result::GetDuration() const {
  cl_ulong start_time = 0;
  cl_ulong end_time = 0;
  Err err;

  if (!event_) {
    throw error::NotFound("No associated timing information");
  }

  err = clGetEventProfilingInfo(event_.get(), CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, nullptr);
  Err::Check(err, "Unable to read profiling info for CL_PROFILING_COMMAND_START");

  err = clGetEventProfilingInfo(event_.get(), CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, nullptr);
  Err::Check(err, "Unable to read profiling info for CL_PROFILING_COMMAND_END");

  return std::chrono::nanoseconds(end_time - start_time);
}

void Result::LogStatistics() const {
  cl_ulong queued_time = 0;
  cl_ulong submit_time = 0;
  cl_ulong start_time = 0;
  cl_ulong end_time = 0;
  Err err;
  std::cout << "WOOF";
  if (!event_) {
    throw error::NotFound("No associated event information");
  }

  err = clGetEventProfilingInfo(event_.get(), CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, nullptr);
  Err::Check(err, "Unable to read profiling info");

  err = clGetEventProfilingInfo(event_.get(), CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, nullptr);
  Err::Check(err, "Unable to read profiling info");

  err = clGetEventProfilingInfo(event_.get(), CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, nullptr);
  Err::Check(err, "Unable to read profiling info");

  err = clGetEventProfilingInfo(event_.get(), CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, nullptr);
  Err::Check(err, "Unable to read profiling info");

  auto queued_dur = gpu::TimeUtil::NanosecondsToDuration(queued_time);
  auto submit_dur = gpu::TimeUtil::NanosecondsToDuration(submit_time);
  auto start_dur = gpu::TimeUtil::NanosecondsToDuration(start_time);
  auto end_dur = gpu::TimeUtil::NanosecondsToDuration(end_time);
  device_state_->clock().LogActivity(ctx_, "tile::hal::opencl::HostQueue", queued_dur, submit_dur);
  device_state_->clock().LogActivity(ctx_, "tile::hal::opencl::DevQueue", submit_dur, start_dur);
  device_state_->clock().LogActivity(ctx_, "tile::hal::opencl::Executing", start_dur, end_dur);
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
