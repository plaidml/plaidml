// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/result.h"

#include <google/protobuf/util/time_util.h>

#include <utility>

#include "base/util/compat.h"
#include "base/util/error.h"
#include "tile/lang/semprinter.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

namespace {

namespace gpu = google::protobuf::util;

void LogActivity(const context::Context& ctx, std::shared_ptr<DeviceState> device_state, const ResultInfo& info) {
  auto queued_dur = gpu::TimeUtil::NanosecondsToDuration(info.queued_time);
  auto submit_dur = gpu::TimeUtil::NanosecondsToDuration(info.submit_time);
  auto start_dur = gpu::TimeUtil::NanosecondsToDuration(info.start_time);
  auto end_dur = gpu::TimeUtil::NanosecondsToDuration(info.end_time);
  device_state->clock().LogActivity(ctx, "tile::hal::opencl::HostQueue", queued_dur, submit_dur);
  device_state->clock().LogActivity(ctx, "tile::hal::opencl::DevQueue", submit_dur, start_dur);
  device_state->clock().LogActivity(ctx, "tile::hal::opencl::Executing", start_dur, end_dur);
}

std::unique_ptr<ResultInfo> MakeResultInfo(const CLObj<cl_event>& event) {
  Err err;

  if (!event) {
    throw error::NotFound("No associated event information");
  }

  auto info = compat::make_unique<ResultInfo>();

  err = clGetEventInfo(event.get(), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(info->status), &info->status, nullptr);
  Err::Check(err, "Unable to get command execution status");

  err = clGetEventProfilingInfo(event.get(), CL_PROFILING_COMMAND_QUEUED, sizeof(info->queued_time), &info->queued_time,
                                nullptr);
  Err::Check(err, "Unable to read profiling info for CL_PROFILING_COMMAND_QUEUED");

  err = clGetEventProfilingInfo(event.get(), CL_PROFILING_COMMAND_SUBMIT, sizeof(info->submit_time), &info->submit_time,
                                nullptr);
  Err::Check(err, "Unable to read profiling info for CL_PROFILING_COMMAND_SUBMIT");

  err = clGetEventProfilingInfo(event.get(), CL_PROFILING_COMMAND_START, sizeof(info->start_time), &info->start_time,
                                nullptr);
  Err::Check(err, "Unable to read profiling info for CL_PROFILING_COMMAND_START");

  err =
      clGetEventProfilingInfo(event.get(), CL_PROFILING_COMMAND_END, sizeof(info->end_time), &info->end_time, nullptr);
  Err::Check(err, "Unable to read profiling info for CL_PROFILING_COMMAND_END");

  info->execution_duration = std::chrono::nanoseconds(info->end_time - info->start_time);

  return info;
}

}  // namespace

Result::Result(const context::Context& ctx, std::shared_ptr<DeviceState> device_state, CLObj<cl_event> event)
    : ctx_{ctx}, device_state_{std::move(device_state)}, event_{std::move(event)} {}

std::chrono::high_resolution_clock::duration Result::GetDuration() const {
  std::call_once(once_, [this]() { info_ = MakeResultInfo(event_); });
  return info_->execution_duration;
}

void Result::LogStatistics() const {
  std::call_once(once_, [this]() { info_ = MakeResultInfo(event_); });
  if (info_->status < 0) {
    Err err(info_->status);
    LOG(ERROR) << "Event " << event_.get() << " failed with: " << err.str();
  } else {
    auto duration = info_->execution_duration.count();
    VLOG(1) << "Result: dur=" << duration;
    LogActivity(ctx_, device_state_, *info_);
  }
}

KernelResult::KernelResult(const context::Context& ctx, std::shared_ptr<DeviceState> device_state,
                           CLObj<cl_event> event, const lang::KernelInfo& ki)
    : ctx_{ctx}, device_state_{std::move(device_state)}, event_{std::move(event)}, ki_(ki) {}

std::chrono::high_resolution_clock::duration KernelResult::GetDuration() const {
  std::call_once(once_, [this]() { info_ = MakeResultInfo(event_); });
  return info_->execution_duration;
}

void KernelResult::LogStatistics() const {
  std::call_once(once_, [this]() { info_ = MakeResultInfo(event_); });
  if (info_->status < 0) {
    Err err(info_->status);
    LOG(ERROR) << "Kernel " << ki_.kname << " failed with: " << err.str();

    sem::Print emit_debug(*ki_.kfunc);
    LOG(ERROR) << "Generic debug kernel:";
    LOG(ERROR) << ki_.comments;
    LOG(ERROR) << emit_debug.str();

    Err::Check(err, "Kernel execution failure");
  } else {
    auto duration = info_->execution_duration.count();
    if (duration == 0) {
      // Prevent division by 0
      duration = 1;
    }
    VLOG(1) << "Ran " << ki_.kname << ": dur=" << duration << " GFL/s=" << ki_.tot_flops / duration
            << " GBP/s=" << ki_.tot_bytes / duration;
    LogActivity(ctx_, device_state_, *info_);
  }
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
