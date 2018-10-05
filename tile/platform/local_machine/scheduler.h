// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <string>

#include "tile/base/schedule.h"
#include "tile/lang/generate.h"
#include "tile/platform/local_machine/local_machine.pb.h"
#include "tile/proto/hal.pb.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// A scheduler schedules a sequence of kernels for execution.
class Scheduler {
 public:
  virtual ~Scheduler() {}

  virtual schedule::Schedule BuildSchedule(const tile::proto::Program& program, const lang::KernelList& kl) = 0;

  virtual const char* name() const = 0;
};

// Creates a basic schedule with the steps defined by the supplied
// KernelList, but with no dependency information.  The result isn't
// suitable for execution, but makes a useful starting point for other
// scheduler algorithms.  The resulting schedule will have been
// indexed.
schedule::Schedule ToScheduleSteps(const tile::proto::Program& program, const lang::KernelList& kl);

// Adds the schedule's step's data dependencies to the schedule.
void AddDataflowDeps(schedule::Schedule* schedule);

// Adds linear dependencies to a schedule, with the given delta.
void AddLinearDeps(schedule::Schedule* schedule, std::size_t delta);

// Validates a schedule -- i.e. for the supplied kernel list, validate that:
// * All kernels are run exactly once,
// * All kernels have the correct number of outputs and inputs,
// * All steps only have dependencies on earlier steps,
// * Tensor values are produced before they're consumed,
// * Tensor values that overlap spatially do not overlap temporaly,
// * Tensor values fit within their assigned allocations,
// * Tensor values are completely copied,
// * Input buffers are read-only,
// * All program outputs are written, and
// * All output buffers end up with the correct program output values.
//
// If any of these conditions do not hold, an error::Internal exception is raised.
void ValidateSchedule(const tile::proto::Program& program, const lang::KernelList& kl,
                      const schedule::Schedule& schedule);

// Writes information about a schedule to the debug log, and
// optionally updates a CompilationInfo proto's tmp_sizes and
// alloc_sizes fields based on the schedule.
void SummarizeSchedule(hal::proto::CompilationInfo* cinfo, const tile::proto::Program& program,
                       const lang::KernelList& kl, const schedule::Schedule& schedule);

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
