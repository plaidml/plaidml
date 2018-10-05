// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/tdep_scheduler.h"

namespace vertexai {
namespace tile {
namespace local_machine {

TransitiveDepScheduler::TransitiveDepScheduler(const std::shared_ptr<Placer>& placer, std::size_t max_in_flight)
    : placer_{placer}, max_in_flight_{max_in_flight} {}

schedule::Schedule TransitiveDepScheduler::BuildSchedule(const tile::proto::Program& program,
                                                         const lang::KernelList& kl) {
  schedule::Schedule schedule = ToScheduleSteps(program, kl);
  AddDataflowDeps(&schedule);
  if (max_in_flight_) {
    AddLinearDeps(&schedule, max_in_flight_);
  }
  placer_->PlaceSchedule(program, &schedule)->Apply();
  return schedule;
}

const char* TransitiveDepScheduler::name() const { return "TransitiveDep"; }

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
