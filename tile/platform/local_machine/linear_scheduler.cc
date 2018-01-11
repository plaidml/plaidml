// Copyright 2017, Vertex.AI.

#include "tile/platform/local_machine/linear_scheduler.h"

namespace vertexai {
namespace tile {
namespace local_machine {

LinearScheduler::LinearScheduler(const std::shared_ptr<Placer>& placer) : placer_{placer} {}

Schedule LinearScheduler::BuildSchedule(const tile::proto::Program& program, const lang::KernelList& kl) {
  Schedule schedule = ToScheduleSteps(program, kl);
  AddLinearDeps(&schedule, 1);
  placer_->PlaceSchedule(&schedule)->Apply();
  return schedule;
}

const char* LinearScheduler::name() const { return "Linear"; }

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
