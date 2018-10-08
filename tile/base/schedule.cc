// Copyright 2017-2018 Intel Corporation.

#include "tile/base/schedule.h"

#include "base/util/error.h"

namespace vertexai {
namespace tile {
namespace schedule {

void Alloc::Log(el::base::type::ostream_t& os) const {
  if (is_input()) {
    os << "Input: " << input << ' ';
  }
  if (is_output()) {
    os << "Output: " << output << ' ';
  }
  if (is_tmp()) {
    os << "Tmp: ";
  }
  os << byte_size << " bytes";
  if (safe_self_alias_allocs.size()) {
    os << " May-alias:";
    for (const auto& ap : safe_self_alias_allocs) {
      os << " a" << ap->idx;
    }
  }
}

void Step::Log(el::base::type::ostream_t& os) const {
  switch (tag) {
    case Tag::kRun:
      os << "Run: k" << kidx;
      break;
    case Tag::kCopy:
      os << "Copy(" << byte_count << ')';
      break;
    default:
      os << "<InvalidStep>";
  }
  os << " (";
  bool first = true;
  for (const auto& input : inputs) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << 'a' << input->idx;
  }
  os << ") -> (";
  first = true;
  for (const auto& output : outputs) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << 'a' << output.allocp->idx;
  }
  os << ')';
  if (deps.size()) {
    os << " deps=[";
    bool first = true;
    for (auto dep : deps) {
      if (!first) {
        os << ", ";
      }
      first = false;
      os << 's' << dep->idx;
    }
    os << ']';
  }
}

void Schedule::Reindex() {
  std::size_t idx = 0;
  for (auto& alloc : allocs) {
    alloc.idx = idx++;
  }
  idx = 0;
  for (auto& step : steps) {
    step.idx = idx++;
  }
}

void Schedule::Log(el::base::type::ostream_t& os) const {
  for (const auto& alloc : allocs) {
    os << "  a" << alloc.idx << ": ";
    alloc.Log(os);
    os << '\n';
  }
  for (const auto& step : steps) {
    os << "  s" << step.idx << ": ";
    step.Log(os);
    os << '\n';
  }
}

void ScheduleToProto(proto::Schedule* pb, const Schedule& schedule) {
  for (const auto& alloc : schedule.allocs) {
    auto* apb = pb->add_allocs();
    apb->set_size(alloc.byte_size);
    apb->set_input(alloc.input);
    apb->set_output(alloc.output);
  }

  std::size_t sidx = 0;
  for (const auto& step : schedule.steps) {
    auto* spb = pb->add_steps();
    for (auto dep : step.deps) {
      spb->add_deps(dep->idx);
    }
    switch (step.tag) {
      case Step::Tag::kRun: {
        auto* run_pb = spb->mutable_run();
        run_pb->set_kidx(step.kidx);
        for (const auto& output : step.outputs) {
          run_pb->add_output_aidxs(output.allocp->idx);
        }
        for (const auto& input : step.inputs) {
          run_pb->add_input_aidxs(input->idx);
        }
        break;
      }
      case Step::Tag::kCopy: {
        auto* copy_pb = spb->mutable_copy();
        if (step.outputs.size() != 1) {
          throw error::Internal{
              "In schedule proto construction: copy step " + std::to_string(sidx) +
              " has an incorrect output count; expected: 1, got: " + std::to_string(step.outputs.size())};
        }
        if (step.inputs.size() != 1) {
          throw error::Internal{
              "In schedule proto construction: copy step " + std::to_string(sidx) +
              " has an incorrect input count; expected: 1, got: " + std::to_string(step.inputs.size())};
        }
        copy_pb->set_from_aidx(step.inputs[0]->idx);
        copy_pb->set_to_aidx(step.outputs[0].allocp->idx);
        copy_pb->set_count_bytes(step.byte_count);
        break;
      }
      default:
        throw error::Internal{"In schedule proto construction: step " + std::to_string(sidx) + " has an invalid tag"};
    }
    sidx++;
  }
}

}  // namespace schedule
}  // namespace tile
}  // namespace vertexai
