// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "base/util/logging.h"
#include "tile/proto/schedule.pb.h"

namespace vertexai {
namespace tile {
namespace schedule {

// Describes a memory allocation for use by a program.
struct Alloc {
  bool is_input() const { return input.length() > 0; }
  bool is_output() const { return output.length() > 0; }
  bool is_tmp() const { return !is_input() && !is_output(); }

  void Log(el::base::type::ostream_t& os) const;  // NOLINT

  std::size_t idx = 0;
  std::uint64_t byte_size = 0;
  std::set<Alloc*> safe_self_alias_allocs;
  std::string input;   // If non-empty, this is a program input.
  std::string output;  // If non-empty, this is a program output.
};

inline MAKE_LOGGABLE(Alloc, alloc, os) {
  alloc.Log(os);
  return os;
}

struct OutputInfo {
  Alloc* allocp;
  bool add_dep;
};

// A particular step to take in evaluating a program.
struct Step {
  enum class Tag { kRun, kCopy };

  explicit Step(Tag tag_) : tag{tag_} {}

  void Log(el::base::type::ostream_t& os) const;  // NOLINT

  Tag tag;
  std::size_t idx = 0;
  std::set<Step*> deps;
  std::vector<OutputInfo> outputs;
  std::vector<Alloc*> inputs;

  std::size_t kidx = 0;          // Used for run steps
  std::uint64_t byte_count = 0;  // Used for copy steps
};

inline MAKE_LOGGABLE(Step, step, os) {
  step.Log(os);
  return os;
}

// An execution schedule, describing how to run a particular program.
struct Schedule {
  void Reindex();
  void Log(el::base::type::ostream_t& os) const;  // NOLINT

  std::list<Alloc> allocs;
  std::list<Step> steps;
};

inline MAKE_LOGGABLE(Schedule, schedule, os) {
  schedule.Log(os);
  return os;
}

// Serializes a schedule to a protocol buffer.
void ScheduleToProto(proto::Schedule* pb, const Schedule& schedule);

}  // namespace schedule
}  // namespace tile
}  // namespace vertexai
