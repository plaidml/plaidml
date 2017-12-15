// Copyright 2017, Vertex.AI.

#include "tile/platform/local_machine/schedule.h"

namespace vertexai {
namespace tile {
namespace local_machine {

void TmpAlloc::Accept(AllocVisitor* visitor) const { visitor->Visit(*this); }

void TmpAlloc::log(std::ostream& os) const {
  os << "Tmp: " << byte_size << " bytes";
  switch (location) {
    case ON_DEVICE:
      os << " on-device";
      break;
    case ON_HOST:
      os << " on-host";
      break;
  }
}

void ProgramInputAlloc::Accept(AllocVisitor* visitor) const { visitor->Visit(*this); }

void ProgramInputAlloc::log(std::ostream& os) const { os << "ProgramInput: " << name; }

void ProgramOutputAlloc::Accept(AllocVisitor* visitor) const { visitor->Visit(*this); }

void ProgramOutputAlloc::log(std::ostream& os) const { os << "ProgramOutput: " << name; }

void Step::PrintDeps(std::ostream& os) const {
  if (!deps.size()) {
    return;
  }
  os << " deps=[";
  bool first = true;
  for (auto dep : deps) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << 's' << (*dep)->idx;
  }
  os << ']';
}

void RunStep::Accept(StepVisitor* visitor) const { visitor->Visit(*this); }

void RunStep::log(std::ostream& os) const {
  os << "Run: k" << kidx << "(";
  bool first = true;
  for (const auto& input : inputs) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << 'a' << (*input)->idx;
  }
  os << ") -> (";
  first = true;
  for (const auto& output : outputs) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << 'a' << (*output.allocp)->idx;
  }
  os << ')';
  PrintDeps(os);
}

void CopyStep::Accept(StepVisitor* visitor) const { visitor->Visit(*this); }

void CopyStep::log(std::ostream& os) const {
  os << "Copy: a" << (*from)->idx << " -> a" << (*to.allocp)->idx << ", " << byte_count << " bytes";
  PrintDeps(os);
}

void Schedule::Reindex() {
  std::size_t idx = 0;
  for (auto& alloc : allocs) {
    alloc->idx = idx++;
  }
  idx = 0;
  for (auto& step : steps) {
    step->idx = idx++;
  }
}

void Schedule::log(std::ostream& os) const {
  for (const auto& alloc : allocs) {
    os << "  a" << alloc->idx << ": ";
    alloc->log(os);
    os << '\n';
  }
  for (const auto& step : steps) {
    os << "  s" << step->idx << ": ";
    step->log(os);
    os << '\n';
  }
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
