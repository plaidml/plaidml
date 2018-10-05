// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <exception>
#include <list>
#include <memory>
#include <mutex>
#include <vector>

#include "tile/base/hal.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// Records the liveness of a particular chunk of memory -- the set of events that need to reach a final state in
// order to know whether the memory's contents to be valid, and whether the memory's contents are valid at all.
class MemDeps final : public std::enable_shared_from_this<MemDeps> {
 public:
  // Get the current read dependencies, adding them to the supplied vector.  This will raise an exception if the
  // MemDeps has been poisoned.
  void GetReadDependencies(std::vector<std::shared_ptr<hal::Event>>* deps);

  // Adds to this memory chunk's read dependency, blocking read operations from taking place until the supplied event
  // has reached a final state.  This also clears the poison state of the MemDeps.
  void AddReadDependency(std::shared_ptr<hal::Event> event);

  // Poisons the memory chunk, if not already poisoned.  This causes the indicated exception to be thrown from
  // subsequent calls to Capture.
  void Poison(std::exception_ptr ep) noexcept;

 private:
  std::mutex mu_;
  std::list<std::shared_ptr<hal::Event>> events_;
  std::exception_ptr ep_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
