// Copyright 2017, Vertex.AI.

#pragma once

#include <cstddef>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/context/context.h"
#include "tile/platform/local_machine/buffer.h"
#include "tile/platform/local_machine/mem_chunk.h"
#include "tile/platform/local_machine/program.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// Shim encapsulates a set of modifications to a program schedule and
// the overall state of the system at runtime, adjusting the schedule
// to fit with the current system state and adjusting the current
// system state to take into account the effect of evaluating the
// program (e.g. dealiasing input and output buffers).
class Shim {
 public:
  // Construct the Shim.  This should be done at the start of queueing
  // the program's steps.
  Shim(const context::Context& ctx, const Program* program, std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
       std::map<std::string, std::shared_ptr<tile::Buffer>> outputs);

  // Destroys the Shim, applying its effects to the system state.
  // This should be done after queueing the program's steps.
  ~Shim();

  // Translate an input or output for a step.
  std::shared_ptr<MemChunk> LookupAlloc(std::size_t sidx, AllocPtr alloc) const;

  // Handle execution errors.
  void SetLaunchException(std::exception_ptr ep) const noexcept;

 private:
  struct AliasUpdate {
    std::shared_ptr<Buffer> buffer;
    std::shared_ptr<MemChunk> chunk;
  };

  std::vector<std::shared_ptr<MemChunk>> chunk_infos_;
  std::list<AliasUpdate> updates_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
