// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/shim.h"

#include <unordered_set>
#include <utility>

#include "base/util/error.h"
#include "tile/platform/local_machine/buffer.h"

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

// Builds a memory allocation map for a particular program run.
std::pair<std::vector<std::shared_ptr<MemChunk>>, std::list<Shim::AliasUpdate>> BuildChunkMap(
    const context::Context& ctx, const Program* program,
    const std::map<std::string, std::shared_ptr<tile::Buffer>>& inputs,
    const std::map<std::string, std::shared_ptr<tile::Buffer>>& outputs) {
  std::vector<std::shared_ptr<MemChunk>> chunk_infos;
  std::list<Shim::AliasUpdate> updates;
  chunk_infos.reserve(program->schedule().allocs.size());
  for (const auto& alloc : program->schedule().allocs) {
    std::shared_ptr<MemChunk> chunk;
    if (alloc.is_input()) {
      // This is a program input.  If the input has a chunk, we have to use it --
      // by definition, this is the input data to the program.  Note that the input
      // might not have a chunk; this is unusual, but it's technically allowed;
      // this can be useful when a caller's just testing to see whether it's correctly
      // composed a Tile program.
      auto iit = inputs.find(alloc.input);
      if (iit == inputs.end()) {
        throw error::NotFound{"Missing program input: " + alloc.input};
      }
      std::shared_ptr<Buffer> input_buffer = Buffer::Downcast(iit->second, program->devinfo());
      input_buffer->EnsureChunk(ctx);
      chunk = input_buffer->chunk();

      if (alloc.is_output()) {
        // The chunk is also being used as a program output; the corresponding output buffer
        // must wind up pointing to this chunk if launch succeeds, regardless of whether it
        // already has a chunk.
        auto oit = outputs.find(alloc.output);
        if (oit == outputs.end()) {
          throw error::NotFound{"Missing program output: " + alloc.output};
        }
        std::shared_ptr<Buffer> output_buffer = Buffer::Downcast(oit->second, program->devinfo());
        updates.emplace_back(Shim::AliasUpdate{std::move(output_buffer), chunk});
      }
    } else if (alloc.is_output()) {
      // This is a program output, but not a program input.  So we'll be creating a new chunk
      // for it here -- typically, the output buffer will not already have a chunk, but if it does,
      // it's okay to go ahead and replace it iff launch succeeds.
      auto oit = outputs.find(alloc.output);
      if (oit == outputs.end()) {
        throw error::NotFound{"Missing program output: " + alloc.output};
      }
      std::shared_ptr<Buffer> output_buffer = Buffer::Downcast(oit->second, program->devinfo());
      chunk = program->output_mem_strategy()->MakeChunk(ctx, output_buffer->size());
      updates.emplace_back(Shim::AliasUpdate{std::move(output_buffer), chunk});
    } else {
      // This is neither a program input nor a program output; the alloc is purely internal
      // to the program.  Make a temporary buffer for it.
      chunk = program->tmp_mem_strategy()->MakeChunk(ctx, alloc.byte_size);
    }

    chunk_infos.emplace_back(std::move(chunk));
  }
  return std::make_pair(std::move(chunk_infos), std::move(updates));
}

}  // namespace

Shim::Shim(                                   //
    const context::Context& ctx,              //
    const std::shared_ptr<Program>& program,  //
    std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
    std::map<std::string, std::shared_ptr<tile::Buffer>> outputs)
    : program_{program} {
  std::tie(chunk_infos_, updates_) = BuildChunkMap(ctx, program.get(), inputs, outputs);
}

Shim::~Shim() {
  // Release the resource manually first
  // Then tell program that the resource is released
  chunk_infos_.clear();
  updates_.clear();
  program_->Release();
}

std::shared_ptr<MemChunk> Shim::LookupAlloc(std::size_t /* sidx */, schedule::Alloc* alloc) const {
  return chunk_infos_[alloc->idx];
}

void Shim::SetLaunchException(std::exception_ptr ep) const noexcept {
  // Any error in the launch poisons all output buffers.
  for (const auto& chunk : chunk_infos_) {
    chunk->deps()->Poison(ep);
  }
}

void Shim::OnLaunchSuccess() noexcept {
  // Apply updates to outputs.
  for (const auto& update : updates_) {
    update.buffer->RemapTo(std::move(update.chunk));
  }
}

}  // namespace local_machine
}  // namespace tile

}  // namespace vertexai
