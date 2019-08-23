// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/shim.h"

#include <unordered_set>

#include "base/util/error.h"
#include "tile/platform/local_machine/buffer.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// Builds a memory allocation map for a particular program run.
void Shim::BuildChunkMap(const std::map<std::string, std::shared_ptr<tile::Buffer>>& inputs,
                         const std::map<std::string, std::shared_ptr<tile::Buffer>>& outputs) {
  chunk_infos_.reserve(program_->schedule().allocs.size());
  size_t user_size = 0;
  size_t tmp_size = 0;
  for (const auto& alloc : program_->schedule().allocs) {
    std::shared_ptr<MemChunk> chunk = {nullptr};
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
      std::shared_ptr<Buffer> input_buffer = Buffer::Downcast(iit->second, program_->devinfo());
      input_buffer->EnsureChunk(ctx_);
      chunk = input_buffer->chunk();

      if (alloc.is_output()) {
        // The chunk is also being used as a program output; the corresponding output buffer
        // must wind up pointing to this chunk if launch succeeds, regardless of whether it
        // already has a chunk.
        auto oit = outputs.find(alloc.output);
        if (oit == outputs.end()) {
          throw error::NotFound{"Missing program output: " + alloc.output};
        }
        std::shared_ptr<Buffer> output_buffer = Buffer::Downcast(oit->second, program_->devinfo());
        updates_.emplace_back(Shim::AliasUpdate{std::move(output_buffer), chunk});
      }
      user_size += alloc.byte_size;
    } else if (alloc.is_output()) {
      // This is a program output, but not a program input.  So we'll be creating a new chunk
      // for it here -- typically, the output buffer will not already have a chunk, but if it does,
      // it's okay to go ahead and replace it iff launch succeeds.
      auto oit = outputs.find(alloc.output);
      if (oit == outputs.end()) {
        throw error::NotFound{"Missing program output: " + alloc.output};
      }
      std::shared_ptr<Buffer> output_buffer = Buffer::Downcast(oit->second, program_->devinfo());
      chunk = program_->output_mem_strategy()->MakeChunk(ctx_, output_buffer->size());
      updates_.emplace_back(Shim::AliasUpdate{std::move(output_buffer), chunk});
      user_size += alloc.byte_size;
    } else {
      // We do not allocate memory for temp buffers now. We decide if they should be pre-allocated laster.
      tmp_size += alloc.byte_size;
    }
    chunk_infos_.emplace_back(std::move(chunk));
  }

  pre_alloc_ = (user_size + tmp_size <= program_->devinfo()->settings.max_global_mem());
  if (pre_alloc_) {
    // We can pre-allocate the tmp buffers right now
    for (auto& alloc : program_->schedule().allocs) {
      if (alloc.is_tmp()) {
        AllocateChunk(const_cast<schedule::Alloc*>(&alloc));
      }
    }
  }
}

Shim::Shim(const context::Context& ctx, const Program* program,
           std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
           std::map<std::string, std::shared_ptr<tile::Buffer>> outputs) :
           ctx_{ctx}, program_{program} {
  // Count the uses of the allocs
  alloc_uses_.clear();
  for (const auto& step : program->schedule().steps) {
    for (const auto& alloc : step.inputs) {
      const auto& it = alloc_uses_.find(alloc->idx);
      if (it == alloc_uses_.end()) {
        alloc_uses_[alloc->idx] = 1;
      }
      else {
        ++it->second;
      }
    }
    for (const auto& oi : step.outputs) {
      auto alloc = oi.allocp;
      const auto& it = alloc_uses_.find(alloc->idx);
      if (it == alloc_uses_.end()) {
        alloc_uses_[alloc->idx] = 1;
      }
      else {
        ++it->second;
      }
    }
  }
  BuildChunkMap(inputs, outputs);
}

std::shared_ptr<MemChunk> Shim::LookupAlloc(std::size_t /* sidx */, schedule::Alloc* alloc) const {
  return chunk_infos_[alloc->idx];
}

std::shared_ptr<MemChunk> Shim::AllocateChunk(schedule::Alloc* alloc) {
  chunk_infos_[alloc->idx] = program_->tmp_mem_strategy()->MakeChunk(ctx_, alloc->byte_size, alloc_uses_.at(alloc->idx));
  return chunk_infos_[alloc->idx];
}

void Shim::FreeChunk(size_t idx) {
  chunk_infos_[idx] = nullptr;
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
