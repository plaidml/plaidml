// Copyright 2017, Vertex.AI.

#include "tile/platform/local_machine/shim.h"

#include <unordered_set>

#include "base/util/error.h"
#include "tile/platform/local_machine/buffer.h"

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

// Builds a memory allocation map for a particular program run.
class ChunkMap final : private AllocVisitor {
 public:
  static std::vector<std::shared_ptr<MemChunk>> Build(
      const context::Context& ctx, const Program* program,
      const std::map<std::string, std::shared_ptr<tile::Buffer>>& inputs,
      const std::map<std::string, std::shared_ptr<tile::Buffer>>& outputs) {
    ChunkMap chunk_map{ctx, program, &inputs, &outputs};
    chunk_map.chunk_infos_.reserve(program->schedule().allocs.size());
    for (const auto& alloc : program->schedule().allocs) {
      alloc->Accept(&chunk_map);
    }
    return chunk_map.chunk_infos_;
  }

 private:
  ChunkMap(const context::Context& ctx, const Program* program,
           const std::map<std::string, std::shared_ptr<tile::Buffer>>* inputs,
           const std::map<std::string, std::shared_ptr<tile::Buffer>>* outputs)
      : ctx_{ctx}, program_{program}, inputs_{inputs}, outputs_{outputs} {}

  void Visit(const TmpAlloc& tmp_alloc) final {
    chunk_infos_.emplace_back(program_->tmp_mem_strategy()->MakeChunk(ctx_, tmp_alloc.byte_size));
  }

  void Visit(const ProgramInputAlloc& input_alloc) final {
    auto it = inputs_->find(input_alloc.name);
    if (it == inputs_->end()) {
      throw error::NotFound{"Missing program input: " + input_alloc.name};
    }
    std::shared_ptr<MemChunk> chunk = Buffer::Downcast(it->second, program_->devinfo())->chunk();
    chunk_infos_.emplace_back(std::move(chunk));
  }

  void Visit(const ProgramOutputAlloc& output_alloc) final {
    auto it = outputs_->find(output_alloc.name);
    if (it == outputs_->end()) {
      throw error::NotFound{"Missing program output: " + output_alloc.name};
    }
    std::shared_ptr<MemChunk> chunk = Buffer::Downcast(it->second, program_->devinfo())->chunk();
    chunk_infos_.emplace_back(std::move(chunk));
  }

  context::Context ctx_;
  const Program* program_;
  const std::map<std::string, std::shared_ptr<tile::Buffer>>* inputs_;
  const std::map<std::string, std::shared_ptr<tile::Buffer>>* outputs_;
  std::vector<std::shared_ptr<MemChunk>> chunk_infos_;
};

}  // namespace

Shim::Shim(const context::Context& ctx, const Program* program,
           std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
           std::map<std::string, std::shared_ptr<tile::Buffer>> outputs) {
  // Dealias all aliased outputs, and remember the remapping so that
  // we can correctly update the original buffers.
  std::unordered_set<tile::Buffer*> input_buffers;
  for (const auto& kvp : inputs) {
    std::shared_ptr<Buffer> buffer = Buffer::Downcast(kvp.second, program->devinfo());
    input_buffers.insert(buffer.get());
  }

  for (auto& kvp : outputs) {
    std::shared_ptr<Buffer> buffer = Buffer::Downcast(kvp.second, program->devinfo());
    if (input_buffers.count(buffer.get())) {
      auto chunk = program->output_mem_strategy()->MakeChunk(ctx, buffer->size());
      kvp.second = std::make_shared<Buffer>(program->devinfo(), chunk);
      updates_.emplace_back(AliasUpdate{std::move(buffer), std::move(chunk)});
    }
  }

  // Build the final chunk information map.
  chunk_infos_ = ChunkMap::Build(ctx, program, inputs, outputs);
}

Shim::~Shim() {
  // Apply dealiasing to inputs (which still point to the original buffers).
  for (const auto& update : updates_) {
    update.buffer->RemapTo(std::move(update.chunk));
  }
}

std::shared_ptr<MemChunk> Shim::LookupAlloc(std::size_t /* sidx */, AllocPtr alloc) const {
  return chunk_infos_[(*alloc)->idx];
}

void Shim::SetLaunchException(std::exception_ptr ep) const noexcept {
  // Any error in the launch poisons all buffers.
  for (const auto& chunk : chunk_infos_) {
    chunk->deps()->Poison(ep);
  }
}

}  // namespace local_machine
}  // namespace tile

}  // namespace vertexai
