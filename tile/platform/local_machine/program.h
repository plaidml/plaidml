// Copyright 2017, Vertex.AI.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tile/base/buffer.h"
#include "tile/base/program.h"
#include "tile/platform/local_machine/devinfo.h"
#include "tile/platform/local_machine/mem_strategy.h"
#include "tile/proto/tile.pb.h"

namespace vertexai {
namespace tile {
namespace local_machine {

class Program final : public tile::Program {
 public:
  enum class KernelParamType { kInput, kOutput, kTmpInput, kTmpOutput, kSynthetic };

  struct KernelParam {
    KernelParam(KernelParamType _ty, std::string _name, std::size_t _tidx = 0) : ty{_ty}, name{_name}, tidx{_tidx} {}

    KernelParamType ty;
    std::string name;      // Used for inputs and outputs
    std::size_t tidx = 0;  // Used for temporaries.
  };

  struct BoundKernel {
    std::unique_ptr<hal::Kernel> kernel;
    std::vector<KernelParam> params;
    lang::KernelInfo info;
  };

  Program(const context::Context& ctx, const tile::proto::Program& program, const std::shared_ptr<DevInfo>& devinfo,
          const std::shared_ptr<MemStrategy>& output_mem_strategy, const std::shared_ptr<MemStrategy>& tmp_mem_strategy,
          hal::Memory* tmp_memory);

  boost::future<void> Run(const context::Context& ctx, std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
                          std::map<std::string, std::shared_ptr<tile::Buffer>> outputs) final;

  const std::shared_ptr<DevInfo>& devinfo() const { return devinfo_; }
  const std::shared_ptr<MemStrategy>& output_mem_strategy() const { return output_mem_strategy_; }
  const std::shared_ptr<MemStrategy>& tmp_mem_strategy() const { return tmp_mem_strategy_; }
  const std::vector<BoundKernel>& kernels() const { return kernels_; }
  const std::unordered_map<std::string, std::size_t>& last_input_use() const { return last_input_use_; }
  const std::vector<std::size_t>& tmp_locs() const { return tmp_locs_; }
  const std::vector<std::size_t>& alloc_sizes() const { return alloc_sizes_; }

 private:
  struct TmpInfo {
    std::size_t first_writer_kidx = 0;
    std::size_t last_writer_kidx = 0;
    std::size_t size = 0;
  };

  // Compiles kernel semantic trees down to executable code, and loads
  // the code onto the underlying hardware, initializing the kernels_
  // vector.
  void LoadKernels(const context::Context& ctx, std::vector<lang::KernelInfo> kernel_infos);

  // Fills in the params fields of the kernels in the kernels_ vector
  // (i.e. translating the string temporary names to unique integers),
  // and returns a vector containing information about each temporary
  // allocation.
  //
  // Additionally, this fills in the last_input_use_ map, which is
  // used at runtime to dealias buffers that are used as both inputs
  // and outputs.  (It's legal to use a single buffer as both an input
  // and an output to a program, but if this creates a RAW dependency,
  // the implementation needs to allocate a new HAL buffer for the
  // output memory and remap the output buffer to that HAL buffer.)
  std::vector<TmpInfo> AllocTemporaries(const tile::proto::Program& program, const lang::ShapeMap& shape_map);

  // Adds synthetic dependencies between all kernels.  This is useful
  // when the underlying device queue is synchronous, as it maximizes
  // device memory reuse and removes explicit inter-kernel
  // synchronization (which has some overhead).
  void AddInterKernelDeps(size_t max_in_flight);

  // Schedules the temporary buffers used by the program: i.e. which
  // allocations need to be made in order to run the program, the size
  // of each allocation, and the assignment from each temporary to its
  // allocation.
  //
  // The effect of this is to initialize tmp_locs_ and alloc_sizes_.
  void ScheduleTemporaries(std::vector<TmpInfo> tmps);

  // Logs the temporary buffer assignments.
  void LogTemporaries(hal::proto::CompilationInfo* cinfo);

  // Validates the temporary buffer assignments -- i.e. that
  // temporaries that overlap spatially do not overlap temporaly.
  // This exists because scheduling temporaries is tricky, and it's
  // nice to build confidence in the resulting assignments.
  void ValidateTemporaries();

  std::shared_ptr<DevInfo> devinfo_;
  std::shared_ptr<MemStrategy> output_mem_strategy_;
  std::shared_ptr<MemStrategy> tmp_mem_strategy_;
  std::vector<BoundKernel> kernels_;
  std::unordered_map<std::string, std::size_t> last_input_use_;
  std::vector<std::size_t> tmp_locs_;
  std::vector<std::size_t> alloc_sizes_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
