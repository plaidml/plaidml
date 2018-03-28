// Copyright 2017, Vertex.AI.

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "tile/base/buffer.h"
#include "tile/base/program.h"
#include "tile/platform/local_machine/devinfo.h"
#include "tile/platform/local_machine/mem_strategy.h"
#include "tile/platform/local_machine/scheduler.h"
#include "tile/proto/tile.pb.h"

namespace vertexai {
namespace tile {
namespace local_machine {

class Program final : public tile::Program {
 public:
  Program(const context::Context& ctx, const tile::proto::Program& program, const std::shared_ptr<DevInfo>& devinfo,
          const std::shared_ptr<Scheduler>& scheduler, const std::shared_ptr<MemStrategy>& output_mem_strategy,
          const std::shared_ptr<MemStrategy>& tmp_mem_strategy, hal::Memory* tmp_memory,
          const lang::TileOptimizer& optimizer);

  boost::future<void> Run(const context::Context& ctx, std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
                          std::map<std::string, std::shared_ptr<tile::Buffer>> outputs) final;

  const std::shared_ptr<DevInfo>& devinfo() const { return devinfo_; }
  const std::shared_ptr<MemStrategy>& output_mem_strategy() const { return output_mem_strategy_; }
  const std::shared_ptr<MemStrategy>& tmp_mem_strategy() const { return tmp_mem_strategy_; }
  const Schedule& schedule() const { return schedule_; }
  const lang::KernelList& kernel_list() const { return kernel_list_; }
  const std::vector<std::unique_ptr<hal::Kernel>>& kernels() const { return kernels_; }

 private:
  std::shared_ptr<DevInfo> devinfo_;
  std::shared_ptr<MemStrategy> output_mem_strategy_;
  std::shared_ptr<MemStrategy> tmp_mem_strategy_;
  lang::KernelList kernel_list_;
  Schedule schedule_;
  std::vector<std::unique_ptr<hal::Kernel>> kernels_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
