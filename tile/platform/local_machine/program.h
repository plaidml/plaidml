// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "tile/base/buffer.h"
#include "tile/base/program.h"
#include "tile/base/schedule.h"
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
  const schedule::Schedule& schedule() const { return schedule_; }
  const lang::KernelList& kernel_list() const { return kernel_list_; }
  const std::unique_ptr<hal::Executable>& executable() const { return executable_; }

 private:
  std::shared_ptr<DevInfo> devinfo_;
  std::shared_ptr<MemStrategy> output_mem_strategy_;
  std::shared_ptr<MemStrategy> tmp_mem_strategy_;
  lang::KernelList kernel_list_;
  schedule::Schedule schedule_;
  std::unique_ptr<hal::Executable> executable_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
