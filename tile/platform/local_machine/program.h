// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <condition_variable>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>

#include "tile/base/buffer.h"
#include "tile/base/program.h"
#include "tile/base/schedule.h"
#include "tile/lang/runinfo.h"
#include "tile/platform/local_machine/devinfo.h"
#include "tile/platform/local_machine/mem_strategy.h"
#include "tile/platform/local_machine/scheduler.h"
#include "tile/proto/tile.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// The percentage of the device's memory that programs will try to use.
// This value seems to work pretty well on most devices.
// TODO: Either autotune this, or move it to the per-device configuration.
constexpr float kGoalMemPercentage = .85;

class Program final : public tile::Program, public std::enable_shared_from_this<Program> {
 public:
  Program(const context::Context& ctx, const tile::proto::Program& program, const std::shared_ptr<DevInfo>& devinfo,
          const std::shared_ptr<Scheduler>& scheduler, const std::shared_ptr<MemStrategy>& output_mem_strategy,
          const std::shared_ptr<MemStrategy>& tmp_mem_strategy, hal::Memory* tmp_memory,
          const lang::TileOptimizer& optimizer, ConstBufferManager* const_bufs);

  Program(const context::Context& ctx,                              //
          const std::shared_ptr<stripe::Program>& stripe,           //
          const std::string& target_id,                             //
          const std::shared_ptr<DevInfo>& devinfo,                  //
          const std::shared_ptr<Scheduler>& scheduler,              //
          const std::shared_ptr<MemStrategy>& output_mem_strategy,  //
          const std::shared_ptr<MemStrategy>& tmp_mem_strategy,     //
          hal::Memory* tmp_memory,                                  //
          ConstBufferManager* const_bufs);

  boost::future<void> Run(          //
      const context::Context& ctx,  //
      std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
      std::map<std::string, std::shared_ptr<tile::Buffer>> outputs) final;

  // The maximum available memory
  std::size_t MaxAvailableMemory() final;

  // Release resource used by the program
  void Release() final;

  const std::shared_ptr<DevInfo>& devinfo() const { return devinfo_; }
  const std::shared_ptr<MemStrategy>& output_mem_strategy() const { return output_mem_strategy_; }
  const std::shared_ptr<MemStrategy>& tmp_mem_strategy() const { return tmp_mem_strategy_; }
  const schedule::Schedule& schedule() const { return schedule_; }
  const lang::KernelList& kernel_list() const { return kernel_list_; }
  const std::unique_ptr<hal::Executable>& executable() const { return executable_; }

 private:
  void Initialize(                   //
      const context::Context& ctx,   //
      tile::proto::Program program,  //
      const std::shared_ptr<Scheduler>& scheduler);

 private:
  std::shared_ptr<DevInfo> devinfo_;
  std::shared_ptr<MemStrategy> output_mem_strategy_;
  std::shared_ptr<MemStrategy> tmp_mem_strategy_;
  lang::KernelList kernel_list_;
  schedule::Schedule schedule_;
  std::map<std::string, std::shared_ptr<tile::Buffer>> const_bufs_;
  std::unique_ptr<hal::Executable> executable_;
  std::size_t alloc_mem_;
  std::size_t num_runs_;
  hal::Memory* memory_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
