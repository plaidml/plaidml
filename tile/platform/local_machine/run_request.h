// Copyright 2017, Vertex.AI.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/context/context.h"
#include "tile/base/buffer.h"
#include "tile/base/hal.h"
#include "tile/platform/local_machine/program.h"
#include "tile/platform/local_machine/shim.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// Represents the state of a Program::Run request.
class RunRequest {
 public:
  static boost::future<void> Run(const context::Context& ctx, const Program* program,
                                 std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
                                 std::map<std::string, std::shared_ptr<tile::Buffer>> outputs);

  void AddProgramDoneDep(const std::shared_ptr<hal::Event>& event);
  void AddKernelInfo(std::size_t kidx, std::shared_ptr<hal::Event> event);

  const Program* program() const { return program_; }
  const Shim* shim() const { return shim_.get(); }

 private:
  struct KernelLogInfo {
    std::shared_ptr<hal::Event> done;
    std::string kname;
    std::size_t tot_bytes;
    std::size_t tot_flops;
  };

  RunRequest(const Program* program, std::unique_ptr<Shim> shim);
  RunRequest(const RunRequest&) = delete;
  RunRequest(RunRequest&&) = default;
  RunRequest& operator=(const RunRequest&) = delete;
  RunRequest& operator=(RunRequest&&) = default;

  static void LogRequest(const Program* program, const std::map<std::string, std::shared_ptr<tile::Buffer>>& inputs,
                         const std::map<std::string, std::shared_ptr<tile::Buffer>>& outputs);

  boost::future<void> LogResults(const context::Context& ctx,
                                 boost::future<std::vector<std::shared_ptr<hal::Result>>> results);

  const Program* program_;
  std::vector<KernelLogInfo> kernel_log_info_;
  std::unique_ptr<Shim> shim_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
