// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/context/context.h"
#include "tile/base/buffer.h"
#include "tile/base/hal.h"
#include "tile/platform/local_machine/program.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// Represents the state of a Program::Run request.
class RunRequest {
 public:
  static boost::future<void> Run(               //
      const context::Context& ctx,              //
      const std::shared_ptr<Program>& program,  //
      std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
      std::map<std::string, std::shared_ptr<tile::Buffer>> outputs);

  void AddProgramDoneDep(const std::shared_ptr<hal::Event>& event);

  const Program* program() const { return program_.get(); }

 private:
  struct KernelLogInfo {
    std::shared_ptr<hal::Event> done;
    std::string kname;
    std::size_t tot_bytes;
    std::size_t tot_flops;
  };

  explicit RunRequest(const std::shared_ptr<Program>& program) : program_{program} {}

  static void LogRequest(                       //
      const std::shared_ptr<Program>& program,  //
      const std::map<std::string, std::shared_ptr<tile::Buffer>>& inputs,
      const std::map<std::string, std::shared_ptr<tile::Buffer>>& outputs);

  boost::future<void> LogResults(   //
      const context::Context& ctx,  //
      boost::future<std::vector<std::shared_ptr<hal::Result>>> results);

  const std::shared_ptr<Program> program_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
