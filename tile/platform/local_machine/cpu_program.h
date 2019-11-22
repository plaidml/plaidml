// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <map>
#include <memory>
#include <string>

#include "tile/base/program.h"
#include "tile/lang/runinfo.h"
#include "tile/proto/tile.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace targets {
namespace cpu {

class Native;

}  // namespace cpu
}  // namespace targets

namespace local_machine {

class CpuProgram final : public tile::Program {
 public:
  CpuProgram(                        //
      const std::string& target,     //
      const lang::RunInfo& runinfo,  //
      ConstBufferManager* const_bufs);

  CpuProgram(                                          //
      const std::string& target,                       //
      const std::shared_ptr<stripe::Program>& stripe,  //
      ConstBufferManager* const_bufs);

  ~CpuProgram();

  boost::future<void> Run(          //
      const context::Context& ctx,  //
      std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
      std::map<std::string, std::shared_ptr<tile::Buffer>> outputs) final;

  // The maximum available memory
  std::size_t MaxAvailableMemory() final;

  // Release resource used by the program
  void Release() final;

 private:
  std::unique_ptr<tile::targets::cpu::Native> executable_;
  std::shared_ptr<stripe::Block> source_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
