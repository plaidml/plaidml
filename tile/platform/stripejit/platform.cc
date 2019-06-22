// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/stripejit/platform.h"

#include <memory>
#include <utility>

#include "tile/platform/stripejit/program.h"

namespace vertexai {
namespace tile {
namespace stripejit {

std::shared_ptr<tile::Buffer> Platform::MakeBuffer(const context::Context& ctx, const std::string& device_id,
                                                   std::uint64_t size) {
  return std::make_shared<SimpleBuffer>(size);
}

std::unique_ptr<tile::Program> Platform::MakeProgram(const context::Context& ctx, const tile::proto::Program& program,
                                                     ConstBufferManager* const_bufs) {
  return std::make_unique<Program>(ctx, program, const_bufs);
}

std::shared_ptr<tile::Program> Platform::MakeProgram(const context::Context& ctx,   //
                                                     const std::string& device_id,  //
                                                     const lang::RunInfo& runinfo,  //
                                                     ConstBufferManager* const_bufs) {
  throw std::runtime_error("Not implemented");
}

void Platform::ListDevices(const context::Context& ctx, const tile::proto::ListDevicesRequest& request,
                           tile::proto::ListDevicesResponse* response) {
  // register one single device
  tile::proto::Device* dev = response->add_devices();
  dev->set_dev_id("llvm_cpu.0");
  dev->set_description("CPU (via LLVM)");
}

}  // namespace stripejit
}  // namespace tile
}  // namespace vertexai
