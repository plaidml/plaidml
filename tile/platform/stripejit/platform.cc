// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/stripejit/platform.h"

#include <memory>
#include <utility>

#include "tile/lang/parser.h"
#include "tile/platform/stripejit/program.h"
#include "tile/proto/support.h"

namespace vertexai {
namespace tile {
namespace stripejit {

std::shared_ptr<tile::Buffer> Platform::MakeBuffer(  //
    const context::Context& ctx,                     //
    const std::string& device_id,                    //
    std::uint64_t size) {
  return std::make_shared<SimpleBuffer>(size);
}

std::unique_ptr<tile::Program> Platform::MakeProgram(  //
    const context::Context& ctx,                       //
    const tile::proto::Program& program,               //
    ConstBufferManager* const_bufs) {
  lang::Parser parser;
  lang::RunInfo runinfo;
  runinfo.program = parser.Parse(program.code());
  runinfo.input_shapes = FromProto(program.inputs());
  runinfo.output_shapes = FromProto(program.outputs());
  runinfo.program_name = "stripe_program";
  return std::make_unique<Program>("llvm_cpu", runinfo, const_bufs);
}

std::shared_ptr<tile::Program> Platform::MakeProgram(const context::Context& ctx,   //
                                                     const std::string& device_id,  //
                                                     const std::string& target_id,  //
                                                     const lang::RunInfo& runinfo,  //
                                                     ConstBufferManager* const_bufs) {
  return std::make_unique<Program>(target_id, runinfo, const_bufs);
}

void Platform::ListDevices(                          //
    const context::Context& ctx,                     //
    const tile::proto::ListDevicesRequest& request,  //
    tile::proto::ListDevicesResponse* response) {
  // register one single device
  tile::proto::Device* dev = response->add_devices();
  dev->set_dev_id("llvm_cpu.0");
  dev->set_description("CPU (via LLVM)");
}

std::vector<std::string> Platform::ListDevices() {  //
  return {"llvm_cpu.0"};
}

}  // namespace stripejit
}  // namespace tile
}  // namespace vertexai
