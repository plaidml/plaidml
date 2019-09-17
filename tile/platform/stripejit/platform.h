// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "tile/base/hal.h"
#include "tile/base/platform.h"

namespace vertexai {
namespace tile {
namespace stripejit {

class Platform : public tile::Platform {
 public:
  Platform() {}

  std::shared_ptr<tile::Buffer> MakeBuffer(  //
      const context::Context& ctx,           //
      const std::string& device,             //
      std::uint64_t size) final;

  std::unique_ptr<tile::Program> MakeProgram(  //
      const context::Context& ctx,             //
      const proto::Program& program,           //
      ConstBufferManager* const_bufs) final;

  std::shared_ptr<tile::Program> MakeProgram(           //
      const context::Context& ctx,                      //
      const std::string& device,                        //
      const std::string& target,                        //
      const std::shared_ptr<stripe::Program>& program,  //
      ConstBufferManager* const_bufs) final;

  std::vector<std::string> ListDevices() final;

  void ListDevices(                              //
      const context::Context& ctx,               //
      const proto::ListDevicesRequest& request,  //
      proto::ListDevicesResponse* response) final;

  void RegisterCostModel(const lang::TileCostFunction& cost_fn) final{};
};

}  // namespace stripejit
}  // namespace tile
}  // namespace vertexai
