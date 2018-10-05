// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <list>
#include <memory>
#include <string>

#include "base/context/context.h"
#include "tile/base/buffer.h"
#include "tile/base/program.h"
#include "tile/lang/generate.h"
#include "tile/proto/tile.pb.h"

namespace vertexai {
namespace tile {

// The Tile Platform interface definition.
// Platforms are things capable of executing Tile programs.
class Platform {
 public:
  virtual ~Platform() {}

  // Allocates a memory buffer on the target device.
  // The initial buffer contents are undefined.
  virtual std::shared_ptr<Buffer> MakeBuffer(const context::Context& ctx, const std::string& device_id,
                                             std::uint64_t size) = 0;

  // Builds (pre-compiling if possible) a program for executing the supplied Program
  virtual std::unique_ptr<Program> MakeProgram(const context::Context& ctx, const proto::Program& program) = 0;

  virtual void ListDevices(const context::Context& ctx, const proto::ListDevicesRequest& request,
                           proto::ListDevicesResponse* response) = 0;

  virtual void RegisterCostModel(const lang::TileCostFunction& cost_fn) = 0;
};

}  // namespace tile
}  // namespace vertexai
