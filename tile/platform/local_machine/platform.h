// Copyright 2017, Vertex.AI.

#pragma once

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "tile/base/hal.h"
#include "tile/base/platform.h"
#include "tile/platform/local_machine/devinfo.h"
#include "tile/platform/local_machine/local_machine.pb.h"
#include "tile/platform/local_machine/mem_strategy.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// Platform implements tile::Platform by maintaining a map from subdevice IDs
// to information about the underlying device and strategies for manipulating
// the device.
class Platform : public tile::Platform {
 public:
  // Contains the platform API handlers for a particular device.
  struct PlatformDev {
    std::string id;
    std::shared_ptr<DevInfo> devinfo;
    std::shared_ptr<MemStrategy> mem_strategy;
    hal::Memory* tmp_mem_source;
  };

  Platform(const context::Context& ctx, const proto::Platform& config);
  Platform(const context::Context& ctx, const proto::Platform& config, const std::set<std::string>& config_ids,
           const std::set<std::string>& device_ids);

  std::shared_ptr<tile::Buffer> MakeBuffer(const context::Context& ctx, const std::string& device_id,
                                           std::uint64_t size) final;

  std::unique_ptr<tile::Program> MakeProgram(const context::Context& ctx, const tile::proto::Program& program) final;

  void ListDevices(const context::Context& ctx, const tile::proto::ListDevicesRequest& request,
                   tile::proto::ListDevicesResponse* response) final;

 private:
  const PlatformDev& LookupDevice(const std::string& id);

  std::vector<std::unique_ptr<hal::Driver>> drivers_;
  std::unordered_map<std::string, PlatformDev> devs_;
  std::unordered_map<std::string, PlatformDev> unmatched_devs_;
  std::set<std::string> config_ids;
  std::set<std::string> device_ids;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
