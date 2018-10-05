// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/platform.h"

#include <google/protobuf/util/json_util.h>

#include <algorithm>
#include <memory>
#include <utility>

#include <boost/process/environment.hpp>

#include "base/util/compat.h"
#include "base/util/error.h"
#include "base/util/factory.h"
#include "base/util/logging.h"
#include "base/util/type_url.h"
#include "tile/hal/util/selector.h"
#include "tile/hal/util/settings.h"
#include "tile/platform/local_machine/block_placer.h"
#include "tile/platform/local_machine/buffer.h"
#include "tile/platform/local_machine/direct_mem_strategy.h"
#include "tile/platform/local_machine/fifo_scheduler.h"
#include "tile/platform/local_machine/loose_scheduler.h"
#include "tile/platform/local_machine/program.h"
#include "tile/platform/local_machine/tdep_scheduler.h"
#include "tile/platform/local_machine/tmp_mem_strategy.h"

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

// The percentage of the device's memory that programs will try to use.
// This value seems to work pretty well on most devices.
// TODO: Either autotune this, or move it to the per-device configuration.
constexpr float kGoalMemPercentage = .85;

void GetMemStrategy(const std::shared_ptr<DevInfo>& devinfo, Platform::PlatformDev* pd) {
  if (devinfo->dev->executor() && devinfo->dev->executor()->shared_memory()) {
    IVLOG(1, "Using shared memory for data transfer");
    pd->mem_strategy = std::make_shared<DirectMemStrategy>(devinfo, devinfo->dev->executor()->shared_memory());
    pd->tmp_mem_source = devinfo->dev->executor()->shared_memory();
    return;
  }

  if (devinfo->dev->executor() && devinfo->dev->executor()->device_memory()) {
    IVLOG(1, "Using device memory and direct memory strategy");
    pd->mem_strategy = std::make_shared<DirectMemStrategy>(devinfo, devinfo->dev->executor()->device_memory());
    pd->tmp_mem_source = devinfo->dev->executor()->device_memory();
    return;
  }

  IVLOG(1, "Using host memory for data transfer");
  pd->mem_strategy = std::make_shared<DirectMemStrategy>(devinfo, devinfo->devset->host_memory());
  pd->tmp_mem_source = devinfo->devset->host_memory();
}

bool MatchConfig(const proto::Platform& config, const hal::proto::HardwareInfo& info,
                 hal::proto::HardwareSettings* settings) {
  for (const auto& hardware_config : config.hardware_configs()) {
    if (hal::selector::Match(hardware_config.sel(), info)) {
      auto overrides = hardware_config.settings();
      // Note: Booleans can only be overriden from false to true.
      // This is because protobuf treats an unspecified boolean as a false.
      settings->MergeFrom(overrides);
      return true;
    }
  }
  return false;
}

}  // namespace

Platform::Platform(const context::Context& ctx, const proto::Platform& config) {
  auto env = boost::this_process::environment();
  if (env.count("PLAIDML_DEBUG")) {
    LOG(INFO) << "Press any key after attaching a debugger to pid: " << boost::this_process::get_id();
    std::getchar();
  }

  for (auto& item : FactoryRegistrar<hal::Driver>::Instance()->Factories()) {
    try {
      VLOG(1) << "Creating HAL: " << item.second.name;
      auto driver = item.second.factory(ctx);
      drivers_.emplace_back(std::move(driver));
    } catch (const std::exception& ex) {
      VLOG(1) << "Failed to initialize HAL: " << ex.what();
    }
  }

  for (const auto& driver : drivers_) {
    for (const auto& devset : driver->device_sets()) {
      for (const auto& dev : devset->devices()) {
        if (dev->executor()) {
          const hal::proto::HardwareInfo& info = dev->executor()->info();
          hal::proto::HardwareSettings settings = info.settings();
          // TODO(T1101): Move ids into the hal

          // Loop over identical devices and ensure each one gets a unique id
          int ididx = 0;
          std::string id;
          do {
            std::stringstream ss;
            ss << info.name() << "." << ididx++;
            id = ss.str();
            std::replace(id.begin(), id.end(), ' ', '_');
            std::transform(id.begin(), id.end(), id.begin(), ::tolower);
          } while (devs_.find(id) != devs_.end());

          bool found_hardware_config = MatchConfig(config, info, &settings);
          dev->Initialize(settings);
          auto devinfo = std::make_shared<DevInfo>(DevInfo{devset, dev, settings});
          PlatformDev pd{id, devinfo};
          if (!found_hardware_config) {
            unmatched_devs_[id] = std::move(pd);
            continue;
          }
          VLOG(1) << settings.DebugString();
          GetMemStrategy(devinfo, &pd);

          auto memory = (dev->executor() && dev->executor()->device_memory() ? dev->executor()->device_memory()
                                                                             : devset->host_memory());
          if (dev->executor() && dev->executor()->is_synchronous()) {
            IVLOG(1, "Device is synchronous");
          }
          auto size_goal = memory->size_goal() * kGoalMemPercentage;
          IVLOG(1, "Using fifo scheduler; size_goal=" << size_goal);
          pd.scheduler = std::make_shared<fifo_scheduler::FifoScheduler>(memory->ArenaBufferAlignment(),
                                                                         std::lround(std::floor(size_goal)), settings);
          devs_[id] = std::move(pd);
        }
      }
    }
  }
}

void Platform::RegisterCostModel(const lang::TileCostFunction& cost_fn) { tile_optimizer_.RegisterModel(cost_fn); }

std::shared_ptr<tile::Buffer> Platform::MakeBuffer(const context::Context& ctx, const std::string& device_id,
                                                   std::uint64_t size) {
  auto& platform_dev = LookupDevice(device_id);
  return std::make_shared<Buffer>(platform_dev.devinfo, platform_dev.mem_strategy, size);
}

std::unique_ptr<tile::Program> Platform::MakeProgram(const context::Context& ctx, const tile::proto::Program& program) {
  auto& platform_dev = LookupDevice(program.dev_id());
  return compat::make_unique<Program>(
      ctx, program, platform_dev.devinfo, platform_dev.scheduler, platform_dev.mem_strategy,
      std::make_shared<TmpMemStrategy>(platform_dev.devinfo, platform_dev.tmp_mem_source), platform_dev.tmp_mem_source,
      tile_optimizer_);
}

void _fill_device(const Platform::PlatformDev& pdev, tile::proto::Device* dev) {
  google::protobuf::util::JsonPrintOptions options;
  options.add_whitespace = true;
  // options.preserve_proto_field_names = true;
  dev->set_dev_id(pdev.id);
  dev->set_description(pdev.devinfo->dev->description());
  std::string buf;
  google::protobuf::util::MessageToJsonString(pdev.devinfo->dev->executor()->info().info(), &buf, options);
  dev->set_details(buf);
  buf.clear();
  google::protobuf::util::MessageToJsonString(pdev.devinfo->settings, &buf, options);
  dev->set_config(buf);
}

void Platform::ListDevices(const context::Context& ctx, const tile::proto::ListDevicesRequest& request,
                           tile::proto::ListDevicesResponse* response) {
  for (const auto& dev : devs_) {
    _fill_device(dev.second, response->add_devices());
  }
  for (const auto& dev : unmatched_devs_) {
    _fill_device(dev.second, response->add_unmatched_devices());
  }
}

const Platform::PlatformDev& Platform::LookupDevice(const std::string& id) {
  if (!id.length()) {
    if (!devs_.size()) {
      throw error::NotFound{"No Tile compute devices available"};
    }
    return devs_.begin()->second;
  }
  auto it = devs_.find(id);
  if (it == devs_.end()) {
    throw error::NotFound{std::string("Unable to find Tile device \"") + id + "\""};
  }
  return it->second;
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
