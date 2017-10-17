// Copyright 2017, Vertex.AI.

#include "tile/platform/local_machine/platform.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "base/util/any_factory_map.h"
#include "base/util/compat.h"
#include "base/util/error.h"
#include "base/util/logging.h"
#include "base/util/type_url.h"
#include "tile/hal/util/selector.h"
#include "tile/hal/util/settings.h"
#include "tile/platform/local_machine/buffer.h"
#include "tile/platform/local_machine/copy_mem_strategy.h"
#include "tile/platform/local_machine/direct_mem_strategy.h"
#include "tile/platform/local_machine/program.h"
#include "tile/platform/local_machine/tmp_mem_strategy.h"

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

void GetMemStrategy(const std::shared_ptr<DevInfo>& devinfo, Platform::PlatformDev* pd) {
  if (devinfo->dev->executor()->shared_memory()) {
    IVLOG(1, "Using shared memory for data transfer");
    pd->mem_strategy = std::make_shared<DirectMemStrategy>(devinfo, devinfo->dev->executor()->shared_memory());
    pd->tmp_mem_source = devinfo->dev->executor()->shared_memory();
    return;
  }
  if (devinfo->devset->host_memory() && devinfo->dev->executor() && devinfo->dev->executor()->device_memory()) {
    IVLOG(1, "Using device memory and explicit memory copies for data transfer");
    pd->mem_strategy = std::make_shared<CopyMemStrategy>(devinfo);
    pd->tmp_mem_source = devinfo->dev->executor()->device_memory();
    return;
  }
  IVLOG(1, "Using host memory for data transfer");
  pd->mem_strategy = std::make_shared<DirectMemStrategy>(devinfo, devinfo->devset->host_memory());
  pd->tmp_mem_source = devinfo->devset->host_memory();
}

}  // namespace

Platform::Platform(const context::Context& ctx, const proto::Platform& config) {
  for (auto hal_config : config.hals()) {
    drivers_.emplace_back(AnyFactoryMap<hal::Driver>::Instance()->MakeInstance(ctx, hal_config));
  }

  bool skip_device = false;
  for (const auto& driver : drivers_) {
    for (const auto& devset : driver->device_sets()) {
      for (const auto& dev : devset->devices()) {
        if (dev->executor()) {
          const hal::proto::HardwareInfo& info = dev->executor()->info();
          hal::proto::HardwareSettings settings = info.settings();
          bool found_hardware_config = false;
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

          for (const auto& hardware_config : config.hardware_configs()) {
            if (hal::selector::Match(hardware_config.sel(), info)) {
              settings.MergeFrom(hardware_config.settings());
              found_hardware_config = true;
            }
          }
          if (!found_hardware_config) {
            skip_device = true;
            LOG(WARNING) << "No settings found for hardware device \"" << info.name() << "\", vendor \""
                         << info.vendor() << "\"";
          }
          auto devinfo = std::make_shared<DevInfo>(DevInfo{devset, dev, settings});
          PlatformDev pd{devinfo};
          if (skip_device) {
            unmatched_devs_[id] = std::move(pd);
            continue;
          }
          LOG(INFO) << "Initializing device " << id << ": \"" << info.name() << "\", vendor \"" << info.vendor()
                    << "\"";
          LOG(INFO) << "  " << id << " Threads:" << settings.threads();
          LOG(INFO) << "  " << id << " Vector size:" << settings.vec_size();
          LOG(INFO) << "  " << id << " Memory width:" << settings.mem_width();
          LOG(INFO) << "  " << id << " Memory Max:" << settings.max_mem();
          LOG(INFO) << "  " << id << " Register Max:" << settings.max_regs();
          LOG(INFO) << "  " << id << " Goal Groups:" << settings.goal_groups();
          LOG(INFO) << "  " << id << " Goal Flops/byte:" << settings.goal_flops_per_byte();
          LOG(INFO) << "  " << id
                    << (settings.use_global() ? " Running from system memory" : " Running from device memory");
          GetMemStrategy(devinfo, &pd);
          devs_[id] = std::move(pd);
        }
      }
    }
  }
}

std::shared_ptr<tile::Buffer> Platform::MakeBuffer(const context::Context& ctx, const std::string& device_id,
                                                   std::uint64_t size) {
  auto& platform_dev = LookupDevice(device_id);
  return std::make_shared<Buffer>(platform_dev.devinfo, platform_dev.mem_strategy->MakeChunk(ctx, size));
}

std::unique_ptr<tile::Program> Platform::MakeProgram(const context::Context& ctx, const tile::proto::Program& program) {
  auto& platform_dev = LookupDevice(program.dev_id());
  return compat::make_unique<Program>(
      ctx, program, platform_dev.devinfo, platform_dev.mem_strategy,
      std::make_shared<TmpMemStrategy>(platform_dev.devinfo, platform_dev.tmp_mem_source), platform_dev.tmp_mem_source);
}

void Platform::ListDevices(const context::Context& ctx, const tile::proto::ListDevicesRequest& request,
                           tile::proto::ListDevicesResponse* response) {
  for (const auto& dev : devs_) {
    auto desc = response->add_devices();
    desc->set_dev_id(dev.first);
    desc->set_description(dev.second.devinfo->dev->description());
  }
  for (const auto& dev : unmatched_devs_) {
    auto desc = response->add_unmatched_devices();
    desc->set_dev_id(dev.first);
    desc->set_description(dev.second.devinfo->dev->description());
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
