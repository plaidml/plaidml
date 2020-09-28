// Copyright 2020 Intel Corporation
#include <vector>

#include "pmlc/rt/runtime.h"
#include "pmlc/rt/runtime_registry.h"
#include "pmlc/util/logging.h"

#include "pmlc/rt/opencl/opencl_device.h"

namespace pmlc::rt::opencl {

class OpenCLRuntime final : public pmlc::rt::Runtime {
public:
  OpenCLRuntime() {
    std::vector<cl::Device> supportedDevices =
        pmlc::rt::opencl::getSupportedDevices();
    for (cl::Device &device : supportedDevices)
      devices.emplace_back(std::make_shared<OpenCLDevice>(device));
  }

  ~OpenCLRuntime() {}

  std::size_t deviceCount() const noexcept final { return devices.size(); }
  std::shared_ptr<pmlc::rt::Device> device(std::size_t idx) override {
    return devices.at(idx);
  }

private:
  std::vector<std::shared_ptr<OpenCLDevice>> devices;
};

pmlc::rt::RuntimeRegistration<OpenCLRuntime> reg{"opencl"};

} // namespace pmlc::rt::opencl
