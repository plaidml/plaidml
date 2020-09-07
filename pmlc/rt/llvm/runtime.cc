// Copyright 2020, Intel Corporation

#include "pmlc/rt/runtime_registry.h"

#include <stdexcept>

namespace {

class Runtime final : public pmlc::rt::Runtime {
public:
  std::size_t deviceCount() const noexcept final { return 1; }
  std::shared_ptr<pmlc::rt::Device> device(std::size_t idx) {
    if (idx) {
      throw std::out_of_range{"Invalid device index"};
    }
    return dev;
  }

private:
  std::shared_ptr<pmlc::rt::Device> dev = std::make_shared<pmlc::rt::Device>();
};

pmlc::rt::RuntimeRegistration<Runtime> reg{"llvm_cpu"};

} // namespace
