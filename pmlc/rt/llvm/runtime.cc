// Copyright 2020, Intel Corporation

#include <stdexcept>

#include "pmlc/rt/llvm/device.h"
#include "pmlc/rt/llvm/register.h"
#include "pmlc/rt/runtime_registry.h"

namespace pmlc::rt::llvm {
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
  std::shared_ptr<Device> dev = std::make_shared<Device>();
};

} // namespace

extern void registerBuiltins();
extern void registerOpenMP();

void registerRuntime() {
  registerBuiltins();
  registerOpenMP();
  registerRuntime("llvm_cpu", std::make_shared<Runtime>());
}

} // namespace pmlc::rt::llvm
