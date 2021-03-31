// Copyright 2020 Intel Corporation

#pragma once

#include <memory>

#include "pmlc/rt/device.h"

namespace pmlc::rt::llvm {

class Device final : public pmlc::rt::Device,
                     public std::enable_shared_from_this<Device> {
public:
  std::unique_ptr<Executable>
  compile(const std::shared_ptr<pmlc::compiler::Program> &program) final;
};

} // namespace pmlc::rt::llvm
