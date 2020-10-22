// Copyright 2020, Intel Corporation

#pragma once

#include <memory>
#include <stdexcept>
#include <utility>

#include "pmlc/rt/executable.h"
#include "pmlc/util/buffer.h"

namespace pmlc::rt {

// Device represents a PlaidML device, capable of evaluating a PlaidML program.
class Device {
public:
  virtual ~Device() = default;

  virtual std::unique_ptr<Executable>
  compile(const std::shared_ptr<pmlc::compiler::Program> &program) = 0;
  double execTimeInMS{0.0};
};

} // namespace pmlc::rt
