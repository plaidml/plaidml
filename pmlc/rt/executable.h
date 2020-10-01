// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <vector>

#include "pmlc/compiler/program.h"

namespace pmlc::rt {

class Executable {
public:
  static std::unique_ptr<Executable>
  fromProgram(const std::shared_ptr<pmlc::compiler::Program> &program,
              mlir::StringRef deviceID, mlir::ArrayRef<void *> bufptrs);

  virtual ~Executable() = default;

  virtual void invoke() = 0;
};

} // namespace pmlc::rt
