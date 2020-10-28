// Copyright 2019, Intel Corporation

#pragma once

#include <functional>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include "pmlc/compiler/program.h"
#include "pmlc/util/buffer.h"

namespace pmlc::compiler {

using TargetFactory = llvm::function_ref<TargetPtr()>;

void registerTarget(llvm::StringRef name, TargetFactory factory);

TargetPtr resolveTarget(llvm::StringRef name);

std::vector<llvm::StringRef> listTargets();

struct TargetRegistration {
  TargetRegistration(llvm::StringRef name, TargetFactory factory) {
    registerTarget(name, factory);
  }
};

} // namespace pmlc::compiler
