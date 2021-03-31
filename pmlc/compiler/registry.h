// Copyright 2019, Intel Corporation

#pragma once

#include <functional>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include "pmlc/compiler/program.h"
#include "pmlc/util/buffer.h"

namespace pmlc::compiler {

void registerTarget(llvm::StringRef name, TargetPtr target);

TargetPtr resolveTarget(llvm::StringRef name);

std::vector<llvm::StringRef> listTargets();

} // namespace pmlc::compiler
