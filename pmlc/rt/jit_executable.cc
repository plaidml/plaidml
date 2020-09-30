// Copyright 2020 Intel Corporation

#include "pmlc/rt/jit_executable.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/rt/device_id.h"
#include "pmlc/rt/internal.h"
#include "pmlc/rt/runtime.h"
#include "pmlc/rt/symbol_registry.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

using pmlc::compiler::Program;

namespace pmlc::rt {

// TODO: Move JitExecutable and related code from executable.cc to here.  (We're
// keeping it in executable.cc temporarily to simplify code review of the
// refactoring that causes us to use it.)

} // namespace pmlc::rt
