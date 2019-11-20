// Copyright 2019, Intel Corporation

#pragma once

#include <functional>
#include <vector>

#include "llvm/ADT/SetVector.h"

#include "mlir/IR/Value.h"

namespace pmlc {
namespace util {

using TransitiveFilter = std::function<bool(mlir::Value*)>;

std::vector<mlir::Value*> getBackwardSlice(
    const llvm::SetVector<mlir::Value*>& values,  //
    bool enter_regions = false,                   //
    TransitiveFilter filter = [](mlir::Value*) { return true; });

}  // namespace util
}  // namespace pmlc
