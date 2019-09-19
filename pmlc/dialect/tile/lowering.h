// Copyright 2019, Intel Corporation

#pragma once

#include <memory>

#include "mlir/IR/Module.h"

namespace pmlc {
namespace dialect {
namespace tile {

struct TileProgram;
struct StripeProgram;

mlir::OwningModuleRef LowerIntoStripe(  //
    mlir::MLIRContext* context,         //
    TileProgram* program);

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
