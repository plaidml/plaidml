// Copyright 2020, Intel Corporation

#include "pmlc/dialect/tile/transforms/constant_types.h"
#include "pmlc/dialect/tile/transforms/contraction.h"

#include "mlir/Pass/Pass.h"

namespace pmlc::dialect::tile {

static mlir::PassRegistration<ComputeBoundsPass> compute_bounds_pass(  //
    "tile-compute-bounds",                                             //
    "Compute bounds for contractions");

static mlir::PassRegistration<ConstantTypesPass> constant_types_pass(  //
    "tile-constant-types",                                             //
    "Set constants to specified types");

}  // namespace pmlc::dialect::tile
