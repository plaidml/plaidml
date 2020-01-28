// Copyright 2020, Intel Corporation

#include "pmlc/dialect/tile/transforms/contraction.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace pmlc::dialect::tile {

static mlir::PassRegistration<ComputeBoundsPass> compute_bounds_pass(  //
    "tile-compute-bounds",                                             //
    "Compute bounds for contractions");

}  // namespace pmlc::dialect::tile
