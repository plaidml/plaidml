// Copyright 2020, Intel Corporation

#include "pmlc/dialect/tile/transforms/contraction.h"

#include "mlir/Pass/Pass.h"

namespace pmlc::dialect::tile {

static mlir::PassRegistration<ComputeBoundsPass>
    pass("tile-compute-bounds", "Compute bounds for contractions");

} // namespace pmlc::dialect::tile
