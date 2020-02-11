// Copyright 2020, Intel Corporation

#include "pmlc/dialect/tile/transforms/contraction.h"
#include "pmlc/dialect/tile/transforms/passes.h"

#include "mlir/Pass/Pass.h"

namespace pmlc::dialect::tile {

static mlir::PassRegistration<ComputeBoundsPass>
    compute_bounds_pass("tile-compute-bounds",
                        "Compute bounds for contractions");

static mlir::PassRegistration<PaddingPass>
    padding_pass("tile-padding", "Pad outputs to remove constraints on inputs");

} // namespace pmlc::dialect::tile
