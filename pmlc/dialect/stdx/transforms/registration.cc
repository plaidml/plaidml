// Copyright 2020, Intel Corporation

#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/stdx/transforms/boundscheck.h"

namespace pmlc::dialect::stdx {

static mlir::PassRegistration<BoundsCheckPass>
    compute_bounds_pass("stdx-check-bounds",
                        "Check bounds for Load and Store Ops");

} // namespace pmlc::dialect::stdx
