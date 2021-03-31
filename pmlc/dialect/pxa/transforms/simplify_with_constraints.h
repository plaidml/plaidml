// Copyright 2020, Intel Corporation
#pragma once

namespace mlir {
class Operation;
} // namespace mlir
namespace pmlc::dialect::pxa {

/// Simplifies in-place affine maps used by operation, using
/// constraints of surrounding `affine.parallel`s.
void simplifyAffineMapsWithConstraints(mlir::Operation *op);

} // namespace pmlc::dialect::pxa
