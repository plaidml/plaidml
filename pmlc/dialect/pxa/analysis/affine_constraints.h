// Copyright 2020, Intel Corporation
#pragma once

#include <utility>

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LLVM.h"

namespace pmlc::dialect::pxa {

/// Adds `affine.parallel` operand to constraints system.
/// It is expected that "constraints" already hold dimension
/// for operand.
/// TODO: This shoud be integrated into FlatAffineConstraints,
///       same way addAffineForDomain already is.
mlir::LogicalResult
addAffineParallelIVDomain(mlir::AffineParallelOp parallelOp, unsigned idx,
                          mlir::FlatAffineConstraints &constraints);

/// Calculates lower inclusive bound for given expression,
/// under constraints for dimensions it uses.
mlir::Optional<int64_t> getLowerBound(mlir::AffineExpr expr,
                                      mlir::FlatAffineConstraints &constraints);

/// Calculates upper inclusive bound for given expression,
/// under constraints for dimensions it uses.
mlir::Optional<int64_t> getUpperBound(mlir::AffineExpr expr,
                                      mlir::FlatAffineConstraints &constraints);

/// Calculates [lower, upper] bounds (both sides inclusive),
/// under constraints for input dimensions.
std::pair<mlir::Optional<int64_t>, mlir::Optional<int64_t>>
getLowerUpperBounds(mlir::AffineExpr expr,
                    mlir::FlatAffineConstraints &constraints);

mlir::AffineExpr
simplifyExprWithConstraints(mlir::AffineExpr expr,
                            mlir::FlatAffineConstraints &constraints);

/// Simplifies affine map given integral constraints.
/// Returns the same map if it cannot be simplified further.
mlir::AffineMap
simplifyMapWithConstraints(mlir::AffineMap map,
                           mlir::FlatAffineConstraints &constraints);
/// Fully composes affine map, gathers constraints and simplifies
/// using gathered constraints.
mlir::AffineValueMap simplifyMapWithConstraints(mlir::AffineValueMap map);

mlir::FlatAffineConstraints
gatherAffineMapConstraints(mlir::AffineValueMap map);

} // namespace pmlc::dialect::pxa
