// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::conversion::pxa_to_affine {

std::unique_ptr<mlir::Pass> createLowerPXAToAffinePass();

} // namespace pmlc::conversion::pxa_to_affine
