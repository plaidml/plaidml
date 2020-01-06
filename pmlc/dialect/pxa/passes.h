// Copyright 2019, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
}  // namespace mlir

namespace pmlc::dialect::pxa {

std::unique_ptr<mlir::Pass> createLowerPXAToAffinePass();

}  // namespace pmlc::dialect::pxa
