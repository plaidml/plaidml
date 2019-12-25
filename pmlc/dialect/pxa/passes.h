// Copyright 2019, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
}  // namespace mlir

namespace pmlc::dialect::pxa {

std::unique_ptr<mlir::Pass> createLowerToPXAPass();
std::unique_ptr<mlir::Pass> createLowerToAffinePass();

}  // namespace pmlc::dialect::pxa
