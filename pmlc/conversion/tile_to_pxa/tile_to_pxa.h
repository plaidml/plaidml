// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::conversion::tile_to_pxa {

std::unique_ptr<mlir::Pass> createLowerTileToPXAPass();

} // namespace pmlc::conversion::tile_to_pxa
