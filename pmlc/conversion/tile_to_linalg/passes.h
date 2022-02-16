// Copyright 2021, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::conversion::tile_to_linalg {

std::unique_ptr<mlir::Pass> createLowerTileToLinalgPass();

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/conversion/tile_to_linalg/passes.h.inc"

} // namespace pmlc::conversion::tile_to_linalg
