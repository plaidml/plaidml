// Copyright 2021, Intel Corporation

#include "pmlc/conversion/tile_to_linalg/pass_detail.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/util/logging.h"

namespace pmlc::conversion::tile_to_linalg {

namespace layer = dialect::layer;
namespace tile = dialect::tile;

using namespace mlir; // NOLINT

void populateTileToLinalgSpecialPatterns(mlir::RewritePatternSet &patterns) {
  // TODO: support specials
}

} // namespace pmlc::conversion::tile_to_linalg
