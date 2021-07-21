// Copyright 2021, Intel Corporation

#include "pmlc/conversion/linalg_to_pxa/pass_detail.h"
#include "pmlc/util/logging.h"

namespace pmlc::conversion::linalg_to_pxa {

namespace layer = dialect::layer;
namespace pxa = dialect::pxa;

using namespace mlir; // NOLINT

void populateLinalgToPXASpecialPatterns(mlir::RewritePatternSet &patterns) {}

} // namespace pmlc::conversion::linalg_to_pxa
