// Copyright 2020 Intel Corporation

#include "pmlc/all_dialects.h"

#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

#include "pmlc/dialect/layer/ir/ops.h"
#include "pmlc/dialect/linalgx/ir/ops.h"
#include "pmlc/dialect/pml/ir/dialect.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/xsmm/ir/ops.h"

using namespace mlir; // NOLINT [build/namespaces]

namespace pmlc {

// Add all the MLIR dialects to the provided registry.
void registerAllDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<
      mlir::mhlo::MhloDialect, 
      pmlc::dialect::layer::LayerDialect,
      pmlc::dialect::linalgx::LinalgXDialect, 
      pmlc::dialect::pml::PMLDialect,
      pmlc::dialect::pxa::PXADialect, 
      pmlc::dialect::stdx::StdXDialect,
      pmlc::dialect::tile::TileDialect, 
      pmlc::dialect::xsmm::XSMMDialect>();
  // clang-format on
}

} // end namespace pmlc
