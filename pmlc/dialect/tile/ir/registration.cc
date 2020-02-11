// Copyright 2020, Intel Corporation

#include "pmlc/dialect/tile/ir/dialect.h"

namespace pmlc::dialect::tile {

// Static initialization for Tile dialect registration.
static mlir::DialectRegistration<Dialect> theDialect;

}  // namespace pmlc::dialect::tile
