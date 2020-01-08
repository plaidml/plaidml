// Copyright 2020, Intel Corporation

#include "pmlc/dialect/pxa/ir/dialect.h"

namespace pmlc::dialect::pxa {

// Static initialization for PXA dialect registration.
static mlir::DialectRegistration<Dialect> theDialect;

}  // namespace pmlc::dialect::pxa
