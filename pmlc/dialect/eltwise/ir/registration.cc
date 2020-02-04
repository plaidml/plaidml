// Copyright 2020, Intel Corporation

#include "pmlc/dialect/eltwise/ir/dialect.h"

namespace pmlc::dialect::eltwise {

// Static initialization for Eltwise dialect registration.
static mlir::DialectRegistration<Dialect> theDialect;

} // namespace pmlc::dialect::eltwise
