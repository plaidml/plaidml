// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/Dialect.h"

namespace pmlc {

// Add all the MLIR dialects to the provided registry.
void registerAllDialects(mlir::DialectRegistry &registry);

} // namespace pmlc
