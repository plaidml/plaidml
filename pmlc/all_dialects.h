// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/Dialect.h"

// Add all the MLIR dialects to the provided registry.
void registerAllDialects(mlir::DialectRegistry &registry);
