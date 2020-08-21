// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/Dialect.h"

// Add all the MLIR dialects to the provided registry.
void registerAllDialects(mlir::DialectRegistry &registry);

// This function should be called before creating any MLIRContext if one expect
// all the possible dialects to be made available to the context automatically.
void registerAllDialects();
