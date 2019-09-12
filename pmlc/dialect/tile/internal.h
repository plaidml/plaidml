// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Module.h"

namespace pmlc {
namespace dialect {
namespace tile {

struct TileProgram {
  mlir::OwningModuleRef module;
  mlir::BlockAndValueMapping mapper;

  explicit TileProgram(mlir::ModuleOp module) : module(module) {}
};

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
