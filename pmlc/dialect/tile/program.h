// Copyright 2019, Intel Corporation

#pragma once

#include <string>
#include <vector>

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Module.h"

#include "pmlc/util/buffer.h"

namespace pmlc::dialect::tile {

struct ProgramArgument {
  bool isInput;
  mlir::Value value;
  mlir::RankedTensorType shape;
  pmlc::util::BufferPtr buffer;
};

struct TileProgram {
  std::string entry;
  mlir::OwningModuleRef module;
  std::vector<mlir::Value> outputs;
  std::vector<ProgramArgument> arguments;

  explicit TileProgram(mlir::ModuleOp module) : module(module) {}
};

}  // namespace pmlc::dialect::tile
