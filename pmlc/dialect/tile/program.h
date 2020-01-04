// Copyright 2019, Intel Corporation

#pragma once

#include <string>
#include <vector>

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Module.h"

#include "tile/base/buffer.h"

namespace pmlc {
namespace dialect {
namespace tile {

struct ProgramArgument {
  bool isInput;
  mlir::Value value;
  mlir::RankedTensorType shape;
  vertexai::tile::BufferPtr buffer;
};

struct TileProgram {
  std::string entry;
  mlir::OwningModuleRef module;
  std::vector<mlir::Value> outputs;
  std::vector<ProgramArgument> arguments;

  explicit TileProgram(mlir::ModuleOp module) : module(module) {}
};

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
