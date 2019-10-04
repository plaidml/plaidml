// Copyright 2019, Intel Corporation

#pragma once

#include <unordered_map>

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Module.h"

#include "tile/base/buffer.h"

namespace pmlc {
namespace dialect {
namespace tile {

struct TileProgram {
  mlir::OwningModuleRef module;
  mlir::BlockAndValueMapping mapper;
  std::unordered_map<mlir::Value*, vertexai::tile::BufferPtr> ioMap;

  explicit TileProgram(mlir::ModuleOp module) : module(module) {}
};

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
