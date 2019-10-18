// Copyright 2019 Intel Corporation.

#pragma once

// TODO: Need relevant includes
// TODO: Several of these can move to the *.cc file

#include <map>
#include <string>
#include <unordered_map>

// #include "pmlc/dialect/eltwise/dialect.h"
// #include "pmlc/dialect/eltwise/ops.h"
#include "pmlc/dialect/tile/ops.h"
// #include "pmlc/dialect/tile/program.h"

namespace mlir {
class Value;
class Operation;
}  // namespace mlir

namespace pmlc::dialect::tile {
class TileBuilder;

// Deriv is a function that builds the gradients ("dXs") from the following inputs:
//  1. Y: The Value for the node
//  2. dY: The Value for the already-computed gradient of the node's output
//  3. Xs: The Values for the node's inputs
using Deriv = std::function<llvm::SmallVector<mlir::Value*, 3>(  // TODO: Size?
    mlir::Value* Y,                                              //
    mlir::Value* dY,                                             //
    const llvm::SmallVector<mlir::Value*, 3>& Xs,                // TODO: Size?
    void* user_fn,                                               //
    void* user_ctx)>;

// A DerivEntry bundles the Deriv with the FFI-processed user function & context needed to call it from Values
struct DerivEntry {
  Deriv fn;
  void* user_fn;
  void* user_ctx;
};

class DerivRegistry {
 public:
  static DerivRegistry* Instance() {
    static DerivRegistry registry;
    return &registry;
  }

  void Register(const std::string& name, const Deriv& fn, void* user_fn, void* user_ctx) {  //
    if (registry_.count(name)) {
      throw std::runtime_error("Attempted to register deriv '" + name + "', which was already in the DerivRegistry");
    }
    registry_[name] = DerivEntry{fn, user_fn, user_ctx};
  }

  DerivEntry Resolve(const std::string& name) const {
    auto it = registry_.find(name);
    if (it == registry_.end()) {
      throw std::runtime_error("Invalid derivative: Unknown function " + name);
    }
    return it->second;
  }

 private:
  std::unordered_map<std::string, DerivEntry> registry_;
};

class Gradient {
 public:
  explicit Gradient(mlir::Value* loss, TileBuilder* builder);
  void ComputeOperandDerivs(mlir::Value* val);
  mlir::Value* GetDerivative(mlir::Value* val);

 private:
  mlir::Value* DeriveEltwise(mlir::Value* dout, mlir::Value* out, size_t idx);
  mlir::Value* DeriveContraction(mlir::Value* dout, mlir::Value* out, size_t idx);
  mlir::Value* DeriveSpecial(const mlir::Value* dout, SpecialOp* op, size_t val);  // TODO
  void AddToGradient(Value* source_op, Value* deriv);

  TileBuilder* builder_;
  std::map<mlir::Value*, mlir::Value*> grads_;
};

}  // namespace pmlc::dialect::tile
