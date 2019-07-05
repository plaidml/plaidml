// Copyright 2019 Intel Corporation.

#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "tile/lang/ast/ast.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace ast {

// ExprDeriv is a function that builds the gradient ("dX") from the following inputs:
//  1. Y: The Expr for the node
//  2. dY: The Expr for the already-computed gradient of the node's output
//  3. Xs: The Exprs for the node's inputs
using ExprDeriv = std::function<std::vector<ExprPtr>(  //
    const ExprPtr& Y,                                  //
    const ExprPtr& dY,                                 //
    const std::vector<ExprPtr>& Xs,                    //
    void* user_fn,                                     //
    void* user_ctx)>;

struct ExprDerivEntry {
  ExprDeriv fn;
  void* user_fn;
  void* user_ctx;
};

class DerivRegistry {
 public:
  static DerivRegistry* Instance() {
    static DerivRegistry registry;
    return &registry;
  }

  void Register(const std::string& name, const ExprDeriv& fn, void* user_fn, void* user_ctx) {  //
    registry_[name] = ExprDerivEntry{fn, user_fn, user_ctx};
  }

  ExprDerivEntry Resolve(const std::string& name) const {
    auto it = registry_.find(name);
    if (it == registry_.end()) {
      throw std::runtime_error("Invalid derivative: Unknown function " + name);
    }
    return it->second;
  }

 private:
  std::unordered_map<std::string, ExprDerivEntry> registry_;
};

std::vector<ExprPtr> ComputeGradients(  //
    const std::vector<ExprPtr>& wrts,   //
    const ExprPtr& result,              //
    const ExprPtr& loss);

}  // namespace ast
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
