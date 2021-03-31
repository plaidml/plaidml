// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"

#include "pmlc/ast/ast.h"
#include "pmlc/ast/eval.h"

namespace pmlc::ast {

struct Intrinsic {
  virtual ~Intrinsic() = default;

  virtual util::TensorShapes
  getShapes(Evaluator *evaluator, llvm::ArrayRef<ExprNodePtr> operands,
            llvm::ArrayRef<util::TensorShape> shapes) const = 0;
};

class IntrinsicRegistry {
public:
  static IntrinsicRegistry *Instance() {
    static IntrinsicRegistry registry;
    return &registry;
  }

  void add(llvm::StringRef name, std::unique_ptr<Intrinsic> op) {
    registry_[name] = std::move(op);
  }

  const Intrinsic *resolve(llvm::StringRef name) {
    auto it = registry_.find(name);
    if (it == registry_.end()) {
      return nullptr;
    }
    return it->second.get();
  }

private:
  llvm::StringMap<std::unique_ptr<Intrinsic>> registry_;
};

void registerOps();

} // namespace pmlc::ast
