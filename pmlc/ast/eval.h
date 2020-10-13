// Copyright 2020 Intel Corporation

#pragma once

#include <unordered_map>

#include "llvm/ADT/ArrayRef.h"

#include "pmlc/ast/ast.h"
#include "pmlc/util/enums.h"

namespace pmlc::ast {

bool isAmbiguousDataType(util::DataType dtype);

util::DataType inferElementType(llvm::ArrayRef<util::TensorShape> shapes);

util::TensorShape inferShape(llvm::ArrayRef<util::TensorShape> operands,
                             util::DataType override = util::DataType::invalid);

//
// Evaluator
//

class Evaluator {
public:
  int64_t evaluate(const DimNodePtr &node);
  int64_t evaluate(const DimNode *node);

  util::TensorShape getShape(const ExprNodePtr &node);
  util::TensorShape getShape(const ExprNode *node);

  util::TensorShape getShape(const ExprNodePtr &node, size_t ordinal);
  util::TensorShape getShape(const ExprNode *node, size_t ordinal);

  llvm::ArrayRef<util::TensorShape> getShapes(const ExprNodePtr &node);
  llvm::ArrayRef<util::TensorShape> getShapes(const ExprNode *node);

  void verify(const ExprNodePtr &node);

private:
  util::TensorShapes computeShapes(const ExprNode *node);

private:
  std::unordered_map<const DimNode *, int64_t> dimsCache;
  std::unordered_map<const ExprNode *, util::TensorShapes> shapesCache;
};

} // namespace pmlc::ast
