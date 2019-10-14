// Copyright 2019 Intel Corporation.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tile/lang/ast/ast.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace ast {

PolyExprPtr MakeOp(IntOp op, const std::vector<PolyExprPtr>& args);
DimExprPtr MakeOp(IntOp op, const std::vector<DimExprPtr>& args);
ExprPtr MakeCall(const std::string& fn, const std::vector<ExprPtr>& args);
ExprPtr MakeGradOverride(const std::shared_ptr<ExprDerivEntry>& fn, const std::vector<ExprPtr>& ins,
                         const ExprPtr& out);

}  // namespace ast
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
