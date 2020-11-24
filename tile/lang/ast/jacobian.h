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

std::vector<ExprPtr> ComputeJacobian(const std::vector<ExprPtr>& wrts, const ExprPtr& loss);

}  // namespace ast
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
