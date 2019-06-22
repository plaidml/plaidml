#pragma once

#include <memory>
#include <string>

#include "tile/lang/ast.h"

extern "C" {

struct tile_string {
  std::string str;
};

struct tile_shape {
  vertexai::tile::TensorShape shape;
};

struct tile_expr {
  std::shared_ptr<vertexai::tile::lang::Expr> expr;
};

struct tile_poly_expr {
  std::shared_ptr<vertexai::tile::lang::PolyExpr> expr;
};

struct tile_program {
  vertexai::tile::lang::ProgramEvaluation eval;
};

}  // extern "C"
