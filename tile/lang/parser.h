#pragma once

#include <string>

#include "tile/lang/ops.h"

namespace vertexai {
namespace tile {
namespace lang {

class Parser final {
 public:
  Program Parse(const std::string& code, const std::string& id = "") const;
  Program ParseExpr(const std::string& code, int64_t start_tmp = 0) const;
  Polynomial<Rational> ParsePolynomial(const std::string& poly) const;
  Contraction ParseContraction(const std::string& contract) const;
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
