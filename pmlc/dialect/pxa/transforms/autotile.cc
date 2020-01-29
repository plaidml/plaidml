// Copyright 2020, Intel Corporation

#include "pmlc/dialect/pxa/transforms/autotile.h"

namespace pmlc::dialect::pxa {

std::vector<int64_t> PowerOfTwoGenerator(int64_t range) {
  std::vector<int64_t> out;
  for (int64_t r = 1; r <= range; r *= 2) {
    out.push_back(r);
  }
  return out;
}

std::vector<int64_t> EvenDivisionGenerator(int64_t range) {
  std::vector<int64_t> out;
  // TODO: Something less dumb here
  for (int64_t r = 1; r <= range; r++) {
    if (range % r != 0) {
      continue;
    }
    out.push_back(r);
  }
  return out;
}

static mlir::PassRegistration<AutotilePass> pass(  //
    "autotile-10", "Tile all dimensions by 10");

}  // namespace pmlc::dialect::pxa
