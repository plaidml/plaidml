// Copyright 2021, Intel Corporation

#pragma once

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/DebugStringHelper.h"

namespace pmlc::dialect::pml {
struct AxisDim;
} // namespace pmlc::dialect::pml

#define GET_ATTRDEF_CLASSES
#include "pmlc/dialect/pml/ir/attrdef.h.inc"

#include "pmlc/dialect/pml/ir/dialect.h.inc"

namespace pmlc::dialect::pml {

struct AxisDim {
  AxisAttr axis;
  size_t dim;
};

inline std::ostream &operator<<(std::ostream &os, AxisAttr attr) {
  mlir::Attribute base = attr;
  os << debugString(base);
  return os;
}

inline std::ostream &operator<<(std::ostream &os, ScheduleAttr attr) {
  mlir::Attribute base = attr;
  os << debugString(base);
  return os;
}

static constexpr mlir::StringLiteral kScheduleAttrName = "schedule";

} // namespace pmlc::dialect::pml
