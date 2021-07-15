// Copyright 2021, Intel Corporation

#pragma once

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/DebugStringHelper.h"

namespace pmlc::util {
struct AxisDim;
} // namespace pmlc::util

#define GET_ATTRDEF_CLASSES
#include "pmlc/util/schedule_attrdef.h.inc"

#include "pmlc/util/dialect.h.inc"

namespace pmlc::util {

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

} // namespace pmlc::util
