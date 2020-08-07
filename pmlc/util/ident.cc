// Copyright 2019, Intel Corporation

#include "pmlc/util/ident.h"

#include <limits>

namespace mlir {

static Attribute typedFloat(double num, Type type) {
  auto value = llvm::APFloat(num);
  auto floatType = type.cast<FloatType>();
  bool losesInfo = false;
  value.convert(floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                &losesInfo);
  return FloatAttr::get(type, value);
}

static Attribute typedInf(Type type, bool neg) {
  auto floatType = type.cast<FloatType>();
  auto value = llvm::APFloat::getInf(floatType.getFloatSemantics(), neg);
  return FloatAttr::get(type, value);
}

static Attribute getAggAttr(AtomicRMWKind agg, Type type) {
  // If it's an assign, treat like an add (zero)
  if (agg == AtomicRMWKind::assign) {
    if (type.isa<FloatType>()) {
      agg = AtomicRMWKind::addf;
    } else {
      agg = AtomicRMWKind::addi;
    }
  }

  switch (agg) {
  case AtomicRMWKind::assign:
  case AtomicRMWKind::addf:
    return typedFloat(0.0, type);
  case AtomicRMWKind::addi:
    return IntegerAttr::get(type, static_cast<int64_t>(0));
  case AtomicRMWKind::mulf:
    return typedFloat(1.0, type);
  case AtomicRMWKind::muli:
    return IntegerAttr::get(type, static_cast<int64_t>(1));
  case AtomicRMWKind::maxf:
    return typedInf(type, true);
  case AtomicRMWKind::maxs:
    return IntegerAttr::get(type, std::numeric_limits<int64_t>::min());
  case AtomicRMWKind::maxu:
    return IntegerAttr::get(type, static_cast<int64_t>(0));
  case AtomicRMWKind::minf:
    return typedInf(type, false);
  case AtomicRMWKind::mins:
    return IntegerAttr::get(type, std::numeric_limits<int64_t>::max());
  case AtomicRMWKind::minu:
    return IntegerAttr::get(type, std::numeric_limits<uint64_t>::max());
  }
  llvm_unreachable("Unable to compute identity for aggregation");
}

Value createIdentity(OpBuilder &builder, Location &loc, AtomicRMWKind agg,
                     Type type) {
  Attribute attr = getAggAttr(agg, type);
  return builder.create<mlir::ConstantOp>(loc, type, attr);
}

} // namespace mlir
