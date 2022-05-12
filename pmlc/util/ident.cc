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

static Attribute getAggAttr(arith::AtomicRMWKind agg, Type type) {
  // If it's an assign, treat like an add (zero)
  if (agg == arith::AtomicRMWKind::assign) {
    if (type.isa<FloatType>()) {
      agg = arith::AtomicRMWKind::addf;
    } else {
      agg = arith::AtomicRMWKind::addi;
    }
  }

  switch (agg) {
  case arith::AtomicRMWKind::assign:
  case arith::AtomicRMWKind::addf:
    return typedFloat(0.0, type);
  case arith::AtomicRMWKind::addi:
    return IntegerAttr::get(type, static_cast<int64_t>(0));
  case arith::AtomicRMWKind::mulf:
    return typedFloat(1.0, type);
  case arith::AtomicRMWKind::muli:
    return IntegerAttr::get(type, static_cast<int64_t>(1));
  case arith::AtomicRMWKind::maxf:
    return typedInf(type, true);
  case arith::AtomicRMWKind::maxs:
    return IntegerAttr::get(type, std::numeric_limits<int64_t>::min());
  case arith::AtomicRMWKind::maxu:
    return IntegerAttr::get(type, static_cast<int64_t>(0));
  case arith::AtomicRMWKind::minf:
    return typedInf(type, false);
  case arith::AtomicRMWKind::mins:
    return IntegerAttr::get(type, std::numeric_limits<int64_t>::max());
  case arith::AtomicRMWKind::minu:
    return IntegerAttr::get(type, std::numeric_limits<uint64_t>::max());
  }
  llvm_unreachable("Unable to compute identity for aggregation");
}

Value createIdentity(OpBuilder &builder, Location loc, arith::AtomicRMWKind agg,
                     Type type) {
  Attribute attr = getAggAttr(agg, type);
  return builder.create<arith::ConstantOp>(loc, type, attr);
}

} // namespace mlir
