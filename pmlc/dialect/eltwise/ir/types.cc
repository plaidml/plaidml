// Copyright 2019, Intel Corporation

#include "pmlc/dialect/eltwise/ir/types.h"

#include "mlir/IR/StandardTypes.h"

namespace pmlc::dialect::eltwise {

using util::DataType;

using mlir::FloatType;
using mlir::IntegerType;
using mlir::MLIRContext;
using mlir::Type;

Type ScalarType::toStandard() {
  MLIRContext* ctx = getContext();
  switch (type()) {
    case DataType::u1:
      return IntegerType::get(1, ctx);
    case DataType::i8:
    case DataType::u8:
      return IntegerType::get(8, ctx);
    case DataType::i16:
    case DataType::u16:
      return IntegerType::get(16, ctx);
    case DataType::i32:
    case DataType::u32:
      return IntegerType::get(32, ctx);
    case DataType::i64:
    case DataType::u64:
      return IntegerType::get(64, ctx);
    case DataType::bf16:
      return FloatType::getBF16(ctx);
    case DataType::f16:
      return FloatType::getF16(ctx);
    case DataType::f32:
      return FloatType::getF32(ctx);
    case DataType::f64:
      return FloatType::getF64(ctx);
    default:
      return {};
  }
}

}  // namespace pmlc::dialect::eltwise
