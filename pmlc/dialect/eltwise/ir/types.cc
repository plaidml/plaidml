// Copyright 2019, Intel Corporation

#include "pmlc/dialect/eltwise/ir/types.h"

#include "mlir/IR/StandardTypes.h"

namespace pmlc::dialect::eltwise {

using vertexai::tile::DataType;

using mlir::FloatType;
using mlir::IntegerType;
using mlir::MLIRContext;
using mlir::Type;

Type ScalarType::toStandard() {
  MLIRContext* ctx = getContext();
  switch (type()) {
    case DataType::BOOLEAN:
      return IntegerType::get(1, ctx);
    case DataType::INT8:
    case DataType::UINT8:
      return IntegerType::get(8, ctx);
    case DataType::INT16:
    case DataType::UINT16:
      return IntegerType::get(16, ctx);
    case DataType::INT32:
    case DataType::UINT32:
      return IntegerType::get(32, ctx);
    case DataType::INT64:
    case DataType::UINT64:
      return IntegerType::get(64, ctx);
    case DataType::FLOAT16:
      return FloatType::getF16(ctx);
    case DataType::FLOAT32:
      return FloatType::getF32(ctx);
    case DataType::FLOAT64:
      return FloatType::getF64(ctx);
    case DataType::BFLOAT16:
      return FloatType::getBF16(ctx);
    default:
      return {};
  }
}

}  // namespace pmlc::dialect::eltwise
