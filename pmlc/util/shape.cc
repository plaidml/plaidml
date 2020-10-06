// Copyright 2020 Intel Corporation

#include "pmlc/util/shape.h"

#include <sstream>

#include "mlir/IR/StandardTypes.h"

namespace pmlc::util {

static size_t getByteSize(DataType dtype) {
  switch (dtype) {
  case DataType::i1:
  case DataType::si8:
  case DataType::ui8:
    return 1;
  case DataType::si16:
  case DataType::ui16:
  case DataType::bf16:
  case DataType::f16:
    return 2;
  case DataType::si32:
  case DataType::ui32:
  case DataType::f32:
    return 4;
  case DataType::si64:
  case DataType::ui64:
  case DataType::f64:
    return 8;
  default:
    break;
  }
  llvm_unreachable("Invalid DataType for getByteSize");
}

static DataType convertFromType(mlir::Type type) {
  if (type.isInteger(1))
    return DataType::i1;
  if (type.isSignedInteger(8))
    return DataType::si8;
  if (type.isUnsignedInteger(8))
    return DataType::ui8;
  if (type.isSignedInteger(16))
    return DataType::si16;
  if (type.isUnsignedInteger(16))
    return DataType::ui16;
  if (type.isSignedInteger(32))
    return DataType::si32;
  if (type.isUnsignedInteger(32))
    return DataType::ui32;
  if (type.isSignedInteger(64))
    return DataType::si64;
  if (type.isUnsignedInteger(64))
    return DataType::ui64;
  if (type.isBF16())
    return DataType::bf16;
  if (type.isF16())
    return DataType::f16;
  if (type.isF32())
    return DataType::f32;
  if (type.isF64())
    return DataType::f64;
  llvm_unreachable("Invalid mlir::Type");
}

TensorShape TensorShape::fromType(mlir::Type type) {
  auto rankedTensorType = type.cast<mlir::RankedTensorType>();
  return TensorShape(convertFromType(rankedTensorType.getElementType()),
                     rankedTensorType.getShape());
}

std::string TensorShape::str() const {
  std::stringstream ss;
  for (int64_t dim : sizes) {
    if (dim) {
      ss << dim;
    } else {
      ss << '?';
    }
    ss << 'x';
  }
  ss << util::stringifyDataType(elementType).str();
  if (strides.size()) {
    ss << ", [";
    for (auto item : llvm::enumerate(strides)) {
      if (item.index()) {
        ss << 'x';
      }
      ss << item.value();
    }
    ss << ']';
  }
  return ss.str();
}

size_t TensorShape::getByteSize() const {
  size_t product = 1;
  for (size_t dim : sizes) {
    product *= dim;
  }
  return product * util::getByteSize(elementType);
}

bool TensorShape::operator==(const TensorShape &rhs) const {
  return std::tie(elementType, sizes, strides) ==
         std::tie(rhs.elementType, rhs.sizes, rhs.strides);
}

} // namespace pmlc::util
