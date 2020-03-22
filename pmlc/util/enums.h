// Copyright 2019, Intel Corporation

#pragma once

#include <algorithm>

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include "pmlc/util/enums.h.inc"

namespace pmlc::util {

// inline bool isFloat(DataType dtype) {
//   switch (dtype) {
//   case DataType::bf16:
//   case DataType::f16:
//   case DataType::f32:
//   case DataType::f64:
//     return true;
//   default:
//     return false;
//   }
// }

// inline bool isSigned(DataType dtype) {
//   switch (dtype) {
//   case DataType::i8:
//   case DataType::i16:
//   case DataType::i32:
//   case DataType::i64:
//     return true;
//   default:
//     return false;
//   }
// }

// inline bool isUnsigned(DataType dtype) {
//   switch (dtype) {
//   case DataType::u1:
//   case DataType::u8:
//   case DataType::u16:
//   case DataType::u32:
//   case DataType::u64:
//     return true;
//   default:
//     return false;
//   }
// }

// inline bool isInteger(DataType dtype) {
//   return isSigned(dtype) || isUnsigned(dtype);
// }

// inline DataType promoteTypes(DataType lhs, DataType rhs) {
//   return std::max(lhs, rhs);
// }

} // namespace pmlc::util
