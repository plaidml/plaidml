// Copyright 2019, Intel Corporation

#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include "pmlc/util/enums.h.inc"

namespace pmlc::util {

inline bool isFloat(DataType dtype) {
  switch (dtype) {
    case DataType::bf16:
    case DataType::f16:
    case DataType::f32:
    case DataType::f64:
      return true;
    default:
      return false;
  }
}

inline bool isSigned(DataType dtype) {
  switch (dtype) {
    case DataType::i8:
    case DataType::i16:
    case DataType::i32:
    case DataType::i64:
      return true;
    default:
      return false;
  }
}

inline bool isUnsigned(DataType dtype) {
  switch (dtype) {
    case DataType::u1:
    case DataType::u8:
    case DataType::u16:
    case DataType::u32:
    case DataType::u64:
      return true;
    default:
      return false;
  }
}

inline DataType from_string(const std::string& dtype_str) {
  auto data_type = pmlc::util::symbolizeDataType(dtype_str).getValueOr(DataType::invalid);

  static std::unordered_map<std::string, DataType> long_str_to_type{
      {"bool", DataType::u1},       {"int8", DataType::i8},          {"uint8", DataType::u8},
      {"int16", DataType::i16},     {"uint16", DataType::u16},       {"int32", DataType::i32},
      {"uint32", DataType::u32},    {"int64", DataType::i64},        {"uint64", DataType::u64},
      {"bfloat16", DataType::bf16}, {"float16", DataType::f16},      {"float32", DataType::f32},
      {"float64", DataType::f64},   {"<invalid>", DataType::invalid}};

  if (data_type == DataType::invalid && long_str_to_type.find(dtype_str) != long_str_to_type.end()) {
    return long_str_to_type[dtype_str];
  }
  return data_type;
}

inline bool isInteger(DataType dtype) {  //
  return isSigned(dtype) || isUnsigned(dtype);
}

inline DataType promoteTypes(DataType lhs, DataType rhs) {  //
  return std::max(lhs, rhs);
}

}  // namespace pmlc::util
