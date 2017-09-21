#pragma once

#include <boost/multi_array.hpp>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "tile/lang/flat.h"
#include "tile/lang/ops.h"

namespace vertexai {
namespace tile {
namespace lang {

template <class T, int dims>
using Tensor = boost::multi_array<T, dims>;

template <class T>
DataType GetDataType() {
  return DataType();
}
template <>
DataType GetDataType<int8_t>() {
  return DataType::INT8;
}
template <>
DataType GetDataType<int16_t>() {
  return DataType::INT16;
}
template <>
DataType GetDataType<int32_t>() {
  return DataType::INT32;
}
template <>
DataType GetDataType<int64_t>() {
  return DataType::INT64;
}
template <>
DataType GetDataType<uint8_t>() {
  return DataType::UINT8;
}
template <>
DataType GetDataType<uint16_t>() {
  return DataType::UINT16;
}
template <>
DataType GetDataType<uint32_t>() {
  return DataType::UINT32;
}
template <>
DataType GetDataType<uint64_t>() {
  return DataType::UINT64;
}
template <>
DataType GetDataType<float>() {
  return DataType::FLOAT32;
}
template <>
DataType GetDataType<double>() {
  return DataType::FLOAT64;
}

template <class T, size_t dims>
TensorShape ShapeOf(const Tensor<T, dims>& t) {
  std::vector<TensorDimension> ndims;
  for (size_t i = 0; i < dims; i++) {
    ndims.emplace_back(t.strides()[i], t.shape()[i]);
  }
  return TensorShape(GetDataType<T>(), ndims);
}

template <typename T1, typename T2, typename T3>
void ExecuteRecursive(const FlatContraction& f, T1* out, const T2* in1, const T3* in2, std::vector<int64_t>* vars,
                      unsigned depth) {
  if (depth == f.ranges.size()) {
    for (const FlatConstraint& fc : f.constraints) {
      int64_t tot = 0;
      for (size_t i = 0; i < fc.lhs.size(); i++) {
        tot += fc.lhs[i] * (*vars)[i];
      }
      if (tot > fc.rhs) {
        IVLOG(4, "Skipping at vars " << *vars);
        return;
      }
    }
    T1 comb = (in2 == NULL ? *in1 : (f.comb_op == CombinationOp::MULTIPLY ? (*in1 * *in2) : (*in1 + *in2)));
    IVLOG(3, "At vars " << *vars << ", in1 = " << *in1 << ", in2 = " << (in2 == NULL ? "NULL" : std::to_string(*in2))
                        << ", comb = " << comb << ", out = " << *out);
    if (f.agg_op == AggregationOp::SUM) {
      *out += comb;
    } else {
      *out = std::max(*out, comb);
    }
    return;
  }
  for (size_t i = 0; i < f.ranges[depth]; i++) {
    (*vars)[depth] = i;
    ExecuteRecursive(f, out, in1, in2, vars, depth + 1);
    out += f.access[0].strides[depth];
    in1 += f.access[1].strides[depth];
    if (in2 != NULL) {
      in2 += f.access[2].strides[depth];
    }
  }
}

template <typename T1, typename T2, typename T3, size_t D1, size_t D2, size_t D3>
void Execute(const FlatContraction& f, Tensor<T1, D1>& out, const Tensor<T2, D2>& in1,  // NOLINT(runtime/references)
             const Tensor<T3, D3>& in2) {
  std::vector<int64_t> vars(f.ranges.size());
  ExecuteRecursive(f, out.origin() + f.access[0].offset, in1.origin() + f.access[1].offset,
                   in2.origin() + f.access[2].offset, &vars, 0);
}

template <typename T1, typename T2, size_t D1, size_t D2>
void Execute(const FlatContraction& f, Tensor<T1, D1>& out, const Tensor<T2, D2>& in1) {  // NOLINT(runtime/references)
  std::vector<int64_t> vars(f.ranges.size());
  ExecuteRecursive<T1, T2, int>(f, out.origin() + f.access[0].offset, in1.origin() + f.access[1].offset, NULL, &vars,
                                0);
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
