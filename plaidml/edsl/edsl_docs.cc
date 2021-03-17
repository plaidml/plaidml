// Copyright 2020 Intel Corporation.
// Note:
//    This file is being used by sphinx docs to pull in code blocks.
//    Code blocks are pulled into docs/usage/*.rst
//    Any changes made here may upset the docs.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "plaidml/edsl/edsl.h"
#include "plaidml/testenv.h"

namespace plaidml::edsl {

class DocCppEdsl : public TestFixture {};

// sum_over_axis_start
Tensor SumOverAxis(const Tensor& I) {
  TensorDim M, N;
  TensorIndex m, n;
  I.bind_dims(M, N);
  return Contraction().outShape(N).outAccess(n).sum(I(m, n));  // contraction
}
// sum_over_axis_end

// max_over_axis_start
Tensor MaxOverAxis(const Tensor& I) {
  TensorDim M, N;
  TensorIndex m, n;
  I.bind_dims(M, N);
  return Contraction().outShape(N).outAccess(n).max(I(m, n));
}
// max_over_axis_end

// matmul_start
Tensor MatMul(const Tensor& A, const Tensor& B) {
  TensorDim I, J, K;
  TensorIndex i, j, k;
  A.bind_dims(I, K);
  B.bind_dims(K, J);
  return Contraction().outShape(I, J).outAccess(i, j).sum(A(i, k) * B(k, j));
}
// matmul_end

// global_min_start
Tensor GlobalMin(const Tensor& I) {
  TensorIndex i, j, k;
  auto Neg = -I;
  Tensor O = -Tensor(Contraction().max(Neg(i, j, k)));
  return O;
}
// global_min_end

// avg_start
Tensor Avg(const Tensor& I) {
  TensorDim X, Y;
  TensorIndex x, y;
  I.bind_dims(X, Y);
  Tensor Sum = Contraction().outShape(Y).outAccess(y).sum(I(x, y));
  return Sum / X;
}
// avg_end

// avg_stages_start
Tensor AvgStages(const Tensor& I) {
  TensorDim X, Y;
  TensorIndex x, y;
  I.bind_dims(X, Y);
  Tensor Sum = Contraction().sum(I(x, y));
  auto PartialMean = Sum / X;
  return PartialMean / Y;
}
// avg_stages_end

// avg_merge_start
Tensor AvgMerge(const Tensor& I) {
  TensorDim X, Y;
  TensorIndex x, y;
  I.bind_dims(X, Y);
  Tensor Sum = Contraction().sum(I(x, y));
  return Sum / (X * Y);
}
// avg_merge_end

// skip_start
Tensor Skip(const Tensor& I) {
  TensorDim M, N;
  TensorIndex i, j;
  I.bind_dims(M, N);
  return Contraction().outShape(N).outAccess(2 * i).sum(I(2 * i, j));
}
// skip_end

// cumsum_start
Tensor CumSum(const Tensor& I) {
  TensorDim N;
  TensorIndex i, k;
  I.bind_dims(N);
  return Contraction().outShape(N).outAccess(i).sum(I(k)).add_constraint(i - k < N);
}
// cumsum_end

// layer_start
Tensor Layer(const Tensor& A, const Tensor& B) {
  std::string name = "Sum";
  std::function<Tensor()> function = [&]() { return A + B; };
  std::vector<Tensor> inputs = {A, B};
  return layer(name, inputs, function);
}
// layer_end

// trace_start
Tensor Trace(const Tensor& A, const Tensor& B) {
  auto At = trace(A, "Pre-summation");
  auto C = At + B;
  return trace(C, "Post-summation");
}
// trace_end

TEST_F(DocCppEdsl, SumOveAxis) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("sum_over_axis", {I}, {SumOverAxis(I)}));
}

TEST_F(DocCppEdsl, MaxOverAxis) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("max_over_axis", {I}, {MaxOverAxis(I)}));
}

TEST_F(DocCppEdsl, MatMul) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("mat_mul", {A, B}, {MatMul(A, B)}));
}

TEST_F(DocCppEdsl, GlobalMin) {
  auto I = Placeholder(DType::FLOAT32, {10, 10, 10}, "I");
  runProgram(makeProgram("global_min", {I}, {GlobalMin(I)}));
}

TEST_F(DocCppEdsl, Avg) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("avg", {I}, {Avg(I)}));
}

TEST_F(DocCppEdsl, AvgStages) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("avg_stages", {I}, {AvgStages(I)}));
}

TEST_F(DocCppEdsl, AvgMerge) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("avg_merge", {I}, {AvgMerge(I)}));
}

TEST_F(DocCppEdsl, Skip) {
  auto I = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("skip", {I}, {Skip(I)}));
}

TEST_F(DocCppEdsl, CumSum) {
  auto I = Placeholder(DType::FLOAT32, {10}, "I");
  runProgram(makeProgram("cumsum", {I}, {CumSum(I)}));
}

TEST_F(DocCppEdsl, Layer) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("layer", {A, B}, {Layer(A, B)}));
}

TEST_F(DocCppEdsl, Trace) {
  auto A = Placeholder(DType::UINT64, {3, 3});
  auto B = Placeholder(DType::UINT64, {3, 3});
  runProgram(makeProgram("trace", {A, B}, {Trace(A, B)}));
}

}  // namespace plaidml::edsl
