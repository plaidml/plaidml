#include "tile/lib/lib.h"

#include <boost/format.hpp>

namespace vertexai {
namespace tile {
namespace lib {

lang::RunInfo LoadMatMul(const std::string& name, const TensorShape& i1, const TensorShape& i2) {
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = "function (A[M, K], B[K, N]) -> (C) { C[m, n : M, N] = +(A[m, k] * B[k, n]); }";
  runinfo.input_shapes.emplace("A", i1);
  runinfo.input_shapes.emplace("B", i2);
  runinfo.output_shapes.emplace("C", SimpleShape(i1.type, {i1.dims[0].size, i2.dims[1].size}));
  return runinfo;
}

lang::RunInfo LoadConstCalc(const std::string& name, const TensorShape& output) {
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = R"***(
function () -> (O) {
  N = 1;
  F = 0.0;
  F2 = 3.7;
  Simple[i : N] = =(F[]);
  DoubleN[i : N] = =(N[]);
  Partial = Simple + DoubleN;
  O = Partial + F2;
})***";
  runinfo.output_shapes.emplace("O", output);
  return runinfo;
}

lang::RunInfo LoadConv1d(const std::string& name,    //
                         const TensorShape& input,   //
                         const TensorShape& kernel,  //
                         const TensorShape& output) {
  auto center = kernel.dims[0].size / 2;
  auto code = R"***(
function (I[X, CI], K[KX, CI, CO]) -> (O) {
  [[pid(res2a_branch2a)]] O[x, co : X, CO] = +(I[x + kx - %1%, ci] * K[kx, ci, co]);
})***";
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = str(boost::format(code) % center);
  runinfo.input_shapes.insert(std::make_pair("I", input));
  runinfo.input_shapes.insert(std::make_pair("K", kernel));
  runinfo.output_shapes.insert(std::make_pair("O", output));
  return runinfo;
}

lang::RunInfo LoadConv2d(const std::string& name,    //
                         const TensorShape& input,   //
                         const TensorShape& kernel,  //
                         const TensorShape& output) {
  auto center = kernel.dims[0].size / 2;
  auto code = R"***(
function (I[X, Y, CI], K[KX, KY, CI, CO]) -> (O) {
  [[pid(res2a_branch2a)]] O[x, y, co : X, Y, CO] = +(I[x + kx - %1%, y + ky - %1%, ci] * K[kx, ky, ci, co]);
})***";
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = str(boost::format(code) % center);
  runinfo.input_shapes.insert(std::make_pair("I", input));
  runinfo.input_shapes.insert(std::make_pair("K", kernel));
  runinfo.output_shapes.insert(std::make_pair("O", output));
  return runinfo;
}

lang::RunInfo LoadConv2dRelu(const std::string& name,    //
                             const TensorShape& input,   //
                             const TensorShape& kernel,  //
                             const TensorShape& output) {
  auto center = kernel.dims[0].size / 2;
  auto code = R"***(
function (I[X, Y, CI], K[KX, KY, CI, CO]) -> (O) {
  [[pid(res2a_branch2a)]] O[x, y, co : X, Y, CO] = +(I[x + kx - %1%, y + ky - %1%, ci] * K[i, ky, ci, co]);
  [[pid(relu)]] R = zelu(O);
})***";
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = str(boost::format(code) % center);
  runinfo.input_shapes.insert(std::make_pair("I", input));
  runinfo.input_shapes.insert(std::make_pair("K", kernel));
  runinfo.output_shapes.insert(std::make_pair("R", output));
  return runinfo;
}

lang::RunInfo LoadConv2dBnRelu(const std::string& name,      //
                               const TensorShape& input,     //
                               const TensorShape& kernel,    //
                               const TensorShape& channels,  //
                               const TensorShape& output) {
  auto center = kernel.dims[0].size / 2;
  auto code = R"***(
function (I[X, Y, CI], K[KX, KY, CI, CO], B[CO], S[CO]) -> (R) {
  [[pid(res2a_branch2a)]] O[x, y, co : X, Y, CO] = +(I[x + kx - %1%, y + ky - %1%, ci] * K[kx, ky, ci, co]);
  [[pid(bias_add)]] BO = O + B;
  [[pid(scale)]] BS = BO * S;
  [[pid(relu)]] R = zelu(BS);
})***";
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = str(boost::format(code) % center);
  runinfo.input_shapes.insert(std::make_pair("I", input));
  runinfo.input_shapes.insert(std::make_pair("K", kernel));
  runinfo.input_shapes.insert(std::make_pair("B", channels));
  runinfo.input_shapes.insert(std::make_pair("S", channels));
  runinfo.output_shapes.insert(std::make_pair("R", output));
  return runinfo;
}

lang::RunInfo LoadConv2d3Deep(const std::string& name,     //
                              const TensorShape& input,    //
                              const TensorShape& kernel1,  //
                              const TensorShape& kernel2,  //
                              const TensorShape& kernel3,  //
                              const TensorShape& output) {
  auto center1 = kernel1.dims[0].size / 2;
  auto center2 = kernel2.dims[0].size / 2;
  auto center3 = kernel3.dims[0].size / 2;
  auto code = R"***(
function (I[X, Y, CI], K1[KX, KY, C1, C2], K2[KX, KY, C2, C3], K3[KX, KY, C3, C4]) -> (O3) {
  O1[x, y, co : X, Y, C2] = +(I[x + kx - %1%, y + ky - %1%, ci] * K1[kx, ky, ci, co]);
  O2[x, y, co : X, Y, C3] = +(O1[x + kx - %2%, y + ky - %2%, ci] * K2[kx, ky, ci, co]);
  O3[x, y, co : X, Y, C4] = +(O2[x + kx - %3%, y + ky - %3%, ci] * K3[kx, ky, ci, co]);
})***";
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = str(boost::format(code) % center1 % center2 % center3);
  runinfo.input_shapes.insert(std::make_pair("I", input));
  runinfo.input_shapes.insert(std::make_pair("K1", kernel1));
  runinfo.input_shapes.insert(std::make_pair("K2", kernel2));
  runinfo.input_shapes.insert(std::make_pair("K3", kernel3));
  runinfo.output_shapes.insert(std::make_pair("O3", output));
  return runinfo;
}

}  // namespace lib
}  // namespace tile
}  // namespace vertexai
