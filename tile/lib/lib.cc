#include "tile/lib/lib.h"

#include <boost/format.hpp>

#include "tile/util/tile_file.h"

namespace vertexai {
namespace tile {
namespace lib {

namespace {

std::shared_ptr<lang::BufferBase> MakeBuffer(const TensorShape& shape) {
  auto buffer = std::make_shared<util::SimpleBuffer>();
  buffer->bytes.resize(shape.byte_size());
  return buffer;
}

}  // namespace

lang::RunInfo LoadMatMul(const std::string& name, const TensorShape& i1, const TensorShape& i2) {
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = "function (A[M, K], B[K, N]) -> (C) { C[m, n : M, N] = +(A[m, k] * B[k, n]); }";
  runinfo.input_shapes.emplace("A", i1);
  runinfo.input_shapes.emplace("B", i2);
  runinfo.output_shapes.emplace("C", SimpleShape(i1.type, {i1.dims[0].size, i2.dims[1].size}));
  return runinfo;
}

lang::RunInfo LoadEWAdd(const std::string& name, const TensorShape& i1, const TensorShape& i2) {
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = "function (A, B) -> (C) { C = A + B; }";
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
  runinfo.const_inputs = {"K"};
  runinfo.input_buffers = {{"K", MakeBuffer(kernel)}};
  return runinfo;
}

lang::RunInfo LoadConv2d(const std::string& name,    //
                         const TensorShape& input,   //
                         const TensorShape& kernel,  //
                         const TensorShape& output) {
  auto center = kernel.dims[0].size / 2;
  auto code = R"***(
function (I[N, X, Y, CI], K[KX, KY, CI, CO]) -> (O) {
  [[pid(res2a_branch2a)]] O[n, x0, x1, co : N, X, Y, CO] = +(I[n, x0 + kx - %1%, x1 + ky - %1%, ci] * K[kx, ky, ci, co]);
})***";
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = str(boost::format(code) % center);
  runinfo.input_shapes.insert(std::make_pair("I", input));
  runinfo.input_shapes.insert(std::make_pair("K", kernel));
  runinfo.output_shapes.insert(std::make_pair("O", output));
  runinfo.const_inputs = {"K"};
  runinfo.input_buffers = {{"K", MakeBuffer(kernel)}};
  return runinfo;
}

lang::RunInfo LoadConv2dRelu(const std::string& name,    //
                             const TensorShape& input,   //
                             const TensorShape& kernel,  //
                             const TensorShape& output) {
  auto center = kernel.dims[0].size / 2;
  auto code = R"***(
function (I[N, X, Y, CI], K[KX, KY, CI, CO]) -> (O) {
  [[pid(res2a_branch2a)]] O[n, x0, x1, co : N, X, Y, CO] = +(I[n, x0 + kx - %1%, x1 + ky - %1%, ci] * K[kx, ky, ci, co]);
  [[pid(relu)]] R = zelu(O);
})***";
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = str(boost::format(code) % center);
  runinfo.input_shapes.insert(std::make_pair("I", input));
  runinfo.input_shapes.insert(std::make_pair("K", kernel));
  runinfo.output_shapes.insert(std::make_pair("R", output));
  runinfo.const_inputs = {"K"};
  runinfo.input_buffers = {{"K", MakeBuffer(kernel)}};
  return runinfo;
}

lang::RunInfo LoadConv2dBnRelu(const std::string& name,      //
                               const TensorShape& input,     //
                               const TensorShape& kernel,    //
                               const TensorShape& channels,  //
                               const TensorShape& output) {
  auto center = kernel.dims[0].size / 2;
  auto code = R"***(
function (I[N, X, Y, CI], K[KX, KY, CI, CO], B[CO], S[CO]) -> (R) {
  [[pid(res2a_branch2a)]] O[n, x0, x1, co : N, X, Y, CO] = +(I[n, x0 + kx - %1%, x1 + ky - %1%, ci] * K[kx, ky, ci, co]);
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
  runinfo.const_inputs = {"K"};
  runinfo.input_buffers = {
      {"K", MakeBuffer(kernel)},
      {"B", MakeBuffer(channels)},
      {"S", MakeBuffer(channels)},
  };
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
function (I[N, X, Y, CI], K1[KX, KY, C1, C2], K2[KX, KY, C2, C3], K3[KX, KY, C3, C4]) -> (O3) {
  O1[n, x0, x1, co : N, X, Y, C2] = +(I[n, x0 + kx - %1%, x1 + ky - %1%, ci] * K1[kx, ky, ci, co]);
  O2[n, x0, x1, co : N, X, Y, C3] = +(O1[n, x0 + kx - %2%, x1 + ky - %2%, ci] * K2[kx, ky, ci, co]);
  O3[n, x0, x1, co : N, X, Y, C4] = +(O2[n, x0 + kx - %3%, x1 + ky - %3%, ci] * K3[kx, ky, ci, co]);
})***";
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = str(boost::format(code) % center1 % center2 % center3);
  runinfo.input_shapes.insert(std::make_pair("I", input));
  runinfo.input_shapes.insert(std::make_pair("K1", kernel1));
  runinfo.input_shapes.insert(std::make_pair("K2", kernel2));
  runinfo.input_shapes.insert(std::make_pair("K3", kernel3));
  runinfo.output_shapes.insert(std::make_pair("O3", output));
  runinfo.const_inputs = {"K1", "K2", "K3"};
  runinfo.input_buffers = {
      {"K1", MakeBuffer(kernel1)},
      {"K2", MakeBuffer(kernel2)},
      {"K3", MakeBuffer(kernel3)},
  };
  return runinfo;
}

lang::RunInfo LoadDilatedConv2d(const std::string& name,    //
                                const TensorShape& input,   //
                                const TensorShape& kernel,  //
                                const TensorShape& output) {
  auto code = R"***(
function (I[N, Lx, Ly, CI], K[LKx, LKy, CI, CO]) -> (O) {
    O[n, x, y, co: N, Lx - 2 * (LKx - 1), Ly - 3 * (LKy - 1), CO] = +(I[n, x + 2 * kx, y + 3 * ky, ci] * K[kx, ky, ci, co]);
})***";
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = code;
  runinfo.input_shapes.insert(std::make_pair("I", input));
  runinfo.input_shapes.insert(std::make_pair("K", kernel));
  runinfo.output_shapes.insert(std::make_pair("O", output));
  runinfo.const_inputs = {"K"};
  runinfo.input_buffers = {
      {"K", MakeBuffer(kernel)},
  };
  return runinfo;
}

lang::RunInfo LoadLarsMomentum4d(const std::string& name,     //
                                 const TensorShape& x_shape,  //
                                 const TensorShape& lr_shape) {
  // Note: X/Grad/Veloc/NewX/NewVeloc should all have the same shape for the
  // semantics of this operation to be correct, so we only pass in 1 shape for
  // all of them.
  double lars_coeff = 1. / 1024.;
  double lars_weight_decay = 1. / 2048.;
  double momentum = 1. / 8.;
  auto code = R"***(
function (X[X0, X1, X2, X3], Grad[Grad0, Grad1, Grad2, Grad3], Veloc[Veloc0, Veloc1, Veloc2, Veloc3], LR[]) -> (NewX, NewVeloc) {
    XSqr = X * X;
    X_MS[] = +(XSqr[x0, x1, x2, x3]);
    XNorm = sqrt(X_MS);
    GradSqr = Grad * Grad;
    Grad_MS[] = +(GradSqr[g0, g1, g2, g3]);
    GradNorm = sqrt(Grad_MS);
    LocLR = LR * %1% * XNorm / (GradNorm + %2% * XNorm);
    NewVeloc = %3% * Veloc + LocLR * (Grad + %2% * X);
    NewX = X - NewVeloc;
})***";
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = str(boost::format(code) % lars_coeff % lars_weight_decay % momentum);
  runinfo.input_shapes.insert(std::make_pair("X", x_shape));
  runinfo.input_shapes.insert(std::make_pair("Grad", x_shape));
  runinfo.input_shapes.insert(std::make_pair("Veloc", x_shape));
  runinfo.input_shapes.insert(std::make_pair("LR", lr_shape));
  runinfo.output_shapes.insert(std::make_pair("NewX", x_shape));
  runinfo.output_shapes.insert(std::make_pair("NewVeloc", x_shape));
  runinfo.input_buffers = {
      {"X", MakeBuffer(x_shape)},
      {"Grad", MakeBuffer(x_shape)},
      {"Veloc", MakeBuffer(x_shape)},
      {"LR", MakeBuffer(lr_shape)},
  };
  return runinfo;
}

lang::RunInfo LoadPow(const std::string& name,  //
                      const TensorShape& i1,    //
                      const TensorShape& i2,    //
                      const TensorShape& output) {
  auto code = R"***(
function (X, Y) -> (O) {
    O = pow(X, Y);
})***";
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = code;
  runinfo.input_shapes.insert(std::make_pair("X", i1));
  runinfo.input_shapes.insert(std::make_pair("Y", i2));
  runinfo.output_shapes.insert(std::make_pair("O", output));
  runinfo.input_buffers = {
      {"X", MakeBuffer(i1)},
      {"Y", MakeBuffer(i2)},
  };
  return runinfo;
}

lang::RunInfo LoadLayerNorm4dAx2(const std::string& name,  //
                                 const TensorShape& input) {
  // Note: I/G/B/O should all have the same shape, so pass in one shape to share
  const TensorShape epsilon_shape = SimpleShape(DataType::FLOAT32, {});
  auto code = R"***(
function (I[I0, I1, I2, I3], G[I0, I1, I2, I3], B[I0, I1, I2, I3], Epsilon[]) -> (O) {
    H = I2 * I3;
    Sum[i0, i1, 0, 0: I0, I1, 1, 1] = +(I[i0, i1, i2, i3]);
    Mu = Sum / H;
    Diff = I - Mu;
    SqDiff = Diff * Diff;
    SumSqDiff[i0, i1, 0, 0: I0, I1, 1, 1] = +(SqDiff[i0, i1, i2, i3]);
    Stdev = sqrt(SumSqDiff + Epsilon) / H;
    O = (G / Stdev) * (I - Mu) + B;
})***";
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = code;
  runinfo.input_shapes.insert(std::make_pair("I", input));
  runinfo.input_shapes.insert(std::make_pair("G", input));
  runinfo.input_shapes.insert(std::make_pair("B", input));
  runinfo.input_shapes.insert(std::make_pair("Epsilon", epsilon_shape));
  runinfo.output_shapes.insert(std::make_pair("O", input));
  runinfo.input_buffers = {
      {"I", MakeBuffer(input)},
      {"G", MakeBuffer(input)},
      {"B", MakeBuffer(input)},
      {"Epsilon", MakeBuffer(epsilon_shape)},
  };
  return runinfo;
}

lang::RunInfo LoadPolygonBoxTransform(const std::string& name,  //
                                      const TensorShape& input) {
  // Note: I and O have the same shape
  auto code = R"***(
function (I[N, C, H, W]) -> (O) {
    Widx = index(I, 3);
    TEpartial[2*n, c, h, w: N,C,H,W] = =(I[2*n, c, h, w]);
    TE = 4 * Widx - TEpartial;
    Hidx = index(I, 2);
    TOpartial[2*n + 1, c, h, w: N,C,H,W] = =(I[2*n+1, c, h, w]);
    TO = 4 * Hidx - TOpartial;
    O = TE + TO;
})***";
  lang::RunInfo runinfo;
  runinfo.program_name = name;
  runinfo.code = code;
  runinfo.input_shapes.insert(std::make_pair("I", input));
  runinfo.output_shapes.insert(std::make_pair("O", input));
  runinfo.input_buffers = {
      {"I", MakeBuffer(input)},
  };
  return runinfo;
}

}  // namespace lib
}  // namespace tile
}  // namespace vertexai
