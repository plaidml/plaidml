#include "tile/lib/lib.h"

#include <boost/format.hpp>

#include "tile/lang/tile_cc.h"
#include "tile/util/tile_file.h"

namespace vertexai {
namespace tile {
namespace lib {

using namespace lang;  // NOLINT

namespace {

std::shared_ptr<BufferBase> MakeBuffer(const TensorShape& shape) {
  auto buffer = std::make_shared<util::SimpleBuffer>();
  buffer->bytes.resize(shape.byte_size());
  return buffer;
}

Tensor MatMul(const Tensor& A, const Tensor& B) {
  auto M = A[0], N = B[1];
  Index k("k"), m("m"), n("n");
  Tensor C("C");
  C({m, n}, {M, N}) += A({m, k}) * B({k, n});
  return C;
}

Tensor Convolution1(const Tensor& I, const Tensor& K) {
  auto X = I[0], CO = K[2];
  auto kc = K.shape().dims[0].size / 2;
  Index x("x"), kx("kx"), co("co"), ci("ci");
  Tensor O("O");
  O({x, co}, {X, CO}) += I({x + kx - kc, ci}) * K({kx, ci, co});
  return O;
}

Tensor Convolution2(const Tensor& I, const Tensor& K) {
  auto N = I[0], H = I[1], W = I[2];
  auto KH = K[0], KW = K[1], CO = K[3];
  auto kc0 = K.shape().dims[0].size / 2;
  auto kc1 = K.shape().dims[1].size / 2;
  Index n("n"), x0("x0"), x1("x1"), kx("kx"), ky("ky"), co("co"), ci("ci");
  Tensor O("O");
  O({n, x0, x1, co}, {N, H - (KH - 1), W - (KW - 1), CO}) +=
      I({n, x0 + kx - kc0, x1 + ky - kc1, ci}) * K({kx, ky, ci, co});
  return O;
}

Tensor DilatedConvolution2(const Tensor& I, const Tensor& K) {
  auto N = I[0], Lx = I[1], Ly = I[2], LKx = K[0], LKy = K[1], CO = K[3];
  Tensor O("O");
  Index n, x, y, kx, ky, ci, co;
  O({n, x, y, co}, {N, Lx - 2 * (LKx - 1), Ly - 3 * (LKy - 1), CO}) +=
      I({n, x + 2 * kx, y + 3 * ky, ci}) * K({kx, ky, ci, co});
  return O;
}

Tensor Relu(const Tensor& X) { return Call("relu", {X}); }

}  // namespace

RunInfo LoadMatMul(const std::string& name, const TensorShape& i1, const TensorShape& i2) {
  Tensor A(i1, "A");
  Tensor B(i2, "B");
  return Evaluate(name, {MatMul(A, B)});
}

RunInfo LoadEltwiseAdd(const std::string& name, const TensorShape& i1, const TensorShape& i2) {
  Tensor A(i1, "A");
  Tensor B(i2, "B");
  return Evaluate(name, {A + B});
}

RunInfo LoadConstCalc(const std::string& name) {
  Tensor N(1);
  Tensor F(0.0);
  Tensor F2(3.7);
  Index i;
  Tensor Simple;
  Simple({i}, {1}) = F({});
  Tensor DoubleN;
  DoubleN({i}, {1}) = N({});
  Tensor Partial = Simple + DoubleN;
  Tensor O = Partial + F2;
  return Evaluate(name, {O});
}

RunInfo LoadConv1d(const std::string& name,   //
                   const TensorShape& input,  //
                   const TensorShape& kernel) {
  Tensor I(input, "I");
  Tensor K(kernel, "K");
  auto runinfo = Evaluate(name, {Convolution1(I, K)});
  runinfo.const_inputs = {"K"};
  runinfo.input_buffers = {{"K", MakeBuffer(kernel)}};
  return runinfo;
}

RunInfo LoadConv2d(const std::string& name,   //
                   const TensorShape& input,  //
                   const TensorShape& kernel) {
  Tensor I(input, "I");
  Tensor K(kernel, "K");
  auto runinfo = Evaluate(name, {Convolution2(I, K)});
  runinfo.const_inputs = {"K"};
  runinfo.input_buffers = {{"K", MakeBuffer(kernel)}};
  return runinfo;
}

RunInfo LoadConv2dRelu(const std::string& name,   //
                       const TensorShape& input,  //
                       const TensorShape& kernel) {
  Tensor I(input, "I");
  Tensor K(kernel, "K");
  auto runinfo = Evaluate(name, {Relu(Convolution2(I, K))});
  runinfo.const_inputs = {"K"};
  runinfo.input_buffers = {{"K", MakeBuffer(kernel)}};
  return runinfo;
}

RunInfo LoadConv2dBnRelu(const std::string& name,    //
                         const TensorShape& input,   //
                         const TensorShape& kernel,  //
                         const TensorShape& channels) {
  Tensor I(input, "I");
  Tensor K(kernel, "K");
  Tensor B(channels, "B");
  Tensor S(channels, "S");
  auto O = Convolution2(I, K);
  auto R = Relu((O + B) * S);
  auto runinfo = Evaluate(name, {R});
  runinfo.const_inputs = {"K"};
  runinfo.input_buffers = {
      {"K", MakeBuffer(kernel)},
      {"B", MakeBuffer(channels)},
      {"S", MakeBuffer(channels)},
  };
  return runinfo;
}

RunInfo LoadConv2d3Deep(const std::string& name,     //
                        const TensorShape& input,    //
                        const TensorShape& kernel1,  //
                        const TensorShape& kernel2,  //
                        const TensorShape& kernel3) {
  Tensor I(input, "I");
  Tensor K1(input, "K1");
  Tensor K2(input, "K2");
  Tensor K3(input, "K3");
  auto O1 = Convolution2(I, K1);
  auto O2 = Convolution2(O1, K2);
  auto O3 = Convolution2(O2, K3);
  auto runinfo = Evaluate(name, {O3});
  runinfo.const_inputs = {"K1", "K2", "K3"};
  runinfo.input_buffers = {
      {"K1", MakeBuffer(kernel1)},
      {"K2", MakeBuffer(kernel2)},
      {"K3", MakeBuffer(kernel3)},
  };
  return runinfo;
}

RunInfo LoadDilatedConv2d(const std::string& name,   //
                          const TensorShape& input,  //
                          const TensorShape& kernel) {
  Tensor I(input);
  Tensor K(kernel);
  return Evaluate(name, {DilatedConvolution2(I, K)});
}

Tensor Normalize(const Tensor& X) {
  auto XSqr = X * X;
  Tensor X_MS;
  {
    std::vector<Index> idxs(X.shape().dims.size());
    X_MS({}) += XSqr(idxs);
  }
  return sqrt(X_MS);
}

std::tuple<Tensor, Tensor> LarsMomentum(const Tensor& X,           //
                                        const Tensor& Grad,        //
                                        const Tensor& Veloc,       //
                                        const Tensor& LR,          //
                                        double lars_coeff,         //
                                        double lars_weight_decay,  //
                                        double momentum) {
  auto XNorm = Normalize(X);
  auto GradNorm = Normalize(Grad);
  auto LocLR = LR * lars_coeff * XNorm / (GradNorm + lars_weight_decay * XNorm);
  auto NewVeloc = momentum * Veloc + LocLR * (Grad + lars_weight_decay * X);
  return std::make_tuple(X - NewVeloc, NewVeloc);
}

RunInfo LoadLarsMomentum4d(const std::string& name,     //
                           const TensorShape& x_shape,  //
                           const TensorShape& lr_shape) {
  // Note: X/Grad/Veloc/NewX/NewVeloc should all have the same shape for the
  // semantics of this operation to be correct, so we only pass in 1 shape for
  // all of them.
  double lars_coeff = 1. / 1024.;
  double lars_weight_decay = 1. / 2048.;
  double momentum = 1. / 8.;
  Tensor X(x_shape);
  Tensor Grad(x_shape);
  Tensor Veloc(x_shape);
  Tensor LR(lr_shape);
  auto R = LarsMomentum(X, Grad, Veloc, LR, lars_coeff, lars_weight_decay, momentum);
  return Evaluate("lars_momentum4d", {std::get<0>(R), std::get<1>(R)});
}

RunInfo LoadPow(const std::string& name,  //
                const TensorShape& i1,    //
                const TensorShape& i2) {
  Tensor X(i1, "X");
  Tensor Y(i2, "Y");
  auto runinfo = Evaluate(name, {pow(X, Y)});
  runinfo.input_buffers = {
      {"X", MakeBuffer(i1)},
      {"Y", MakeBuffer(i2)},
  };
  return runinfo;
}

Tensor Norm4dAx2(const Tensor& I, const Tensor& G, const Tensor& B, const Tensor& Epsilon) {
  int64_t H = I[2] * I[3];
  Tensor Sum;
  Index i0, i1, i2, i3;
  Sum({i0, i1, 0, 0}, {I[0], I[1], 1, 1}) += I({i0, i1, i2, i3});
  auto Mu = Sum / H;
  auto Diff = I - Mu;
  auto SqDiff = Diff * Diff;
  Tensor SumSqDiff;
  SumSqDiff({i0, i1, 0, 0}, {I[0], I[1], 1, 1}) += SqDiff({i0, i1, i2, i3});
  auto Stdev = sqrt(SumSqDiff + Epsilon) / H;
  return (G / Stdev) * (I - Mu) + B;
}

RunInfo LoadLayerNorm4dAx2(const std::string& name,  //
                           const TensorShape& input) {
  // Note: I/G/B/O should all have the same shape, so pass in one shape to share
  Tensor I(input);
  Tensor G(input);
  Tensor B(input);
  Tensor Epsilon(SimpleShape(DataType::FLOAT32, {}));
  return Evaluate(name, {Norm4dAx2(I, G, B, Epsilon)});
}

Tensor PolygonBoxTransform(const Tensor& I) {
  Tensor TEpartial;
  Tensor TOpartial;
  auto N = I[0], C = I[1], H = I[2], W = I[3];
  Index n, c, h, w;
  auto Widx = index(I, 3);
  TEpartial({2 * n, c, h, w}, {N, C, H, W}) = I({2 * n, c, h, w});
  auto TE = 4 * Widx - TEpartial;
  TOpartial({2 * n + 1, c, h, w}, {N, C, H, W}) = I({2 * n + 1, c, h, w});
  auto Hidx = index(I, 2);
  auto TO = 4 * Hidx - TOpartial;
  return TE + TO;
}

RunInfo LoadPolygonBoxTransform(const std::string& name,  //
                                const TensorShape& input) {
  // Note: I and O have the same shape
  Tensor I(input);
  return Evaluate(name, {PolygonBoxTransform(I)});
}

}  // namespace lib
}  // namespace tile
}  // namespace vertexai
