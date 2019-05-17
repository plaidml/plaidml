#include "tile/lib/lib.h"

#include <boost/format.hpp>

#include "base/util/stream_container.h"
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
  TensorDim M, N, K;
  A.match_dims(M, K);
  B.match_dims(K, N);
  TensorIndex k("k"), m("m"), n("n");
  Tensor C("C", M, N);
  C(m, n) += A(m, k) * B(k, n);
  return C;
}

Tensor DilatedConvolution2(const Tensor& I, const Tensor& K) {
  TensorDim N, Lx, Ly, LKx, LKy, CI, CO;
  I.match_dims(N, Lx, Ly, CI);
  K.match_dims(LKx, LKy, CI, CO);
  Tensor O("O", N, Lx - 2 * (LKx - 1), Ly - 3 * (LKy - 1), CO);
  TensorIndex n, x, y, kx, ky, ci, co;
  O(n, x, y, co) += I(n, x + 2 * kx, y + 3 * ky, ci) * K(kx, ky, ci, co);
  return O;
}

Tensor Relu(const Tensor& X) { return Call("relu", X); }

Tensor Sin(const Tensor& X) { return Call("sin", X); }

Tensor Tanh(const Tensor& X) { return Call("tanh", X); }

}  // namespace

Tensor Convolution(const Tensor& I,                     //
                   const Tensor& K,                     //
                   const std::vector<size_t>& O_sizes,  //
                   std::vector<size_t> strides,         //
                   ConvolutionFormat I_format,          //
                   ConvolutionFormat K_format) {
  TensorDim N, CI, CO;
  auto I_shape = I.shape();
  auto K_shape = K.shape();
  auto rank = I_shape.dims.size() - 2;
  if (strides.empty()) {
    for (size_t i = 0; i < rank; i++) {
      strides.push_back(1);
    }
  } else if (strides.size() != rank) {
    throw std::runtime_error(
        str(boost::format("Convolution strides length inconsistent with input shape: %1% (rank %2%) v %3% (rank %4%)") %
            StreamContainer(strides) % strides.size() % I_shape % rank));
  }
  TensorIndex n("n"), co("co"), ci("ci");
  std::vector<TensorDim> I_dims = {N};
  std::vector<TensorDim> I_spatial_dims(rank);
  std::vector<TensorDim> K_dims;
  std::vector<TensorDim> K_spatial_dims(rank);
  std::vector<TensorDim> O_dims;
  for (const auto& size : O_sizes) {
    O_dims.emplace_back(size);
  }
  std::vector<TensorIndex> K_idxs;
  std::vector<TensorIndex> I_idxs = {n};
  std::vector<TensorIndex> O_idxs = {n};
  size_t K_spatial_dims_offset = 0;
  if (K_format == ConvolutionFormat::ChannelsFirst) {
    K_spatial_dims_offset = 2;
    K_idxs.push_back(co);
    K_idxs.push_back(ci);
    K_dims.push_back(CO);
    K_dims.push_back(CI);
  }
  if (I_format == ConvolutionFormat::ChannelsFirst) {
    I_idxs.push_back(ci);
    O_idxs.push_back(co);
    I_dims.push_back(CI);
  }
  K_dims.insert(std::end(K_dims), std::begin(K_spatial_dims), std::end(K_spatial_dims));
  I_dims.insert(std::end(I_dims), std::begin(I_spatial_dims), std::end(I_spatial_dims));
  for (size_t i = 0; i < rank; i++) {
    TensorIndex x(str(boost::format("x%1%") % i));
    TensorIndex k(str(boost::format("k%1%") % i));
    auto K_dim = K_shape.dims[K_spatial_dims_offset + i].size;
    I_idxs.emplace_back(strides[i] * x + k - K_dim / 2);
    K_idxs.push_back(k);
    O_idxs.push_back(x);
  }
  if (I_format == ConvolutionFormat::ChannelsLast) {
    I_idxs.push_back(ci);
    I_dims.push_back(CI);
    O_idxs.push_back(co);
  }
  if (K_format == ConvolutionFormat::ChannelsLast) {
    K_idxs.push_back(ci);
    K_idxs.push_back(co);
    K_dims.push_back(CI);
    K_dims.push_back(CO);
  }
  I.match_dims(I_dims);
  K.match_dims(K_dims);
  Tensor O("O", O_dims);
  O(O_idxs) += I(I_idxs) * K(K_idxs);
  return O;
}

RunInfo LoadMatMul(const std::string& name, const TensorShape& i1, const TensorShape& i2) {
  Tensor A("A", i1);
  Tensor B("B", i2);
  return Evaluate(name, {MatMul(A, B)});
}

RunInfo LoadMatMulIntermediate(const std::string& name, const TensorShape& i1, const TensorShape& i2,
                               const TensorShape& i3) {
  Tensor A("A", i1);
  Tensor B("B", i2);
  Tensor C("C", i2);
  Tensor D = MatMul(A, B);
  Tensor E = D + C;
  return Evaluate(name, {D, E});
}

RunInfo LoadEltwiseMulFlip(const std::string& name, const TensorShape& i1, const TensorShape& i2) {
  Tensor A{"A", i1}, B{"B", i2};
  return Evaluate(name, {~(A * B)});
}

RunInfo LoadMatMulAmongEltwise(const std::string& name, const TensorShape& i1, const TensorShape& i2,
                               const TensorShape& i3) {
  Tensor A("A", i1);
  Tensor B("B", i2);
  Tensor C("C", i3);
  Tensor NegA = -A;
  Tensor NegB = -B;
  Tensor P = MatMul(NegA, NegB);
  return Evaluate(name, {P + C});
}

RunInfo LoadEltwiseAdd(const std::string& name, const TensorShape& i1, const TensorShape& i2) {
  Tensor A("A", i1);
  Tensor B("B", i2);
  return Evaluate(name, {A + B});
}

RunInfo LoadEltwiseMultiAdd(const std::string& name, const TensorShape& i1, const TensorShape& i2,
                            const TensorShape& i3, const TensorShape& i4) {
  Tensor A("A", i1);
  Tensor B("B", i2);
  Tensor C("C", i3);
  Tensor D("D", i4);
  return Evaluate(name, {A + B + C + D});
}

RunInfo LoadEltwiseDiv(const std::string& name, const TensorShape& i1, const TensorShape& i2) {
  Tensor A("A", i1);
  Tensor B("B", i2);
  return Evaluate(name, {A / B});
}

RunInfo LoadEltwiseMul(const std::string& name, const TensorShape& i1, const TensorShape& i2) {
  Tensor A("A", i1);
  Tensor B("B", i2);
  return Evaluate(name, {A * B});
}

RunInfo LoadEltwiseMultiMul(const std::string& name, const TensorShape& i1, const TensorShape& i2,
                            const TensorShape& i3, const TensorShape& i4) {
  Tensor A("A", i1);
  Tensor B("B", i2);
  Tensor C("C", i3);
  Tensor D("D", i4);
  return Evaluate(name, {A * B * C * D});
}

RunInfo LoadSin(const std::string& name, const TensorShape& i1) {
  Tensor A("A", i1);
  return Evaluate(name, {Sin(A)});
}

RunInfo LoadTanh(const std::string& name, const TensorShape& i1) {
  Tensor A("A", i1);
  return Evaluate(name, {Tanh(A)});
}

RunInfo LoadMulThenNeg(const std::string& name, const TensorShape& i1, const TensorShape& i2) {
  Tensor A("A", i1);
  Tensor B("B", i2);
  Tensor C = A * B;
  return Evaluate(name, {-C});
}

RunInfo LoadNegThenMul(const std::string& name, const TensorShape& i1, const TensorShape& i2) {
  Tensor A("A", i1);
  Tensor B("B", i2);
  Tensor NegA = -A;
  Tensor NegB = -B;
  return Evaluate(name, {NegA * NegB});
}

RunInfo LoadConstCalc(const std::string& name) {
  Tensor N(1);
  Tensor F(0.0);
  Tensor F2(3.7);
  TensorIndex i;
  auto Simple = TensorOutput(1);
  Simple(i) = F();
  auto DoubleN = TensorOutput(1);
  DoubleN(i) = N();
  Tensor Partial = Simple + DoubleN;
  Tensor O = Partial + F2;
  return Evaluate(name, {O});
}

RunInfo LoadConv1d(const std::string& name,    //
                   const TensorShape& input,   //
                   const TensorShape& kernel,  //
                   const std::vector<size_t>& output) {
  Tensor I("I", input);
  Tensor K("K", kernel);
  auto runinfo = Evaluate(name, {Convolution(I, K, output)});
  runinfo.const_inputs = {"K"};
  runinfo.input_buffers = {{"K", MakeBuffer(kernel)}};
  return runinfo;
}

RunInfo LoadConv2d(const std::string& name,    //
                   const TensorShape& input,   //
                   const TensorShape& kernel,  //
                   const std::vector<size_t>& output) {
  Tensor I("I", input);
  Tensor K("K", kernel);
  auto runinfo = Evaluate(name, {Convolution(I, K, output)});
  runinfo.const_inputs = {"K"};
  runinfo.input_buffers = {{"K", MakeBuffer(kernel)}};
  return runinfo;
}

RunInfo LoadConv2dRelu(const std::string& name,    //
                       const TensorShape& input,   //
                       const TensorShape& kernel,  //
                       const std::vector<size_t>& output) {
  Tensor I("I", input);
  Tensor K("K", kernel);
  auto runinfo = Evaluate(name, {Relu(Convolution(I, K, output))});
  runinfo.const_inputs = {"K"};
  runinfo.input_buffers = {{"K", MakeBuffer(kernel)}};
  return runinfo;
}

RunInfo LoadConv2dBnRelu(const std::string& name,      //
                         const TensorShape& input,     //
                         const TensorShape& kernel,    //
                         const TensorShape& channels,  //
                         const std::vector<size_t>& output) {
  Tensor I("I", input);
  Tensor K("K", kernel);
  Tensor B("B", channels);
  Tensor S("S", channels);
  auto O = Convolution(I, K, output);
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
  Tensor I("I", input);
  Tensor K1("K1", input);
  Tensor K2("K2", input);
  Tensor K3("K3", input);
  auto I_dims = input.sizes();
  auto O1 = Convolution(I, K1, {I_dims[0], I_dims[1], I_dims[2], kernel1.dims[3].size});
  auto O2 = Convolution(O1, K2, {I_dims[0], I_dims[1], I_dims[2], kernel2.dims[3].size});
  auto O3 = Convolution(O2, K3, {I_dims[0], I_dims[1], I_dims[2], kernel3.dims[3].size});
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
    std::vector<TensorIndex> idxs(X.shape().dims.size());
    X_MS() += XSqr(idxs);
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
  Tensor X("X", i1);
  Tensor Y("Y", i2);
  auto runinfo = Evaluate(name, {pow(X, Y)});
  runinfo.input_buffers = {
      {"X", MakeBuffer(i1)},
      {"Y", MakeBuffer(i2)},
  };
  return runinfo;
}

Tensor Norm4dAx2(const Tensor& I, const Tensor& G, const Tensor& B, const Tensor& Epsilon) {
  TensorDim I0, I1, I2, I3;
  I.match_dims(I0, I1, I2, I3);
  int64_t H = I.dims(2) * I.dims(3);
  auto Sum = TensorOutput(I0, I1, 1, 1);
  TensorIndex i0, i1, i2, i3;
  Sum(i0, i1, 0, 0) += I(i0, i1, i2, i3);
  auto Mu = Sum / H;
  auto Diff = I - Mu;
  auto SqDiff = Diff * Diff;
  auto SumSqDiff = TensorOutput(I0, I1, 1, 1);
  SumSqDiff(i0, i1, 0, 0) += SqDiff(i0, i1, i2, i3);
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
  TensorDim N, C, H, W;
  I.match_dims(N, C, H, W);
  auto TEpartial = TensorOutput(N, C, H, W);
  auto TOpartial = TensorOutput(N, C, H, W);
  TensorIndex n, c, h, w;
  auto Widx = index(I, 3);
  TEpartial(2 * n, c, h, w) = I(2 * n, c, h, w);
  auto TE = 4 * Widx - TEpartial;
  TOpartial(2 * n + 1, c, h, w) = I(2 * n + 1, c, h, w);
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
