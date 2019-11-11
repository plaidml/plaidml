#include "tile/lib/lib.h"

#include <memory>
#include <tuple>

#include <boost/format.hpp>

#include "base/util/stream_container.h"
#include "plaidml2/edsl/autodiff.h"
#include "tile/util/tile_file.h"

namespace vertexai {
namespace tile {
namespace lib {

using namespace plaidml::edsl;  // NOLINT

namespace {

lang::RunInfo Evaluate(const std::string& name, const std::vector<Tensor>& vars) {
  Program program(name, vars);
  return *static_cast<const tile::lang::RunInfo*>(program.runinfo());
}

std::shared_ptr<lang::BufferBase> MakeBuffer(const LogicalShape& shape) {
  std::vector<size_t> sizes;
  for (const auto& dim : shape.int_dims()) {
    sizes.push_back(dim);
  }
  auto tensor_shape = tile::SimpleShape(static_cast<DataType>(shape.dtype()), sizes);
  auto buffer = std::make_shared<util::SimpleBuffer>();
  buffer->bytes.resize(tensor_shape.byte_size());
  return buffer;
}

Tensor MatMul(const Tensor& A, const Tensor& B) {
  TensorDim M, N, K;
  A.bind_dims(M, K);
  B.bind_dims(K, N);
  TensorIndex k("k"), m("m"), n("n");
  auto C = NamedTensorOutput("C", M, N);
  C(m, n) += A(m, k) * B(k, n);
  return C;
}

Tensor DilatedConvolution2(const Tensor& I, const Tensor& K) {
  TensorDim N, Lx, Ly, LKx, LKy, CI, CO;
  I.bind_dims(N, Lx, Ly, CI);
  K.bind_dims(LKx, LKy, CI, CO);
  auto O = NamedTensorOutput("O", N, Lx - 2 * (LKx - 1), Ly - 3 * (LKy - 1), CO);
  TensorIndex n, x, y, kx, ky, ci, co;
  O(n, x, y, co) += I(n, x + 2 * kx, y + 3 * ky, ci) * K(kx, ky, ci, co);
  return O;
}

Tensor MaxPool2d(const Tensor& I, std::vector<size_t> pool_size, std::vector<size_t> strides) {
  TensorDim N, X, Y, C;
  auto I_shape = I.shape();
  auto ndims = I_shape.ndims() - 2;
  if (ndims != 2) {
    throw std::runtime_error("MaxPool2d requires exactly 2 spatial dimensions");
  }
  while (pool_size.size() < ndims) {
    pool_size.push_back(2);
  }
  if (pool_size.size() != ndims) {
    throw std::runtime_error("Pool window size needs as many dims as tensor spatial dims");
  }
  while (strides.size() < ndims) {
    strides.push_back(pool_size[strides.size()]);
  }
  if (strides.size() != ndims) {
    throw std::runtime_error("Pool strides must have as many dims as tensor spatial dims");
  }
  TensorIndex n("n"), x("x"), y("y"), kx("kx"), ky("ky"), c("c");
  std::vector<TensorDim> I_dims = {N, X, Y, C};
  I.bind_dims(I_dims);
  auto O = NamedTensorOutput("O", N, X / strides[0], Y / strides[1], C);
  std::vector<Constraint> constraints{kx < pool_size[0], ky < pool_size[1]};
  O(n, x, y, c) >= I(n, strides[0] * x + kx, strides[1] * y + ky, c);
  O.add_constraints(constraints);
  return O;
}

Tensor Relu(const Tensor& X) { return Call("relu", X); }

Tensor Sin(const Tensor& X) { return Call("sin", X); }

Tensor Tanh(const Tensor& X) { return Call("tanh", X); }

}  // namespace

Tensor Convolution(const Tensor& I,                      //
                   const Tensor& K,                      //
                   const std::vector<int64_t>& O_sizes,  //
                   std::vector<size_t> strides,          //
                   ConvolutionFormat I_format,           //
                   ConvolutionFormat K_format) {
  TensorDim N, CI, CO;
  auto I_shape = I.shape();
  auto K_shape = K.shape();
  IVLOG(1, "I.shape(): " << I_shape);
  IVLOG(1, "K.shape(): " << K_shape);
  auto ndims = I_shape.ndims() - 2;
  if (strides.empty()) {
    for (size_t i = 0; i < ndims; i++) {
      strides.push_back(1);
    }
  } else if (strides.size() != ndims) {
    throw std::runtime_error(str(
        boost::format("Convolution strides length inconsistent with input shape: %1% (ndims %2%) v %3% (ndims %4%)") %
        StreamContainer(strides) % strides.size() % I_shape % ndims));
  }
  TensorIndex n("n"), co("co"), ci("ci");
  std::vector<TensorDim> I_dims = {N};
  std::vector<TensorDim> I_spatial_dims(ndims);
  std::vector<TensorDim> K_dims;
  std::vector<TensorDim> K_spatial_dims(ndims);
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
  if (I_format == ConvolutionFormat::ChannelsLast) {
    I_dims.push_back(CI);
  }
  if (K_format == ConvolutionFormat::ChannelsLast) {
    K_dims.push_back(CI);
    K_dims.push_back(CO);
  }
  I.bind_dims(I_dims);
  K.bind_dims(K_dims);
  for (size_t i = 0; i < ndims; i++) {
    TensorIndex x(str(boost::format("x%1%") % i));
    TensorIndex k(str(boost::format("k%1%") % i));
    IVLOG(1, "Adding " << i);
    I_idxs.emplace_back(strides[i] * x + k - K_dims[K_spatial_dims_offset + i] / 2);
    K_idxs.push_back(k);
    O_idxs.push_back(x);
  }
  if (I_format == ConvolutionFormat::ChannelsLast) {
    I_idxs.push_back(ci);
    O_idxs.push_back(co);
  }
  if (K_format == ConvolutionFormat::ChannelsLast) {
    K_idxs.push_back(ci);
    K_idxs.push_back(co);
  }
  Tensor O("O", O_dims);
  O(O_idxs) += I(I_idxs) * K(K_idxs);
  return O;
}

lang::RunInfo LoadMatMul(const std::string& name, const LogicalShape& i1, const LogicalShape& i2) {
  auto A = Placeholder(i1, "A");
  auto B = Placeholder(i2, "B");
  return Evaluate(name, {MatMul(A, B)});
}

lang::RunInfo LoadMatMulIntermediate(const std::string& name, const LogicalShape& i1, const LogicalShape& i2,
                                     const LogicalShape& i3) {
  auto A = Placeholder(i1, "A");
  auto B = Placeholder(i2, "B");
  auto C = Placeholder(i3, "C");
  Tensor D = MatMul(A, B);
  Tensor E = D + C;
  return Evaluate(name, {D, E});
}

lang::RunInfo LoadEltwiseMulFlip(const std::string& name, const LogicalShape& i1, const LogicalShape& i2) {
  auto A = Placeholder(i1, "A");
  auto B = Placeholder(i2, "B");
  return Evaluate(name, {~(A * B)});
}

lang::RunInfo LoadMatMulAmongEltwise(const std::string& name, const LogicalShape& i1, const LogicalShape& i2,
                                     const LogicalShape& i3) {
  auto A = Placeholder(i1, "A");
  auto B = Placeholder(i2, "B");
  auto C = Placeholder(i3, "C");
  Tensor NegA = -A;
  Tensor NegB = -B;
  Tensor P = MatMul(NegA, NegB);
  return Evaluate(name, {P + C});
}

lang::RunInfo LoadEltwiseAdd(const std::string& name, const LogicalShape& i1, const LogicalShape& i2) {
  auto A = Placeholder(i1, "A");
  auto B = Placeholder(i2, "B");
  return Evaluate(name, {A + B});
}

lang::RunInfo LoadEltwiseMultiAdd(const std::string& name, const LogicalShape& i1, const LogicalShape& i2,
                                  const LogicalShape& i3, const LogicalShape& i4) {
  auto A = Placeholder(i1, "A");
  auto B = Placeholder(i2, "B");
  auto C = Placeholder(i3, "C");
  auto D = Placeholder(i4, "D");
  return Evaluate(name, {A + B + C + D});
}

lang::RunInfo LoadEltwiseDiv(const std::string& name, const LogicalShape& i1, const LogicalShape& i2) {
  auto A = Placeholder(i1, "A");
  auto B = Placeholder(i2, "B");
  return Evaluate(name, {A / B});
}

lang::RunInfo LoadConstScalarMul(const std::string& name, const double s, const LogicalShape& i1) {
  Tensor scalar(s);
  auto A = Placeholder(i1, "A");
  return Evaluate(name, {scalar * A});
}

lang::RunInfo LoadEltwiseMul(const std::string& name, const LogicalShape& i1, const LogicalShape& i2) {
  auto A = Placeholder(i1, "A");
  auto B = Placeholder(i2, "B");
  return Evaluate(name, {A * B});
}

lang::RunInfo LoadEltwiseMultiMul(const std::string& name, const LogicalShape& i1, const LogicalShape& i2,
                                  const LogicalShape& i3, const LogicalShape& i4) {
  auto A = Placeholder(i1, "A");
  auto B = Placeholder(i2, "B");
  auto C = Placeholder(i3, "C");
  auto D = Placeholder(i4, "D");
  return Evaluate(name, {A * B * C * D});
}

lang::RunInfo LoadSin(const std::string& name, const LogicalShape& i1) {
  auto A = Placeholder(i1, "A");
  return Evaluate(name, {Sin(A)});
}

lang::RunInfo LoadTanh(const std::string& name, const LogicalShape& i1) {
  auto A = Placeholder(i1, "A");
  return Evaluate(name, {Tanh(A)});
}

lang::RunInfo LoadMulThenNeg(const std::string& name, const LogicalShape& i1, const LogicalShape& i2) {
  auto A = Placeholder(i1, "A");
  auto B = Placeholder(i2, "B");
  Tensor C = A * B;
  return Evaluate(name, {-C});
}

lang::RunInfo LoadNegThenMul(const std::string& name, const LogicalShape& i1, const LogicalShape& i2) {
  auto A = Placeholder(i1, "A");
  auto B = Placeholder(i2, "B");
  Tensor NegA = -A;
  Tensor NegB = -B;
  return Evaluate(name, {NegA * NegB});
}

lang::RunInfo LoadConstCalc(const std::string& name) {
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

lang::RunInfo LoadConv1d(const std::string& name,     //
                         const LogicalShape& input,   //
                         const LogicalShape& kernel,  //
                         const std::vector<int64_t>& output) {
  auto I = Placeholder(input, "I");
  auto K = Placeholder(kernel, "K");
  auto runinfo = Evaluate(name, {Convolution(I, K, output)});
  runinfo.const_inputs = {"K"};
  runinfo.input_buffers = {{"K", MakeBuffer(kernel)}};
  return runinfo;
}

lang::RunInfo LoadMaxPool2d(const std::string& name,    //
                            const LogicalShape& input,  //
                            const std::vector<size_t>& pool_size) {
  // This is a max pool with strides == pool_size
  auto I = Placeholder(input, "I");
  auto runinfo = Evaluate(name, {MaxPool2d(I, pool_size, {})});
  return runinfo;
}

lang::RunInfo LoadConv2d(const std::string& name,     //
                         const LogicalShape& input,   //
                         const LogicalShape& kernel,  //
                         const std::vector<int64_t>& output) {
  auto I = Placeholder(input, "I");
  auto K = Placeholder(kernel, "K");
  auto runinfo = Evaluate(name, {Convolution(I, K, output)});
  runinfo.const_inputs = {"K"};
  runinfo.input_buffers = {{"K", MakeBuffer(kernel)}};
  return runinfo;
}

lang::RunInfo LoadConv2dRelu(const std::string& name,     //
                             const LogicalShape& input,   //
                             const LogicalShape& kernel,  //
                             const std::vector<int64_t>& output) {
  auto I = Placeholder(input, "I");
  auto K = Placeholder(kernel, "K");
  auto runinfo = Evaluate(name, {Relu(Convolution(I, K, output))});
  runinfo.const_inputs = {"K"};
  runinfo.input_buffers = {{"K", MakeBuffer(kernel)}};
  return runinfo;
}

lang::RunInfo LoadConv2dBnRelu(const std::string& name,       //
                               const LogicalShape& input,     //
                               const LogicalShape& kernel,    //
                               const LogicalShape& channels,  //
                               const std::vector<int64_t>& output) {
  auto I = Placeholder(input, "I");
  auto K = Placeholder(kernel, "K");
  auto B = Placeholder(channels, "B");
  auto S = Placeholder(channels, "S");
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

lang::RunInfo LoadConv2d3Deep(const std::string& name,      //
                              const LogicalShape& input,    //
                              const LogicalShape& kernel1,  //
                              const LogicalShape& kernel2,  //
                              const LogicalShape& kernel3) {
  auto I = Placeholder(input, "I");
  auto K1 = Placeholder(kernel1, "K1");
  auto K2 = Placeholder(kernel2, "K2");
  auto K3 = Placeholder(kernel3, "K3");
  auto dims = input.int_dims();
  auto O1 = Convolution(I, K1, {dims[0], dims[1], dims[2], kernel1.int_dims()[3]});
  auto O2 = Convolution(O1, K2, {dims[0], dims[1], dims[2], kernel2.int_dims()[3]});
  auto O3 = Convolution(O2, K3, {dims[0], dims[1], dims[2], kernel3.int_dims()[3]});
  auto runinfo = Evaluate(name, {O3});
  runinfo.const_inputs = {"K1", "K2", "K3"};
  runinfo.input_buffers = {
      {"K1", MakeBuffer(kernel1)},
      {"K2", MakeBuffer(kernel2)},
      {"K3", MakeBuffer(kernel3)},
  };
  return runinfo;
}

lang::RunInfo LoadDilatedConv2d(const std::string& name,    //
                                const LogicalShape& input,  //
                                const LogicalShape& kernel) {
  auto I = Placeholder(input, "I");
  auto K = Placeholder(kernel, "K");
  return Evaluate(name, {DilatedConvolution2(I, K)});
}

Tensor Normalize(const Tensor& X) {
  auto XSqr = X * X;
  auto X_MS = TensorOutput();
  std::vector<TensorIndex> idxs(X.shape().ndims());
  X_MS() += XSqr(idxs);
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

lang::RunInfo LoadLarsMomentum4d(const std::string& name,      //
                                 const LogicalShape& x_shape,  //
                                 const LogicalShape& lr_shape) {
  // Note: X/Grad/Veloc/NewX/NewVeloc should all have the same shape for the
  // semantics of this operation to be correct, so we only pass in 1 shape for
  // all of them.
  double lars_coeff = 1. / 1024.;
  double lars_weight_decay = 1. / 2048.;
  double momentum = 1. / 8.;
  auto X = Placeholder(x_shape);
  auto Grad = Placeholder(x_shape);
  auto Veloc = Placeholder(x_shape);
  auto LR = Placeholder(lr_shape);
  auto R = LarsMomentum(X, Grad, Veloc, LR, lars_coeff, lars_weight_decay, momentum);
  return Evaluate("lars_momentum4d", {std::get<0>(R), std::get<1>(R)});
}

lang::RunInfo LoadPow(const std::string& name,  //
                      const LogicalShape& i1,   //
                      const LogicalShape& i2) {
  auto X = Placeholder(i1, "X");
  auto Y = Placeholder(i2, "Y");
  auto runinfo = Evaluate(name, {pow(X, Y)});
  runinfo.input_buffers = {
      {"X", MakeBuffer(i1)},
      {"Y", MakeBuffer(i2)},
  };
  return runinfo;
}

Tensor Norm4dAx2(const Tensor& I, const Tensor& G, const Tensor& B, const Tensor& Epsilon) {
  TensorDim I0, I1, I2, I3;
  I.bind_dims(I0, I1, I2, I3);
  auto H = I2 * I3;
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

lang::RunInfo LoadLayerNorm4dAx2(const std::string& name,  //
                                 const LogicalShape& input) {
  // Note: I/G/B/O should all have the same shape, so pass in one shape to share
  auto I = Placeholder(input);
  auto G = Placeholder(input);
  auto B = Placeholder(input);
  auto Epsilon = Placeholder(PLAIDML_DATA_FLOAT32, {});
  return Evaluate(name, {Norm4dAx2(I, G, B, Epsilon)});
}

Tensor PolygonBoxTransform(const Tensor& I) {
  TensorDim N, C, H, W;
  I.bind_dims(N, C, H, W);
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

lang::RunInfo LoadPolygonBoxTransform(const std::string& name,  //
                                      const LogicalShape& input) {
  // Note: I and O have the same shape
  auto I = Placeholder(input);
  return Evaluate(name, {PolygonBoxTransform(I)});
}

lang::RunInfo LoadSoftmax(const std::string& name,      //
                          const LogicalShape& input) {  //
  auto X1 = Placeholder(input);
  TensorDim I, J;
  X1.bind_dims(I, J);
  TensorIndex i("i"), j("j");
  auto M = NamedTensorOutput("M", I, 1);
  M(i, 0) >= X1(i, j);
  auto E = exp(X1 - M);
  auto N = NamedTensorOutput("N", I, 1);
  N(i, 0) += E(i, j);
  return Evaluate(name, {E / N});
}

Tensor BatchNormalization(const Tensor& I, const Tensor& M, const Tensor& V, const Tensor& G, const Tensor& B,
                          const Tensor& E) {
  return ((I - M) * G / sqrt(V + E)) + B;
}

lang::RunInfo LoadBatchNormalization(const std::string& name,      //
                                     const LogicalShape& input) {  //
  auto b1dims = input.int_dims();
  b1dims[0] = 1;
  auto b1 = LogicalShape(PLAIDML_DATA_FLOAT32, b1dims);
  auto I = Placeholder(b1);
  auto M = Placeholder(b1);
  auto V = Placeholder(b1);
  auto G = Placeholder(b1);
  auto B = Placeholder(b1);
  auto E = Placeholder(PLAIDML_DATA_FLOAT32, {});
  return Evaluate(name, {BatchNormalization(I, M, V, G, B, E)});
}

}  // namespace lib
}  // namespace tile
}  // namespace vertexai
