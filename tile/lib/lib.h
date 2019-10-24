#pragma once

#include <string>
#include <vector>

#include "plaidml2/edsl/edsl.h"
#include "tile/lang/runinfo.h"

namespace vertexai {
namespace tile {
namespace lib {

using plaidml::edsl::LogicalShape;
using plaidml::edsl::Tensor;

enum class ConvolutionFormat {
  ChannelsFirst,
  ChannelsLast,
};

// TODO: add support for AutoPadding
Tensor Convolution(const Tensor& I,                                               //
                   const Tensor& K,                                               //
                   const std::vector<int64_t>& O_dims,                            //
                   std::vector<size_t> strides = {},                              //
                   ConvolutionFormat I_format = ConvolutionFormat::ChannelsLast,  //
                   ConvolutionFormat K_format = ConvolutionFormat::ChannelsLast);

lang::RunInfo LoadMaxPool2d(const std::string& name,    //
                            const LogicalShape& input,  //
                            const std::vector<size_t>& pool_size);

lang::RunInfo LoadMatMul(const std::string& name,     //
                         const LogicalShape& input1,  //
                         const LogicalShape& input2);

lang::RunInfo LoadMatMulIntermediate(const std::string& name,  //
                                     const LogicalShape& i1,   //
                                     const LogicalShape& i2,   //
                                     const LogicalShape& i3);

lang::RunInfo LoadMatMulAmongEltwise(const std::string& name,  //
                                     const LogicalShape& i1,   //
                                     const LogicalShape& i2,   //
                                     const LogicalShape& i3);

lang::RunInfo LoadEltwiseAdd(const std::string& name,  //
                             const LogicalShape& i1,   //
                             const LogicalShape& i2);

lang::RunInfo LoadEltwiseMultiAdd(const std::string& name,  //
                                  const LogicalShape& i1,   //
                                  const LogicalShape& i2,   //
                                  const LogicalShape& i3,   //
                                  const LogicalShape& i4);

lang::RunInfo LoadEltwiseMulFlip(const std::string& name, const LogicalShape& i1, const LogicalShape& i2);

lang::RunInfo LoadEltwiseDiv(const std::string& name,  //
                             const LogicalShape& i1,   //
                             const LogicalShape& i2);

lang::RunInfo LoadEltwiseMul(const std::string& name,  //
                             const LogicalShape& i1,   //
                             const LogicalShape& i2);

lang::RunInfo LoadEltwiseMultiMul(const std::string& name,  //
                                  const LogicalShape& i1,   //
                                  const LogicalShape& i2,   //
                                  const LogicalShape& i3,   //
                                  const LogicalShape& i4);

lang::RunInfo LoadSin(const std::string& name, const LogicalShape& i1);

lang::RunInfo LoadTanh(const std::string& name, const LogicalShape& i1);

lang::RunInfo LoadMulThenNeg(const std::string& name,  //
                             const LogicalShape& i1,   //
                             const LogicalShape& i2);

lang::RunInfo LoadNegThenMul(const std::string& name,  //
                             const LogicalShape& i1,   //
                             const LogicalShape& i2);

lang::RunInfo LoadConstCalc(const std::string& name);

lang::RunInfo LoadConstScalarMul(const std::string& name,  //
                                 const double s,           //
                                 const LogicalShape& i1);

lang::RunInfo LoadConv1d(const std::string& name,     //
                         const LogicalShape& input,   //
                         const LogicalShape& kernel,  //
                         const std::vector<int64_t>& output);

lang::RunInfo LoadConv2d(const std::string& name,     //
                         const LogicalShape& input,   //
                         const LogicalShape& kernel,  //
                         const std::vector<int64_t>& output);

lang::RunInfo LoadConv2dRelu(const std::string& name,     //
                             const LogicalShape& input,   //
                             const LogicalShape& kernel,  //
                             const std::vector<int64_t>& output);

lang::RunInfo LoadConv2dBnRelu(const std::string& name,       //
                               const LogicalShape& input,     //
                               const LogicalShape& kernel,    //
                               const LogicalShape& channels,  //
                               const std::vector<int64_t>& output);

lang::RunInfo LoadConv2d3Deep(const std::string& name,      //
                              const LogicalShape& input,    //
                              const LogicalShape& kernel1,  //
                              const LogicalShape& kernel2,  //
                              const LogicalShape& kernel3);

lang::RunInfo LoadDilatedConv2d(const std::string& name,    //
                                const LogicalShape& input,  //
                                const LogicalShape& kernel);

lang::RunInfo LoadLarsMomentum4d(const std::string& name,      //
                                 const LogicalShape& x_shape,  //
                                 const LogicalShape& lr_shape);

lang::RunInfo LoadPow(const std::string& name,  //
                      const LogicalShape& i1,   //
                      const LogicalShape& i2);

lang::RunInfo LoadLayerNorm4dAx2(const std::string& name,  //
                                 const LogicalShape& input);

lang::RunInfo LoadBatchNormalization(const std::string& name,  //
                                      const LogicalShape& input);

lang::RunInfo LoadPolygonBoxTransform(const std::string& name,  //
                                      const LogicalShape& input);

lang::RunInfo LoadSoftmax(const std::string& name,  //
                          const LogicalShape& input);

}  // namespace lib
}  // namespace tile
}  // namespace vertexai
