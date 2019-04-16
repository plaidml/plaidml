#pragma once

#include <string>

#include "tile/lang/compose.h"

namespace vertexai {
namespace tile {
namespace lib {

lang::RunInfo LoadMatMul(const std::string& name,    //
                         const TensorShape& input1,  //
                         const TensorShape& input2);

lang::RunInfo LoadMatMulIntermediate(const std::string& name,  //
                                     const TensorShape& i1,    //
                                     const TensorShape& i2,    //
                                     const TensorShape& i3);

lang::RunInfo LoadMatMulAmongEltwise(const std::string& name,  //
                                     const TensorShape& i1,    //
                                     const TensorShape& i2,    //
                                     const TensorShape& i3);

lang::RunInfo LoadEltwiseAdd(const std::string& name,  //
                             const TensorShape& i1,    //
                             const TensorShape& i2);

lang::RunInfo LoadEltwiseMultiAdd(const std::string& name,  //
                                  const TensorShape& i1,    //
                                  const TensorShape& i2,    //
                                  const TensorShape& i3,    //
                                  const TensorShape& i4);

lang::RunInfo LoadEltwiseDiv(const std::string& name,  //
                             const TensorShape& i1,    //
                             const TensorShape& i2);

lang::RunInfo LoadEltwiseMul(const std::string& name,  //
                             const TensorShape& i1,    //
                             const TensorShape& i2);

lang::RunInfo LoadEltwiseMultiMul(const std::string& name,  //
                                  const TensorShape& i1,    //
                                  const TensorShape& i2,    //
                                  const TensorShape& i3,    //
                                  const TensorShape& i4);

lang::RunInfo LoadSin(const std::string& name, const TensorShape& i1);

lang::RunInfo LoadTanh(const std::string& name, const TensorShape& i1);

lang::RunInfo LoadMulThenNeg(const std::string& name,  //
                             const TensorShape& i1,    //
                             const TensorShape& i2);

lang::RunInfo LoadNegThenMul(const std::string& name,  //
                             const TensorShape& i1,    //
                             const TensorShape& i2);

lang::RunInfo LoadConstCalc(const std::string& name);

lang::RunInfo LoadConv1d(const std::string& name,   //
                         const TensorShape& input,  //
                         const TensorShape& kernel);

lang::RunInfo LoadConv2d(const std::string& name,   //
                         const TensorShape& input,  //
                         const TensorShape& kernel);

lang::RunInfo LoadConv2dRelu(const std::string& name,   //
                             const TensorShape& input,  //
                             const TensorShape& kernel);

lang::RunInfo LoadConv2dBnRelu(const std::string& name,    //
                               const TensorShape& input,   //
                               const TensorShape& kernel,  //
                               const TensorShape& channels);

lang::RunInfo LoadConv2d3Deep(const std::string& name,     //
                              const TensorShape& input,    //
                              const TensorShape& kernel1,  //
                              const TensorShape& kernel2,  //
                              const TensorShape& kernel3);

lang::RunInfo LoadDilatedConv2d(const std::string& name,   //
                                const TensorShape& input,  //
                                const TensorShape& kernel);

lang::RunInfo LoadLarsMomentum4d(const std::string& name,     //
                                 const TensorShape& x_shape,  //
                                 const TensorShape& lr_shape);

lang::RunInfo LoadPow(const std::string& name,  //
                      const TensorShape& i1,    //
                      const TensorShape& i2);

lang::RunInfo LoadLayerNorm4dAx2(const std::string& name,  //
                                 const TensorShape& input);

lang::RunInfo LoadPolygonBoxTransform(const std::string& name,  //
                                      const TensorShape& input);
}  // namespace lib
}  // namespace tile
}  // namespace vertexai
