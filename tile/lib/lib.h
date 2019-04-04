#pragma once

#include <string>

#include "tile/lang/compose.h"

namespace vertexai {
namespace tile {
namespace lib {

lang::RunInfo LoadMatMul(const std::string& name,    //
                         const TensorShape& input1,  //
                         const TensorShape& input2);

lang::RunInfo LoadEWAdd(const std::string& name, const TensorShape& i1, const TensorShape& i2);

lang::RunInfo LoadConstCalc(const std::string& name,  //
                            const TensorShape& output);

lang::RunInfo LoadConv1d(const std::string& name,    //
                         const TensorShape& input,   //
                         const TensorShape& kernel,  //
                         const TensorShape& output);

lang::RunInfo LoadConv2d(const std::string& name,    //
                         const TensorShape& input,   //
                         const TensorShape& kernel,  //
                         const TensorShape& output);

lang::RunInfo LoadConv2dRelu(const std::string& name,    //
                             const TensorShape& input,   //
                             const TensorShape& kernel,  //
                             const TensorShape& output);

lang::RunInfo LoadConv2dBnRelu(const std::string& name,      //
                               const TensorShape& input,     //
                               const TensorShape& kernel,    //
                               const TensorShape& channels,  //
                               const TensorShape& output);

lang::RunInfo LoadConv2d3Deep(const std::string& name,     //
                              const TensorShape& input,    //
                              const TensorShape& kernel1,  //
                              const TensorShape& kernel2,  //
                              const TensorShape& kernel3,  //
                              const TensorShape& output);

lang::RunInfo LoadDilatedConv2d(const std::string& name,    //
                                const TensorShape& input,   //
                                const TensorShape& kernel,  //
                                const TensorShape& output);

lang::RunInfo LoadLarsMomentum4d(const std::string& name,     //
                                 const TensorShape& x_shape,  //
                                 const TensorShape& lr_shape);

lang::RunInfo LoadPow(const std::string& name,  //
                      const TensorShape& i1,    //
                      const TensorShape& i2,    //
                      const TensorShape& output);

lang::RunInfo LoadLayerNorm4dAx2(const std::string& name,  //
                                 const TensorShape& input);

lang::RunInfo LoadPolygonBoxTransform(const std::string& name,  //
                                      const TensorShape& input);
}  // namespace lib
}  // namespace tile
}  // namespace vertexai
