#pragma once

#include <string>

#include "tile/lang/compose.h"

namespace vertexai {
namespace tile {
namespace lib {

lang::RunInfo LoadMatMul(const std::string& name,    //
                         const TensorShape& input1,  //
                         const TensorShape& input2);

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

}  // namespace lib
}  // namespace tile
}  // namespace vertexai
