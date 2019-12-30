#pragma once

#include <string>
#include <vector>

#include "plaidml2/edsl/edsl.h"

namespace vertexai::tile::lib {

using plaidml::edsl::LogicalShape;
using plaidml::edsl::Program;
using plaidml::edsl::Tensor;

enum class ConvolutionFormat {
  ChannelsFirst,
  ChannelsLast,
};

// TODO: add support for AutoPadding
Tensor Convolution(                                                //
    const Tensor& I,                                               //
    const Tensor& K,                                               //
    const std::vector<int64_t>& O_dims,                            //
    std::vector<size_t> strides = {},                              //
    ConvolutionFormat I_format = ConvolutionFormat::ChannelsLast,  //
    ConvolutionFormat K_format = ConvolutionFormat::ChannelsLast);

Program LoadMaxPool2d(          //
    const std::string& name,    //
    const LogicalShape& input,  //
    const std::vector<size_t>& pool_size);

Program LoadMatMul(              //
    const std::string& name,     //
    const LogicalShape& input1,  //
    const LogicalShape& input2);

Program LoadMatMulIntermediate(  //
    const std::string& name,     //
    const LogicalShape& i1,      //
    const LogicalShape& i2,      //
    const LogicalShape& i3);

Program LoadMatMulAmongEltwise(  //
    const std::string& name,     //
    const LogicalShape& i1,      //
    const LogicalShape& i2,      //
    const LogicalShape& i3);

Program LoadEltwiseAdd(       //
    const std::string& name,  //
    const LogicalShape& i1,   //
    const LogicalShape& i2);

Program LoadEltwiseMultiAdd(  //
    const std::string& name,  //
    const LogicalShape& i1,   //
    const LogicalShape& i2,   //
    const LogicalShape& i3,   //
    const LogicalShape& i4);

Program LoadEltwiseMulFlip(const std::string& name, const LogicalShape& i1, const LogicalShape& i2);

Program LoadEltwiseDiv(       //
    const std::string& name,  //
    const LogicalShape& i1,   //
    const LogicalShape& i2);

Program LoadEltwiseMul(       //
    const std::string& name,  //
    const LogicalShape& i1,   //
    const LogicalShape& i2);

Program LoadEltwiseMultiMul(  //
    const std::string& name,  //
    const LogicalShape& i1,   //
    const LogicalShape& i2,   //
    const LogicalShape& i3,   //
    const LogicalShape& i4);

Program LoadSin(const std::string& name, const LogicalShape& i1);

Program LoadTanh(const std::string& name, const LogicalShape& i1);

Program LoadMulThenNeg(       //
    const std::string& name,  //
    const LogicalShape& i1,   //
    const LogicalShape& i2);

Program LoadNegThenMul(       //
    const std::string& name,  //
    const LogicalShape& i1,   //
    const LogicalShape& i2);

Program LoadConstCalc(const std::string& name);

Program LoadConstScalarMul(   //
    const std::string& name,  //
    const double s,           //
    const LogicalShape& i1);

Program LoadConv1d(              //
    const std::string& name,     //
    const LogicalShape& input,   //
    const LogicalShape& kernel,  //
    const std::vector<int64_t>& output);

Program LoadConv2d(              //
    const std::string& name,     //
    const LogicalShape& input,   //
    const LogicalShape& kernel,  //
    const std::vector<int64_t>& output);

Program LoadConv2dRelu(          //
    const std::string& name,     //
    const LogicalShape& input,   //
    const LogicalShape& kernel,  //
    const std::vector<int64_t>& output);

Program LoadConv2dBnRelu(          //
    const std::string& name,       //
    const LogicalShape& input,     //
    const LogicalShape& kernel,    //
    const LogicalShape& channels,  //
    const std::vector<int64_t>& output);

Program LoadConv2d3Deep(          //
    const std::string& name,      //
    const LogicalShape& input,    //
    const LogicalShape& kernel1,  //
    const LogicalShape& kernel2,  //
    const LogicalShape& kernel3);

Program LoadDilatedConv2d(      //
    const std::string& name,    //
    const LogicalShape& input,  //
    const LogicalShape& kernel);

Program LoadLarsMomentum4d(       //
    const std::string& name,      //
    const LogicalShape& x_shape,  //
    const LogicalShape& lr_shape);

Program LoadPow(              //
    const std::string& name,  //
    const LogicalShape& i1,   //
    const LogicalShape& i2);

Program LoadLayerNorm4dAx2(   //
    const std::string& name,  //
    const LogicalShape& input);

Program LoadBatchNormalization(  //
    const std::string& name,     //
    const LogicalShape& input);

Program LoadPolygonBoxTransform(  //
    const std::string& name,      //
    const LogicalShape& input);

Program LoadSoftmax(          //
    const std::string& name,  //
    const LogicalShape& input);

}  // namespace vertexai::tile::lib
