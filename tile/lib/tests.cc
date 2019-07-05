#include "tile/lib/tests.h"

#include "tile/lib/lib.h"

namespace vertexai {
namespace tile {
namespace lib {

using plaidml::edsl::LogicalShape;
using plaidml::edsl::Tensor;

std::pair<std::string, std::function<lang::RunInfo()>> MakeEntry(
    const std::string& name,  //
    std::function<lang::RunInfo(const std::string& name)> fn) {
  return std::make_pair(name, std::bind(fn, name));
}

std::map<std::string, std::function<lang::RunInfo()>>* InternalTests() {
  plaidml::edsl::init();
  static std::map<std::string, std::function<lang::RunInfo()>> tests = {
      MakeEntry("matmul",
                [](const std::string& name) {
                  return LoadMatMul(name,                                             //
                                    LogicalShape(PLAIDML_DATA_FLOAT32, {100, 100}),   //
                                    LogicalShape(PLAIDML_DATA_FLOAT32, {100, 100}));  //
                }),
      MakeEntry("matmul_1",
                [](const std::string& name) {
                  return LoadMatMul(name,                                         //
                                    LogicalShape(PLAIDML_DATA_FLOAT32, {1, 1}),   //
                                    LogicalShape(PLAIDML_DATA_FLOAT32, {1, 1}));  //
                }),
      MakeEntry("matmul_2",
                [](const std::string& name) {
                  return LoadMatMul(name,                                         //
                                    LogicalShape(PLAIDML_DATA_FLOAT32, {2, 2}),   //
                                    LogicalShape(PLAIDML_DATA_FLOAT32, {2, 2}));  //
                }),
      MakeEntry("matmul_32",
                [](const std::string& name) {
                  return LoadMatMul(name,                                           //
                                    LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),   //
                                    LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}));  //
                }),
      MakeEntry("matmul_big",
                [](const std::string& name) {
                  return LoadMatMul(                                      //
                      name,                                               //
                      LogicalShape(PLAIDML_DATA_FLOAT32, {1000, 1000}),   //
                      LogicalShape(PLAIDML_DATA_FLOAT32, {1000, 1000}));  //
                }),
      MakeEntry("matmul_grad",
                [](const std::string& name) {
                  return LoadMatMulGradient(                            //
                      name,                                             //
                      LogicalShape(PLAIDML_DATA_FLOAT32, {100, 100}),   //
                      LogicalShape(PLAIDML_DATA_FLOAT32, {100, 100}));  //
                }),
      // TODO: multimatmul_grad has small shapes due to Crest compile times; note that Crest eventually segfaults
      // regardless
      MakeEntry("multimatmul_grad",
                [](const std::string& name) {
                  return LoadMultiMatMulGradient(                     //
                      name,                                           //
                      LogicalShape(PLAIDML_DATA_FLOAT32, {10, 10}),   //
                      LogicalShape(PLAIDML_DATA_FLOAT32, {10, 10}));  //
                }),
      MakeEntry("matmulsqrt_grad",
                [](const std::string& name) {
                  return LoadMatMulSqrtGradient(                        //
                      name,                                             //
                      LogicalShape(PLAIDML_DATA_FLOAT32, {100, 100}),   //
                      LogicalShape(PLAIDML_DATA_FLOAT32, {100, 100}));  //
                }),
      MakeEntry("matmul_4k",
                [](const std::string& name) {
                  return LoadMatMul(                                    //
                      name,                                             //
                      LogicalShape(PLAIDML_DATA_FLOAT32, {4096, 32}),   //
                      LogicalShape(PLAIDML_DATA_FLOAT32, {4096, 32}));  //
                }),
      MakeEntry("matmul_intermediate",
                [](const std::string& name) {
                  return LoadMatMulIntermediate(name,                                             //
                                                LogicalShape(PLAIDML_DATA_FLOAT32, {100, 100}),   //
                                                LogicalShape(PLAIDML_DATA_FLOAT32, {100, 100}),   //
                                                LogicalShape(PLAIDML_DATA_FLOAT32, {100, 100}));  //
                }),
      MakeEntry("matmul_intermediate_32",
                [](const std::string& name) {
                  return LoadMatMulIntermediate(name,                                           //
                                                LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),   //
                                                LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),   //
                                                LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}));  //
                }),
      MakeEntry("matmul_among_eltwise",
                [](const std::string& name) {
                  return LoadMatMulAmongEltwise(name,                                             //
                                                LogicalShape(PLAIDML_DATA_FLOAT32, {100, 100}),   //
                                                LogicalShape(PLAIDML_DATA_FLOAT32, {100, 100}),   //
                                                LogicalShape(PLAIDML_DATA_FLOAT32, {100, 100}));  //
                }),
      MakeEntry("matmul_among_eltwise_32",
                [](const std::string& name) {
                  return LoadMatMulAmongEltwise(name,                                           //
                                                LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),   //
                                                LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),   //
                                                LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}));  //
                }),
      MakeEntry("matmul_among_eltwise_4k",
                [](const std::string& name) {
                  return LoadMatMulAmongEltwise(name,                                              //
                                                LogicalShape(PLAIDML_DATA_FLOAT32, {4096, 4096}),  //
                                                LogicalShape(PLAIDML_DATA_FLOAT32, {4096, 32}),    //
                                                LogicalShape(PLAIDML_DATA_FLOAT32, {4096, 32}));   //
                }),
      MakeEntry("eltwise_add",
                [](const std::string& name) {
                  return LoadEltwiseAdd(name,                                               //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}),   //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("eltwise_add_32",
                [](const std::string& name) {
                  return LoadEltwiseAdd(name,                                           //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),   //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}));  //
                }),
      MakeEntry("eltwise_add_2k",
                [](const std::string& name) {
                  return LoadEltwiseAdd(name,                                             //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {2048, 32}),   //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {2048, 32}));  //
                }),
      MakeEntry("eltwise_add_4k",
                [](const std::string& name) {
                  return LoadEltwiseAdd(name,                                             //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {4096, 32}),   //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {4096, 32}));  //
                }),
      MakeEntry("eltwise_mul_flip",
                [](const std::string& name) {
                  return LoadEltwiseMulFlip(name, LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),
                                            LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}));
                }),
      MakeEntry("eltwise_mul_flip_4k",
                [](const std::string& name) {
                  return LoadEltwiseMulFlip(name, LogicalShape(PLAIDML_DATA_FLOAT32, {4096, 32}),
                                            LogicalShape(PLAIDML_DATA_FLOAT32, {4096, 32}));
                }),
      MakeEntry("eltwise_add_16k",
                [](const std::string& name) {
                  return LoadEltwiseAdd(name,                                              //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {16384, 32}),   //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {16384, 32}));  //
                }),
      MakeEntry("eltwise_multi_add_32",
                [](const std::string& name) {
                  return LoadEltwiseMultiAdd(name,                                           //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}));  //
                }),
      MakeEntry("eltwise_multi_add",
                [](const std::string& name) {
                  return LoadEltwiseMultiAdd(name,                                               //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("eltwise_multi_add_128",
                [](const std::string& name) {
                  return LoadEltwiseMultiAdd(name,                                            //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {128, 32}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {128, 32}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {128, 32}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {128, 32}));  //
                }),
      MakeEntry("eltwise_mul",
                [](const std::string& name) {
                  return LoadEltwiseMul(name,                                               //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}),   //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("eltwise_mul_32",
                [](const std::string& name) {
                  return LoadEltwiseMul(name,                                           //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),   //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}));  //
                }),
      MakeEntry("eltwise_mul_4k",
                [](const std::string& name) {
                  return LoadEltwiseMul(name,                                             //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {4096, 32}),   //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {4096, 32}));  //
                }),
      MakeEntry("eltwise_multi_mul_32",
                [](const std::string& name) {
                  return LoadEltwiseMultiMul(name,                                           //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}));  //
                }),
      MakeEntry("eltwise_multi_mul_128",
                [](const std::string& name) {
                  return LoadEltwiseMultiMul(name,                                            //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {128, 32}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {128, 32}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {128, 32}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {128, 32}));  //
                }),
      MakeEntry("eltwise_multi_mul",
                [](const std::string& name) {
                  return LoadEltwiseMultiMul(name,                                               //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}),   //
                                             LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("eltwise_div",
                [](const std::string& name) {
                  return LoadEltwiseAdd(name,                                               //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}),   //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("sin",
                [](const std::string& name) {
                  return LoadSin(name,                                               //
                                 LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("tanh",
                [](const std::string& name) {
                  return LoadTanh(name,                                               //
                                  LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("mulneg_32",
                [](const std::string& name) {
                  return LoadMulThenNeg(name,                                           //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),   //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}));  //
                }),
      MakeEntry("mulneg",
                [](const std::string& name) {
                  return LoadMulThenNeg(name,                                               //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}),   //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("negmul_32",
                [](const std::string& name) {
                  return LoadNegThenMul(name,                                           //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}),   //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {32, 32}));  //
                }),
      MakeEntry("negmul",
                [](const std::string& name) {
                  return LoadNegThenMul(name,                                               //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}),   //
                                        LogicalShape(PLAIDML_DATA_FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("const_test", [](const std::string& name) { return LoadConstCalc(name); }),
      MakeEntry("const_scalar_mul",
                [](const std::string& name) {
                  return LoadConstScalarMul(name, 4.7, LogicalShape(PLAIDML_DATA_FLOAT32, {256, 256}));
                }),
      MakeEntry("dilated_conv2d",
                [](const std::string& name) {
                  return LoadDilatedConv2d(name,                                              //
                                           LogicalShape(PLAIDML_DATA_INT8, {1, 56, 56, 64}),  //
                                           LogicalShape(PLAIDML_DATA_INT8, {3, 3, 64, 64}));  //
                }),
      MakeEntry("layer_test1",
                [](const std::string& name) {
                  return LoadConv2d(name,                                              //
                                    LogicalShape(PLAIDML_DATA_INT8, {1, 56, 56, 64}),  //
                                    LogicalShape(PLAIDML_DATA_INT8, {1, 1, 64, 64}),   //
                                    {1, 56, 56, 64});                                  //
                }),
      MakeEntry("layer_test2",
                [](const std::string& name) {
                  return LoadConv2d(name,                                              //
                                    LogicalShape(PLAIDML_DATA_INT8, {1, 56, 56, 64}),  //
                                    LogicalShape(PLAIDML_DATA_INT8, {3, 3, 64, 64}),   //
                                    {1, 56, 56, 64});                                  //
                }),
      MakeEntry("layer_test3",
                [](const std::string& name) {
                  return LoadConv2dRelu(name,                                              //
                                        LogicalShape(PLAIDML_DATA_INT8, {1, 56, 56, 64}),  //
                                        LogicalShape(PLAIDML_DATA_INT8, {3, 3, 64, 64}),   //
                                        {1, 56, 56, 64});                                  //
                }),
      MakeEntry("layer_test4",
                [](const std::string& name) {
                  return LoadConv2dBnRelu(name,                                              //
                                          LogicalShape(PLAIDML_DATA_INT8, {1, 56, 56, 64}),  //
                                          LogicalShape(PLAIDML_DATA_INT8, {3, 3, 64, 64}),   //
                                          LogicalShape(PLAIDML_DATA_INT8, {64}),             //
                                          {1, 56, 56, 64});                                  //
                }),
      MakeEntry("layer_test4",
                [](const std::string& name) {
                  return LoadConv2dBnRelu(name,                                                 //
                                          LogicalShape(PLAIDML_DATA_FLOAT32, {1, 56, 56, 64}),  //
                                          LogicalShape(PLAIDML_DATA_FLOAT32, {3, 3, 64, 64}),   //
                                          LogicalShape(PLAIDML_DATA_FLOAT32, {64}),             //
                                          {1, 56, 56, 64});                                     //
                }),
      MakeEntry("layer_test5",
                [](const std::string& name) {
                  return LoadConv2d(name,                                                //
                                    LogicalShape(PLAIDML_DATA_INT8, {1, 7, 7, 2048}),    //
                                    LogicalShape(PLAIDML_DATA_INT8, {1, 1, 2048, 512}),  //
                                    {1, 7, 7, 512});                                     //
                }),
      MakeEntry("layer_test6",
                [](const std::string& name) {
                  return LoadConv2d3Deep(name,                                              //
                                         LogicalShape(PLAIDML_DATA_INT8, {1, 56, 56, 64}),  // I
                                         LogicalShape(PLAIDML_DATA_INT8, {3, 3, 64, 64}),   // K1
                                         LogicalShape(PLAIDML_DATA_INT8, {3, 3, 64, 64}),   // K2
                                         LogicalShape(PLAIDML_DATA_INT8, {3, 3, 64, 64}));  // K2
                }),
      MakeEntry("layer_test7",
                [](const std::string& name) {
                  return LoadConv2d3Deep(name,                                                  //
                                         LogicalShape(PLAIDML_DATA_INT8, {1, 1024, 1024, 32}),  // I
                                         LogicalShape(PLAIDML_DATA_INT8, {3, 3, 32, 64}),       // K1
                                         LogicalShape(PLAIDML_DATA_INT8, {3, 3, 64, 64}),       // K2
                                         LogicalShape(PLAIDML_DATA_INT8, {3, 3, 64, 64}));      // K2
                }),
      MakeEntry("layer_test8",
                [](const std::string& name) {
                  return LoadConv2dBnRelu(name,                                              //
                                          LogicalShape(PLAIDML_DATA_INT8, {1, 55, 55, 63}),  //
                                          LogicalShape(PLAIDML_DATA_INT8, {3, 3, 63, 63}),   //
                                          LogicalShape(PLAIDML_DATA_INT8, {63}),             //
                                          {1, 55, 55, 63});                                  //
                }),
      MakeEntry("lars_momentum_test",
                [](const std::string& name) {
                  return LoadLarsMomentum4d(name,                                              //
                                            LogicalShape(PLAIDML_DATA_FLOAT32, {4, 7, 3, 9}),  //
                                            LogicalShape(PLAIDML_DATA_FLOAT32, {}));           //
                }),
      MakeEntry("pow_test",
                [](const std::string& name) {
                  return LoadPow(name,                                           //
                                 LogicalShape(PLAIDML_DATA_FLOAT32, {3, 2, 3}),  //
                                 LogicalShape(PLAIDML_DATA_FLOAT32, {2, 1}));    //
                }),
      MakeEntry("layer_norm_test",
                [](const std::string& name) {
                  return LoadLayerNorm4dAx2(name,                                               //
                                            LogicalShape(PLAIDML_DATA_FLOAT32, {4, 7, 5, 3}));  //
                }),
      MakeEntry("polygon_box_transform_test",
                [](const std::string& name) {
                  return LoadPolygonBoxTransform(name,                                               //
                                                 LogicalShape(PLAIDML_DATA_FLOAT32, {4, 5, 7, 3}));  //
                }),
      MakeEntry("softmax",
                [](const std::string& name) {
                  return LoadSoftmax(name,                                         //
                                     LogicalShape(PLAIDML_DATA_FLOAT32, {4, 5}));  //
                }),
  };
  return &tests;
}  // namespace lib

void RegisterTest(const std::string& name, std::function<lang::RunInfo()> factory) {
  auto tests = InternalTests();
  tests->emplace(name, factory);
}

boost::optional<lang::RunInfo> CreateTest(const std::string& name) {
  auto tests = InternalTests();
  auto it = tests->find(name);
  if (it == tests->end()) {
    return boost::none;
  }
  return it->second();
}

}  // namespace lib
}  // namespace tile
}  // namespace vertexai
