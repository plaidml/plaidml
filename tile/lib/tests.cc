#include "tile/lib/tests.h"

#include "tile/lib/lib.h"

namespace vertexai {
namespace tile {
namespace lib {

std::pair<std::string, std::function<lang::RunInfo()>> MakeEntry(
    const std::string& name,  //
    std::function<lang::RunInfo(const std::string& name)> fn) {
  return std::make_pair(name, std::bind(fn, name));
}

std::map<std::string, std::function<lang::RunInfo()>>* InternalTests() {
  static std::map<std::string, std::function<lang::RunInfo()>> tests = {
      MakeEntry("matmul",
                [](const std::string& name) {
                  return LoadMatMul(name,                                         //
                                    SimpleShape(DataType::FLOAT32, {100, 100}),   //
                                    SimpleShape(DataType::FLOAT32, {100, 100}));  //
                }),
      MakeEntry("matmul_32",
                [](const std::string& name) {
                  return LoadMatMul(name,                                       //
                                    SimpleShape(DataType::FLOAT32, {32, 32}),   //
                                    SimpleShape(DataType::FLOAT32, {32, 32}));  //
                }),
      MakeEntry("matmul_big",
                [](const std::string& name) {
                  return LoadMatMul(name, SimpleShape(DataType::FLOAT32, {1000, 1000}),  //
                                    SimpleShape(DataType::FLOAT32, {1000, 1000}));       //
                }),
      MakeEntry("matmul_4k",
                [](const std::string& name) {
                  return LoadMatMul(name, SimpleShape(DataType::FLOAT32, {4096, 32}),  //
                                    SimpleShape(DataType::FLOAT32, {4096, 32}));       //
                }),
      MakeEntry("matmul_intermediate",
                [](const std::string& name) {
                  return LoadMatMulIntermediate(name,                                         //
                                                SimpleShape(DataType::FLOAT32, {100, 100}),   //
                                                SimpleShape(DataType::FLOAT32, {100, 100}),   //
                                                SimpleShape(DataType::FLOAT32, {100, 100}));  //
                }),
      MakeEntry("matmul_intermediate_32",
                [](const std::string& name) {
                  return LoadMatMulIntermediate(name,                                       //
                                                SimpleShape(DataType::FLOAT32, {32, 32}),   //
                                                SimpleShape(DataType::FLOAT32, {32, 32}),   //
                                                SimpleShape(DataType::FLOAT32, {32, 32}));  //
                }),
      MakeEntry("matmul_among_eltwise",
                [](const std::string& name) {
                  return LoadMatMulAmongEltwise(name,                                         //
                                                SimpleShape(DataType::FLOAT32, {100, 100}),   //
                                                SimpleShape(DataType::FLOAT32, {100, 100}),   //
                                                SimpleShape(DataType::FLOAT32, {100, 100}));  //
                }),
      MakeEntry("matmul_among_eltwise_32",
                [](const std::string& name) {
                  return LoadMatMulAmongEltwise(name,                                       //
                                                SimpleShape(DataType::FLOAT32, {32, 32}),   //
                                                SimpleShape(DataType::FLOAT32, {32, 32}),   //
                                                SimpleShape(DataType::FLOAT32, {32, 32}));  //
                }),
      MakeEntry("matmul_among_eltwise_4k",
                [](const std::string& name) {
                  return LoadMatMulAmongEltwise(name,                                          //
                                                SimpleShape(DataType::FLOAT32, {4096, 4096}),  //
                                                SimpleShape(DataType::FLOAT32, {4096, 32}),    //
                                                SimpleShape(DataType::FLOAT32, {4096, 32}));   //
                }),
      MakeEntry("eltwise_add",
                [](const std::string& name) {
                  return LoadEltwiseAdd(name,                                           //
                                        SimpleShape(DataType::FLOAT32, {1024, 1024}),   //
                                        SimpleShape(DataType::FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("eltwise_add_32",
                [](const std::string& name) {
                  return LoadEltwiseAdd(name,                                       //
                                        SimpleShape(DataType::FLOAT32, {32, 32}),   //
                                        SimpleShape(DataType::FLOAT32, {32, 32}));  //
                }),
      MakeEntry("eltwise_add_2k",
                [](const std::string& name) {
                  return LoadEltwiseAdd(name,                                         //
                                        SimpleShape(DataType::FLOAT32, {2048, 32}),   //
                                        SimpleShape(DataType::FLOAT32, {2048, 32}));  //
                }),
      MakeEntry("eltwise_add_4k",
                [](const std::string& name) {
                  return LoadEltwiseAdd(name,                                         //
                                        SimpleShape(DataType::FLOAT32, {4096, 32}),   //
                                        SimpleShape(DataType::FLOAT32, {4096, 32}));  //
                }),
      MakeEntry("eltwise_mul_flip",
                [](const std::string& name) {
                  return LoadEltwiseMulFlip(name, SimpleShape(DataType::FLOAT32, {32, 32}),
                                            SimpleShape(DataType::FLOAT32, {32, 32}));
                }),
      MakeEntry("eltwise_mul_flip_4k",
                [](const std::string& name) {
                  return LoadEltwiseMulFlip(name, SimpleShape(DataType::FLOAT32, {4096, 32}),
                                            SimpleShape(DataType::FLOAT32, {4096, 32}));
                }),
      MakeEntry("eltwise_add_16k",
                [](const std::string& name) {
                  return LoadEltwiseAdd(name,                                          //
                                        SimpleShape(DataType::FLOAT32, {16384, 32}),   //
                                        SimpleShape(DataType::FLOAT32, {16384, 32}));  //
                }),
      MakeEntry("eltwise_multi_add_32",
                [](const std::string& name) {
                  return LoadEltwiseMultiAdd(name,                                       //
                                             SimpleShape(DataType::FLOAT32, {32, 32}),   //
                                             SimpleShape(DataType::FLOAT32, {32, 32}),   //
                                             SimpleShape(DataType::FLOAT32, {32, 32}),   //
                                             SimpleShape(DataType::FLOAT32, {32, 32}));  //
                }),
      MakeEntry("eltwise_multi_add",
                [](const std::string& name) {
                  return LoadEltwiseMultiAdd(name,                                           //
                                             SimpleShape(DataType::FLOAT32, {1024, 1024}),   //
                                             SimpleShape(DataType::FLOAT32, {1024, 1024}),   //
                                             SimpleShape(DataType::FLOAT32, {1024, 1024}),   //
                                             SimpleShape(DataType::FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("eltwise_multi_add_128",
                [](const std::string& name) {
                  return LoadEltwiseMultiAdd(name,                                        //
                                             SimpleShape(DataType::FLOAT32, {128, 32}),   //
                                             SimpleShape(DataType::FLOAT32, {128, 32}),   //
                                             SimpleShape(DataType::FLOAT32, {128, 32}),   //
                                             SimpleShape(DataType::FLOAT32, {128, 32}));  //
                }),
      MakeEntry("eltwise_mul",
                [](const std::string& name) {
                  return LoadEltwiseMul(name,                                           //
                                        SimpleShape(DataType::FLOAT32, {1024, 1024}),   //
                                        SimpleShape(DataType::FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("eltwise_mul_32",
                [](const std::string& name) {
                  return LoadEltwiseMul(name,                                       //
                                        SimpleShape(DataType::FLOAT32, {32, 32}),   //
                                        SimpleShape(DataType::FLOAT32, {32, 32}));  //
                }),
      MakeEntry("eltwise_mul_4k",
                [](const std::string& name) {
                  return LoadEltwiseMul(name,                                         //
                                        SimpleShape(DataType::FLOAT32, {4096, 32}),   //
                                        SimpleShape(DataType::FLOAT32, {4096, 32}));  //
                }),
      MakeEntry("eltwise_multi_mul_32",
                [](const std::string& name) {
                  return LoadEltwiseMultiMul(name,                                       //
                                             SimpleShape(DataType::FLOAT32, {32, 32}),   //
                                             SimpleShape(DataType::FLOAT32, {32, 32}),   //
                                             SimpleShape(DataType::FLOAT32, {32, 32}),   //
                                             SimpleShape(DataType::FLOAT32, {32, 32}));  //
                }),
      MakeEntry("eltwise_multi_mul_128",
                [](const std::string& name) {
                  return LoadEltwiseMultiMul(name,                                        //
                                             SimpleShape(DataType::FLOAT32, {128, 32}),   //
                                             SimpleShape(DataType::FLOAT32, {128, 32}),   //
                                             SimpleShape(DataType::FLOAT32, {128, 32}),   //
                                             SimpleShape(DataType::FLOAT32, {128, 32}));  //
                }),
      MakeEntry("eltwise_multi_mul",
                [](const std::string& name) {
                  return LoadEltwiseMultiMul(name,                                           //
                                             SimpleShape(DataType::FLOAT32, {1024, 1024}),   //
                                             SimpleShape(DataType::FLOAT32, {1024, 1024}),   //
                                             SimpleShape(DataType::FLOAT32, {1024, 1024}),   //
                                             SimpleShape(DataType::FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("eltwise_div",
                [](const std::string& name) {
                  return LoadEltwiseAdd(name,                                           //
                                        SimpleShape(DataType::FLOAT32, {1024, 1024}),   //
                                        SimpleShape(DataType::FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("sin",
                [](const std::string& name) {
                  return LoadSin(name,                                           //
                                 SimpleShape(DataType::FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("tanh",
                [](const std::string& name) {
                  return LoadTanh(name,                                           //
                                  SimpleShape(DataType::FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("mulneg_32",
                [](const std::string& name) {
                  return LoadMulThenNeg(name,                                       //
                                        SimpleShape(DataType::FLOAT32, {32, 32}),   //
                                        SimpleShape(DataType::FLOAT32, {32, 32}));  //
                }),
      MakeEntry("mulneg",
                [](const std::string& name) {
                  return LoadMulThenNeg(name,                                           //
                                        SimpleShape(DataType::FLOAT32, {1024, 1024}),   //
                                        SimpleShape(DataType::FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("negmul_32",
                [](const std::string& name) {
                  return LoadNegThenMul(name,                                       //
                                        SimpleShape(DataType::FLOAT32, {32, 32}),   //
                                        SimpleShape(DataType::FLOAT32, {32, 32}));  //
                }),
      MakeEntry("negmul",
                [](const std::string& name) {
                  return LoadNegThenMul(name,                                           //
                                        SimpleShape(DataType::FLOAT32, {1024, 1024}),   //
                                        SimpleShape(DataType::FLOAT32, {1024, 1024}));  //
                }),
      MakeEntry("const_test", [](const std::string& name) { return LoadConstCalc(name); }),
      MakeEntry("dilated_conv2d",
                [](const std::string& name) {
                  return LoadDilatedConv2d(name,                                                  //
                                           SimpleShape(DataType::INT8, {1, 56, 56, 64}),          //
                                           SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"));  //
                }),
      MakeEntry("layer_test1",
                [](const std::string& name) {
                  return LoadConv2d(name,                                                 //
                                    SimpleShape(DataType::INT8, {1, 56, 56, 64}),         //
                                    SimpleShape(DataType::INT8, {1, 1, 64, 64}, "HWCK"),  //
                                    {1, 56, 56, 64});                                     //
                }),
      MakeEntry("layer_test2",
                [](const std::string& name) {
                  return LoadConv2d(name,                                                 //
                                    SimpleShape(DataType::INT8, {1, 56, 56, 64}),         //
                                    SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),  //
                                    {1, 56, 56, 64});                                     //
                }),
      MakeEntry("layer_test3",
                [](const std::string& name) {
                  return LoadConv2dRelu(name,                                                 //
                                        SimpleShape(DataType::INT8, {1, 56, 56, 64}),         //
                                        SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),  //
                                        {1, 56, 56, 64});                                     //
                }),
      MakeEntry("layer_test4",
                [](const std::string& name) {
                  return LoadConv2dBnRelu(name,                                                 //
                                          SimpleShape(DataType::INT8, {1, 56, 56, 64}),         //
                                          SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),  //
                                          SimpleShape(DataType::INT8, {64}),                    //
                                          {1, 56, 56, 64});                                     //
                }),
      MakeEntry("layer_test4",
                [](const std::string& name) {
                  return LoadConv2dBnRelu(name,                                             //
                                          SimpleShape(DataType::FLOAT32, {1, 56, 56, 64}),  //
                                          SimpleShape(DataType::FLOAT32, {3, 3, 64, 64}),   //
                                          SimpleShape(DataType::FLOAT32, {64}),             //
                                          {1, 56, 56, 64});                                 //
                }),
      MakeEntry("layer_test5",
                [](const std::string& name) {
                  return LoadConv2d(name,                                                    //
                                    SimpleShape(DataType::INT8, {1, 7, 7, 2048}),            //
                                    SimpleShape(DataType::INT8, {1, 1, 2048, 512}, "HWCK"),  //
                                    {1, 7, 7, 512});                                         //
                }),
      MakeEntry("layer_test6",
                [](const std::string& name) {
                  return LoadConv2d3Deep(name,                                                  //
                                         SimpleShape(DataType::INT8, {1, 56, 56, 64}),          // I
                                         SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),   // K1
                                         SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),   // K2
                                         SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"));  // K2
                }),
      MakeEntry("layer_test7",
                [](const std::string& name) {
                  return LoadConv2d3Deep(name,                                                  //
                                         SimpleShape(DataType::INT8, {1, 1024, 1024, 32}),      // I
                                         SimpleShape(DataType::INT8, {3, 3, 32, 64}, "HWCK"),   // K1
                                         SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),   // K2
                                         SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"));  // K2
                }),
      MakeEntry("layer_test8",
                [](const std::string& name) {
                  return LoadConv2dBnRelu(name,                                                 //
                                          SimpleShape(DataType::INT8, {1, 55, 55, 63}),         //
                                          SimpleShape(DataType::INT8, {3, 3, 63, 63}, "HWCK"),  //
                                          SimpleShape(DataType::INT8, {63}),                    //
                                          {1, 55, 55, 63});                                     //
                }),
      MakeEntry("lars_momentum_test",
                [](const std::string& name) {
                  return LoadLarsMomentum4d(name,                                          //
                                            SimpleShape(DataType::FLOAT32, {4, 7, 3, 9}),  //
                                            SimpleShape(DataType::FLOAT32, {}));           //
                }),
      MakeEntry("pow_test",
                [](const std::string& name) {
                  return LoadPow(name,                                       //
                                 SimpleShape(DataType::FLOAT32, {3, 2, 3}),  //
                                 SimpleShape(DataType::FLOAT32, {2, 1}));    //
                }),
      MakeEntry("layer_norm_test",
                [](const std::string& name) {
                  return LoadLayerNorm4dAx2(name,                                           //
                                            SimpleShape(DataType::FLOAT32, {4, 7, 5, 3}));  //
                }),
      MakeEntry("polygon_box_transform_test",
                [](const std::string& name) {
                  return LoadPolygonBoxTransform(name,                                           //
                                                 SimpleShape(DataType::FLOAT32, {4, 5, 7, 3}));  //
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
