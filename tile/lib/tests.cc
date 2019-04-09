#include "tile/lib/tests.h"

#include "tile/lib/lib.h"

namespace vertexai {
namespace tile {
namespace lib {

std::pair<std::string, std::function<lang::RunInfo()>> MakeEntry(
    const std::string& name,  //
    std::function<lang::RunInfo(const std::string& name)> fn) {
  return std::make_pair("$" + name, std::bind(fn, name));
}

std::map<std::string, std::function<lang::RunInfo()>> InternalTests() {
  static std::map<std::string, std::function<lang::RunInfo()>> tests = {
      MakeEntry("matmul",
                [](const std::string& name) {
                  return LoadMatMul(name,                                         //
                                    SimpleShape(DataType::FLOAT32, {100, 100}),   //
                                    SimpleShape(DataType::FLOAT32, {100, 100}));  //
                }),
      MakeEntry("matmul_big",
                [](const std::string& name) {
                  return LoadMatMul(name, SimpleShape(DataType::FLOAT32, {1000, 1000}),  //
                                    SimpleShape(DataType::FLOAT32, {1000, 1000}));       //
                }),
      MakeEntry("eltwise_add",
                [](const std::string& name) {
                  return LoadEltwiseAdd(name,                                           //
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
                  return LoadConv2d(name,                                                  //
                                    SimpleShape(DataType::INT8, {1, 56, 56, 64}),          //
                                    SimpleShape(DataType::INT8, {1, 1, 64, 64}, "HWCK"));  //
                }),
      MakeEntry("layer_test2",
                [](const std::string& name) {
                  return LoadConv2d(name,                                                  //
                                    SimpleShape(DataType::INT8, {1, 56, 56, 64}),          //
                                    SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"));  //
                }),
      MakeEntry("layer_test3",
                [](const std::string& name) {
                  return LoadConv2dRelu(name,                                                  //
                                        SimpleShape(DataType::INT8, {1, 56, 56, 64}),          //
                                        SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"));  //
                }),
      MakeEntry("layer_test4",
                [](const std::string& name) {
                  return LoadConv2dBnRelu(name,                                                 //
                                          SimpleShape(DataType::INT8, {1, 56, 56, 64}),         //
                                          SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),  //
                                          SimpleShape(DataType::INT8, {64}));                   //
                }),
      MakeEntry("layer_test4",
                [](const std::string& name) {
                  return LoadConv2dBnRelu(name,                                             //
                                          SimpleShape(DataType::FLOAT32, {1, 56, 56, 64}),  //
                                          SimpleShape(DataType::FLOAT32, {3, 3, 64, 64}),   //
                                          SimpleShape(DataType::FLOAT32, {64}));            //
                }),
      MakeEntry("layer_test5",
                [](const std::string& name) {
                  return LoadConv2d(name,                                                     //
                                    SimpleShape(DataType::INT8, {1, 7, 7, 2048}),             //
                                    SimpleShape(DataType::INT8, {1, 1, 2048, 512}, "HWCK"));  //
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
                                          SimpleShape(DataType::INT8, {63}));                   //
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
  return tests;
}  // namespace lib

}  // namespace lib
}  // namespace tile
}  // namespace vertexai
