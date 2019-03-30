#include "tile/lib/tests.h"

#include "tile/lib/lib.h"

namespace vertexai {
namespace tile {
namespace lib {

std::map<std::string, lang::RunInfo> InternalTests() {
  static std::map<std::string, lang::RunInfo> tests = {
      {
          "$matmul",
          lib::LoadMatMul("matmul",                                    //
                          SimpleShape(DataType::FLOAT32, {100, 100}),  //
                          SimpleShape(DataType::FLOAT32, {100, 100}))  //
      },                                                               //
      {
          "$matmul_big",
          lib::LoadMatMul("matmul_big",                                  //
                          SimpleShape(DataType::FLOAT32, {1000, 1000}),  //
                          SimpleShape(DataType::FLOAT32, {1000, 1000}))  //
      },                                                                 //
      {
          "$ew_add",
          lib::LoadEWAdd("ew_add",                                      //
                         SimpleShape(DataType::FLOAT32, {1024, 1024}),  //
                         SimpleShape(DataType::FLOAT32, {1024, 1024}))  //
      },
      {
          "$const_test",
          lib::LoadConstCalc("const_test",                         //
                             SimpleShape(DataType::FLOAT32, {1}))  //
      },
      {
          "$dilated_conv2d",
          lib::LoadDilatedConv2d("dilated_conv2d",                                     //
                                 SimpleShape(DataType::INT8, {1, 56, 56, 64}),         //
                                 SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),  //
                                 SimpleShape(DataType::INT8, {1, 52, 50, 64}))         //
      },
      {
          "$i8x8x256_o8x8x64_k1x1s1",
          lib::LoadConv2d("i8x8x256_o8x8x64_k1x1s1",                             //
                          SimpleShape(DataType::INT8, {1, 8, 8, 256}),           //
                          SimpleShape(DataType::INT8, {1, 1, 256, 64}, "HWCK"),  //
                          SimpleShape(DataType::INT8, {1, 8, 8, 64}))            //
      },
      {
          "$layer_test1",
          lib::LoadConv2d("layer_test1",                                        //
                          SimpleShape(DataType::INT8, {1, 56, 56, 64}),         //
                          SimpleShape(DataType::INT8, {1, 1, 64, 64}, "HWCK"),  //
                          SimpleShape(DataType::INT8, {1, 56, 56, 64}))         //
      },
      {
          "$layer_test2",
          lib::LoadConv2d("layer_test2",                                        //
                          SimpleShape(DataType::INT8, {1, 56, 56, 64}),         //
                          SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),  //
                          SimpleShape(DataType::INT8, {1, 56, 56, 64}))         //
      },
      {
          "$layer_test3",
          lib::LoadConv2dRelu("layer_test3",                                        //
                              SimpleShape(DataType::INT8, {1, 56, 56, 64}),         //
                              SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),  //
                              SimpleShape(DataType::INT8, {1, 56, 56, 64}))         //
      },
      {
          "$layer_test4",
          lib::LoadConv2dBnRelu("layer_test4",                                        //
                                SimpleShape(DataType::INT8, {1, 56, 56, 64}),         //
                                SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),  //
                                SimpleShape(DataType::INT8, {64}),                    //
                                SimpleShape(DataType::INT8, {1, 56, 56, 64}))         //
      },
      {
          "$layer_test4_float",
          lib::LoadConv2dBnRelu("layer_test4",                                    //
                                SimpleShape(DataType::FLOAT32, {1, 56, 56, 64}),  //
                                SimpleShape(DataType::FLOAT32, {3, 3, 64, 64}),   //
                                SimpleShape(DataType::FLOAT32, {64}),             //
                                SimpleShape(DataType::FLOAT32, {1, 56, 56, 64}))  //
      },
      {
          "$layer_test5",
          lib::LoadConv2d("layer_test5",                                           //
                          SimpleShape(DataType::INT8, {1, 7, 7, 2048}),            //
                          SimpleShape(DataType::INT8, {1, 1, 2048, 512}, "HWCK"),  //
                          SimpleShape(DataType::INT8, {1, 7, 7, 512}))             //
      },
      {
          "$layer_test6",
          lib::LoadConv2d3Deep("layer_test6",                                        //
                               SimpleShape(DataType::INT8, {1, 56, 56, 64}),         // I
                               SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),  // K1
                               SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),  // K2
                               SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),  // K2
                               SimpleShape(DataType::INT8, {1, 56, 56, 64}))         // O
      },
      {
          "$layer_test7",
          lib::LoadConv2d3Deep("layer_test7",                                        //
                               SimpleShape(DataType::INT8, {1, 1024, 1024, 32}),     // I
                               SimpleShape(DataType::INT8, {3, 3, 32, 64}, "HWCK"),  // K1
                               SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),  // K2
                               SimpleShape(DataType::INT8, {3, 3, 64, 64}, "HWCK"),  // K2
                               SimpleShape(DataType::INT8, {1, 1024, 1024, 64}))     // O
      },
      {
          "$layer_test8",
          lib::LoadConv2dBnRelu("layer_test8",                                        //
                                SimpleShape(DataType::INT8, {1, 55, 55, 63}),         //
                                SimpleShape(DataType::INT8, {3, 3, 63, 63}, "HWCK"),  //
                                SimpleShape(DataType::INT8, {63}),                    //
                                SimpleShape(DataType::INT8, {1, 55, 55, 63}))         //
      },
  };
  return tests;
}

}  // namespace lib
}  // namespace tile
}  // namespace vertexai
