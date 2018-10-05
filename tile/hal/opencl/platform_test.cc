// Copyright 2018, Intel Corporation.

#include <gtest/gtest.h>

#include "tile/base/platform_test.h"
#include "tile/platform/local_machine/platform.h"

namespace vertexai {
namespace tile {
namespace testing {
namespace {

Param supported_params[] = {
    {DataType::INT8, 1},   //
    {DataType::INT8, 2},   //
    {DataType::INT8, 4},   //
    {DataType::INT8, 8},   //
    {DataType::INT8, 16},  //

    {DataType::INT16, 1},   //
    {DataType::INT16, 2},   //
    {DataType::INT16, 4},   //
    {DataType::INT16, 8},   //
    {DataType::INT16, 16},  //

    {DataType::INT32, 1},   //
    {DataType::INT32, 2},   //
    {DataType::INT32, 4},   //
    {DataType::INT32, 8},   //
    {DataType::INT32, 16},  //

    {DataType::INT64, 1},  //
    {DataType::INT64, 2},  //

    {DataType::UINT8, 1},   //
    {DataType::UINT8, 2},   //
    {DataType::UINT8, 4},   //
    {DataType::UINT8, 8},   //
    {DataType::UINT8, 16},  //

    {DataType::UINT16, 1},   //
    {DataType::UINT16, 2},   //
    {DataType::UINT16, 4},   //
    {DataType::UINT16, 8},   //
    {DataType::UINT16, 16},  //

    {DataType::UINT32, 1},   //
    {DataType::UINT32, 2},   //
    {DataType::UINT32, 4},   //
    {DataType::UINT32, 8},   //
    {DataType::UINT32, 16},  //

    {DataType::UINT64, 1},  //
    {DataType::UINT64, 2},  //

    {DataType::FLOAT16, 1},   //
    {DataType::FLOAT16, 2},   //
    {DataType::FLOAT16, 4},   //
    {DataType::FLOAT16, 8},   //
    {DataType::FLOAT16, 16},  //

    {DataType::FLOAT32, 1},   //
    {DataType::FLOAT32, 2},   //
    {DataType::FLOAT32, 4},   //
    {DataType::FLOAT32, 8},   //
    {DataType::FLOAT32, 16},  //

    {DataType::FLOAT64, 1},  //
    {DataType::FLOAT64, 2},  //
};

std::vector<FactoryParam> SupportedParams() {
  std::vector<FactoryParam> params;
  for (const Param& param : supported_params) {
    auto factory = [param] {
      context::Context ctx;
      local_machine::proto::Platform config;
      auto hw_config = config.add_hardware_configs();
      hw_config->mutable_sel()->set_value(true);
      hw_config->mutable_settings()->set_vec_size(param.vec_size);
      return compat::make_unique<local_machine::Platform>(ctx, config);
    };
    params.push_back({factory, param});
  }
  return params;
}

INSTANTIATE_TEST_CASE_P(OpenCl, PlatformTest, ::testing::ValuesIn(SupportedParams()));

}  // namespace
}  // namespace testing
}  // namespace tile
}  // namespace vertexai
