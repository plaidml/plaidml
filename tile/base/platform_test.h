// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <vector>

#include "base/util/compat.h"
#include "tile/base/platform.h"

namespace vertexai {
namespace tile {
namespace testing {

typedef std::function<std::unique_ptr<tile::Platform>()> PlatformFactory;

struct Param {
  DataType dtype;
  std::size_t vec_size;
};

struct FactoryParam {
  PlatformFactory factory;
  Param param;
};

// Platform implementation conformance tests.
//
// To test a platform, #include this header (linking with the :platform_tests
// target), and use INSTANTIATE_TEST_CASE_P to instantiate the conformance
// tests with factories producing Platform instances -- e.g.
//
//    Param supported_params[] = {
//        {DataType::FLOAT32, 1},  //
//        {DataType::FLOAT32, 2},  //
//    };
//
//    std::vector<FactoryParam> SupportedParams() {
//      std::vector<FactoryParam> params;
//      for (const Param& param : supported_params) {
//        auto factory = [param] {
//          context::Context ctx;
//          local_machine::proto::Platform config;
//          auto hw_config = config.add_hardware_configs();
//          hw_config->mutable_sel()->set_value(true);
//          hw_config->mutable_settings()->set_vec_size(param.vec_size);
//          return std::make_unique<local_machine::Platform>(ctx, config);
//        };
//        params.push_back({factory, param});
//      }
//      return params;
//    }
class PlatformTest : public ::testing::TestWithParam<FactoryParam> {
 protected:
  void SetUp() final;

  std::shared_ptr<Program> MakeProgram(proto::TileScanningParameters* params,  //
                                       const char* code,                       //
                                       const TensorShape& shape);
  std::shared_ptr<Buffer> MakeInput(const TensorShape& shape,  //
                                    const std::vector<int>& data);
  std::shared_ptr<Buffer> MakeOutput(const TensorShape& shape);
  void CheckExpected(const TensorShape& shape,            //
                     const std::shared_ptr<Buffer>& buf,  //
                     const std::vector<int>& expected);

 protected:
  context::Context ctx_;
  Param param_;
  std::unique_ptr<tile::Platform> platform_;
};

}  // namespace testing
}  // namespace tile
}  // namespace vertexai
