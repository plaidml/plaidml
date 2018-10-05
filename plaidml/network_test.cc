// Copyright 2018 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#ifdef WITH_CALLGRIND
#include <valgrind/callgrind.h>
#endif  // defined(WITH_CALLGRIND)

#include <fstream>

#include "base/util/logging.h"
#include "plaidml/plaidml++.h"
#include "testing/plaidml_config.h"
#include "tile/proto/tile.pb.h"

using ::testing::Eq;
using ::testing::Ne;

namespace gp = ::google::protobuf;
namespace gpi = ::google::protobuf::io;

namespace vertexai {
namespace plaidml {
namespace {

// The network tests validate that the implementation is capable of
// running a particular network.  They're intended to be useful for
// performance analysis, memory/thread correctness validation, and
// running the implementation under a debugger.  They do not validate
// computation correctness; they are completely self-contained, but
// they run over random data, and do not verify that output
// correctness.
class NetworkTest : public ::testing::TestWithParam<const char*> {
 protected:
  void SetUp() final {
#ifdef WITH_CALLGRIND
    CALLGRIND_STOP_INSTRUMENTATION;
    CALLGRIND_ZERO_STATS;
#endif  // WITH_CALLGRIND

    vai_clear_status();

    std::vector<device_config> configs = enumerate_devices(vertexai::testing::PlaidMLConfig());
    ctx_ = std::make_shared<vertexai::ctx>();

    for (size_t i = 0; i < configs.size(); i++) {
      std::cout << i << ": " << configs[i].name() << "->" << configs[i].description() << std::endl;
    }

    ASSERT_THAT(configs.size(), Ne(0));

    dev_ = configs[0].open();
  }

  tensor<float> AllocTensor(const tile::proto::TensorShape& shape_proto) {
    shape<float> shp{ctx_};
    for (const auto& dim : shape_proto.dimensions()) {
      shp.add_dimension(dim.size(), dim.stride());
    }
    return dev_.allocate(shp);
  }

  std::shared_ptr<ctx> ctx_;
  device dev_;
};

TEST_P(NetworkTest, Runs) {
  tile::proto::Program program;

  {
    std::ifstream prog_stream{GetParam()};
    ASSERT_TRUE(prog_stream) << "Failed to open " << GetParam();

    gpi::IstreamInputStream prog_zcs{&prog_stream};
    ASSERT_THAT(gp::TextFormat::Parse(&prog_zcs, &program), Eq(true));
  }

  invoker call(ctx_, function(program.code()));

  for (const auto& in : program.inputs()) {
    call.set_input(in.first.c_str(), AllocTensor(in.second));
  }

  std::vector<tensor<float>> outputs;

  for (const auto& out : program.outputs()) {
    auto t = AllocTensor(out.second);
    call.set_output(out.first.c_str(), t);
    outputs.emplace_back(std::move(t));
  }

#ifdef WITH_CALLGRIND
  const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();

  std::string name = test_info->test_case_name();
  name = name.substr(0, name.find('/'));
  LOG(INFO) << "Warmup " << name;
#endif  // WITH_CALLGRIND

  call.invoke();

  for (auto& out : outputs) {
    out.map(map_for_read);
  }

#ifdef WITH_CALLGRIND
  LOG(INFO) << "Callgrind " << name;
  CALLGRIND_START_INSTRUMENTATION;

  call.invoke();

  for (auto& out : outputs) {
    out.map(map_for_read);
  }

  CALLGRIND_DUMP_STATS_AT(name.c_str());
  CALLGRIND_STOP_INSTRUMENTATION;
#endif  // WITH_CALLGRIND
}

// This is an Xception network produced via Keras, with a batch size of 16.
INSTANTIATE_TEST_CASE_P(Xception, NetworkTest, ::testing::Values("plaidml/testdata/xception.tpb"));

// This is a resnet50 network produced via Keras, with a batch size of 64.
INSTANTIATE_TEST_CASE_P(Resnet50, NetworkTest, ::testing::Values("plaidml/testdata/resnet50.tpb"));

}  // namespace
}  // namespace plaidml
}  // namespace vertexai
