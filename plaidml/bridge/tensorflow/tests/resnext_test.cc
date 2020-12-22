// Tests that show HLO Module conversion to PlaidML Program.

#include <gtest/gtest.h>

#include <fstream>

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"

#include "plaidml/bridge/tensorflow/tests/archive_generated.h"
#include "plaidml/bridge/tensorflow/tests/codegen_test.h"

using plaidml::MultiBuffer;
namespace zoo = plaidml::zoo;

namespace xla {
namespace plaidml {
namespace {

struct ResNextTestSpec {
  PrimitiveType primitive_type;
};

string ResNextTestSpecToString(const ::testing::TestParamInfo<ResNextTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLResNextOperationTest : public PlaidMLCodegenTest, public ::testing::WithParamInterface<ResNextTestSpec> {};

TEST_P(PlaidMLResNextOperationTest, ResNext50) {
  auto data = ReadFile("plaidml/bridge/tensorflow/tests/resnext.pml");
  zoo::ArchiveT archive;
  zoo::GetArchive(data.data())->UnPackTo(&archive);

  std::vector<MultiBuffer> inputs;
  std::vector<MultiBuffer> outputs;

  // FIXME: Placeholders created for the purpose of internal computations are needlessly becoming inputs to the Program.
  std::vector<float> const_0 = {0};
  for (int i = 0; i < 4; i++) {
    inputs.emplace_back(const_0);
  }

  VLOG(0) << "Archive: " << archive.name;
  VLOG(0) << "Inputs: " << archive.inputs.size();
  for (const auto& buffer : archive.inputs) {
    VLOG(2) << "  " << buffer->name;
    inputs.emplace_back(convertBuffer(buffer->data));
  }

  VLOG(0) << "Outputs: " << archive.outputs.size();
  for (const auto& buffer : archive.outputs) {
    VLOG(2) << "  " << buffer->name;
    outputs.emplace_back(convertBuffer(buffer->data));
  }

  auto hlo_module =
      HloRunner::ReadModuleFromBinaryProtoFile("plaidml/bridge/tensorflow/tests/resnext_hlo.pb", DebugOptions())
          .ValueOrDie();

  CompileAndCheck(std::move(hlo_module), {{inputs, outputs}}, /*tolerance=*/1e-02);
}

std::vector<ResNextTestSpec> GetResNextTestCases() {
  std::vector<ResNextTestSpec> result;
  result.push_back({F32});
  return result;
}

INSTANTIATE_TEST_SUITE_P(All, PlaidMLResNextOperationTest, ::testing::ValuesIn(GetResNextTestCases()),
                         ResNextTestSpecToString);
}  // namespace
}  // namespace plaidml
}  // namespace xla
