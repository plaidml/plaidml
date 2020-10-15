// Tests that show HLO Module conversion to PlaidML Program.

#include <gtest/gtest.h>

#include <fstream>

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"

#include "plaidml/bridge/tensorflow/service/compiler.h"
#include "plaidml/bridge/tensorflow/service/graph_util.h"
#include "plaidml/bridge/tensorflow/tests/archive_generated.h"
#include "plaidml/bridge/tensorflow/tests/codegen_test.h"

using plaidml::edsl::MultiBuffer;
namespace zoo = plaidml::zoo;

static MultiBuffer convertBuffer(const zoo::DataUnion& data) {
  switch (data.type) {
    case zoo::Data_I8Data:
      return data.AsI8Data()->data;
    case zoo::Data_I16Data:
      return data.AsI16Data()->data;
    case zoo::Data_I32Data:
      return data.AsI32Data()->data;
    case zoo::Data_I64Data:
      return data.AsI64Data()->data;
    case zoo::Data_U8Data:
      return data.AsU8Data()->data;
    case zoo::Data_U16Data:
      return data.AsU16Data()->data;
    case zoo::Data_U32Data:
      return data.AsU32Data()->data;
    case zoo::Data_U64Data:
      return data.AsU64Data()->data;
    case zoo::Data_F16Data:
      return data.AsF16Data()->data;
    case zoo::Data_F32Data:
      return data.AsF32Data()->data;
    case zoo::Data_F64Data:
      return data.AsF64Data()->data;
    default:
      break;
  }
  throw std::runtime_error("Invalid data_type");
}

static std::vector<char> ReadFile(const std::string& path) {
  std::ifstream fs;
  fs.open(path, std::ios::binary | std::ios::in);
  fs.seekg(0, std::ios::end);
  int length = fs.tellg();
  fs.seekg(0, std::ios::beg);
  std::vector<char> buf(length);
  fs.read(buf.data(), buf.size());
  fs.close();
  return buf;
}

namespace xla {
namespace plaidml {
namespace {

struct I3DTestSpec {
  PrimitiveType primitive_type;
};

string I3DTestSpecToString(const ::testing::TestParamInfo<I3DTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLI3DOperationTest : public PlaidMLCodegenTest, public ::testing::WithParamInterface<I3DTestSpec> {};

TEST_P(PlaidMLI3DOperationTest, SimpleI3D) {
  auto data = ReadFile("plaidml/bridge/tensorflow/tests/i3d.pml");
  zoo::ArchiveT archive;
  zoo::GetArchive(data.data())->UnPackTo(&archive);

  std::vector<MultiBuffer> inputs;
  std::vector<MultiBuffer> outputs;
  std::unordered_map<std::string, MultiBuffer> weights;

  VLOG(0) << "Archive: " << archive.name;
  VLOG(0) << "Inputs: " << archive.inputs.size();
  for (const auto& buffer : archive.inputs) {
    VLOG(2) << "  " << buffer->name;
    inputs.emplace_back(convertBuffer(buffer->data));
  }

  VLOG(0) << "Weights: " << archive.weights.size();
  for (const auto& buffer : archive.weights) {
    VLOG(0) << "  " << buffer->name;
    weights.emplace(buffer->name, convertBuffer(buffer->data));
  }

  VLOG(0) << "Outputs: " << archive.outputs.size();
  for (const auto& buffer : archive.outputs) {
    VLOG(2) << "  " << buffer->name;
    outputs.emplace_back(convertBuffer(buffer->data));
  }

  auto lookup = [&](const char* key) {
    auto it = weights.find(key);
    if (it == weights.end()) {
      throw std::runtime_error(std::string("Key not found: ") + key);
    }
    return it->second;
  };

  std::vector<MultiBuffer> args = {inputs[0]};

  auto hlo_module =
      ImportFrozenGraph("plaidml/bridge/tensorflow/tests/i3d_frozen_graph.pb", {"Placeholder"},  // x.op.name
                        {"module_apply_default/RGB/inception_i3d/Mean"}                          // y.op.name
                        )
          .ValueOrDie();
  CompileAndCheck(std::move(hlo_module), {{args, outputs}});
}

std::vector<I3DTestSpec> GetI3DTestCases() {
  std::vector<I3DTestSpec> result;
  result.push_back({F32});
  return result;
}

INSTANTIATE_TEST_SUITE_P(All, PlaidMLI3DOperationTest, ::testing::ValuesIn(GetI3DTestCases()), I3DTestSpecToString);

}  // namespace
}  // namespace plaidml
}  // namespace xla
