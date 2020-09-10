// Tests that show HLO Module conversion to PlaidML Program.

#include <gtest/gtest.h>

#include <fstream>

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"

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

struct ResNextTestSpec {
  PrimitiveType primitive_type;
};

string ResNextTestSpecToString(const ::testing::TestParamInfo<ResNextTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLResNextOperationTest : public PlaidMLCodegenTest, public ::testing::WithParamInterface<ResNextTestSpec> {};

TEST_P(PlaidMLResNextOperationTest, SimpleResNext) {
  auto data = ReadFile("plaidml/bridge/tensorflow/tests/resnext.pml");
  zoo::ArchiveT archive;
  zoo::GetArchive(data.data())->UnPackTo(&archive);

  std::vector<MultiBuffer> inputs;
  std::vector<MultiBuffer> outputs;
  std::unordered_map<std::string, MultiBuffer> weights;

  VLOG(0) << "Archive: " << archive.name;
  VLOG(2) << "Model:\n" << archive.model;
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

  std::vector<MultiBuffer> args = {std::vector<float>{49},  // pool1
                                   std::vector<float>{0},   // stage4_unit3_relu
                                   lookup("stage4_unit3_bn3_mean"),
                                   lookup("stage4_unit3_bn3_scale"),
                                   lookup("stage4_unit3_bn3_var"),
                                   std::vector<float>{2e-05},  // stage4_unit3_bn3/add
                                   lookup("stage4_unit3_bn3_bias"),
                                   lookup("stage4_unit3_conv3_weight"),
                                   std::vector<float>{0},  // stage4_unit3_relu2
                                   lookup("stage4_unit3_bn2_mean"),
                                   lookup("stage4_unit3_bn2_scale"),
                                   lookup("stage4_unit3_bn2_var"),
                                   std::vector<float>{2e-05},  // stage4_unit3_bn2/add
                                   lookup("stage4_unit3_bn2_bias"),
                                   lookup("stage4_unit3_conv2_weight"),
                                   std::vector<float>{0},  // stage4_unit3_relu1
                                   lookup("stage4_unit3_bn1_mean"),
                                   lookup("stage4_unit3_bn1_scale"),
                                   lookup("stage4_unit3_bn1_var"),
                                   std::vector<float>{2e-05},  // stage4_unit3_bn1/add
                                   lookup("stage4_unit3_bn1_bias"),
                                   lookup("stage4_unit3_conv1_weight"),
                                   std::vector<float>{0},  // stage4_unit2_relu
                                   lookup("stage4_unit2_bn3_mean"),
                                   lookup("stage4_unit2_bn3_scale"),
                                   lookup("stage4_unit2_bn3_var"),
                                   std::vector<float>{2e-05},  // stage4_unit2_bn3/add
                                   lookup("stage4_unit2_bn3_bias"),
                                   lookup("stage4_unit2_conv3_weight"),
                                   std::vector<float>{0},  // stage4_unit2_relu2
                                   lookup("stage4_unit2_bn2_mean"),
                                   lookup("stage4_unit2_bn2_scale"),
                                   lookup("stage4_unit2_bn2_var"),
                                   std::vector<float>{2e-05},  // stage4_unit2_bn2/add
                                   lookup("stage4_unit2_bn2_bias"),
                                   lookup("stage4_unit2_conv2_weight"),
                                   std::vector<float>{0},  // stage4_unit2_relu1
                                   lookup("stage4_unit2_bn1_mean"),
                                   lookup("stage4_unit2_bn1_scale"),
                                   lookup("stage4_unit2_bn1_var"),
                                   std::vector<float>{2e-05},  // stage4_unit2_bn1/add
                                   lookup("stage4_unit2_bn1_bias"),
                                   lookup("stage4_unit2_conv1_weight"),
                                   std::vector<float>{0},  // stage4_unit1_relu
                                   lookup("stage4_unit1_sc_bn_mean"),
                                   lookup("stage4_unit1_sc_bn_scale"),
                                   lookup("stage4_unit1_sc_bn_var"),
                                   std::vector<float>{2e-05},  // stage4_unit1_sc_bn/add
                                   lookup("stage4_unit1_sc_bn_bias"),
                                   lookup("stage4_unit1_sc_weight"),
                                   std::vector<float>{0},  // stage3_unit6_relu
                                   lookup("stage3_unit6_bn3_mean"),
                                   lookup("stage3_unit6_bn3_scale"),
                                   lookup("stage3_unit6_bn3_var"),
                                   std::vector<float>{2e-05},  // stage3_unit6_bn3/add
                                   lookup("stage3_unit6_bn3_bias"),
                                   lookup("stage3_unit6_conv3_weight"),
                                   std::vector<float>{0},  // stage3_unit6_relu2
                                   lookup("stage3_unit6_bn2_mean"),
                                   lookup("stage3_unit6_bn2_scale"),
                                   lookup("stage3_unit6_bn2_var"),
                                   std::vector<float>{2e-05},  // stage3_unit6_bn2/add
                                   lookup("stage3_unit6_bn2_bias"),
                                   lookup("stage3_unit6_conv2_weight"),
                                   std::vector<float>{0},  // stage3_unit6_relu1
                                   lookup("stage3_unit6_bn1_mean"),
                                   lookup("stage3_unit6_bn1_scale"),
                                   lookup("stage3_unit6_bn1_var"),
                                   std::vector<float>{2e-05},  // stage3_unit6_bn1/add
                                   lookup("stage3_unit6_bn1_bias"),
                                   lookup("stage3_unit6_conv1_weight"),
                                   std::vector<float>{0},  // stage3_unit5_relu
                                   lookup("stage3_unit5_bn3_mean"),
                                   lookup("stage3_unit5_bn3_scale"),
                                   lookup("stage3_unit5_bn3_var"),
                                   std::vector<float>{2e-05},  // stage3_unit5_bn3/add
                                   lookup("stage3_unit5_bn3_bias"),
                                   lookup("stage3_unit5_conv3_weight"),
                                   std::vector<float>{0},  // stage3_unit5_relu2
                                   lookup("stage3_unit5_bn2_mean"),
                                   lookup("stage3_unit5_bn2_scale"),
                                   lookup("stage3_unit5_bn2_var"),
                                   std::vector<float>{2e-05},  // stage3_unit5_bn2/add
                                   lookup("stage3_unit5_bn2_bias"),
                                   lookup("stage3_unit5_conv2_weight"),
                                   std::vector<float>{0},  // stage3_unit5_relu1
                                   lookup("stage3_unit5_bn1_mean"),
                                   lookup("stage3_unit5_bn1_scale"),
                                   lookup("stage3_unit5_bn1_var"),
                                   std::vector<float>{2e-05},  // stage3_unit5_bn1/add
                                   lookup("stage3_unit5_bn1_bias"),
                                   lookup("stage3_unit5_conv1_weight"),
                                   std::vector<float>{0},  // stage3_unit4_relu
                                   lookup("stage3_unit4_bn3_mean"),
                                   lookup("stage3_unit4_bn3_scale"),
                                   lookup("stage3_unit4_bn3_var"),
                                   std::vector<float>{2e-05},  // stage3_unit4_bn3/add
                                   lookup("stage3_unit4_bn3_bias"),
                                   lookup("stage3_unit4_conv3_weight"),
                                   std::vector<float>{0},  // stage3_unit4_relu2
                                   lookup("stage3_unit4_bn2_mean"),
                                   lookup("stage3_unit4_bn2_scale"),
                                   lookup("stage3_unit4_bn2_var"),
                                   std::vector<float>{2e-05},  // stage3_unit4_bn2/add
                                   lookup("stage3_unit4_bn2_bias"),
                                   lookup("stage3_unit4_conv2_weight"),
                                   std::vector<float>{0},  // stage3_unit4_relu1
                                   lookup("stage3_unit4_bn1_mean"),
                                   lookup("stage3_unit4_bn1_scale"),
                                   lookup("stage3_unit4_bn1_var"),
                                   std::vector<float>{2e-05},  // stage3_unit4_bn1/add
                                   lookup("stage3_unit4_bn1_bias"),
                                   lookup("stage3_unit4_conv1_weight"),
                                   std::vector<float>{0},  // stage3_unit3_relu
                                   lookup("stage3_unit3_bn3_mean"),
                                   lookup("stage3_unit3_bn3_scale"),
                                   lookup("stage3_unit3_bn3_var"),
                                   std::vector<float>{2e-05},  // stage3_unit3_bn3/add
                                   lookup("stage3_unit3_bn3_bias"),
                                   lookup("stage3_unit3_conv3_weight"),
                                   std::vector<float>{0},  // stage3_unit3_relu2
                                   lookup("stage3_unit3_bn2_mean"),
                                   lookup("stage3_unit3_bn2_scale"),
                                   lookup("stage3_unit3_bn2_var"),
                                   std::vector<float>{2e-05},  // stage3_unit3_bn2/add
                                   lookup("stage3_unit3_bn2_bias"),
                                   lookup("stage3_unit3_conv2_weight"),
                                   std::vector<float>{0},  // stage3_unit3_relu1
                                   lookup("stage3_unit3_bn1_mean"),
                                   lookup("stage3_unit3_bn1_scale"),
                                   lookup("stage3_unit3_bn1_var"),
                                   std::vector<float>{2e-05},  // stage3_unit3_bn1/add
                                   lookup("stage3_unit3_bn1_bias"),
                                   lookup("stage3_unit3_conv1_weight"),
                                   std::vector<float>{0},  // stage3_unit2_relu
                                   lookup("stage3_unit2_bn3_mean"),
                                   lookup("stage3_unit2_bn3_scale"),
                                   lookup("stage3_unit2_bn3_var"),
                                   std::vector<float>{2e-05},  // stage3_unit2_bn3/add
                                   lookup("stage3_unit2_bn3_bias"),
                                   lookup("stage3_unit2_conv3_weight"),
                                   std::vector<float>{0},  // stage3_unit2_relu2
                                   lookup("stage3_unit2_bn2_mean"),
                                   lookup("stage3_unit2_bn2_scale"),
                                   lookup("stage3_unit2_bn2_var"),
                                   std::vector<float>{2e-05},  // stage3_unit2_bn2/add
                                   lookup("stage3_unit2_bn2_bias"),
                                   lookup("stage3_unit2_conv2_weight"),
                                   std::vector<float>{0},  // stage3_unit2_relu1
                                   lookup("stage3_unit2_bn1_mean"),
                                   lookup("stage3_unit2_bn1_scale"),
                                   lookup("stage3_unit2_bn1_var"),
                                   std::vector<float>{2e-05},  // stage3_unit2_bn1/add
                                   lookup("stage3_unit2_bn1_bias"),
                                   lookup("stage3_unit2_conv1_weight"),
                                   std::vector<float>{0},  // stage3_unit1_relu
                                   lookup("stage3_unit1_sc_bn_mean"),
                                   lookup("stage3_unit1_sc_bn_scale"),
                                   lookup("stage3_unit1_sc_bn_var"),
                                   std::vector<float>{2e-05},  // stage3_unit1_sc_bn/add
                                   lookup("stage3_unit1_sc_bn_bias"),
                                   lookup("stage3_unit1_sc_weight"),
                                   std::vector<float>{0},  // stage2_unit4_relu
                                   lookup("stage2_unit4_bn3_mean"),
                                   lookup("stage2_unit4_bn3_scale"),
                                   lookup("stage2_unit4_bn3_var"),
                                   std::vector<float>{2e-05},  // stage2_unit4_bn3/add
                                   lookup("stage2_unit4_bn3_bias"),
                                   lookup("stage2_unit4_conv3_weight"),
                                   std::vector<float>{0},  // stage2_unit4_relu2
                                   lookup("stage2_unit4_bn2_mean"),
                                   lookup("stage2_unit4_bn2_scale"),
                                   lookup("stage2_unit4_bn2_var"),
                                   std::vector<float>{2e-05},  // stage2_unit4_bn2/add
                                   lookup("stage2_unit4_bn2_bias"),
                                   lookup("stage2_unit4_conv2_weight"),
                                   std::vector<float>{0},  // stage2_unit4_relu1
                                   lookup("stage2_unit4_bn1_mean"),
                                   lookup("stage2_unit4_bn1_scale"),
                                   lookup("stage2_unit4_bn1_var"),
                                   std::vector<float>{2e-05},  // stage2_unit4_bn1/add
                                   lookup("stage2_unit4_bn1_bias"),
                                   lookup("stage2_unit4_conv1_weight"),
                                   std::vector<float>{0},  // stage2_unit3_relu
                                   lookup("stage2_unit3_bn3_mean"),
                                   lookup("stage2_unit3_bn3_scale"),
                                   lookup("stage2_unit3_bn3_var"),
                                   std::vector<float>{2e-05},  // stage2_unit3_bn3/add
                                   lookup("stage2_unit3_bn3_bias"),
                                   lookup("stage2_unit3_conv3_weight"),
                                   std::vector<float>{0},  // stage2_unit3_relu2
                                   lookup("stage2_unit3_bn2_mean"),
                                   lookup("stage2_unit3_bn2_scale"),
                                   lookup("stage2_unit3_bn2_var"),
                                   std::vector<float>{2e-05},  // stage2_unit3_bn2/add
                                   lookup("stage2_unit3_bn2_bias"),
                                   lookup("stage2_unit3_conv2_weight"),
                                   std::vector<float>{0},  // stage2_unit3_relu1
                                   lookup("stage2_unit3_bn1_mean"),
                                   lookup("stage2_unit3_bn1_scale"),
                                   lookup("stage2_unit3_bn1_var"),
                                   std::vector<float>{2e-05},  // stage2_unit3_bn1/add
                                   lookup("stage2_unit3_bn1_bias"),
                                   lookup("stage2_unit3_conv1_weight"),
                                   std::vector<float>{0},  // stage2_unit2_relu
                                   lookup("stage2_unit2_bn3_mean"),
                                   lookup("stage2_unit2_bn3_scale"),
                                   lookup("stage2_unit2_bn3_var"),
                                   std::vector<float>{2e-05},  // stage2_unit2_bn3/add
                                   lookup("stage2_unit2_bn3_bias"),
                                   lookup("stage2_unit2_conv3_weight"),
                                   std::vector<float>{0},  // stage2_unit2_relu2
                                   lookup("stage2_unit2_bn2_mean"),
                                   lookup("stage2_unit2_bn2_scale"),
                                   lookup("stage2_unit2_bn2_var"),
                                   std::vector<float>{2e-05},  // stage2_unit2_bn2/add
                                   lookup("stage2_unit2_bn2_bias"),
                                   lookup("stage2_unit2_conv2_weight"),
                                   std::vector<float>{0},  // stage2_unit2_relu1
                                   lookup("stage2_unit2_bn1_mean"),
                                   lookup("stage2_unit2_bn1_scale"),
                                   lookup("stage2_unit2_bn1_var"),
                                   std::vector<float>{2e-05},  // stage2_unit2_bn1/add
                                   lookup("stage2_unit2_bn1_bias"),
                                   lookup("stage2_unit2_conv1_weight"),
                                   std::vector<float>{0},  // stage2_unit1_relu
                                   lookup("stage2_unit1_sc_bn_mean"),
                                   lookup("stage2_unit1_sc_bn_scale"),
                                   lookup("stage2_unit1_sc_bn_var"),
                                   std::vector<float>{2e-05},  // stage2_unit1_sc_bn/add
                                   lookup("stage2_unit1_sc_bn_bias"),
                                   lookup("stage2_unit1_sc_weight"),
                                   std::vector<float>{0},  // stage1_unit3_relu
                                   lookup("stage1_unit3_bn3_mean"),
                                   lookup("stage1_unit3_bn3_scale"),
                                   lookup("stage1_unit3_bn3_var"),
                                   std::vector<float>{2e-05},  // stage1_unit3_bn3/add
                                   lookup("stage1_unit3_bn3_bias"),
                                   lookup("stage1_unit3_conv3_weight"),
                                   std::vector<float>{0},  // stage1_unit3_relu2
                                   lookup("stage1_unit3_bn2_mean"),
                                   lookup("stage1_unit3_bn2_scale"),
                                   lookup("stage1_unit3_bn2_var"),
                                   std::vector<float>{2e-05},  // stage1_unit3_bn2/add
                                   lookup("stage1_unit3_bn2_bias"),
                                   lookup("stage1_unit3_conv2_weight"),
                                   std::vector<float>{0},  // stage1_unit3_relu1
                                   lookup("stage1_unit3_bn1_mean"),
                                   lookup("stage1_unit3_bn1_scale"),
                                   lookup("stage1_unit3_bn1_var"),
                                   std::vector<float>{2e-05},  // stage1_unit3_bn1/add
                                   lookup("stage1_unit3_bn1_bias"),
                                   lookup("stage1_unit3_conv1_weight"),
                                   std::vector<float>{0},  // stage1_unit2_relu
                                   lookup("stage1_unit2_bn3_mean"),
                                   lookup("stage1_unit2_bn3_scale"),
                                   lookup("stage1_unit2_bn3_var"),
                                   std::vector<float>{2e-05},  // stage1_unit2_bn3/add
                                   lookup("stage1_unit2_bn3_bias"),
                                   lookup("stage1_unit2_conv3_weight"),
                                   std::vector<float>{0},  // stage1_unit2_relu2
                                   lookup("stage1_unit2_bn2_mean"),
                                   lookup("stage1_unit2_bn2_scale"),
                                   lookup("stage1_unit2_bn2_var"),
                                   std::vector<float>{2e-05},  // stage1_unit2_bn2/add
                                   lookup("stage1_unit2_bn2_bias"),
                                   lookup("stage1_unit2_conv2_weight"),
                                   std::vector<float>{0},  // stage1_unit2_relu1
                                   lookup("stage1_unit2_bn1_mean"),
                                   lookup("stage1_unit2_bn1_scale"),
                                   lookup("stage1_unit2_bn1_var"),
                                   std::vector<float>{2e-05},  // stage1_unit2_bn1/add
                                   lookup("stage1_unit2_bn1_bias"),
                                   lookup("stage1_unit2_conv1_weight"),
                                   std::vector<float>{0},  // stage1_unit1_relu
                                   lookup("stage1_unit1_sc_bn_mean"),
                                   lookup("stage1_unit1_sc_bn_scale"),
                                   lookup("stage1_unit1_sc_bn_var"),
                                   std::vector<float>{2e-05},  // stage1_unit1_sc_bn/add
                                   lookup("stage1_unit1_sc_bn_bias"),
                                   lookup("stage1_unit1_sc_weight"),
                                   std::vector<float>{0},  // relu0
                                   lookup("bn0_mean"),
                                   lookup("bn0_scale"),
                                   std::vector<float>{2e-05},  // bn0/add
                                   lookup("bn0_var"),
                                   lookup("bn0_bias"),
                                   lookup("conv0_weight"),
                                   lookup("bn_data_mean"),
                                   lookup("bn_data_var"),
                                   std::vector<float>{2e-05},  // bn_data/add
                                   lookup("bn_data_bias"),
                                   inputs[0],
                                   lookup("stage1_unit1_bn3_mean"),
                                   lookup("stage1_unit1_bn3_scale"),
                                   lookup("stage1_unit1_bn3_var"),
                                   std::vector<float>{2e-05},  // stage1_unit1_bn3/add
                                   lookup("stage1_unit1_bn3_bias"),
                                   lookup("stage1_unit1_conv3_weight"),
                                   std::vector<float>{0},  // stage1_unit1_relu2
                                   lookup("stage1_unit1_bn2_mean"),
                                   lookup("stage1_unit1_bn2_scale"),
                                   lookup("stage1_unit1_bn2_var"),
                                   std::vector<float>{2e-05},  // stage1_unit1_bn2/add
                                   lookup("stage1_unit1_bn2_bias"),
                                   lookup("stage1_unit1_conv2_weight"),
                                   std::vector<float>{0},  // stage1_unit1_relu1
                                   lookup("stage1_unit1_bn1_mean"),
                                   lookup("stage1_unit1_bn1_scale"),
                                   lookup("stage1_unit1_bn1_var"),
                                   std::vector<float>{2e-05},  // stage1_unit1_bn1/add
                                   lookup("stage1_unit1_bn1_bias"),
                                   lookup("stage1_unit1_conv1_weight"),
                                   lookup("stage2_unit1_bn3_mean"),
                                   lookup("stage2_unit1_bn3_scale"),
                                   lookup("stage2_unit1_bn3_var"),
                                   std::vector<float>{2e-05},  // stage2_unit1_bn3/add
                                   lookup("stage2_unit1_bn3_bias"),
                                   lookup("stage2_unit1_conv3_weight"),
                                   std::vector<float>{0},  // stage2_unit1_relu2
                                   lookup("stage2_unit1_bn2_mean"),
                                   lookup("stage2_unit1_bn2_scale"),
                                   lookup("stage2_unit1_bn2_var"),
                                   std::vector<float>{2e-05},  // stage2_unit1_bn2/add
                                   lookup("stage2_unit1_bn2_bias"),
                                   lookup("stage2_unit1_conv2_weight"),
                                   std::vector<float>{0},  // stage2_unit1_relu1
                                   lookup("stage2_unit1_bn1_mean"),
                                   lookup("stage2_unit1_bn1_scale"),
                                   lookup("stage2_unit1_bn1_var"),
                                   std::vector<float>{2e-05},  // stage2_unit1_bn1/add
                                   lookup("stage2_unit1_bn1_bias"),
                                   lookup("stage2_unit1_conv1_weight"),
                                   lookup("stage3_unit1_bn3_mean"),
                                   lookup("stage3_unit1_bn3_scale"),
                                   lookup("stage3_unit1_bn3_var"),
                                   std::vector<float>{2e-05},  // stage3_unit1_bn3/add
                                   lookup("stage3_unit1_bn3_bias"),
                                   lookup("stage3_unit1_conv3_weight"),
                                   std::vector<float>{0},  // stage3_unit1_relu2
                                   lookup("stage3_unit1_bn2_mean"),
                                   lookup("stage3_unit1_bn2_scale"),
                                   lookup("stage3_unit1_bn2_var"),
                                   std::vector<float>{2e-05},  // stage3_unit1_bn2/add
                                   lookup("stage3_unit1_bn2_bias"),
                                   lookup("stage3_unit1_conv2_weight"),
                                   std::vector<float>{0},  // stage3_unit1_relu1
                                   lookup("stage3_unit1_bn1_mean"),
                                   lookup("stage3_unit1_bn1_scale"),
                                   lookup("stage3_unit1_bn1_var"),
                                   std::vector<float>{2e-05},  // stage3_unit1_bn1/add
                                   lookup("stage3_unit1_bn1_bias"),
                                   lookup("stage3_unit1_conv1_weight"),
                                   lookup("stage4_unit1_bn3_mean"),
                                   lookup("stage4_unit1_bn3_scale"),
                                   lookup("stage4_unit1_bn3_var"),
                                   std::vector<float>{2e-05},  // stage4_unit1_bn3/add
                                   lookup("stage4_unit1_bn3_bias"),
                                   lookup("stage4_unit1_conv3_weight"),
                                   std::vector<float>{0},  // stage4_unit1_relu2
                                   lookup("stage4_unit1_bn2_mean"),
                                   lookup("stage4_unit1_bn2_scale"),
                                   lookup("stage4_unit1_bn2_var"),
                                   std::vector<float>{2e-05},  // stage4_unit1_bn2/add
                                   lookup("stage4_unit1_bn2_bias"),
                                   lookup("stage4_unit1_conv2_weight"),
                                   std::vector<float>{0},  // stage4_unit1_relu1
                                   lookup("stage4_unit1_bn1_mean"),
                                   lookup("stage4_unit1_bn1_scale"),
                                   lookup("stage4_unit1_bn1_var"),
                                   std::vector<float>{2e-05},  // stage4_unit1_bn1/add
                                   lookup("stage4_unit1_bn1_bias"),
                                   lookup("stage4_unit1_conv1_weight")};

  std::string hlo_text = R"(
    HloModule cluster_0__XlaCompiledKernel_true__XlaNumConstantArgs_70__XlaNumResourceArgs_0_.2963

%max_F32.569 (lhs.570: f32[], rhs.571: f32[]) -> f32[] {
  %lhs.570 = f32[] parameter(0)
  %rhs.571 = f32[] parameter(1)
  ROOT %maximum.572 = f32[] maximum(f32[] %lhs.570, f32[] %rhs.571)
}

%add_F32.2930 (lhs.2931: f32[], rhs.2932: f32[]) -> f32[] {
  %lhs.2931 = f32[] parameter(0)
  %rhs.2932 = f32[] parameter(1)
  ROOT %add.2933 = f32[] add(f32[] %lhs.2931, f32[] %rhs.2932)
}

ENTRY %cluster_0__XlaCompiledKernel_true__XlaNumConstantArgs_70__XlaNumResourceArgs_0_.2963 (arg0.1: f32[3], arg1.2: f32[2048], arg2.3: f32[64], arg3.4: f32[512], arg4.5: f32[1024], arg5.6: f32[128], arg6.7: f32[256], arg7.8: f32[3,3,16,512], arg8.9: f32[3,3,4,128], arg9.10: f32[128], arg10.11: f32[256], arg11.12: f32[128], arg12.13: f32[3,3,4,128], arg13.14: f32[128], arg14.15: f32[512], arg15.16: f32[256], arg16.17: f32[128], arg17.18: f32[3,3,4,128], arg18.19: f32[128], arg19.20: f32[1024], arg20.21: f32[256], arg21.22: f32[3,3,32,1024], arg22.23: f32[256], arg23.24: f32[512], arg24.25: f32[3,3,8,256], arg25.26: f32[256], arg26.27: f32[1024], arg27.28: f32[512], arg28.29: f32[256], arg29.30: f32[2048], arg30.31: f32[3,3,8,256], arg31.32: f32[256], arg32.33: f32[512], arg33.34: f32[3,3,32,1024], arg34.35: f32[256], arg35.36: f32[3,3,8,256], arg36.37: f32[256], arg37.38: f32[512], arg38.39: f32[256], arg39.40: f32[3,3,8,256], arg40.41: f32[1024], arg41.42: f32[256], arg42.43: f32[512], arg43.44: f32[512], arg44.45: f32[2048], arg45.46: f32[1024], arg46.47: f32[3,3,16,512], arg47.48: f32[512], arg48.49: f32[1024], arg49.50: f32[1024], arg50.51: f32[1024], arg51.52: f32[512], arg52.53: f32[3,3,16,512], arg53.54: f32[512], arg54.55: f32[3,3,32,1024], arg55.56: f32[1024], arg56.57: f32[512], arg57.58: f32[3,3,16,512], arg58.59: f32[512], arg59.60: f32[1024], arg60.61: f32[512], arg61.62: f32[1024], arg62.63: f32[3,3,16,512], arg63.64: f32[512], arg64.65: f32[1024], arg65.66: f32[512], arg66.67: f32[2048], arg67.68: f32[3,3,16,512], arg68.69: f32[512], arg69.70: f32[1024], arg70.71: f32[3], arg71.72: f32[2048], arg72.73: f32[64], arg73.74: f32[512], arg74.75: f32[1024], arg75.76: f32[128], arg76.77: f32[256], arg77.78: f32[128], arg78.79: f32[256], arg79.80: f32[128], arg80.81: f32[128], arg81.82: f32[512], arg82.83: f32[256], arg83.84: f32[128], arg84.85: f32[128], arg85.86: f32[1024], arg86.87: f32[256], arg87.88: f32[256], arg88.89: f32[512], arg89.90: f32[256], arg90.91: f32[1024], arg91.92: f32[512], arg92.93: f32[256], arg93.94: f32[2048], arg94.95: f32[256], arg95.96: f32[512], arg96.97: f32[256], arg97.98: f32[256], arg98.99: f32[512], arg99.100: f32[256], arg100.101: f32[1024], arg101.102: f32[256], arg102.103: f32[512], arg103.104: f32[512], arg104.105: f32[2048], arg105.106: f32[1024], arg106.107: f32[512], arg107.108: f32[1024], arg108.109: f32[1024], arg109.110: f32[1024], arg110.111: f32[512], arg111.112: f32[512], arg112.113: f32[1024], arg113.114: f32[512], arg114.115: f32[512], arg115.116: f32[1024], arg116.117: f32[512], arg117.118: f32[1024], arg118.119: f32[512], arg119.120: f32[1024], arg120.121: f32[512], arg121.122: f32[2048], arg122.123: f32[512], arg123.124: f32[1024], arg124.125: f32[3], arg125.126: f32[2048], arg126.127: f32[64], arg127.128: f32[512], arg128.129: f32[1024], arg129.130: f32[128], arg130.131: f32[256], arg131.132: f32[128], arg132.133: f32[256], arg133.134: f32[128], arg134.135: f32[128], arg135.136: f32[512], arg136.137: f32[256], arg137.138: f32[128], arg138.139: f32[128], arg139.140: f32[1024], arg140.141: f32[256], arg141.142: f32[256], arg142.143: f32[512], arg143.144: f32[256], arg144.145: f32[1024], arg145.146: f32[512], arg146.147: f32[256], arg147.148: f32[2048], arg148.149: f32[256], arg149.150: f32[512], arg150.151: f32[256], arg151.152: f32[256], arg152.153: f32[512], arg153.154: f32[256], arg154.155: f32[1024], arg155.156: f32[256], arg156.157: f32[512], arg157.158: f32[512], arg158.159: f32[2048], arg159.160: f32[1024], arg160.161: f32[512], arg161.162: f32[1024], arg162.163: f32[1024], arg163.164: f32[1024], arg164.165: f32[512], arg165.166: f32[512], arg166.167: f32[1024], arg167.168: f32[512], arg168.169: f32[512], arg169.170: f32[1024], arg170.171: f32[512], arg171.172: f32[1024], arg172.173: f32[512], arg173.174: f32[1024], arg174.175: f32[512], arg175.176: f32[2048], arg176.177: f32[512], arg177.178: f32[1024], arg178.179: f32[2048], arg179.180: f32[64], arg180.181: f32[512], arg181.182: f32[1024], arg182.183: f32[128], arg183.184: f32[256], arg184.185: f32[128], arg185.186: f32[256], arg186.187: f32[128], arg187.188: f32[128], arg188.189: f32[512], arg189.190: f32[256], arg190.191: f32[128], arg191.192: f32[128], arg192.193: f32[1024], arg193.194: f32[256], arg194.195: f32[256], arg195.196: f32[512], arg196.197: f32[256], arg197.198: f32[1024], arg198.199: f32[512], arg199.200: f32[256], arg200.201: f32[2048], arg201.202: f32[256], arg202.203: f32[512], arg203.204: f32[256], arg204.205: f32[256], arg205.206: f32[512], arg206.207: f32[256], arg207.208: f32[1024], arg208.209: f32[256], arg209.210: f32[512], arg210.211: f32[512], arg211.212: f32[2048], arg212.213: f32[1024], arg213.214: f32[512], arg214.215: f32[1024], arg215.216: f32[1024], arg216.217: f32[1024], arg217.218: f32[512], arg218.219: f32[512], arg219.220: f32[1024], arg220.221: f32[512], arg221.222: f32[512], arg222.223: f32[1024], arg223.224: f32[512], arg224.225: f32[1024], arg225.226: f32[512], arg226.227: f32[1024], arg227.228: f32[512], arg228.229: f32[2048], arg229.230: f32[512], arg230.231: f32[1024], arg231.232: f32[7,7,3,64], arg232.233: f32[1,1,64,128], arg233.234: f32[1,1,64,256], arg234.235: f32[1,1,128,256], arg235.236: f32[1,1,256,128], arg236.237: f32[1,1,128,256], arg237.238: f32[1,1,256,128], arg238.239: f32[1,1,128,256], arg239.240: f32[1,1,256,256], arg240.241: f32[1,1,256,512], arg241.242: f32[1,1,256,512], arg242.243: f32[1,1,512,256], arg243.244: f32[1,1,256,512], arg244.245: f32[1,1,512,256], arg245.246: f32[1,1,256,512], arg246.247: f32[1,1,512,256], arg247.248: f32[1,1,256,512], arg248.249: f32[1,1,512,512], arg249.250: f32[1,1,512,1024], arg250.251: f32[1,1,512,1024], arg251.252: f32[1,1,1024,512], arg252.253: f32[1,1,512,1024], arg253.254: f32[1,1,1024,512], arg254.255: f32[1,1,512,1024], arg255.256: f32[1,1,1024,512], arg256.257: f32[1,1,512,1024], arg257.258: f32[1,1,1024,512], arg258.259: f32[1,1,512,1024], arg259.260: f32[1,1,1024,512], arg260.261: f32[1,1,512,1024], arg261.262: f32[1,1,1024,1024], arg262.263: f32[1,1,1024,2048], arg263.264: f32[1,1,1024,2048], arg264.265: f32[1,1,2048,1024], arg265.266: f32[1,1,1024,2048], arg266.267: f32[1,1,2048,1024], arg267.268: f32[1,1,1024,2048], arg268.269: f32[1,224,224,3]) -> f32[1,2048] {
  %constant.2923 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit3_relu"}
  %broadcast.2924 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[] %constant.2923), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit3_relu"}
  %constant.2779 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit2_relu"}
  %broadcast.2780 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[] %constant.2779), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit2_relu"}
  %constant.2635 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit1_relu"}
  %broadcast.2636 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[] %constant.2635), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit1_relu"}
  %constant.2496 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit1_bn3/add"}
  %broadcast.2497 = f32[2048]{0} broadcast(f32[] %constant.2496), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit1_bn3/add"}
  %arg44.45 = f32[2048]{0} parameter(44), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.314 = f32[2048]{0} reshape(f32[2048]{0} %arg44.45)
  %add.2498 = f32[2048]{0} add(f32[2048]{0} %broadcast.2497, f32[2048]{0} %reshape.314), metadata={op_type="AddV2" op_name="stage4_unit1_bn3/add"}
  %rsqrt.2499 = f32[2048]{0} rsqrt(f32[2048]{0} %add.2498), metadata={op_type="Rsqrt" op_name="stage4_unit1_bn3/Rsqrt"}
  %arg104.105 = f32[2048]{0} parameter(104), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.374 = f32[2048]{0} reshape(f32[2048]{0} %arg104.105)
  %multiply.2500 = f32[2048]{0} multiply(f32[2048]{0} %rsqrt.2499, f32[2048]{0} %reshape.374), metadata={op_type="Mul" op_name="stage4_unit1_bn3/mul"}
  %broadcast.2618 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %multiply.2500), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit1_bn3/mul_1"}
  %constant.2614 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit1_relu2"}
  %broadcast.2615 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[] %constant.2614), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit1_relu2"}
  %constant.2508 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit1_relu1"}
  %broadcast.2509 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.2508), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit1_relu1"}
  %constant.2482 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit1_bn1/add"}
  %broadcast.2483 = f32[1024]{0} broadcast(f32[] %constant.2482), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit1_bn1/add"}
  %arg26.27 = f32[1024]{0} parameter(26), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.296 = f32[1024]{0} reshape(f32[1024]{0} %arg26.27)
  %add.2484 = f32[1024]{0} add(f32[1024]{0} %broadcast.2483, f32[1024]{0} %reshape.296), metadata={op_type="AddV2" op_name="stage4_unit1_bn1/add"}
  %rsqrt.2485 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2484), metadata={op_type="Rsqrt" op_name="stage4_unit1_bn1/Rsqrt"}
  %arg90.91 = f32[1024]{0} parameter(90), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.360 = f32[1024]{0} reshape(f32[1024]{0} %arg90.91)
  %multiply.2486 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2485, f32[1024]{0} %reshape.360), metadata={op_type="Mul" op_name="stage4_unit1_bn1/mul"}
  %broadcast.2504 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2486), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit1_bn1/mul_1"}
  %constant.2479 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit6_relu"}
  %broadcast.2480 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.2479), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit6_relu"}
  %constant.2335 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit5_relu"}
  %broadcast.2336 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.2335), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit5_relu"}
  %constant.2191 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit4_relu"}
  %broadcast.2192 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.2191), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit4_relu"}
  %constant.2047 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit3_relu"}
  %broadcast.2048 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.2047), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit3_relu"}
  %constant.1903 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit2_relu"}
  %broadcast.1904 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.1903), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit2_relu"}
  %constant.1759 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit1_relu"}
  %broadcast.1760 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.1759), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit1_relu"}
  %constant.1620 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit1_bn3/add"}
  %broadcast.1621 = f32[1024]{0} broadcast(f32[] %constant.1620), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit1_bn3/add"}
  %arg48.49 = f32[1024]{0} parameter(48), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.318 = f32[1024]{0} reshape(f32[1024]{0} %arg48.49)
  %add.1622 = f32[1024]{0} add(f32[1024]{0} %broadcast.1621, f32[1024]{0} %reshape.318), metadata={op_type="AddV2" op_name="stage3_unit1_bn3/add"}
  %rsqrt.1623 = f32[1024]{0} rsqrt(f32[1024]{0} %add.1622), metadata={op_type="Rsqrt" op_name="stage3_unit1_bn3/Rsqrt"}
  %arg107.108 = f32[1024]{0} parameter(107), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.377 = f32[1024]{0} reshape(f32[1024]{0} %arg107.108)
  %multiply.1624 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.1623, f32[1024]{0} %reshape.377), metadata={op_type="Mul" op_name="stage3_unit1_bn3/mul"}
  %broadcast.1742 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.1624), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit1_bn3/mul_1"}
  %constant.1738 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit1_relu2"}
  %broadcast.1739 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.1738), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit1_relu2"}
  %constant.1632 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit1_relu1"}
  %broadcast.1633 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1632), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit1_relu1"}
  %constant.1606 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit1_bn1/add"}
  %broadcast.1607 = f32[512]{0} broadcast(f32[] %constant.1606), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit1_bn1/add"}
  %arg43.44 = f32[512]{0} parameter(43), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.313 = f32[512]{0} reshape(f32[512]{0} %arg43.44)
  %add.1608 = f32[512]{0} add(f32[512]{0} %broadcast.1607, f32[512]{0} %reshape.313), metadata={op_type="AddV2" op_name="stage3_unit1_bn1/add"}
  %rsqrt.1609 = f32[512]{0} rsqrt(f32[512]{0} %add.1608), metadata={op_type="Rsqrt" op_name="stage3_unit1_bn1/Rsqrt"}
  %arg103.104 = f32[512]{0} parameter(103), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.373 = f32[512]{0} reshape(f32[512]{0} %arg103.104)
  %multiply.1610 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1609, f32[512]{0} %reshape.373), metadata={op_type="Mul" op_name="stage3_unit1_bn1/mul"}
  %broadcast.1628 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1610), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit1_bn1/mul_1"}
  %constant.1603 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit4_relu"}
  %broadcast.1604 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1603), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit4_relu"}
  %constant.1459 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit3_relu"}
  %broadcast.1460 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1459), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit3_relu"}
  %constant.1315 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit2_relu"}
  %broadcast.1316 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1315), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit2_relu"}
  %constant.1171 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit1_relu"}
  %broadcast.1172 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1171), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit1_relu"}
  %constant.1032 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add"}
  %broadcast.1033 = f32[512]{0} broadcast(f32[] %constant.1032), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add"}
  %arg27.28 = f32[512]{0} parameter(27), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.297 = f32[512]{0} reshape(f32[512]{0} %arg27.28)
  %add.1034 = f32[512]{0} add(f32[512]{0} %broadcast.1033, f32[512]{0} %reshape.297), metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add"}
  %rsqrt.1035 = f32[512]{0} rsqrt(f32[512]{0} %add.1034), metadata={op_type="Rsqrt" op_name="stage2_unit1_bn3/Rsqrt"}
  %arg91.92 = f32[512]{0} parameter(91), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.361 = f32[512]{0} reshape(f32[512]{0} %arg91.92)
  %multiply.1036 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1035, f32[512]{0} %reshape.361), metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul"}
  %broadcast.1154 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1036), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul_1"}
  %constant.1150 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit1_relu2"}
  %broadcast.1151 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1150), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit1_relu2"}
  %constant.1044 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit1_relu1"}
  %broadcast.1045 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.1044), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit1_relu1"}
  %constant.1018 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add"}
  %broadcast.1019 = f32[256]{0} broadcast(f32[] %constant.1018), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add"}
  %arg22.23 = f32[256]{0} parameter(22), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.292 = f32[256]{0} reshape(f32[256]{0} %arg22.23)
  %add.1020 = f32[256]{0} add(f32[256]{0} %broadcast.1019, f32[256]{0} %reshape.292), metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add"}
  %rsqrt.1021 = f32[256]{0} rsqrt(f32[256]{0} %add.1020), metadata={op_type="Rsqrt" op_name="stage2_unit1_bn1/Rsqrt"}
  %arg87.88 = f32[256]{0} parameter(87), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.357 = f32[256]{0} reshape(f32[256]{0} %arg87.88)
  %multiply.1022 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1021, f32[256]{0} %reshape.357), metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul"}
  %broadcast.1040 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1022), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul_1"}
  %constant.1015 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit3_relu"}
  %broadcast.1016 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.1015), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit3_relu"}
  %constant.871 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit2_relu"}
  %broadcast.872 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.871), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit2_relu"}
  %constant.727 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit1_relu"}
  %broadcast.728 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.727), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit1_relu"}
  %constant.588 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add"}
  %broadcast.589 = f32[256]{0} broadcast(f32[] %constant.588), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add"}
  %arg10.11 = f32[256]{0} parameter(10), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.280 = f32[256]{0} reshape(f32[256]{0} %arg10.11)
  %add.590 = f32[256]{0} add(f32[256]{0} %broadcast.589, f32[256]{0} %reshape.280), metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add"}
  %rsqrt.591 = f32[256]{0} rsqrt(f32[256]{0} %add.590), metadata={op_type="Rsqrt" op_name="stage1_unit1_bn3/Rsqrt"}
  %arg78.79 = f32[256]{0} parameter(78), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.348 = f32[256]{0} reshape(f32[256]{0} %arg78.79)
  %multiply.592 = f32[256]{0} multiply(f32[256]{0} %rsqrt.591, f32[256]{0} %reshape.348), metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul"}
  %broadcast.710 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.592), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul_1"}
  %constant.706 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit1_relu2"}
  %broadcast.707 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.706), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit1_relu2"}
  %constant.600 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit1_relu1"}
  %broadcast.601 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.600), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit1_relu1"}
  %constant.574 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add"}
  %broadcast.575 = f32[128]{0} broadcast(f32[] %constant.574), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add"}
  %arg5.6 = f32[128]{0} parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.275 = f32[128]{0} reshape(f32[128]{0} %arg5.6)
  %add.576 = f32[128]{0} add(f32[128]{0} %broadcast.575, f32[128]{0} %reshape.275), metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add"}
  %rsqrt.577 = f32[128]{0} rsqrt(f32[128]{0} %add.576), metadata={op_type="Rsqrt" op_name="stage1_unit1_bn1/Rsqrt"}
  %arg75.76 = f32[128]{0} parameter(75), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.345 = f32[128]{0} reshape(f32[128]{0} %arg75.76)
  %multiply.578 = f32[128]{0} multiply(f32[128]{0} %rsqrt.577, f32[128]{0} %reshape.345), metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul"}
  %broadcast.596 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.578), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul_1"}
  %constant.563 = f32[] constant(0), metadata={op_type="Relu" op_name="relu0"}
  %broadcast.564 = f32[1,112,112,64]{3,2,1,0} broadcast(f32[] %constant.563), dimensions={}, metadata={op_type="Relu" op_name="relu0"}
  %arg2.3 = f32[64]{0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.272 = f32[64]{0} reshape(f32[64]{0} %arg2.3)
  %constant.539 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="bn0/add"}
  %broadcast.540 = f32[64]{0} broadcast(f32[] %constant.539), dimensions={}, metadata={op_type="AddV2" op_name="bn0/add"}
  %add.541 = f32[64]{0} add(f32[64]{0} %reshape.272, f32[64]{0} %broadcast.540), metadata={op_type="AddV2" op_name="bn0/add"}
  %rsqrt.542 = f32[64]{0} rsqrt(f32[64]{0} %add.541), metadata={op_type="Rsqrt" op_name="bn0/Rsqrt"}
  %arg72.73 = f32[64]{0} parameter(72), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.342 = f32[64]{0} reshape(f32[64]{0} %arg72.73)
  %multiply.543 = f32[64]{0} multiply(f32[64]{0} %rsqrt.542, f32[64]{0} %reshape.342), metadata={op_type="Mul" op_name="bn0/mul"}
  %broadcast.559 = f32[1,112,112,64]{3,2,1,0} broadcast(f32[64]{0} %multiply.543), dimensions={3}, metadata={op_type="Mul" op_name="bn0/mul_1"}
  %constant.546 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="bn_data/add"}
  %broadcast.547 = f32[3]{0} broadcast(f32[] %constant.546), dimensions={}, metadata={op_type="AddV2" op_name="bn_data/add"}
  %arg0.1 = f32[3]{0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.270 = f32[3]{0} reshape(f32[3]{0} %arg0.1)
  %add.548 = f32[3]{0} add(f32[3]{0} %broadcast.547, f32[3]{0} %reshape.270), metadata={op_type="AddV2" op_name="bn_data/add"}
  %rsqrt.549 = f32[3]{0} rsqrt(f32[3]{0} %add.548), metadata={op_type="Rsqrt" op_name="bn_data/Rsqrt"}
  %broadcast.550 = f32[1,224,224,3]{3,2,1,0} broadcast(f32[3]{0} %rsqrt.549), dimensions={3}, metadata={op_type="Mul" op_name="bn_data/mul"}
  %arg268.269 = f32[1,224,224,3]{3,2,1,0} parameter(268), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.538 = f32[1,224,224,3]{3,2,1,0} reshape(f32[1,224,224,3]{3,2,1,0} %arg268.269)
  %multiply.551 = f32[1,224,224,3]{3,2,1,0} multiply(f32[1,224,224,3]{3,2,1,0} %broadcast.550, f32[1,224,224,3]{3,2,1,0} %reshape.538), metadata={op_type="Mul" op_name="bn_data/mul"}
  %arg124.125 = f32[3]{0} parameter(124), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.394 = f32[3]{0} reshape(f32[3]{0} %arg124.125)
  %arg70.71 = f32[3]{0} parameter(70), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.340 = f32[3]{0} reshape(f32[3]{0} %arg70.71)
  %multiply.552 = f32[3]{0} multiply(f32[3]{0} %rsqrt.549, f32[3]{0} %reshape.340), metadata={op_type="Mul" op_name="bn_data/mul_1"}
  %subtract.553 = f32[3]{0} subtract(f32[3]{0} %reshape.394, f32[3]{0} %multiply.552), metadata={op_type="Sub" op_name="bn_data/sub"}
  %broadcast.554 = f32[1,224,224,3]{3,2,1,0} broadcast(f32[3]{0} %subtract.553), dimensions={3}, metadata={op_type="AddV2" op_name="bn_data/add_1"}
  %add.555 = f32[1,224,224,3]{3,2,1,0} add(f32[1,224,224,3]{3,2,1,0} %multiply.551, f32[1,224,224,3]{3,2,1,0} %broadcast.554), metadata={op_type="AddV2" op_name="bn_data/add_1"}
  %constant.556 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad"}
  %pad.557 = f32[1,230,230,3]{3,2,1,0} pad(f32[1,224,224,3]{3,2,1,0} %add.555, f32[] %constant.556), padding=0_0x3_3x3_3x0_0, metadata={op_type="Pad" op_name="Pad"}
  %arg231.232 = f32[7,7,3,64]{3,2,1,0} parameter(231), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.501 = f32[7,7,3,64]{3,2,1,0} reshape(f32[7,7,3,64]{3,2,1,0} %arg231.232)
  %convolution.558 = f32[1,112,112,64]{3,2,1,0} convolution(f32[1,230,230,3]{3,2,1,0} %pad.557, f32[7,7,3,64]{3,2,1,0} %reshape.501), window={size=7x7 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="conv0"}
  %multiply.560 = f32[1,112,112,64]{3,2,1,0} multiply(f32[1,112,112,64]{3,2,1,0} %broadcast.559, f32[1,112,112,64]{3,2,1,0} %convolution.558), metadata={op_type="Mul" op_name="bn0/mul_1"}
  %arg179.180 = f32[64]{0} parameter(179), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.449 = f32[64]{0} reshape(f32[64]{0} %arg179.180)
  %arg126.127 = f32[64]{0} parameter(126), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.396 = f32[64]{0} reshape(f32[64]{0} %arg126.127)
  %multiply.544 = f32[64]{0} multiply(f32[64]{0} %multiply.543, f32[64]{0} %reshape.396), metadata={op_type="Mul" op_name="bn0/mul_2"}
  %subtract.545 = f32[64]{0} subtract(f32[64]{0} %reshape.449, f32[64]{0} %multiply.544), metadata={op_type="Sub" op_name="bn0/sub"}
  %broadcast.561 = f32[1,112,112,64]{3,2,1,0} broadcast(f32[64]{0} %subtract.545), dimensions={3}, metadata={op_type="AddV2" op_name="bn0/add_1"}
  %add.562 = f32[1,112,112,64]{3,2,1,0} add(f32[1,112,112,64]{3,2,1,0} %multiply.560, f32[1,112,112,64]{3,2,1,0} %broadcast.561), metadata={op_type="AddV2" op_name="bn0/add_1"}
  %maximum.565 = f32[1,112,112,64]{3,2,1,0} maximum(f32[1,112,112,64]{3,2,1,0} %broadcast.564, f32[1,112,112,64]{3,2,1,0} %add.562), metadata={op_type="Relu" op_name="relu0"}
  %constant.566 = f32[] constant(-inf), metadata={op_type="PadV2" op_name="PadV2"}
  %pad.567 = f32[1,114,114,64]{3,2,1,0} pad(f32[1,112,112,64]{3,2,1,0} %maximum.565, f32[] %constant.566), padding=0_0x1_1x1_1x0_0, metadata={op_type="PadV2" op_name="PadV2"}
  %constant.568 = f32[] constant(-inf), metadata={op_type="MaxPool" op_name="pooling0"}
  %reduce-window.573 = f32[1,56,56,64]{3,2,1,0} reduce-window(f32[1,114,114,64]{3,2,1,0} %pad.567, f32[] %constant.568), window={size=1x3x3x1 stride=1x2x2x1}, to_apply=%max_F32.569, metadata={op_type="MaxPool" op_name="pooling0"}
  %arg232.233 = f32[1,1,64,128]{3,2,1,0} parameter(232), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.502 = f32[1,1,64,128]{3,2,1,0} reshape(f32[1,1,64,128]{3,2,1,0} %arg232.233)
  %convolution.595 = f32[1,56,56,128]{3,2,1,0} convolution(f32[1,56,56,64]{3,2,1,0} %reduce-window.573, f32[1,1,64,128]{3,2,1,0} %reshape.502), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv1"}
  %multiply.597 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %broadcast.596, f32[1,56,56,128]{3,2,1,0} %convolution.595), metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul_1"}
  %arg182.183 = f32[128]{0} parameter(182), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.452 = f32[128]{0} reshape(f32[128]{0} %arg182.183)
  %arg129.130 = f32[128]{0} parameter(129), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.399 = f32[128]{0} reshape(f32[128]{0} %arg129.130)
  %multiply.579 = f32[128]{0} multiply(f32[128]{0} %multiply.578, f32[128]{0} %reshape.399), metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul_2"}
  %subtract.580 = f32[128]{0} subtract(f32[128]{0} %reshape.452, f32[128]{0} %multiply.579), metadata={op_type="Sub" op_name="stage1_unit1_bn1/sub"}
  %broadcast.598 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.580), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add_1"}
  %add.599 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.597, f32[1,56,56,128]{3,2,1,0} %broadcast.598), metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add_1"}
  %maximum.602 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.601, f32[1,56,56,128]{3,2,1,0} %add.599), metadata={op_type="Relu" op_name="stage1_unit1_relu1"}
  %constant.603 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_1"}
  %pad.604 = f32[1,58,58,128]{3,2,1,0} pad(f32[1,56,56,128]{3,2,1,0} %maximum.602, f32[] %constant.603), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_1"}
  %slice.605 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [0:4]}, metadata={op_type="Split" op_name="split_1"}
  %arg8.9 = f32[3,3,4,128]{3,2,1,0} parameter(8), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.278 = f32[3,3,4,128]{3,2,1,0} reshape(f32[3,3,4,128]{3,2,1,0} %arg8.9)
  %slice.637 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [0:4]}, metadata={op_type="Split" op_name="split"}
  %convolution.669 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.605, f32[3,3,4,4]{3,2,1,0} %slice.637), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2"}
  %slice.606 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [4:8]}, metadata={op_type="Split" op_name="split_1"}
  %slice.638 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [4:8]}, metadata={op_type="Split" op_name="split"}
  %convolution.670 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.606, f32[3,3,4,4]{3,2,1,0} %slice.638), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_1"}
  %slice.607 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [8:12]}, metadata={op_type="Split" op_name="split_1"}
  %slice.639 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [8:12]}, metadata={op_type="Split" op_name="split"}
  %convolution.681 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.607, f32[3,3,4,4]{3,2,1,0} %slice.639), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_2"}
  %slice.608 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [12:16]}, metadata={op_type="Split" op_name="split_1"}
  %slice.640 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [12:16]}, metadata={op_type="Split" op_name="split"}
  %convolution.692 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.608, f32[3,3,4,4]{3,2,1,0} %slice.640), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_3"}
  %slice.609 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [16:20]}, metadata={op_type="Split" op_name="split_1"}
  %slice.641 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [16:20]}, metadata={op_type="Split" op_name="split"}
  %convolution.695 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.609, f32[3,3,4,4]{3,2,1,0} %slice.641), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_4"}
  %slice.610 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [20:24]}, metadata={op_type="Split" op_name="split_1"}
  %slice.642 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [20:24]}, metadata={op_type="Split" op_name="split"}
  %convolution.696 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.610, f32[3,3,4,4]{3,2,1,0} %slice.642), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_5"}
  %slice.611 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [24:28]}, metadata={op_type="Split" op_name="split_1"}
  %slice.643 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [24:28]}, metadata={op_type="Split" op_name="split"}
  %convolution.697 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.611, f32[3,3,4,4]{3,2,1,0} %slice.643), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_6"}
  %slice.612 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [28:32]}, metadata={op_type="Split" op_name="split_1"}
  %slice.644 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [28:32]}, metadata={op_type="Split" op_name="split"}
  %convolution.698 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.612, f32[3,3,4,4]{3,2,1,0} %slice.644), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_7"}
  %slice.613 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [32:36]}, metadata={op_type="Split" op_name="split_1"}
  %slice.645 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [32:36]}, metadata={op_type="Split" op_name="split"}
  %convolution.699 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.613, f32[3,3,4,4]{3,2,1,0} %slice.645), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_8"}
  %slice.614 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [36:40]}, metadata={op_type="Split" op_name="split_1"}
  %slice.646 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [36:40]}, metadata={op_type="Split" op_name="split"}
  %convolution.700 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.614, f32[3,3,4,4]{3,2,1,0} %slice.646), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_9"}
  %slice.615 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [40:44]}, metadata={op_type="Split" op_name="split_1"}
  %slice.647 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [40:44]}, metadata={op_type="Split" op_name="split"}
  %convolution.671 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.615, f32[3,3,4,4]{3,2,1,0} %slice.647), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_10"}
  %slice.616 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [44:48]}, metadata={op_type="Split" op_name="split_1"}
  %slice.648 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [44:48]}, metadata={op_type="Split" op_name="split"}
  %convolution.672 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.616, f32[3,3,4,4]{3,2,1,0} %slice.648), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_11"}
  %slice.617 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [48:52]}, metadata={op_type="Split" op_name="split_1"}
  %slice.649 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [48:52]}, metadata={op_type="Split" op_name="split"}
  %convolution.673 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.617, f32[3,3,4,4]{3,2,1,0} %slice.649), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_12"}
  %slice.618 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [52:56]}, metadata={op_type="Split" op_name="split_1"}
  %slice.650 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [52:56]}, metadata={op_type="Split" op_name="split"}
  %convolution.674 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.618, f32[3,3,4,4]{3,2,1,0} %slice.650), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_13"}
  %slice.619 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [56:60]}, metadata={op_type="Split" op_name="split_1"}
  %slice.651 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [56:60]}, metadata={op_type="Split" op_name="split"}
  %convolution.675 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.619, f32[3,3,4,4]{3,2,1,0} %slice.651), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_14"}
  %slice.620 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [60:64]}, metadata={op_type="Split" op_name="split_1"}
  %slice.652 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [60:64]}, metadata={op_type="Split" op_name="split"}
  %convolution.676 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.620, f32[3,3,4,4]{3,2,1,0} %slice.652), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_15"}
  %slice.621 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [64:68]}, metadata={op_type="Split" op_name="split_1"}
  %slice.653 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [64:68]}, metadata={op_type="Split" op_name="split"}
  %convolution.677 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.621, f32[3,3,4,4]{3,2,1,0} %slice.653), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_16"}
  %slice.622 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [68:72]}, metadata={op_type="Split" op_name="split_1"}
  %slice.654 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [68:72]}, metadata={op_type="Split" op_name="split"}
  %convolution.678 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.622, f32[3,3,4,4]{3,2,1,0} %slice.654), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_17"}
  %slice.623 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [72:76]}, metadata={op_type="Split" op_name="split_1"}
  %slice.655 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [72:76]}, metadata={op_type="Split" op_name="split"}
  %convolution.679 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.623, f32[3,3,4,4]{3,2,1,0} %slice.655), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_18"}
  %slice.624 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [76:80]}, metadata={op_type="Split" op_name="split_1"}
  %slice.656 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [76:80]}, metadata={op_type="Split" op_name="split"}
  %convolution.680 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.624, f32[3,3,4,4]{3,2,1,0} %slice.656), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_19"}
  %slice.625 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [80:84]}, metadata={op_type="Split" op_name="split_1"}
  %slice.657 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [80:84]}, metadata={op_type="Split" op_name="split"}
  %convolution.682 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.625, f32[3,3,4,4]{3,2,1,0} %slice.657), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_20"}
  %slice.626 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [84:88]}, metadata={op_type="Split" op_name="split_1"}
  %slice.658 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [84:88]}, metadata={op_type="Split" op_name="split"}
  %convolution.683 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.626, f32[3,3,4,4]{3,2,1,0} %slice.658), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_21"}
  %slice.627 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [88:92]}, metadata={op_type="Split" op_name="split_1"}
  %slice.659 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [88:92]}, metadata={op_type="Split" op_name="split"}
  %convolution.684 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.627, f32[3,3,4,4]{3,2,1,0} %slice.659), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_22"}
  %slice.628 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [92:96]}, metadata={op_type="Split" op_name="split_1"}
  %slice.660 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [92:96]}, metadata={op_type="Split" op_name="split"}
  %convolution.685 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.628, f32[3,3,4,4]{3,2,1,0} %slice.660), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_23"}
  %slice.629 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [96:100]}, metadata={op_type="Split" op_name="split_1"}
  %slice.661 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [96:100]}, metadata={op_type="Split" op_name="split"}
  %convolution.686 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.629, f32[3,3,4,4]{3,2,1,0} %slice.661), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_24"}
  %slice.630 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [100:104]}, metadata={op_type="Split" op_name="split_1"}
  %slice.662 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [100:104]}, metadata={op_type="Split" op_name="split"}
  %convolution.687 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.630, f32[3,3,4,4]{3,2,1,0} %slice.662), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_25"}
  %slice.631 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [104:108]}, metadata={op_type="Split" op_name="split_1"}
  %slice.663 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [104:108]}, metadata={op_type="Split" op_name="split"}
  %convolution.688 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.631, f32[3,3,4,4]{3,2,1,0} %slice.663), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_26"}
  %slice.632 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [108:112]}, metadata={op_type="Split" op_name="split_1"}
  %slice.664 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [108:112]}, metadata={op_type="Split" op_name="split"}
  %convolution.689 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.632, f32[3,3,4,4]{3,2,1,0} %slice.664), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_27"}
  %slice.633 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [112:116]}, metadata={op_type="Split" op_name="split_1"}
  %slice.665 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [112:116]}, metadata={op_type="Split" op_name="split"}
  %convolution.690 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.633, f32[3,3,4,4]{3,2,1,0} %slice.665), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_28"}
  %slice.634 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [116:120]}, metadata={op_type="Split" op_name="split_1"}
  %slice.666 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [116:120]}, metadata={op_type="Split" op_name="split"}
  %convolution.691 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.634, f32[3,3,4,4]{3,2,1,0} %slice.666), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_29"}
  %slice.635 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [120:124]}, metadata={op_type="Split" op_name="split_1"}
  %slice.667 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [120:124]}, metadata={op_type="Split" op_name="split"}
  %convolution.693 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.635, f32[3,3,4,4]{3,2,1,0} %slice.667), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_30"}
  %slice.636 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.604), slice={[0:1], [0:58], [0:58], [124:128]}, metadata={op_type="Split" op_name="split_1"}
  %slice.668 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.278), slice={[0:3], [0:3], [0:4], [124:128]}, metadata={op_type="Split" op_name="split"}
  %convolution.694 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.636, f32[3,3,4,4]{3,2,1,0} %slice.668), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_31"}
  %concatenate.701 = f32[1,56,56,128]{3,2,1,0} concatenate(f32[1,56,56,4]{3,2,1,0} %convolution.669, f32[1,56,56,4]{3,2,1,0} %convolution.670, f32[1,56,56,4]{3,2,1,0} %convolution.681, f32[1,56,56,4]{3,2,1,0} %convolution.692, f32[1,56,56,4]{3,2,1,0} %convolution.695, f32[1,56,56,4]{3,2,1,0} %convolution.696, f32[1,56,56,4]{3,2,1,0} %convolution.697, f32[1,56,56,4]{3,2,1,0} %convolution.698, f32[1,56,56,4]{3,2,1,0} %convolution.699, f32[1,56,56,4]{3,2,1,0} %convolution.700, f32[1,56,56,4]{3,2,1,0} %convolution.671, f32[1,56,56,4]{3,2,1,0} %convolution.672, f32[1,56,56,4]{3,2,1,0} %convolution.673, f32[1,56,56,4]{3,2,1,0} %convolution.674, f32[1,56,56,4]{3,2,1,0} %convolution.675, f32[1,56,56,4]{3,2,1,0} %convolution.676, f32[1,56,56,4]{3,2,1,0} %convolution.677, f32[1,56,56,4]{3,2,1,0} %convolution.678, f32[1,56,56,4]{3,2,1,0} %convolution.679, f32[1,56,56,4]{3,2,1,0} %convolution.680, f32[1,56,56,4]{3,2,1,0} %convolution.682, f32[1,56,56,4]{3,2,1,0} %convolution.683, f32[1,56,56,4]{3,2,1,0} %convolution.684, f32[1,56,56,4]{3,2,1,0} %convolution.685, f32[1,56,56,4]{3,2,1,0} %convolution.686, f32[1,56,56,4]{3,2,1,0} %convolution.687, f32[1,56,56,4]{3,2,1,0} %convolution.688, f32[1,56,56,4]{3,2,1,0} %convolution.689, f32[1,56,56,4]{3,2,1,0} %convolution.690, f32[1,56,56,4]{3,2,1,0} %convolution.691, f32[1,56,56,4]{3,2,1,0} %convolution.693, f32[1,56,56,4]{3,2,1,0} %convolution.694), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat"}
  %constant.581 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add"}
  %broadcast.582 = f32[128]{0} broadcast(f32[] %constant.581), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add"}
  %arg9.10 = f32[128]{0} parameter(9), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.279 = f32[128]{0} reshape(f32[128]{0} %arg9.10)
  %add.583 = f32[128]{0} add(f32[128]{0} %broadcast.582, f32[128]{0} %reshape.279), metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add"}
  %rsqrt.584 = f32[128]{0} rsqrt(f32[128]{0} %add.583), metadata={op_type="Rsqrt" op_name="stage1_unit1_bn2/Rsqrt"}
  %arg77.78 = f32[128]{0} parameter(77), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.347 = f32[128]{0} reshape(f32[128]{0} %arg77.78)
  %multiply.585 = f32[128]{0} multiply(f32[128]{0} %rsqrt.584, f32[128]{0} %reshape.347), metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul"}
  %broadcast.702 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.585), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul_1"}
  %multiply.703 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %concatenate.701, f32[1,56,56,128]{3,2,1,0} %broadcast.702), metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul_1"}
  %arg184.185 = f32[128]{0} parameter(184), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.454 = f32[128]{0} reshape(f32[128]{0} %arg184.185)
  %arg131.132 = f32[128]{0} parameter(131), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.401 = f32[128]{0} reshape(f32[128]{0} %arg131.132)
  %multiply.586 = f32[128]{0} multiply(f32[128]{0} %multiply.585, f32[128]{0} %reshape.401), metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul_2"}
  %subtract.587 = f32[128]{0} subtract(f32[128]{0} %reshape.454, f32[128]{0} %multiply.586), metadata={op_type="Sub" op_name="stage1_unit1_bn2/sub"}
  %broadcast.704 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.587), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add_1"}
  %add.705 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.703, f32[1,56,56,128]{3,2,1,0} %broadcast.704), metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add_1"}
  %maximum.708 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.707, f32[1,56,56,128]{3,2,1,0} %add.705), metadata={op_type="Relu" op_name="stage1_unit1_relu2"}
  %arg234.235 = f32[1,1,128,256]{3,2,1,0} parameter(234), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.504 = f32[1,1,128,256]{3,2,1,0} reshape(f32[1,1,128,256]{3,2,1,0} %arg234.235)
  %convolution.709 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,128]{3,2,1,0} %maximum.708, f32[1,1,128,256]{3,2,1,0} %reshape.504), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv3"}
  %multiply.711 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.710, f32[1,56,56,256]{3,2,1,0} %convolution.709), metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul_1"}
  %arg185.186 = f32[256]{0} parameter(185), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.455 = f32[256]{0} reshape(f32[256]{0} %arg185.186)
  %arg132.133 = f32[256]{0} parameter(132), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.402 = f32[256]{0} reshape(f32[256]{0} %arg132.133)
  %multiply.593 = f32[256]{0} multiply(f32[256]{0} %multiply.592, f32[256]{0} %reshape.402), metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul_2"}
  %subtract.594 = f32[256]{0} subtract(f32[256]{0} %reshape.455, f32[256]{0} %multiply.593), metadata={op_type="Sub" op_name="stage1_unit1_bn3/sub"}
  %broadcast.712 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.594), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add_1"}
  %add.713 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.711, f32[1,56,56,256]{3,2,1,0} %broadcast.712), metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add_1"}
  %arg233.234 = f32[1,1,64,256]{3,2,1,0} parameter(233), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.503 = f32[1,1,64,256]{3,2,1,0} reshape(f32[1,1,64,256]{3,2,1,0} %arg233.234)
  %convolution.721 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,64]{3,2,1,0} %reduce-window.573, f32[1,1,64,256]{3,2,1,0} %reshape.503), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_sc"}
  %constant.714 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add"}
  %broadcast.715 = f32[256]{0} broadcast(f32[] %constant.714), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add"}
  %arg6.7 = f32[256]{0} parameter(6), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.276 = f32[256]{0} reshape(f32[256]{0} %arg6.7)
  %add.716 = f32[256]{0} add(f32[256]{0} %broadcast.715, f32[256]{0} %reshape.276), metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add"}
  %rsqrt.717 = f32[256]{0} rsqrt(f32[256]{0} %add.716), metadata={op_type="Rsqrt" op_name="stage1_unit1_sc_bn/Rsqrt"}
  %arg76.77 = f32[256]{0} parameter(76), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.346 = f32[256]{0} reshape(f32[256]{0} %arg76.77)
  %multiply.718 = f32[256]{0} multiply(f32[256]{0} %rsqrt.717, f32[256]{0} %reshape.346), metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul"}
  %broadcast.722 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.718), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul_1"}
  %multiply.723 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %convolution.721, f32[1,56,56,256]{3,2,1,0} %broadcast.722), metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul_1"}
  %arg183.184 = f32[256]{0} parameter(183), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.453 = f32[256]{0} reshape(f32[256]{0} %arg183.184)
  %arg130.131 = f32[256]{0} parameter(130), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.400 = f32[256]{0} reshape(f32[256]{0} %arg130.131)
  %multiply.719 = f32[256]{0} multiply(f32[256]{0} %multiply.718, f32[256]{0} %reshape.400), metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul_2"}
  %subtract.720 = f32[256]{0} subtract(f32[256]{0} %reshape.453, f32[256]{0} %multiply.719), metadata={op_type="Sub" op_name="stage1_unit1_sc_bn/sub"}
  %broadcast.724 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.720), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add_1"}
  %add.725 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.723, f32[1,56,56,256]{3,2,1,0} %broadcast.724), metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add_1"}
  %add.726 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %add.713, f32[1,56,56,256]{3,2,1,0} %add.725), metadata={op_type="AddV2" op_name="add"}
  %maximum.729 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.728, f32[1,56,56,256]{3,2,1,0} %add.726), metadata={op_type="Relu" op_name="stage1_unit1_relu"}
  %constant.744 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add"}
  %broadcast.745 = f32[256]{0} broadcast(f32[] %constant.744), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add"}
  %arg15.16 = f32[256]{0} parameter(15), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.285 = f32[256]{0} reshape(f32[256]{0} %arg15.16)
  %add.746 = f32[256]{0} add(f32[256]{0} %broadcast.745, f32[256]{0} %reshape.285), metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add"}
  %rsqrt.747 = f32[256]{0} rsqrt(f32[256]{0} %add.746), metadata={op_type="Rsqrt" op_name="stage1_unit2_bn3/Rsqrt"}
  %arg82.83 = f32[256]{0} parameter(82), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.352 = f32[256]{0} reshape(f32[256]{0} %arg82.83)
  %multiply.748 = f32[256]{0} multiply(f32[256]{0} %rsqrt.747, f32[256]{0} %reshape.352), metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul"}
  %broadcast.866 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.748), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul_1"}
  %constant.862 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit2_relu2"}
  %broadcast.863 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.862), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit2_relu2"}
  %constant.756 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit2_relu1"}
  %broadcast.757 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.756), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit2_relu1"}
  %constant.730 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add"}
  %broadcast.731 = f32[128]{0} broadcast(f32[] %constant.730), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add"}
  %arg11.12 = f32[128]{0} parameter(11), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.281 = f32[128]{0} reshape(f32[128]{0} %arg11.12)
  %add.732 = f32[128]{0} add(f32[128]{0} %broadcast.731, f32[128]{0} %reshape.281), metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add"}
  %rsqrt.733 = f32[128]{0} rsqrt(f32[128]{0} %add.732), metadata={op_type="Rsqrt" op_name="stage1_unit2_bn1/Rsqrt"}
  %arg79.80 = f32[128]{0} parameter(79), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.349 = f32[128]{0} reshape(f32[128]{0} %arg79.80)
  %multiply.734 = f32[128]{0} multiply(f32[128]{0} %rsqrt.733, f32[128]{0} %reshape.349), metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul"}
  %broadcast.752 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.734), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul_1"}
  %arg235.236 = f32[1,1,256,128]{3,2,1,0} parameter(235), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.505 = f32[1,1,256,128]{3,2,1,0} reshape(f32[1,1,256,128]{3,2,1,0} %arg235.236)
  %convolution.751 = f32[1,56,56,128]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.729, f32[1,1,256,128]{3,2,1,0} %reshape.505), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv1"}
  %multiply.753 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %broadcast.752, f32[1,56,56,128]{3,2,1,0} %convolution.751), metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul_1"}
  %arg186.187 = f32[128]{0} parameter(186), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.456 = f32[128]{0} reshape(f32[128]{0} %arg186.187)
  %arg133.134 = f32[128]{0} parameter(133), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.403 = f32[128]{0} reshape(f32[128]{0} %arg133.134)
  %multiply.735 = f32[128]{0} multiply(f32[128]{0} %multiply.734, f32[128]{0} %reshape.403), metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul_2"}
  %subtract.736 = f32[128]{0} subtract(f32[128]{0} %reshape.456, f32[128]{0} %multiply.735), metadata={op_type="Sub" op_name="stage1_unit2_bn1/sub"}
  %broadcast.754 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.736), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add_1"}
  %add.755 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.753, f32[1,56,56,128]{3,2,1,0} %broadcast.754), metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add_1"}
  %maximum.758 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.757, f32[1,56,56,128]{3,2,1,0} %add.755), metadata={op_type="Relu" op_name="stage1_unit2_relu1"}
  %constant.759 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_2"}
  %pad.760 = f32[1,58,58,128]{3,2,1,0} pad(f32[1,56,56,128]{3,2,1,0} %maximum.758, f32[] %constant.759), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_2"}
  %slice.761 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [0:4]}, metadata={op_type="Split" op_name="split_3"}
  %arg12.13 = f32[3,3,4,128]{3,2,1,0} parameter(12), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.282 = f32[3,3,4,128]{3,2,1,0} reshape(f32[3,3,4,128]{3,2,1,0} %arg12.13)
  %slice.793 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [0:4]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.825 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.761, f32[3,3,4,4]{3,2,1,0} %slice.793), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2"}
  %slice.762 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [4:8]}, metadata={op_type="Split" op_name="split_3"}
  %slice.794 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [4:8]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.826 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.762, f32[3,3,4,4]{3,2,1,0} %slice.794), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_1"}
  %slice.763 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [8:12]}, metadata={op_type="Split" op_name="split_3"}
  %slice.795 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [8:12]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.837 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.763, f32[3,3,4,4]{3,2,1,0} %slice.795), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_2"}
  %slice.764 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [12:16]}, metadata={op_type="Split" op_name="split_3"}
  %slice.796 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [12:16]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.848 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.764, f32[3,3,4,4]{3,2,1,0} %slice.796), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_3"}
  %slice.765 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [16:20]}, metadata={op_type="Split" op_name="split_3"}
  %slice.797 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [16:20]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.851 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.765, f32[3,3,4,4]{3,2,1,0} %slice.797), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_4"}
  %slice.766 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [20:24]}, metadata={op_type="Split" op_name="split_3"}
  %slice.798 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [20:24]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.852 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.766, f32[3,3,4,4]{3,2,1,0} %slice.798), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_5"}
  %slice.767 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [24:28]}, metadata={op_type="Split" op_name="split_3"}
  %slice.799 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [24:28]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.853 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.767, f32[3,3,4,4]{3,2,1,0} %slice.799), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_6"}
  %slice.768 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [28:32]}, metadata={op_type="Split" op_name="split_3"}
  %slice.800 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [28:32]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.854 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.768, f32[3,3,4,4]{3,2,1,0} %slice.800), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_7"}
  %slice.769 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [32:36]}, metadata={op_type="Split" op_name="split_3"}
  %slice.801 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [32:36]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.855 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.769, f32[3,3,4,4]{3,2,1,0} %slice.801), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_8"}
  %slice.770 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [36:40]}, metadata={op_type="Split" op_name="split_3"}
  %slice.802 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [36:40]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.856 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.770, f32[3,3,4,4]{3,2,1,0} %slice.802), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_9"}
  %slice.771 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [40:44]}, metadata={op_type="Split" op_name="split_3"}
  %slice.803 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [40:44]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.827 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.771, f32[3,3,4,4]{3,2,1,0} %slice.803), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_10"}
  %slice.772 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [44:48]}, metadata={op_type="Split" op_name="split_3"}
  %slice.804 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [44:48]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.828 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.772, f32[3,3,4,4]{3,2,1,0} %slice.804), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_11"}
  %slice.773 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [48:52]}, metadata={op_type="Split" op_name="split_3"}
  %slice.805 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [48:52]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.829 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.773, f32[3,3,4,4]{3,2,1,0} %slice.805), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_12"}
  %slice.774 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [52:56]}, metadata={op_type="Split" op_name="split_3"}
  %slice.806 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [52:56]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.830 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.774, f32[3,3,4,4]{3,2,1,0} %slice.806), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_13"}
  %slice.775 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [56:60]}, metadata={op_type="Split" op_name="split_3"}
  %slice.807 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [56:60]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.831 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.775, f32[3,3,4,4]{3,2,1,0} %slice.807), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_14"}
  %slice.776 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [60:64]}, metadata={op_type="Split" op_name="split_3"}
  %slice.808 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [60:64]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.832 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.776, f32[3,3,4,4]{3,2,1,0} %slice.808), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_15"}
  %slice.777 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [64:68]}, metadata={op_type="Split" op_name="split_3"}
  %slice.809 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [64:68]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.833 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.777, f32[3,3,4,4]{3,2,1,0} %slice.809), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_16"}
  %slice.778 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [68:72]}, metadata={op_type="Split" op_name="split_3"}
  %slice.810 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [68:72]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.834 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.778, f32[3,3,4,4]{3,2,1,0} %slice.810), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_17"}
  %slice.779 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [72:76]}, metadata={op_type="Split" op_name="split_3"}
  %slice.811 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [72:76]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.835 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.779, f32[3,3,4,4]{3,2,1,0} %slice.811), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_18"}
  %slice.780 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [76:80]}, metadata={op_type="Split" op_name="split_3"}
  %slice.812 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [76:80]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.836 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.780, f32[3,3,4,4]{3,2,1,0} %slice.812), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_19"}
  %slice.781 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [80:84]}, metadata={op_type="Split" op_name="split_3"}
  %slice.813 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [80:84]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.838 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.781, f32[3,3,4,4]{3,2,1,0} %slice.813), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_20"}
  %slice.782 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [84:88]}, metadata={op_type="Split" op_name="split_3"}
  %slice.814 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [84:88]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.839 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.782, f32[3,3,4,4]{3,2,1,0} %slice.814), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_21"}
  %slice.783 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [88:92]}, metadata={op_type="Split" op_name="split_3"}
  %slice.815 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [88:92]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.840 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.783, f32[3,3,4,4]{3,2,1,0} %slice.815), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_22"}
  %slice.784 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [92:96]}, metadata={op_type="Split" op_name="split_3"}
  %slice.816 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [92:96]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.841 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.784, f32[3,3,4,4]{3,2,1,0} %slice.816), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_23"}
  %slice.785 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [96:100]}, metadata={op_type="Split" op_name="split_3"}
  %slice.817 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [96:100]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.842 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.785, f32[3,3,4,4]{3,2,1,0} %slice.817), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_24"}
  %slice.786 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [100:104]}, metadata={op_type="Split" op_name="split_3"}
  %slice.818 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [100:104]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.843 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.786, f32[3,3,4,4]{3,2,1,0} %slice.818), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_25"}
  %slice.787 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [104:108]}, metadata={op_type="Split" op_name="split_3"}
  %slice.819 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [104:108]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.844 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.787, f32[3,3,4,4]{3,2,1,0} %slice.819), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_26"}
  %slice.788 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [108:112]}, metadata={op_type="Split" op_name="split_3"}
  %slice.820 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [108:112]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.845 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.788, f32[3,3,4,4]{3,2,1,0} %slice.820), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_27"}
  %slice.789 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [112:116]}, metadata={op_type="Split" op_name="split_3"}
  %slice.821 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [112:116]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.846 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.789, f32[3,3,4,4]{3,2,1,0} %slice.821), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_28"}
  %slice.790 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [116:120]}, metadata={op_type="Split" op_name="split_3"}
  %slice.822 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [116:120]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.847 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.790, f32[3,3,4,4]{3,2,1,0} %slice.822), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_29"}
  %slice.791 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [120:124]}, metadata={op_type="Split" op_name="split_3"}
  %slice.823 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [120:124]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.849 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.791, f32[3,3,4,4]{3,2,1,0} %slice.823), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_30"}
  %slice.792 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.760), slice={[0:1], [0:58], [0:58], [124:128]}, metadata={op_type="Split" op_name="split_3"}
  %slice.824 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.282), slice={[0:3], [0:3], [0:4], [124:128]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.850 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.792, f32[3,3,4,4]{3,2,1,0} %slice.824), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_31"}
  %concatenate.857 = f32[1,56,56,128]{3,2,1,0} concatenate(f32[1,56,56,4]{3,2,1,0} %convolution.825, f32[1,56,56,4]{3,2,1,0} %convolution.826, f32[1,56,56,4]{3,2,1,0} %convolution.837, f32[1,56,56,4]{3,2,1,0} %convolution.848, f32[1,56,56,4]{3,2,1,0} %convolution.851, f32[1,56,56,4]{3,2,1,0} %convolution.852, f32[1,56,56,4]{3,2,1,0} %convolution.853, f32[1,56,56,4]{3,2,1,0} %convolution.854, f32[1,56,56,4]{3,2,1,0} %convolution.855, f32[1,56,56,4]{3,2,1,0} %convolution.856, f32[1,56,56,4]{3,2,1,0} %convolution.827, f32[1,56,56,4]{3,2,1,0} %convolution.828, f32[1,56,56,4]{3,2,1,0} %convolution.829, f32[1,56,56,4]{3,2,1,0} %convolution.830, f32[1,56,56,4]{3,2,1,0} %convolution.831, f32[1,56,56,4]{3,2,1,0} %convolution.832, f32[1,56,56,4]{3,2,1,0} %convolution.833, f32[1,56,56,4]{3,2,1,0} %convolution.834, f32[1,56,56,4]{3,2,1,0} %convolution.835, f32[1,56,56,4]{3,2,1,0} %convolution.836, f32[1,56,56,4]{3,2,1,0} %convolution.838, f32[1,56,56,4]{3,2,1,0} %convolution.839, f32[1,56,56,4]{3,2,1,0} %convolution.840, f32[1,56,56,4]{3,2,1,0} %convolution.841, f32[1,56,56,4]{3,2,1,0} %convolution.842, f32[1,56,56,4]{3,2,1,0} %convolution.843, f32[1,56,56,4]{3,2,1,0} %convolution.844, f32[1,56,56,4]{3,2,1,0} %convolution.845, f32[1,56,56,4]{3,2,1,0} %convolution.846, f32[1,56,56,4]{3,2,1,0} %convolution.847, f32[1,56,56,4]{3,2,1,0} %convolution.849, f32[1,56,56,4]{3,2,1,0} %convolution.850), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_1"}
  %constant.737 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add"}
  %broadcast.738 = f32[128]{0} broadcast(f32[] %constant.737), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add"}
  %arg13.14 = f32[128]{0} parameter(13), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.283 = f32[128]{0} reshape(f32[128]{0} %arg13.14)
  %add.739 = f32[128]{0} add(f32[128]{0} %broadcast.738, f32[128]{0} %reshape.283), metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add"}
  %rsqrt.740 = f32[128]{0} rsqrt(f32[128]{0} %add.739), metadata={op_type="Rsqrt" op_name="stage1_unit2_bn2/Rsqrt"}
  %arg80.81 = f32[128]{0} parameter(80), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.350 = f32[128]{0} reshape(f32[128]{0} %arg80.81)
  %multiply.741 = f32[128]{0} multiply(f32[128]{0} %rsqrt.740, f32[128]{0} %reshape.350), metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul"}
  %broadcast.858 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.741), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul_1"}
  %multiply.859 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %concatenate.857, f32[1,56,56,128]{3,2,1,0} %broadcast.858), metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul_1"}
  %arg187.188 = f32[128]{0} parameter(187), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.457 = f32[128]{0} reshape(f32[128]{0} %arg187.188)
  %arg134.135 = f32[128]{0} parameter(134), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.404 = f32[128]{0} reshape(f32[128]{0} %arg134.135)
  %multiply.742 = f32[128]{0} multiply(f32[128]{0} %multiply.741, f32[128]{0} %reshape.404), metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul_2"}
  %subtract.743 = f32[128]{0} subtract(f32[128]{0} %reshape.457, f32[128]{0} %multiply.742), metadata={op_type="Sub" op_name="stage1_unit2_bn2/sub"}
  %broadcast.860 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.743), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add_1"}
  %add.861 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.859, f32[1,56,56,128]{3,2,1,0} %broadcast.860), metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add_1"}
  %maximum.864 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.863, f32[1,56,56,128]{3,2,1,0} %add.861), metadata={op_type="Relu" op_name="stage1_unit2_relu2"}
  %arg236.237 = f32[1,1,128,256]{3,2,1,0} parameter(236), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.506 = f32[1,1,128,256]{3,2,1,0} reshape(f32[1,1,128,256]{3,2,1,0} %arg236.237)
  %convolution.865 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,128]{3,2,1,0} %maximum.864, f32[1,1,128,256]{3,2,1,0} %reshape.506), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv3"}
  %multiply.867 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.866, f32[1,56,56,256]{3,2,1,0} %convolution.865), metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul_1"}
  %arg189.190 = f32[256]{0} parameter(189), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.459 = f32[256]{0} reshape(f32[256]{0} %arg189.190)
  %arg136.137 = f32[256]{0} parameter(136), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.406 = f32[256]{0} reshape(f32[256]{0} %arg136.137)
  %multiply.749 = f32[256]{0} multiply(f32[256]{0} %multiply.748, f32[256]{0} %reshape.406), metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul_2"}
  %subtract.750 = f32[256]{0} subtract(f32[256]{0} %reshape.459, f32[256]{0} %multiply.749), metadata={op_type="Sub" op_name="stage1_unit2_bn3/sub"}
  %broadcast.868 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.750), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add_1"}
  %add.869 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.867, f32[1,56,56,256]{3,2,1,0} %broadcast.868), metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add_1"}
  %add.870 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %maximum.729, f32[1,56,56,256]{3,2,1,0} %add.869), metadata={op_type="AddV2" op_name="add_1"}
  %maximum.873 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.872, f32[1,56,56,256]{3,2,1,0} %add.870), metadata={op_type="Relu" op_name="stage1_unit2_relu"}
  %constant.888 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add"}
  %broadcast.889 = f32[256]{0} broadcast(f32[] %constant.888), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add"}
  %arg20.21 = f32[256]{0} parameter(20), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.290 = f32[256]{0} reshape(f32[256]{0} %arg20.21)
  %add.890 = f32[256]{0} add(f32[256]{0} %broadcast.889, f32[256]{0} %reshape.290), metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add"}
  %rsqrt.891 = f32[256]{0} rsqrt(f32[256]{0} %add.890), metadata={op_type="Rsqrt" op_name="stage1_unit3_bn3/Rsqrt"}
  %arg86.87 = f32[256]{0} parameter(86), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.356 = f32[256]{0} reshape(f32[256]{0} %arg86.87)
  %multiply.892 = f32[256]{0} multiply(f32[256]{0} %rsqrt.891, f32[256]{0} %reshape.356), metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul"}
  %broadcast.1010 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.892), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul_1"}
  %constant.1006 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit3_relu2"}
  %broadcast.1007 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.1006), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit3_relu2"}
  %constant.900 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit3_relu1"}
  %broadcast.901 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.900), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit3_relu1"}
  %constant.874 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add"}
  %broadcast.875 = f32[128]{0} broadcast(f32[] %constant.874), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add"}
  %arg16.17 = f32[128]{0} parameter(16), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.286 = f32[128]{0} reshape(f32[128]{0} %arg16.17)
  %add.876 = f32[128]{0} add(f32[128]{0} %broadcast.875, f32[128]{0} %reshape.286), metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add"}
  %rsqrt.877 = f32[128]{0} rsqrt(f32[128]{0} %add.876), metadata={op_type="Rsqrt" op_name="stage1_unit3_bn1/Rsqrt"}
  %arg83.84 = f32[128]{0} parameter(83), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.353 = f32[128]{0} reshape(f32[128]{0} %arg83.84)
  %multiply.878 = f32[128]{0} multiply(f32[128]{0} %rsqrt.877, f32[128]{0} %reshape.353), metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul"}
  %broadcast.896 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.878), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul_1"}
  %arg237.238 = f32[1,1,256,128]{3,2,1,0} parameter(237), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.507 = f32[1,1,256,128]{3,2,1,0} reshape(f32[1,1,256,128]{3,2,1,0} %arg237.238)
  %convolution.895 = f32[1,56,56,128]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.873, f32[1,1,256,128]{3,2,1,0} %reshape.507), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv1"}
  %multiply.897 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %broadcast.896, f32[1,56,56,128]{3,2,1,0} %convolution.895), metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul_1"}
  %arg190.191 = f32[128]{0} parameter(190), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.460 = f32[128]{0} reshape(f32[128]{0} %arg190.191)
  %arg137.138 = f32[128]{0} parameter(137), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.407 = f32[128]{0} reshape(f32[128]{0} %arg137.138)
  %multiply.879 = f32[128]{0} multiply(f32[128]{0} %multiply.878, f32[128]{0} %reshape.407), metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul_2"}
  %subtract.880 = f32[128]{0} subtract(f32[128]{0} %reshape.460, f32[128]{0} %multiply.879), metadata={op_type="Sub" op_name="stage1_unit3_bn1/sub"}
  %broadcast.898 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.880), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add_1"}
  %add.899 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.897, f32[1,56,56,128]{3,2,1,0} %broadcast.898), metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add_1"}
  %maximum.902 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.901, f32[1,56,56,128]{3,2,1,0} %add.899), metadata={op_type="Relu" op_name="stage1_unit3_relu1"}
  %constant.903 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_3"}
  %pad.904 = f32[1,58,58,128]{3,2,1,0} pad(f32[1,56,56,128]{3,2,1,0} %maximum.902, f32[] %constant.903), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_3"}
  %slice.905 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [0:4]}, metadata={op_type="Split" op_name="split_5"}
  %arg17.18 = f32[3,3,4,128]{3,2,1,0} parameter(17), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.287 = f32[3,3,4,128]{3,2,1,0} reshape(f32[3,3,4,128]{3,2,1,0} %arg17.18)
  %slice.937 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [0:4]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.969 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.905, f32[3,3,4,4]{3,2,1,0} %slice.937), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2"}
  %slice.906 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [4:8]}, metadata={op_type="Split" op_name="split_5"}
  %slice.938 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [4:8]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.970 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.906, f32[3,3,4,4]{3,2,1,0} %slice.938), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_1"}
  %slice.907 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [8:12]}, metadata={op_type="Split" op_name="split_5"}
  %slice.939 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [8:12]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.981 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.907, f32[3,3,4,4]{3,2,1,0} %slice.939), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_2"}
  %slice.908 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [12:16]}, metadata={op_type="Split" op_name="split_5"}
  %slice.940 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [12:16]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.992 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.908, f32[3,3,4,4]{3,2,1,0} %slice.940), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_3"}
  %slice.909 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [16:20]}, metadata={op_type="Split" op_name="split_5"}
  %slice.941 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [16:20]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.995 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.909, f32[3,3,4,4]{3,2,1,0} %slice.941), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_4"}
  %slice.910 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [20:24]}, metadata={op_type="Split" op_name="split_5"}
  %slice.942 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [20:24]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.996 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.910, f32[3,3,4,4]{3,2,1,0} %slice.942), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_5"}
  %slice.911 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [24:28]}, metadata={op_type="Split" op_name="split_5"}
  %slice.943 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [24:28]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.997 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.911, f32[3,3,4,4]{3,2,1,0} %slice.943), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_6"}
  %slice.912 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [28:32]}, metadata={op_type="Split" op_name="split_5"}
  %slice.944 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [28:32]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.998 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.912, f32[3,3,4,4]{3,2,1,0} %slice.944), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_7"}
  %slice.913 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [32:36]}, metadata={op_type="Split" op_name="split_5"}
  %slice.945 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [32:36]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.999 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.913, f32[3,3,4,4]{3,2,1,0} %slice.945), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_8"}
  %slice.914 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [36:40]}, metadata={op_type="Split" op_name="split_5"}
  %slice.946 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [36:40]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1000 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.914, f32[3,3,4,4]{3,2,1,0} %slice.946), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_9"}
  %slice.915 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [40:44]}, metadata={op_type="Split" op_name="split_5"}
  %slice.947 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [40:44]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.971 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.915, f32[3,3,4,4]{3,2,1,0} %slice.947), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_10"}
  %slice.916 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [44:48]}, metadata={op_type="Split" op_name="split_5"}
  %slice.948 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [44:48]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.972 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.916, f32[3,3,4,4]{3,2,1,0} %slice.948), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_11"}
  %slice.917 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [48:52]}, metadata={op_type="Split" op_name="split_5"}
  %slice.949 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [48:52]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.973 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.917, f32[3,3,4,4]{3,2,1,0} %slice.949), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_12"}
  %slice.918 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [52:56]}, metadata={op_type="Split" op_name="split_5"}
  %slice.950 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [52:56]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.974 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.918, f32[3,3,4,4]{3,2,1,0} %slice.950), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_13"}
  %slice.919 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [56:60]}, metadata={op_type="Split" op_name="split_5"}
  %slice.951 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [56:60]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.975 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.919, f32[3,3,4,4]{3,2,1,0} %slice.951), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_14"}
  %slice.920 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [60:64]}, metadata={op_type="Split" op_name="split_5"}
  %slice.952 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [60:64]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.976 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.920, f32[3,3,4,4]{3,2,1,0} %slice.952), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_15"}
  %slice.921 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [64:68]}, metadata={op_type="Split" op_name="split_5"}
  %slice.953 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [64:68]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.977 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.921, f32[3,3,4,4]{3,2,1,0} %slice.953), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_16"}
  %slice.922 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [68:72]}, metadata={op_type="Split" op_name="split_5"}
  %slice.954 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [68:72]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.978 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.922, f32[3,3,4,4]{3,2,1,0} %slice.954), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_17"}
  %slice.923 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [72:76]}, metadata={op_type="Split" op_name="split_5"}
  %slice.955 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [72:76]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.979 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.923, f32[3,3,4,4]{3,2,1,0} %slice.955), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_18"}
  %slice.924 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [76:80]}, metadata={op_type="Split" op_name="split_5"}
  %slice.956 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [76:80]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.980 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.924, f32[3,3,4,4]{3,2,1,0} %slice.956), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_19"}
  %slice.925 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [80:84]}, metadata={op_type="Split" op_name="split_5"}
  %slice.957 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [80:84]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.982 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.925, f32[3,3,4,4]{3,2,1,0} %slice.957), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_20"}
  %slice.926 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [84:88]}, metadata={op_type="Split" op_name="split_5"}
  %slice.958 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [84:88]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.983 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.926, f32[3,3,4,4]{3,2,1,0} %slice.958), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_21"}
  %slice.927 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [88:92]}, metadata={op_type="Split" op_name="split_5"}
  %slice.959 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [88:92]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.984 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.927, f32[3,3,4,4]{3,2,1,0} %slice.959), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_22"}
  %slice.928 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [92:96]}, metadata={op_type="Split" op_name="split_5"}
  %slice.960 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [92:96]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.985 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.928, f32[3,3,4,4]{3,2,1,0} %slice.960), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_23"}
  %slice.929 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [96:100]}, metadata={op_type="Split" op_name="split_5"}
  %slice.961 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [96:100]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.986 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.929, f32[3,3,4,4]{3,2,1,0} %slice.961), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_24"}
  %slice.930 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [100:104]}, metadata={op_type="Split" op_name="split_5"}
  %slice.962 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [100:104]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.987 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.930, f32[3,3,4,4]{3,2,1,0} %slice.962), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_25"}
  %slice.931 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [104:108]}, metadata={op_type="Split" op_name="split_5"}
  %slice.963 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [104:108]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.988 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.931, f32[3,3,4,4]{3,2,1,0} %slice.963), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_26"}
  %slice.932 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [108:112]}, metadata={op_type="Split" op_name="split_5"}
  %slice.964 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [108:112]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.989 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.932, f32[3,3,4,4]{3,2,1,0} %slice.964), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_27"}
  %slice.933 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [112:116]}, metadata={op_type="Split" op_name="split_5"}
  %slice.965 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [112:116]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.990 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.933, f32[3,3,4,4]{3,2,1,0} %slice.965), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_28"}
  %slice.934 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [116:120]}, metadata={op_type="Split" op_name="split_5"}
  %slice.966 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [116:120]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.991 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.934, f32[3,3,4,4]{3,2,1,0} %slice.966), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_29"}
  %slice.935 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [120:124]}, metadata={op_type="Split" op_name="split_5"}
  %slice.967 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [120:124]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.993 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.935, f32[3,3,4,4]{3,2,1,0} %slice.967), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_30"}
  %slice.936 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.904), slice={[0:1], [0:58], [0:58], [124:128]}, metadata={op_type="Split" op_name="split_5"}
  %slice.968 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.287), slice={[0:3], [0:3], [0:4], [124:128]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.994 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.936, f32[3,3,4,4]{3,2,1,0} %slice.968), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_31"}
  %concatenate.1001 = f32[1,56,56,128]{3,2,1,0} concatenate(f32[1,56,56,4]{3,2,1,0} %convolution.969, f32[1,56,56,4]{3,2,1,0} %convolution.970, f32[1,56,56,4]{3,2,1,0} %convolution.981, f32[1,56,56,4]{3,2,1,0} %convolution.992, f32[1,56,56,4]{3,2,1,0} %convolution.995, f32[1,56,56,4]{3,2,1,0} %convolution.996, f32[1,56,56,4]{3,2,1,0} %convolution.997, f32[1,56,56,4]{3,2,1,0} %convolution.998, f32[1,56,56,4]{3,2,1,0} %convolution.999, f32[1,56,56,4]{3,2,1,0} %convolution.1000, f32[1,56,56,4]{3,2,1,0} %convolution.971, f32[1,56,56,4]{3,2,1,0} %convolution.972, f32[1,56,56,4]{3,2,1,0} %convolution.973, f32[1,56,56,4]{3,2,1,0} %convolution.974, f32[1,56,56,4]{3,2,1,0} %convolution.975, f32[1,56,56,4]{3,2,1,0} %convolution.976, f32[1,56,56,4]{3,2,1,0} %convolution.977, f32[1,56,56,4]{3,2,1,0} %convolution.978, f32[1,56,56,4]{3,2,1,0} %convolution.979, f32[1,56,56,4]{3,2,1,0} %convolution.980, f32[1,56,56,4]{3,2,1,0} %convolution.982, f32[1,56,56,4]{3,2,1,0} %convolution.983, f32[1,56,56,4]{3,2,1,0} %convolution.984, f32[1,56,56,4]{3,2,1,0} %convolution.985, f32[1,56,56,4]{3,2,1,0} %convolution.986, f32[1,56,56,4]{3,2,1,0} %convolution.987, f32[1,56,56,4]{3,2,1,0} %convolution.988, f32[1,56,56,4]{3,2,1,0} %convolution.989, f32[1,56,56,4]{3,2,1,0} %convolution.990, f32[1,56,56,4]{3,2,1,0} %convolution.991, f32[1,56,56,4]{3,2,1,0} %convolution.993, f32[1,56,56,4]{3,2,1,0} %convolution.994), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_2"}
  %constant.881 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add"}
  %broadcast.882 = f32[128]{0} broadcast(f32[] %constant.881), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add"}
  %arg18.19 = f32[128]{0} parameter(18), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.288 = f32[128]{0} reshape(f32[128]{0} %arg18.19)
  %add.883 = f32[128]{0} add(f32[128]{0} %broadcast.882, f32[128]{0} %reshape.288), metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add"}
  %rsqrt.884 = f32[128]{0} rsqrt(f32[128]{0} %add.883), metadata={op_type="Rsqrt" op_name="stage1_unit3_bn2/Rsqrt"}
  %arg84.85 = f32[128]{0} parameter(84), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.354 = f32[128]{0} reshape(f32[128]{0} %arg84.85)
  %multiply.885 = f32[128]{0} multiply(f32[128]{0} %rsqrt.884, f32[128]{0} %reshape.354), metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul"}
  %broadcast.1002 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.885), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul_1"}
  %multiply.1003 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %concatenate.1001, f32[1,56,56,128]{3,2,1,0} %broadcast.1002), metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul_1"}
  %arg191.192 = f32[128]{0} parameter(191), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.461 = f32[128]{0} reshape(f32[128]{0} %arg191.192)
  %arg138.139 = f32[128]{0} parameter(138), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.408 = f32[128]{0} reshape(f32[128]{0} %arg138.139)
  %multiply.886 = f32[128]{0} multiply(f32[128]{0} %multiply.885, f32[128]{0} %reshape.408), metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul_2"}
  %subtract.887 = f32[128]{0} subtract(f32[128]{0} %reshape.461, f32[128]{0} %multiply.886), metadata={op_type="Sub" op_name="stage1_unit3_bn2/sub"}
  %broadcast.1004 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.887), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add_1"}
  %add.1005 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.1003, f32[1,56,56,128]{3,2,1,0} %broadcast.1004), metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add_1"}
  %maximum.1008 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.1007, f32[1,56,56,128]{3,2,1,0} %add.1005), metadata={op_type="Relu" op_name="stage1_unit3_relu2"}
  %arg238.239 = f32[1,1,128,256]{3,2,1,0} parameter(238), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.508 = f32[1,1,128,256]{3,2,1,0} reshape(f32[1,1,128,256]{3,2,1,0} %arg238.239)
  %convolution.1009 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,128]{3,2,1,0} %maximum.1008, f32[1,1,128,256]{3,2,1,0} %reshape.508), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv3"}
  %multiply.1011 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.1010, f32[1,56,56,256]{3,2,1,0} %convolution.1009), metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul_1"}
  %arg193.194 = f32[256]{0} parameter(193), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.463 = f32[256]{0} reshape(f32[256]{0} %arg193.194)
  %arg140.141 = f32[256]{0} parameter(140), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.410 = f32[256]{0} reshape(f32[256]{0} %arg140.141)
  %multiply.893 = f32[256]{0} multiply(f32[256]{0} %multiply.892, f32[256]{0} %reshape.410), metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul_2"}
  %subtract.894 = f32[256]{0} subtract(f32[256]{0} %reshape.463, f32[256]{0} %multiply.893), metadata={op_type="Sub" op_name="stage1_unit3_bn3/sub"}
  %broadcast.1012 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.894), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add_1"}
  %add.1013 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.1011, f32[1,56,56,256]{3,2,1,0} %broadcast.1012), metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add_1"}
  %add.1014 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %maximum.873, f32[1,56,56,256]{3,2,1,0} %add.1013), metadata={op_type="AddV2" op_name="add_2"}
  %maximum.1017 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.1016, f32[1,56,56,256]{3,2,1,0} %add.1014), metadata={op_type="Relu" op_name="stage1_unit3_relu"}
  %arg239.240 = f32[1,1,256,256]{3,2,1,0} parameter(239), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.509 = f32[1,1,256,256]{3,2,1,0} reshape(f32[1,1,256,256]{3,2,1,0} %arg239.240)
  %convolution.1039 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.1017, f32[1,1,256,256]{3,2,1,0} %reshape.509), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv1"}
  %multiply.1041 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.1040, f32[1,56,56,256]{3,2,1,0} %convolution.1039), metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul_1"}
  %arg194.195 = f32[256]{0} parameter(194), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.464 = f32[256]{0} reshape(f32[256]{0} %arg194.195)
  %arg141.142 = f32[256]{0} parameter(141), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.411 = f32[256]{0} reshape(f32[256]{0} %arg141.142)
  %multiply.1023 = f32[256]{0} multiply(f32[256]{0} %multiply.1022, f32[256]{0} %reshape.411), metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul_2"}
  %subtract.1024 = f32[256]{0} subtract(f32[256]{0} %reshape.464, f32[256]{0} %multiply.1023), metadata={op_type="Sub" op_name="stage2_unit1_bn1/sub"}
  %broadcast.1042 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1024), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add_1"}
  %add.1043 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.1041, f32[1,56,56,256]{3,2,1,0} %broadcast.1042), metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add_1"}
  %maximum.1046 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.1045, f32[1,56,56,256]{3,2,1,0} %add.1043), metadata={op_type="Relu" op_name="stage2_unit1_relu1"}
  %constant.1047 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_4"}
  %pad.1048 = f32[1,58,58,256]{3,2,1,0} pad(f32[1,56,56,256]{3,2,1,0} %maximum.1046, f32[] %constant.1047), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_4"}
  %slice.1049 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [0:8]}, metadata={op_type="Split" op_name="split_7"}
  %arg24.25 = f32[3,3,8,256]{3,2,1,0} parameter(24), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.294 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg24.25)
  %slice.1081 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1113 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1049, f32[3,3,8,8]{3,2,1,0} %slice.1081), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2"}
  %slice.1050 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [8:16]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1082 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1114 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1050, f32[3,3,8,8]{3,2,1,0} %slice.1082), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_1"}
  %slice.1051 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [16:24]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1083 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1125 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1051, f32[3,3,8,8]{3,2,1,0} %slice.1083), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_2"}
  %slice.1052 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [24:32]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1084 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1136 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1052, f32[3,3,8,8]{3,2,1,0} %slice.1084), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_3"}
  %slice.1053 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [32:40]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1085 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1139 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1053, f32[3,3,8,8]{3,2,1,0} %slice.1085), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_4"}
  %slice.1054 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [40:48]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1086 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1140 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1054, f32[3,3,8,8]{3,2,1,0} %slice.1086), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_5"}
  %slice.1055 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [48:56]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1087 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1141 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1055, f32[3,3,8,8]{3,2,1,0} %slice.1087), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_6"}
  %slice.1056 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [56:64]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1088 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1142 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1056, f32[3,3,8,8]{3,2,1,0} %slice.1088), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_7"}
  %slice.1057 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [64:72]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1089 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1143 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1057, f32[3,3,8,8]{3,2,1,0} %slice.1089), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_8"}
  %slice.1058 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [72:80]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1090 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1144 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1058, f32[3,3,8,8]{3,2,1,0} %slice.1090), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_9"}
  %slice.1059 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [80:88]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1091 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1115 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1059, f32[3,3,8,8]{3,2,1,0} %slice.1091), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_10"}
  %slice.1060 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [88:96]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1092 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1116 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1060, f32[3,3,8,8]{3,2,1,0} %slice.1092), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_11"}
  %slice.1061 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [96:104]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1093 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1117 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1061, f32[3,3,8,8]{3,2,1,0} %slice.1093), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_12"}
  %slice.1062 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [104:112]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1094 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1118 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1062, f32[3,3,8,8]{3,2,1,0} %slice.1094), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_13"}
  %slice.1063 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [112:120]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1095 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1119 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1063, f32[3,3,8,8]{3,2,1,0} %slice.1095), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_14"}
  %slice.1064 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [120:128]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1096 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1120 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1064, f32[3,3,8,8]{3,2,1,0} %slice.1096), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_15"}
  %slice.1065 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [128:136]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1097 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1121 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1065, f32[3,3,8,8]{3,2,1,0} %slice.1097), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_16"}
  %slice.1066 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [136:144]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1098 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1122 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1066, f32[3,3,8,8]{3,2,1,0} %slice.1098), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_17"}
  %slice.1067 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [144:152]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1099 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1123 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1067, f32[3,3,8,8]{3,2,1,0} %slice.1099), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_18"}
  %slice.1068 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [152:160]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1100 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1124 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1068, f32[3,3,8,8]{3,2,1,0} %slice.1100), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_19"}
  %slice.1069 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [160:168]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1101 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1126 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1069, f32[3,3,8,8]{3,2,1,0} %slice.1101), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_20"}
  %slice.1070 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [168:176]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1102 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1127 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1070, f32[3,3,8,8]{3,2,1,0} %slice.1102), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_21"}
  %slice.1071 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [176:184]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1103 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1128 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1071, f32[3,3,8,8]{3,2,1,0} %slice.1103), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_22"}
  %slice.1072 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [184:192]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1104 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1129 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1072, f32[3,3,8,8]{3,2,1,0} %slice.1104), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_23"}
  %slice.1073 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [192:200]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1105 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1130 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1073, f32[3,3,8,8]{3,2,1,0} %slice.1105), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_24"}
  %slice.1074 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [200:208]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1106 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1131 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1074, f32[3,3,8,8]{3,2,1,0} %slice.1106), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_25"}
  %slice.1075 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [208:216]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1107 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1132 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1075, f32[3,3,8,8]{3,2,1,0} %slice.1107), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_26"}
  %slice.1076 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [216:224]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1108 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1133 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1076, f32[3,3,8,8]{3,2,1,0} %slice.1108), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_27"}
  %slice.1077 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [224:232]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1109 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1134 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1077, f32[3,3,8,8]{3,2,1,0} %slice.1109), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_28"}
  %slice.1078 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [232:240]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1110 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1135 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1078, f32[3,3,8,8]{3,2,1,0} %slice.1110), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_29"}
  %slice.1079 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [240:248]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1111 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1137 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1079, f32[3,3,8,8]{3,2,1,0} %slice.1111), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_30"}
  %slice.1080 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1048), slice={[0:1], [0:58], [0:58], [248:256]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1112 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.294), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1138 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1080, f32[3,3,8,8]{3,2,1,0} %slice.1112), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_31"}
  %concatenate.1145 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.1113, f32[1,28,28,8]{3,2,1,0} %convolution.1114, f32[1,28,28,8]{3,2,1,0} %convolution.1125, f32[1,28,28,8]{3,2,1,0} %convolution.1136, f32[1,28,28,8]{3,2,1,0} %convolution.1139, f32[1,28,28,8]{3,2,1,0} %convolution.1140, f32[1,28,28,8]{3,2,1,0} %convolution.1141, f32[1,28,28,8]{3,2,1,0} %convolution.1142, f32[1,28,28,8]{3,2,1,0} %convolution.1143, f32[1,28,28,8]{3,2,1,0} %convolution.1144, f32[1,28,28,8]{3,2,1,0} %convolution.1115, f32[1,28,28,8]{3,2,1,0} %convolution.1116, f32[1,28,28,8]{3,2,1,0} %convolution.1117, f32[1,28,28,8]{3,2,1,0} %convolution.1118, f32[1,28,28,8]{3,2,1,0} %convolution.1119, f32[1,28,28,8]{3,2,1,0} %convolution.1120, f32[1,28,28,8]{3,2,1,0} %convolution.1121, f32[1,28,28,8]{3,2,1,0} %convolution.1122, f32[1,28,28,8]{3,2,1,0} %convolution.1123, f32[1,28,28,8]{3,2,1,0} %convolution.1124, f32[1,28,28,8]{3,2,1,0} %convolution.1126, f32[1,28,28,8]{3,2,1,0} %convolution.1127, f32[1,28,28,8]{3,2,1,0} %convolution.1128, f32[1,28,28,8]{3,2,1,0} %convolution.1129, f32[1,28,28,8]{3,2,1,0} %convolution.1130, f32[1,28,28,8]{3,2,1,0} %convolution.1131, f32[1,28,28,8]{3,2,1,0} %convolution.1132, f32[1,28,28,8]{3,2,1,0} %convolution.1133, f32[1,28,28,8]{3,2,1,0} %convolution.1134, f32[1,28,28,8]{3,2,1,0} %convolution.1135, f32[1,28,28,8]{3,2,1,0} %convolution.1137, f32[1,28,28,8]{3,2,1,0} %convolution.1138), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_3"}
  %constant.1025 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add"}
  %broadcast.1026 = f32[256]{0} broadcast(f32[] %constant.1025), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add"}
  %arg25.26 = f32[256]{0} parameter(25), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.295 = f32[256]{0} reshape(f32[256]{0} %arg25.26)
  %add.1027 = f32[256]{0} add(f32[256]{0} %broadcast.1026, f32[256]{0} %reshape.295), metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add"}
  %rsqrt.1028 = f32[256]{0} rsqrt(f32[256]{0} %add.1027), metadata={op_type="Rsqrt" op_name="stage2_unit1_bn2/Rsqrt"}
  %arg89.90 = f32[256]{0} parameter(89), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.359 = f32[256]{0} reshape(f32[256]{0} %arg89.90)
  %multiply.1029 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1028, f32[256]{0} %reshape.359), metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul"}
  %broadcast.1146 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1029), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul_1"}
  %multiply.1147 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.1145, f32[1,28,28,256]{3,2,1,0} %broadcast.1146), metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul_1"}
  %arg196.197 = f32[256]{0} parameter(196), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.466 = f32[256]{0} reshape(f32[256]{0} %arg196.197)
  %arg143.144 = f32[256]{0} parameter(143), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.413 = f32[256]{0} reshape(f32[256]{0} %arg143.144)
  %multiply.1030 = f32[256]{0} multiply(f32[256]{0} %multiply.1029, f32[256]{0} %reshape.413), metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul_2"}
  %subtract.1031 = f32[256]{0} subtract(f32[256]{0} %reshape.466, f32[256]{0} %multiply.1030), metadata={op_type="Sub" op_name="stage2_unit1_bn2/sub"}
  %broadcast.1148 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1031), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add_1"}
  %add.1149 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1147, f32[1,28,28,256]{3,2,1,0} %broadcast.1148), metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add_1"}
  %maximum.1152 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1151, f32[1,28,28,256]{3,2,1,0} %add.1149), metadata={op_type="Relu" op_name="stage2_unit1_relu2"}
  %arg241.242 = f32[1,1,256,512]{3,2,1,0} parameter(241), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.511 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg241.242)
  %convolution.1153 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.1152, f32[1,1,256,512]{3,2,1,0} %reshape.511), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv3"}
  %multiply.1155 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1154, f32[1,28,28,512]{3,2,1,0} %convolution.1153), metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul_1"}
  %arg198.199 = f32[512]{0} parameter(198), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.468 = f32[512]{0} reshape(f32[512]{0} %arg198.199)
  %arg145.146 = f32[512]{0} parameter(145), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.415 = f32[512]{0} reshape(f32[512]{0} %arg145.146)
  %multiply.1037 = f32[512]{0} multiply(f32[512]{0} %multiply.1036, f32[512]{0} %reshape.415), metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul_2"}
  %subtract.1038 = f32[512]{0} subtract(f32[512]{0} %reshape.468, f32[512]{0} %multiply.1037), metadata={op_type="Sub" op_name="stage2_unit1_bn3/sub"}
  %broadcast.1156 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1038), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add_1"}
  %add.1157 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1155, f32[1,28,28,512]{3,2,1,0} %broadcast.1156), metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add_1"}
  %arg240.241 = f32[1,1,256,512]{3,2,1,0} parameter(240), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.510 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg240.241)
  %convolution.1165 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.1017, f32[1,1,256,512]{3,2,1,0} %reshape.510), window={size=1x1 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_sc"}
  %constant.1158 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add"}
  %broadcast.1159 = f32[512]{0} broadcast(f32[] %constant.1158), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add"}
  %arg23.24 = f32[512]{0} parameter(23), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.293 = f32[512]{0} reshape(f32[512]{0} %arg23.24)
  %add.1160 = f32[512]{0} add(f32[512]{0} %broadcast.1159, f32[512]{0} %reshape.293), metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add"}
  %rsqrt.1161 = f32[512]{0} rsqrt(f32[512]{0} %add.1160), metadata={op_type="Rsqrt" op_name="stage2_unit1_sc_bn/Rsqrt"}
  %arg88.89 = f32[512]{0} parameter(88), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.358 = f32[512]{0} reshape(f32[512]{0} %arg88.89)
  %multiply.1162 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1161, f32[512]{0} %reshape.358), metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul"}
  %broadcast.1166 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1162), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul_1"}
  %multiply.1167 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %convolution.1165, f32[1,28,28,512]{3,2,1,0} %broadcast.1166), metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul_1"}
  %arg195.196 = f32[512]{0} parameter(195), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.465 = f32[512]{0} reshape(f32[512]{0} %arg195.196)
  %arg142.143 = f32[512]{0} parameter(142), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.412 = f32[512]{0} reshape(f32[512]{0} %arg142.143)
  %multiply.1163 = f32[512]{0} multiply(f32[512]{0} %multiply.1162, f32[512]{0} %reshape.412), metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul_2"}
  %subtract.1164 = f32[512]{0} subtract(f32[512]{0} %reshape.465, f32[512]{0} %multiply.1163), metadata={op_type="Sub" op_name="stage2_unit1_sc_bn/sub"}
  %broadcast.1168 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1164), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add_1"}
  %add.1169 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1167, f32[1,28,28,512]{3,2,1,0} %broadcast.1168), metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add_1"}
  %add.1170 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %add.1157, f32[1,28,28,512]{3,2,1,0} %add.1169), metadata={op_type="AddV2" op_name="add_3"}
  %maximum.1173 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1172, f32[1,28,28,512]{3,2,1,0} %add.1170), metadata={op_type="Relu" op_name="stage2_unit1_relu"}
  %constant.1188 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add"}
  %broadcast.1189 = f32[512]{0} broadcast(f32[] %constant.1188), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add"}
  %arg32.33 = f32[512]{0} parameter(32), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.302 = f32[512]{0} reshape(f32[512]{0} %arg32.33)
  %add.1190 = f32[512]{0} add(f32[512]{0} %broadcast.1189, f32[512]{0} %reshape.302), metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add"}
  %rsqrt.1191 = f32[512]{0} rsqrt(f32[512]{0} %add.1190), metadata={op_type="Rsqrt" op_name="stage2_unit2_bn3/Rsqrt"}
  %arg95.96 = f32[512]{0} parameter(95), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.365 = f32[512]{0} reshape(f32[512]{0} %arg95.96)
  %multiply.1192 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1191, f32[512]{0} %reshape.365), metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul"}
  %broadcast.1310 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1192), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul_1"}
  %constant.1306 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit2_relu2"}
  %broadcast.1307 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1306), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit2_relu2"}
  %constant.1200 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit2_relu1"}
  %broadcast.1201 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1200), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit2_relu1"}
  %constant.1174 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add"}
  %broadcast.1175 = f32[256]{0} broadcast(f32[] %constant.1174), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add"}
  %arg28.29 = f32[256]{0} parameter(28), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.298 = f32[256]{0} reshape(f32[256]{0} %arg28.29)
  %add.1176 = f32[256]{0} add(f32[256]{0} %broadcast.1175, f32[256]{0} %reshape.298), metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add"}
  %rsqrt.1177 = f32[256]{0} rsqrt(f32[256]{0} %add.1176), metadata={op_type="Rsqrt" op_name="stage2_unit2_bn1/Rsqrt"}
  %arg92.93 = f32[256]{0} parameter(92), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.362 = f32[256]{0} reshape(f32[256]{0} %arg92.93)
  %multiply.1178 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1177, f32[256]{0} %reshape.362), metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul"}
  %broadcast.1196 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1178), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul_1"}
  %arg242.243 = f32[1,1,512,256]{3,2,1,0} parameter(242), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.512 = f32[1,1,512,256]{3,2,1,0} reshape(f32[1,1,512,256]{3,2,1,0} %arg242.243)
  %convolution.1195 = f32[1,28,28,256]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1173, f32[1,1,512,256]{3,2,1,0} %reshape.512), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv1"}
  %multiply.1197 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %broadcast.1196, f32[1,28,28,256]{3,2,1,0} %convolution.1195), metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul_1"}
  %arg199.200 = f32[256]{0} parameter(199), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.469 = f32[256]{0} reshape(f32[256]{0} %arg199.200)
  %arg146.147 = f32[256]{0} parameter(146), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.416 = f32[256]{0} reshape(f32[256]{0} %arg146.147)
  %multiply.1179 = f32[256]{0} multiply(f32[256]{0} %multiply.1178, f32[256]{0} %reshape.416), metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul_2"}
  %subtract.1180 = f32[256]{0} subtract(f32[256]{0} %reshape.469, f32[256]{0} %multiply.1179), metadata={op_type="Sub" op_name="stage2_unit2_bn1/sub"}
  %broadcast.1198 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1180), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add_1"}
  %add.1199 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1197, f32[1,28,28,256]{3,2,1,0} %broadcast.1198), metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add_1"}
  %maximum.1202 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1201, f32[1,28,28,256]{3,2,1,0} %add.1199), metadata={op_type="Relu" op_name="stage2_unit2_relu1"}
  %constant.1203 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_5"}
  %pad.1204 = f32[1,30,30,256]{3,2,1,0} pad(f32[1,28,28,256]{3,2,1,0} %maximum.1202, f32[] %constant.1203), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_5"}
  %slice.1205 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [0:8]}, metadata={op_type="Split" op_name="split_9"}
  %arg30.31 = f32[3,3,8,256]{3,2,1,0} parameter(30), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.300 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg30.31)
  %slice.1237 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1269 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1205, f32[3,3,8,8]{3,2,1,0} %slice.1237), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2"}
  %slice.1206 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [8:16]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1238 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1270 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1206, f32[3,3,8,8]{3,2,1,0} %slice.1238), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_1"}
  %slice.1207 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [16:24]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1239 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1281 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1207, f32[3,3,8,8]{3,2,1,0} %slice.1239), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_2"}
  %slice.1208 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [24:32]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1240 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1292 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1208, f32[3,3,8,8]{3,2,1,0} %slice.1240), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_3"}
  %slice.1209 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [32:40]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1241 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1295 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1209, f32[3,3,8,8]{3,2,1,0} %slice.1241), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_4"}
  %slice.1210 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [40:48]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1242 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1296 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1210, f32[3,3,8,8]{3,2,1,0} %slice.1242), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_5"}
  %slice.1211 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [48:56]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1243 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1297 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1211, f32[3,3,8,8]{3,2,1,0} %slice.1243), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_6"}
  %slice.1212 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [56:64]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1244 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1298 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1212, f32[3,3,8,8]{3,2,1,0} %slice.1244), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_7"}
  %slice.1213 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [64:72]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1245 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1299 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1213, f32[3,3,8,8]{3,2,1,0} %slice.1245), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_8"}
  %slice.1214 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [72:80]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1246 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1300 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1214, f32[3,3,8,8]{3,2,1,0} %slice.1246), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_9"}
  %slice.1215 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [80:88]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1247 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1271 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1215, f32[3,3,8,8]{3,2,1,0} %slice.1247), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_10"}
  %slice.1216 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [88:96]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1248 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1272 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1216, f32[3,3,8,8]{3,2,1,0} %slice.1248), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_11"}
  %slice.1217 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [96:104]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1249 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1273 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1217, f32[3,3,8,8]{3,2,1,0} %slice.1249), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_12"}
  %slice.1218 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [104:112]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1250 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1274 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1218, f32[3,3,8,8]{3,2,1,0} %slice.1250), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_13"}
  %slice.1219 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [112:120]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1251 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1275 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1219, f32[3,3,8,8]{3,2,1,0} %slice.1251), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_14"}
  %slice.1220 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [120:128]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1252 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1276 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1220, f32[3,3,8,8]{3,2,1,0} %slice.1252), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_15"}
  %slice.1221 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [128:136]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1253 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1277 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1221, f32[3,3,8,8]{3,2,1,0} %slice.1253), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_16"}
  %slice.1222 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [136:144]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1254 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1278 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1222, f32[3,3,8,8]{3,2,1,0} %slice.1254), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_17"}
  %slice.1223 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [144:152]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1255 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1279 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1223, f32[3,3,8,8]{3,2,1,0} %slice.1255), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_18"}
  %slice.1224 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [152:160]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1256 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1280 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1224, f32[3,3,8,8]{3,2,1,0} %slice.1256), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_19"}
  %slice.1225 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [160:168]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1257 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1282 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1225, f32[3,3,8,8]{3,2,1,0} %slice.1257), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_20"}
  %slice.1226 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [168:176]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1258 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1283 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1226, f32[3,3,8,8]{3,2,1,0} %slice.1258), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_21"}
  %slice.1227 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [176:184]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1259 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1284 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1227, f32[3,3,8,8]{3,2,1,0} %slice.1259), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_22"}
  %slice.1228 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [184:192]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1260 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1285 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1228, f32[3,3,8,8]{3,2,1,0} %slice.1260), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_23"}
  %slice.1229 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [192:200]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1261 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1286 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1229, f32[3,3,8,8]{3,2,1,0} %slice.1261), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_24"}
  %slice.1230 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [200:208]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1262 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1287 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1230, f32[3,3,8,8]{3,2,1,0} %slice.1262), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_25"}
  %slice.1231 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [208:216]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1263 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1288 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1231, f32[3,3,8,8]{3,2,1,0} %slice.1263), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_26"}
  %slice.1232 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [216:224]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1264 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1289 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1232, f32[3,3,8,8]{3,2,1,0} %slice.1264), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_27"}
  %slice.1233 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [224:232]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1265 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1290 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1233, f32[3,3,8,8]{3,2,1,0} %slice.1265), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_28"}
  %slice.1234 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [232:240]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1266 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1291 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1234, f32[3,3,8,8]{3,2,1,0} %slice.1266), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_29"}
  %slice.1235 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [240:248]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1267 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1293 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1235, f32[3,3,8,8]{3,2,1,0} %slice.1267), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_30"}
  %slice.1236 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1204), slice={[0:1], [0:30], [0:30], [248:256]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1268 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1294 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1236, f32[3,3,8,8]{3,2,1,0} %slice.1268), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_31"}
  %concatenate.1301 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.1269, f32[1,28,28,8]{3,2,1,0} %convolution.1270, f32[1,28,28,8]{3,2,1,0} %convolution.1281, f32[1,28,28,8]{3,2,1,0} %convolution.1292, f32[1,28,28,8]{3,2,1,0} %convolution.1295, f32[1,28,28,8]{3,2,1,0} %convolution.1296, f32[1,28,28,8]{3,2,1,0} %convolution.1297, f32[1,28,28,8]{3,2,1,0} %convolution.1298, f32[1,28,28,8]{3,2,1,0} %convolution.1299, f32[1,28,28,8]{3,2,1,0} %convolution.1300, f32[1,28,28,8]{3,2,1,0} %convolution.1271, f32[1,28,28,8]{3,2,1,0} %convolution.1272, f32[1,28,28,8]{3,2,1,0} %convolution.1273, f32[1,28,28,8]{3,2,1,0} %convolution.1274, f32[1,28,28,8]{3,2,1,0} %convolution.1275, f32[1,28,28,8]{3,2,1,0} %convolution.1276, f32[1,28,28,8]{3,2,1,0} %convolution.1277, f32[1,28,28,8]{3,2,1,0} %convolution.1278, f32[1,28,28,8]{3,2,1,0} %convolution.1279, f32[1,28,28,8]{3,2,1,0} %convolution.1280, f32[1,28,28,8]{3,2,1,0} %convolution.1282, f32[1,28,28,8]{3,2,1,0} %convolution.1283, f32[1,28,28,8]{3,2,1,0} %convolution.1284, f32[1,28,28,8]{3,2,1,0} %convolution.1285, f32[1,28,28,8]{3,2,1,0} %convolution.1286, f32[1,28,28,8]{3,2,1,0} %convolution.1287, f32[1,28,28,8]{3,2,1,0} %convolution.1288, f32[1,28,28,8]{3,2,1,0} %convolution.1289, f32[1,28,28,8]{3,2,1,0} %convolution.1290, f32[1,28,28,8]{3,2,1,0} %convolution.1291, f32[1,28,28,8]{3,2,1,0} %convolution.1293, f32[1,28,28,8]{3,2,1,0} %convolution.1294), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_4"}
  %constant.1181 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add"}
  %broadcast.1182 = f32[256]{0} broadcast(f32[] %constant.1181), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add"}
  %arg31.32 = f32[256]{0} parameter(31), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.301 = f32[256]{0} reshape(f32[256]{0} %arg31.32)
  %add.1183 = f32[256]{0} add(f32[256]{0} %broadcast.1182, f32[256]{0} %reshape.301), metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add"}
  %rsqrt.1184 = f32[256]{0} rsqrt(f32[256]{0} %add.1183), metadata={op_type="Rsqrt" op_name="stage2_unit2_bn2/Rsqrt"}
  %arg94.95 = f32[256]{0} parameter(94), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.364 = f32[256]{0} reshape(f32[256]{0} %arg94.95)
  %multiply.1185 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1184, f32[256]{0} %reshape.364), metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul"}
  %broadcast.1302 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1185), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul_1"}
  %multiply.1303 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.1301, f32[1,28,28,256]{3,2,1,0} %broadcast.1302), metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul_1"}
  %arg201.202 = f32[256]{0} parameter(201), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.471 = f32[256]{0} reshape(f32[256]{0} %arg201.202)
  %arg148.149 = f32[256]{0} parameter(148), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.418 = f32[256]{0} reshape(f32[256]{0} %arg148.149)
  %multiply.1186 = f32[256]{0} multiply(f32[256]{0} %multiply.1185, f32[256]{0} %reshape.418), metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul_2"}
  %subtract.1187 = f32[256]{0} subtract(f32[256]{0} %reshape.471, f32[256]{0} %multiply.1186), metadata={op_type="Sub" op_name="stage2_unit2_bn2/sub"}
  %broadcast.1304 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1187), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add_1"}
  %add.1305 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1303, f32[1,28,28,256]{3,2,1,0} %broadcast.1304), metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add_1"}
  %maximum.1308 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1307, f32[1,28,28,256]{3,2,1,0} %add.1305), metadata={op_type="Relu" op_name="stage2_unit2_relu2"}
  %arg243.244 = f32[1,1,256,512]{3,2,1,0} parameter(243), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.513 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg243.244)
  %convolution.1309 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.1308, f32[1,1,256,512]{3,2,1,0} %reshape.513), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv3"}
  %multiply.1311 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1310, f32[1,28,28,512]{3,2,1,0} %convolution.1309), metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul_1"}
  %arg202.203 = f32[512]{0} parameter(202), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.472 = f32[512]{0} reshape(f32[512]{0} %arg202.203)
  %arg149.150 = f32[512]{0} parameter(149), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.419 = f32[512]{0} reshape(f32[512]{0} %arg149.150)
  %multiply.1193 = f32[512]{0} multiply(f32[512]{0} %multiply.1192, f32[512]{0} %reshape.419), metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul_2"}
  %subtract.1194 = f32[512]{0} subtract(f32[512]{0} %reshape.472, f32[512]{0} %multiply.1193), metadata={op_type="Sub" op_name="stage2_unit2_bn3/sub"}
  %broadcast.1312 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1194), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add_1"}
  %add.1313 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1311, f32[1,28,28,512]{3,2,1,0} %broadcast.1312), metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add_1"}
  %add.1314 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %maximum.1173, f32[1,28,28,512]{3,2,1,0} %add.1313), metadata={op_type="AddV2" op_name="add_4"}
  %maximum.1317 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1316, f32[1,28,28,512]{3,2,1,0} %add.1314), metadata={op_type="Relu" op_name="stage2_unit2_relu"}
  %constant.1332 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add"}
  %broadcast.1333 = f32[512]{0} broadcast(f32[] %constant.1332), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add"}
  %arg37.38 = f32[512]{0} parameter(37), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.307 = f32[512]{0} reshape(f32[512]{0} %arg37.38)
  %add.1334 = f32[512]{0} add(f32[512]{0} %broadcast.1333, f32[512]{0} %reshape.307), metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add"}
  %rsqrt.1335 = f32[512]{0} rsqrt(f32[512]{0} %add.1334), metadata={op_type="Rsqrt" op_name="stage2_unit3_bn3/Rsqrt"}
  %arg98.99 = f32[512]{0} parameter(98), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.368 = f32[512]{0} reshape(f32[512]{0} %arg98.99)
  %multiply.1336 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1335, f32[512]{0} %reshape.368), metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul"}
  %broadcast.1454 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1336), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul_1"}
  %constant.1450 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit3_relu2"}
  %broadcast.1451 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1450), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit3_relu2"}
  %constant.1344 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit3_relu1"}
  %broadcast.1345 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1344), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit3_relu1"}
  %constant.1318 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add"}
  %broadcast.1319 = f32[256]{0} broadcast(f32[] %constant.1318), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add"}
  %arg34.35 = f32[256]{0} parameter(34), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.304 = f32[256]{0} reshape(f32[256]{0} %arg34.35)
  %add.1320 = f32[256]{0} add(f32[256]{0} %broadcast.1319, f32[256]{0} %reshape.304), metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add"}
  %rsqrt.1321 = f32[256]{0} rsqrt(f32[256]{0} %add.1320), metadata={op_type="Rsqrt" op_name="stage2_unit3_bn1/Rsqrt"}
  %arg96.97 = f32[256]{0} parameter(96), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.366 = f32[256]{0} reshape(f32[256]{0} %arg96.97)
  %multiply.1322 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1321, f32[256]{0} %reshape.366), metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul"}
  %broadcast.1340 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1322), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul_1"}
  %arg244.245 = f32[1,1,512,256]{3,2,1,0} parameter(244), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.514 = f32[1,1,512,256]{3,2,1,0} reshape(f32[1,1,512,256]{3,2,1,0} %arg244.245)
  %convolution.1339 = f32[1,28,28,256]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1317, f32[1,1,512,256]{3,2,1,0} %reshape.514), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv1"}
  %multiply.1341 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %broadcast.1340, f32[1,28,28,256]{3,2,1,0} %convolution.1339), metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul_1"}
  %arg203.204 = f32[256]{0} parameter(203), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.473 = f32[256]{0} reshape(f32[256]{0} %arg203.204)
  %arg150.151 = f32[256]{0} parameter(150), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.420 = f32[256]{0} reshape(f32[256]{0} %arg150.151)
  %multiply.1323 = f32[256]{0} multiply(f32[256]{0} %multiply.1322, f32[256]{0} %reshape.420), metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul_2"}
  %subtract.1324 = f32[256]{0} subtract(f32[256]{0} %reshape.473, f32[256]{0} %multiply.1323), metadata={op_type="Sub" op_name="stage2_unit3_bn1/sub"}
  %broadcast.1342 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1324), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add_1"}
  %add.1343 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1341, f32[1,28,28,256]{3,2,1,0} %broadcast.1342), metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add_1"}
  %maximum.1346 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1345, f32[1,28,28,256]{3,2,1,0} %add.1343), metadata={op_type="Relu" op_name="stage2_unit3_relu1"}
  %constant.1347 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_6"}
  %pad.1348 = f32[1,30,30,256]{3,2,1,0} pad(f32[1,28,28,256]{3,2,1,0} %maximum.1346, f32[] %constant.1347), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_6"}
  %slice.1349 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [0:8]}, metadata={op_type="Split" op_name="split_11"}
  %arg35.36 = f32[3,3,8,256]{3,2,1,0} parameter(35), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.305 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg35.36)
  %slice.1381 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1413 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1349, f32[3,3,8,8]{3,2,1,0} %slice.1381), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2"}
  %slice.1350 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [8:16]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1382 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1414 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1350, f32[3,3,8,8]{3,2,1,0} %slice.1382), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_1"}
  %slice.1351 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [16:24]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1383 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1425 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1351, f32[3,3,8,8]{3,2,1,0} %slice.1383), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_2"}
  %slice.1352 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [24:32]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1384 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1436 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1352, f32[3,3,8,8]{3,2,1,0} %slice.1384), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_3"}
  %slice.1353 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [32:40]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1385 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1439 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1353, f32[3,3,8,8]{3,2,1,0} %slice.1385), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_4"}
  %slice.1354 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [40:48]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1386 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1440 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1354, f32[3,3,8,8]{3,2,1,0} %slice.1386), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_5"}
  %slice.1355 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [48:56]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1387 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1441 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1355, f32[3,3,8,8]{3,2,1,0} %slice.1387), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_6"}
  %slice.1356 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [56:64]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1388 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1442 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1356, f32[3,3,8,8]{3,2,1,0} %slice.1388), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_7"}
  %slice.1357 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [64:72]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1389 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1443 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1357, f32[3,3,8,8]{3,2,1,0} %slice.1389), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_8"}
  %slice.1358 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [72:80]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1390 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1444 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1358, f32[3,3,8,8]{3,2,1,0} %slice.1390), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_9"}
  %slice.1359 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [80:88]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1391 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1415 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1359, f32[3,3,8,8]{3,2,1,0} %slice.1391), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_10"}
  %slice.1360 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [88:96]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1392 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1416 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1360, f32[3,3,8,8]{3,2,1,0} %slice.1392), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_11"}
  %slice.1361 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [96:104]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1393 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1417 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1361, f32[3,3,8,8]{3,2,1,0} %slice.1393), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_12"}
  %slice.1362 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [104:112]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1394 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1418 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1362, f32[3,3,8,8]{3,2,1,0} %slice.1394), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_13"}
  %slice.1363 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [112:120]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1395 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1419 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1363, f32[3,3,8,8]{3,2,1,0} %slice.1395), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_14"}
  %slice.1364 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [120:128]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1396 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1420 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1364, f32[3,3,8,8]{3,2,1,0} %slice.1396), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_15"}
  %slice.1365 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [128:136]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1397 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1421 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1365, f32[3,3,8,8]{3,2,1,0} %slice.1397), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_16"}
  %slice.1366 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [136:144]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1398 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1422 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1366, f32[3,3,8,8]{3,2,1,0} %slice.1398), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_17"}
  %slice.1367 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [144:152]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1399 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1423 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1367, f32[3,3,8,8]{3,2,1,0} %slice.1399), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_18"}
  %slice.1368 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [152:160]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1400 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1424 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1368, f32[3,3,8,8]{3,2,1,0} %slice.1400), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_19"}
  %slice.1369 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [160:168]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1401 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1426 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1369, f32[3,3,8,8]{3,2,1,0} %slice.1401), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_20"}
  %slice.1370 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [168:176]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1402 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1427 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1370, f32[3,3,8,8]{3,2,1,0} %slice.1402), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_21"}
  %slice.1371 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [176:184]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1403 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1428 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1371, f32[3,3,8,8]{3,2,1,0} %slice.1403), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_22"}
  %slice.1372 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [184:192]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1404 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1429 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1372, f32[3,3,8,8]{3,2,1,0} %slice.1404), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_23"}
  %slice.1373 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [192:200]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1405 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1430 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1373, f32[3,3,8,8]{3,2,1,0} %slice.1405), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_24"}
  %slice.1374 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [200:208]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1406 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1431 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1374, f32[3,3,8,8]{3,2,1,0} %slice.1406), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_25"}
  %slice.1375 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [208:216]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1407 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1432 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1375, f32[3,3,8,8]{3,2,1,0} %slice.1407), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_26"}
  %slice.1376 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [216:224]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1408 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1433 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1376, f32[3,3,8,8]{3,2,1,0} %slice.1408), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_27"}
  %slice.1377 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [224:232]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1409 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1434 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1377, f32[3,3,8,8]{3,2,1,0} %slice.1409), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_28"}
  %slice.1378 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [232:240]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1410 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1435 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1378, f32[3,3,8,8]{3,2,1,0} %slice.1410), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_29"}
  %slice.1379 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [240:248]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1411 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1437 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1379, f32[3,3,8,8]{3,2,1,0} %slice.1411), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_30"}
  %slice.1380 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1348), slice={[0:1], [0:30], [0:30], [248:256]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1412 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.305), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1438 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1380, f32[3,3,8,8]{3,2,1,0} %slice.1412), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_31"}
  %concatenate.1445 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.1413, f32[1,28,28,8]{3,2,1,0} %convolution.1414, f32[1,28,28,8]{3,2,1,0} %convolution.1425, f32[1,28,28,8]{3,2,1,0} %convolution.1436, f32[1,28,28,8]{3,2,1,0} %convolution.1439, f32[1,28,28,8]{3,2,1,0} %convolution.1440, f32[1,28,28,8]{3,2,1,0} %convolution.1441, f32[1,28,28,8]{3,2,1,0} %convolution.1442, f32[1,28,28,8]{3,2,1,0} %convolution.1443, f32[1,28,28,8]{3,2,1,0} %convolution.1444, f32[1,28,28,8]{3,2,1,0} %convolution.1415, f32[1,28,28,8]{3,2,1,0} %convolution.1416, f32[1,28,28,8]{3,2,1,0} %convolution.1417, f32[1,28,28,8]{3,2,1,0} %convolution.1418, f32[1,28,28,8]{3,2,1,0} %convolution.1419, f32[1,28,28,8]{3,2,1,0} %convolution.1420, f32[1,28,28,8]{3,2,1,0} %convolution.1421, f32[1,28,28,8]{3,2,1,0} %convolution.1422, f32[1,28,28,8]{3,2,1,0} %convolution.1423, f32[1,28,28,8]{3,2,1,0} %convolution.1424, f32[1,28,28,8]{3,2,1,0} %convolution.1426, f32[1,28,28,8]{3,2,1,0} %convolution.1427, f32[1,28,28,8]{3,2,1,0} %convolution.1428, f32[1,28,28,8]{3,2,1,0} %convolution.1429, f32[1,28,28,8]{3,2,1,0} %convolution.1430, f32[1,28,28,8]{3,2,1,0} %convolution.1431, f32[1,28,28,8]{3,2,1,0} %convolution.1432, f32[1,28,28,8]{3,2,1,0} %convolution.1433, f32[1,28,28,8]{3,2,1,0} %convolution.1434, f32[1,28,28,8]{3,2,1,0} %convolution.1435, f32[1,28,28,8]{3,2,1,0} %convolution.1437, f32[1,28,28,8]{3,2,1,0} %convolution.1438), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_5"}
  %constant.1325 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add"}
  %broadcast.1326 = f32[256]{0} broadcast(f32[] %constant.1325), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add"}
  %arg36.37 = f32[256]{0} parameter(36), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.306 = f32[256]{0} reshape(f32[256]{0} %arg36.37)
  %add.1327 = f32[256]{0} add(f32[256]{0} %broadcast.1326, f32[256]{0} %reshape.306), metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add"}
  %rsqrt.1328 = f32[256]{0} rsqrt(f32[256]{0} %add.1327), metadata={op_type="Rsqrt" op_name="stage2_unit3_bn2/Rsqrt"}
  %arg97.98 = f32[256]{0} parameter(97), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.367 = f32[256]{0} reshape(f32[256]{0} %arg97.98)
  %multiply.1329 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1328, f32[256]{0} %reshape.367), metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul"}
  %broadcast.1446 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1329), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul_1"}
  %multiply.1447 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.1445, f32[1,28,28,256]{3,2,1,0} %broadcast.1446), metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul_1"}
  %arg204.205 = f32[256]{0} parameter(204), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.474 = f32[256]{0} reshape(f32[256]{0} %arg204.205)
  %arg151.152 = f32[256]{0} parameter(151), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.421 = f32[256]{0} reshape(f32[256]{0} %arg151.152)
  %multiply.1330 = f32[256]{0} multiply(f32[256]{0} %multiply.1329, f32[256]{0} %reshape.421), metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul_2"}
  %subtract.1331 = f32[256]{0} subtract(f32[256]{0} %reshape.474, f32[256]{0} %multiply.1330), metadata={op_type="Sub" op_name="stage2_unit3_bn2/sub"}
  %broadcast.1448 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1331), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add_1"}
  %add.1449 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1447, f32[1,28,28,256]{3,2,1,0} %broadcast.1448), metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add_1"}
  %maximum.1452 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1451, f32[1,28,28,256]{3,2,1,0} %add.1449), metadata={op_type="Relu" op_name="stage2_unit3_relu2"}
  %arg245.246 = f32[1,1,256,512]{3,2,1,0} parameter(245), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.515 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg245.246)
  %convolution.1453 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.1452, f32[1,1,256,512]{3,2,1,0} %reshape.515), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv3"}
  %multiply.1455 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1454, f32[1,28,28,512]{3,2,1,0} %convolution.1453), metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul_1"}
  %arg205.206 = f32[512]{0} parameter(205), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.475 = f32[512]{0} reshape(f32[512]{0} %arg205.206)
  %arg152.153 = f32[512]{0} parameter(152), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.422 = f32[512]{0} reshape(f32[512]{0} %arg152.153)
  %multiply.1337 = f32[512]{0} multiply(f32[512]{0} %multiply.1336, f32[512]{0} %reshape.422), metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul_2"}
  %subtract.1338 = f32[512]{0} subtract(f32[512]{0} %reshape.475, f32[512]{0} %multiply.1337), metadata={op_type="Sub" op_name="stage2_unit3_bn3/sub"}
  %broadcast.1456 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1338), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add_1"}
  %add.1457 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1455, f32[1,28,28,512]{3,2,1,0} %broadcast.1456), metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add_1"}
  %add.1458 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %maximum.1317, f32[1,28,28,512]{3,2,1,0} %add.1457), metadata={op_type="AddV2" op_name="add_5"}
  %maximum.1461 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1460, f32[1,28,28,512]{3,2,1,0} %add.1458), metadata={op_type="Relu" op_name="stage2_unit3_relu"}
  %constant.1476 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add"}
  %broadcast.1477 = f32[512]{0} broadcast(f32[] %constant.1476), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add"}
  %arg42.43 = f32[512]{0} parameter(42), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.312 = f32[512]{0} reshape(f32[512]{0} %arg42.43)
  %add.1478 = f32[512]{0} add(f32[512]{0} %broadcast.1477, f32[512]{0} %reshape.312), metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add"}
  %rsqrt.1479 = f32[512]{0} rsqrt(f32[512]{0} %add.1478), metadata={op_type="Rsqrt" op_name="stage2_unit4_bn3/Rsqrt"}
  %arg102.103 = f32[512]{0} parameter(102), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.372 = f32[512]{0} reshape(f32[512]{0} %arg102.103)
  %multiply.1480 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1479, f32[512]{0} %reshape.372), metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul"}
  %broadcast.1598 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1480), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul_1"}
  %constant.1594 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit4_relu2"}
  %broadcast.1595 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1594), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit4_relu2"}
  %constant.1488 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit4_relu1"}
  %broadcast.1489 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1488), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit4_relu1"}
  %constant.1462 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add"}
  %broadcast.1463 = f32[256]{0} broadcast(f32[] %constant.1462), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add"}
  %arg38.39 = f32[256]{0} parameter(38), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.308 = f32[256]{0} reshape(f32[256]{0} %arg38.39)
  %add.1464 = f32[256]{0} add(f32[256]{0} %broadcast.1463, f32[256]{0} %reshape.308), metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add"}
  %rsqrt.1465 = f32[256]{0} rsqrt(f32[256]{0} %add.1464), metadata={op_type="Rsqrt" op_name="stage2_unit4_bn1/Rsqrt"}
  %arg99.100 = f32[256]{0} parameter(99), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.369 = f32[256]{0} reshape(f32[256]{0} %arg99.100)
  %multiply.1466 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1465, f32[256]{0} %reshape.369), metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul"}
  %broadcast.1484 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1466), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul_1"}
  %arg246.247 = f32[1,1,512,256]{3,2,1,0} parameter(246), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.516 = f32[1,1,512,256]{3,2,1,0} reshape(f32[1,1,512,256]{3,2,1,0} %arg246.247)
  %convolution.1483 = f32[1,28,28,256]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1461, f32[1,1,512,256]{3,2,1,0} %reshape.516), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv1"}
  %multiply.1485 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %broadcast.1484, f32[1,28,28,256]{3,2,1,0} %convolution.1483), metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul_1"}
  %arg206.207 = f32[256]{0} parameter(206), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.476 = f32[256]{0} reshape(f32[256]{0} %arg206.207)
  %arg153.154 = f32[256]{0} parameter(153), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.423 = f32[256]{0} reshape(f32[256]{0} %arg153.154)
  %multiply.1467 = f32[256]{0} multiply(f32[256]{0} %multiply.1466, f32[256]{0} %reshape.423), metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul_2"}
  %subtract.1468 = f32[256]{0} subtract(f32[256]{0} %reshape.476, f32[256]{0} %multiply.1467), metadata={op_type="Sub" op_name="stage2_unit4_bn1/sub"}
  %broadcast.1486 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1468), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add_1"}
  %add.1487 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1485, f32[1,28,28,256]{3,2,1,0} %broadcast.1486), metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add_1"}
  %maximum.1490 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1489, f32[1,28,28,256]{3,2,1,0} %add.1487), metadata={op_type="Relu" op_name="stage2_unit4_relu1"}
  %constant.1491 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_7"}
  %pad.1492 = f32[1,30,30,256]{3,2,1,0} pad(f32[1,28,28,256]{3,2,1,0} %maximum.1490, f32[] %constant.1491), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_7"}
  %slice.1493 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [0:8]}, metadata={op_type="Split" op_name="split_13"}
  %arg39.40 = f32[3,3,8,256]{3,2,1,0} parameter(39), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.309 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg39.40)
  %slice.1525 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1557 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1493, f32[3,3,8,8]{3,2,1,0} %slice.1525), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2"}
  %slice.1494 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [8:16]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1526 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1558 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1494, f32[3,3,8,8]{3,2,1,0} %slice.1526), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_1"}
  %slice.1495 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [16:24]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1527 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1569 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1495, f32[3,3,8,8]{3,2,1,0} %slice.1527), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_2"}
  %slice.1496 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [24:32]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1528 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1580 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1496, f32[3,3,8,8]{3,2,1,0} %slice.1528), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_3"}
  %slice.1497 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [32:40]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1529 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1583 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1497, f32[3,3,8,8]{3,2,1,0} %slice.1529), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_4"}
  %slice.1498 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [40:48]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1530 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1584 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1498, f32[3,3,8,8]{3,2,1,0} %slice.1530), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_5"}
  %slice.1499 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [48:56]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1531 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1585 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1499, f32[3,3,8,8]{3,2,1,0} %slice.1531), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_6"}
  %slice.1500 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [56:64]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1532 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1586 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1500, f32[3,3,8,8]{3,2,1,0} %slice.1532), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_7"}
  %slice.1501 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [64:72]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1533 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1587 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1501, f32[3,3,8,8]{3,2,1,0} %slice.1533), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_8"}
  %slice.1502 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [72:80]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1534 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1588 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1502, f32[3,3,8,8]{3,2,1,0} %slice.1534), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_9"}
  %slice.1503 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [80:88]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1535 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1559 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1503, f32[3,3,8,8]{3,2,1,0} %slice.1535), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_10"}
  %slice.1504 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [88:96]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1536 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1560 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1504, f32[3,3,8,8]{3,2,1,0} %slice.1536), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_11"}
  %slice.1505 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [96:104]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1537 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1561 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1505, f32[3,3,8,8]{3,2,1,0} %slice.1537), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_12"}
  %slice.1506 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [104:112]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1538 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1562 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1506, f32[3,3,8,8]{3,2,1,0} %slice.1538), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_13"}
  %slice.1507 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [112:120]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1539 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1563 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1507, f32[3,3,8,8]{3,2,1,0} %slice.1539), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_14"}
  %slice.1508 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [120:128]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1540 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1564 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1508, f32[3,3,8,8]{3,2,1,0} %slice.1540), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_15"}
  %slice.1509 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [128:136]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1541 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1565 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1509, f32[3,3,8,8]{3,2,1,0} %slice.1541), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_16"}
  %slice.1510 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [136:144]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1542 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1566 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1510, f32[3,3,8,8]{3,2,1,0} %slice.1542), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_17"}
  %slice.1511 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [144:152]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1543 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1567 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1511, f32[3,3,8,8]{3,2,1,0} %slice.1543), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_18"}
  %slice.1512 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [152:160]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1544 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1568 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1512, f32[3,3,8,8]{3,2,1,0} %slice.1544), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_19"}
  %slice.1513 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [160:168]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1545 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1570 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1513, f32[3,3,8,8]{3,2,1,0} %slice.1545), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_20"}
  %slice.1514 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [168:176]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1546 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1571 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1514, f32[3,3,8,8]{3,2,1,0} %slice.1546), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_21"}
  %slice.1515 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [176:184]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1547 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1572 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1515, f32[3,3,8,8]{3,2,1,0} %slice.1547), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_22"}
  %slice.1516 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [184:192]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1548 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1573 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1516, f32[3,3,8,8]{3,2,1,0} %slice.1548), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_23"}
  %slice.1517 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [192:200]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1549 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1574 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1517, f32[3,3,8,8]{3,2,1,0} %slice.1549), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_24"}
  %slice.1518 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [200:208]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1550 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1575 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1518, f32[3,3,8,8]{3,2,1,0} %slice.1550), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_25"}
  %slice.1519 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [208:216]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1551 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1576 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1519, f32[3,3,8,8]{3,2,1,0} %slice.1551), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_26"}
  %slice.1520 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [216:224]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1552 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1577 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1520, f32[3,3,8,8]{3,2,1,0} %slice.1552), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_27"}
  %slice.1521 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [224:232]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1553 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1578 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1521, f32[3,3,8,8]{3,2,1,0} %slice.1553), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_28"}
  %slice.1522 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [232:240]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1554 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1579 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1522, f32[3,3,8,8]{3,2,1,0} %slice.1554), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_29"}
  %slice.1523 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [240:248]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1555 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1581 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1523, f32[3,3,8,8]{3,2,1,0} %slice.1555), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_30"}
  %slice.1524 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1492), slice={[0:1], [0:30], [0:30], [248:256]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1556 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.309), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1582 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1524, f32[3,3,8,8]{3,2,1,0} %slice.1556), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_31"}
  %concatenate.1589 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.1557, f32[1,28,28,8]{3,2,1,0} %convolution.1558, f32[1,28,28,8]{3,2,1,0} %convolution.1569, f32[1,28,28,8]{3,2,1,0} %convolution.1580, f32[1,28,28,8]{3,2,1,0} %convolution.1583, f32[1,28,28,8]{3,2,1,0} %convolution.1584, f32[1,28,28,8]{3,2,1,0} %convolution.1585, f32[1,28,28,8]{3,2,1,0} %convolution.1586, f32[1,28,28,8]{3,2,1,0} %convolution.1587, f32[1,28,28,8]{3,2,1,0} %convolution.1588, f32[1,28,28,8]{3,2,1,0} %convolution.1559, f32[1,28,28,8]{3,2,1,0} %convolution.1560, f32[1,28,28,8]{3,2,1,0} %convolution.1561, f32[1,28,28,8]{3,2,1,0} %convolution.1562, f32[1,28,28,8]{3,2,1,0} %convolution.1563, f32[1,28,28,8]{3,2,1,0} %convolution.1564, f32[1,28,28,8]{3,2,1,0} %convolution.1565, f32[1,28,28,8]{3,2,1,0} %convolution.1566, f32[1,28,28,8]{3,2,1,0} %convolution.1567, f32[1,28,28,8]{3,2,1,0} %convolution.1568, f32[1,28,28,8]{3,2,1,0} %convolution.1570, f32[1,28,28,8]{3,2,1,0} %convolution.1571, f32[1,28,28,8]{3,2,1,0} %convolution.1572, f32[1,28,28,8]{3,2,1,0} %convolution.1573, f32[1,28,28,8]{3,2,1,0} %convolution.1574, f32[1,28,28,8]{3,2,1,0} %convolution.1575, f32[1,28,28,8]{3,2,1,0} %convolution.1576, f32[1,28,28,8]{3,2,1,0} %convolution.1577, f32[1,28,28,8]{3,2,1,0} %convolution.1578, f32[1,28,28,8]{3,2,1,0} %convolution.1579, f32[1,28,28,8]{3,2,1,0} %convolution.1581, f32[1,28,28,8]{3,2,1,0} %convolution.1582), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_6"}
  %constant.1469 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add"}
  %broadcast.1470 = f32[256]{0} broadcast(f32[] %constant.1469), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add"}
  %arg41.42 = f32[256]{0} parameter(41), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.311 = f32[256]{0} reshape(f32[256]{0} %arg41.42)
  %add.1471 = f32[256]{0} add(f32[256]{0} %broadcast.1470, f32[256]{0} %reshape.311), metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add"}
  %rsqrt.1472 = f32[256]{0} rsqrt(f32[256]{0} %add.1471), metadata={op_type="Rsqrt" op_name="stage2_unit4_bn2/Rsqrt"}
  %arg101.102 = f32[256]{0} parameter(101), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.371 = f32[256]{0} reshape(f32[256]{0} %arg101.102)
  %multiply.1473 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1472, f32[256]{0} %reshape.371), metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul"}
  %broadcast.1590 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1473), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul_1"}
  %multiply.1591 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.1589, f32[1,28,28,256]{3,2,1,0} %broadcast.1590), metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul_1"}
  %arg208.209 = f32[256]{0} parameter(208), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.478 = f32[256]{0} reshape(f32[256]{0} %arg208.209)
  %arg155.156 = f32[256]{0} parameter(155), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.425 = f32[256]{0} reshape(f32[256]{0} %arg155.156)
  %multiply.1474 = f32[256]{0} multiply(f32[256]{0} %multiply.1473, f32[256]{0} %reshape.425), metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul_2"}
  %subtract.1475 = f32[256]{0} subtract(f32[256]{0} %reshape.478, f32[256]{0} %multiply.1474), metadata={op_type="Sub" op_name="stage2_unit4_bn2/sub"}
  %broadcast.1592 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1475), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add_1"}
  %add.1593 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1591, f32[1,28,28,256]{3,2,1,0} %broadcast.1592), metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add_1"}
  %maximum.1596 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1595, f32[1,28,28,256]{3,2,1,0} %add.1593), metadata={op_type="Relu" op_name="stage2_unit4_relu2"}
  %arg247.248 = f32[1,1,256,512]{3,2,1,0} parameter(247), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.517 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg247.248)
  %convolution.1597 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.1596, f32[1,1,256,512]{3,2,1,0} %reshape.517), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv3"}
  %multiply.1599 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1598, f32[1,28,28,512]{3,2,1,0} %convolution.1597), metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul_1"}
  %arg209.210 = f32[512]{0} parameter(209), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.479 = f32[512]{0} reshape(f32[512]{0} %arg209.210)
  %arg156.157 = f32[512]{0} parameter(156), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.426 = f32[512]{0} reshape(f32[512]{0} %arg156.157)
  %multiply.1481 = f32[512]{0} multiply(f32[512]{0} %multiply.1480, f32[512]{0} %reshape.426), metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul_2"}
  %subtract.1482 = f32[512]{0} subtract(f32[512]{0} %reshape.479, f32[512]{0} %multiply.1481), metadata={op_type="Sub" op_name="stage2_unit4_bn3/sub"}
  %broadcast.1600 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1482), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add_1"}
  %add.1601 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1599, f32[1,28,28,512]{3,2,1,0} %broadcast.1600), metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add_1"}
  %add.1602 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %maximum.1461, f32[1,28,28,512]{3,2,1,0} %add.1601), metadata={op_type="AddV2" op_name="add_6"}
  %maximum.1605 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1604, f32[1,28,28,512]{3,2,1,0} %add.1602), metadata={op_type="Relu" op_name="stage2_unit4_relu"}
  %arg248.249 = f32[1,1,512,512]{3,2,1,0} parameter(248), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.518 = f32[1,1,512,512]{3,2,1,0} reshape(f32[1,1,512,512]{3,2,1,0} %arg248.249)
  %convolution.1627 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1605, f32[1,1,512,512]{3,2,1,0} %reshape.518), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv1"}
  %multiply.1629 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1628, f32[1,28,28,512]{3,2,1,0} %convolution.1627), metadata={op_type="Mul" op_name="stage3_unit1_bn1/mul_1"}
  %arg210.211 = f32[512]{0} parameter(210), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.480 = f32[512]{0} reshape(f32[512]{0} %arg210.211)
  %arg157.158 = f32[512]{0} parameter(157), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.427 = f32[512]{0} reshape(f32[512]{0} %arg157.158)
  %multiply.1611 = f32[512]{0} multiply(f32[512]{0} %multiply.1610, f32[512]{0} %reshape.427), metadata={op_type="Mul" op_name="stage3_unit1_bn1/mul_2"}
  %subtract.1612 = f32[512]{0} subtract(f32[512]{0} %reshape.480, f32[512]{0} %multiply.1611), metadata={op_type="Sub" op_name="stage3_unit1_bn1/sub"}
  %broadcast.1630 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1612), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit1_bn1/add_1"}
  %add.1631 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1629, f32[1,28,28,512]{3,2,1,0} %broadcast.1630), metadata={op_type="AddV2" op_name="stage3_unit1_bn1/add_1"}
  %maximum.1634 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1633, f32[1,28,28,512]{3,2,1,0} %add.1631), metadata={op_type="Relu" op_name="stage3_unit1_relu1"}
  %constant.1635 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_8"}
  %pad.1636 = f32[1,30,30,512]{3,2,1,0} pad(f32[1,28,28,512]{3,2,1,0} %maximum.1634, f32[] %constant.1635), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_8"}
  %slice.1637 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [0:16]}, metadata={op_type="Split" op_name="split_15"}
  %arg46.47 = f32[3,3,16,512]{3,2,1,0} parameter(46), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.316 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg46.47)
  %slice.1669 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1701 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1637, f32[3,3,16,16]{3,2,1,0} %slice.1669), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2"}
  %slice.1638 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [16:32]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1670 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1702 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1638, f32[3,3,16,16]{3,2,1,0} %slice.1670), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_1"}
  %slice.1639 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [32:48]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1671 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1713 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1639, f32[3,3,16,16]{3,2,1,0} %slice.1671), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_2"}
  %slice.1640 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [48:64]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1672 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1724 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1640, f32[3,3,16,16]{3,2,1,0} %slice.1672), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_3"}
  %slice.1641 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [64:80]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1673 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1727 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1641, f32[3,3,16,16]{3,2,1,0} %slice.1673), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_4"}
  %slice.1642 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [80:96]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1674 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1728 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1642, f32[3,3,16,16]{3,2,1,0} %slice.1674), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_5"}
  %slice.1643 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [96:112]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1675 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1729 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1643, f32[3,3,16,16]{3,2,1,0} %slice.1675), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_6"}
  %slice.1644 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [112:128]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1676 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1730 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1644, f32[3,3,16,16]{3,2,1,0} %slice.1676), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_7"}
  %slice.1645 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [128:144]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1677 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1731 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1645, f32[3,3,16,16]{3,2,1,0} %slice.1677), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_8"}
  %slice.1646 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [144:160]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1678 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1732 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1646, f32[3,3,16,16]{3,2,1,0} %slice.1678), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_9"}
  %slice.1647 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [160:176]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1679 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1703 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1647, f32[3,3,16,16]{3,2,1,0} %slice.1679), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_10"}
  %slice.1648 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [176:192]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1680 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1704 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1648, f32[3,3,16,16]{3,2,1,0} %slice.1680), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_11"}
  %slice.1649 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [192:208]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1681 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1705 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1649, f32[3,3,16,16]{3,2,1,0} %slice.1681), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_12"}
  %slice.1650 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [208:224]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1682 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1706 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1650, f32[3,3,16,16]{3,2,1,0} %slice.1682), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_13"}
  %slice.1651 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [224:240]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1683 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1707 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1651, f32[3,3,16,16]{3,2,1,0} %slice.1683), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_14"}
  %slice.1652 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [240:256]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1684 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1708 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1652, f32[3,3,16,16]{3,2,1,0} %slice.1684), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_15"}
  %slice.1653 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [256:272]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1685 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1709 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1653, f32[3,3,16,16]{3,2,1,0} %slice.1685), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_16"}
  %slice.1654 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [272:288]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1686 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1710 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1654, f32[3,3,16,16]{3,2,1,0} %slice.1686), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_17"}
  %slice.1655 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [288:304]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1687 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1711 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1655, f32[3,3,16,16]{3,2,1,0} %slice.1687), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_18"}
  %slice.1656 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [304:320]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1688 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1712 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1656, f32[3,3,16,16]{3,2,1,0} %slice.1688), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_19"}
  %slice.1657 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [320:336]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1689 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1714 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1657, f32[3,3,16,16]{3,2,1,0} %slice.1689), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_20"}
  %slice.1658 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [336:352]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1690 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1715 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1658, f32[3,3,16,16]{3,2,1,0} %slice.1690), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_21"}
  %slice.1659 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [352:368]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1691 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1716 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1659, f32[3,3,16,16]{3,2,1,0} %slice.1691), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_22"}
  %slice.1660 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [368:384]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1692 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1717 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1660, f32[3,3,16,16]{3,2,1,0} %slice.1692), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_23"}
  %slice.1661 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [384:400]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1693 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1718 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1661, f32[3,3,16,16]{3,2,1,0} %slice.1693), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_24"}
  %slice.1662 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [400:416]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1694 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1719 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1662, f32[3,3,16,16]{3,2,1,0} %slice.1694), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_25"}
  %slice.1663 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [416:432]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1695 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1720 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1663, f32[3,3,16,16]{3,2,1,0} %slice.1695), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_26"}
  %slice.1664 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [432:448]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1696 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1721 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1664, f32[3,3,16,16]{3,2,1,0} %slice.1696), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_27"}
  %slice.1665 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [448:464]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1697 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1722 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1665, f32[3,3,16,16]{3,2,1,0} %slice.1697), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_28"}
  %slice.1666 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [464:480]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1698 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1723 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1666, f32[3,3,16,16]{3,2,1,0} %slice.1698), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_29"}
  %slice.1667 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [480:496]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1699 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1725 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1667, f32[3,3,16,16]{3,2,1,0} %slice.1699), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_30"}
  %slice.1668 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1636), slice={[0:1], [0:30], [0:30], [496:512]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1700 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.316), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1726 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1668, f32[3,3,16,16]{3,2,1,0} %slice.1700), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_31"}
  %concatenate.1733 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.1701, f32[1,14,14,16]{3,2,1,0} %convolution.1702, f32[1,14,14,16]{3,2,1,0} %convolution.1713, f32[1,14,14,16]{3,2,1,0} %convolution.1724, f32[1,14,14,16]{3,2,1,0} %convolution.1727, f32[1,14,14,16]{3,2,1,0} %convolution.1728, f32[1,14,14,16]{3,2,1,0} %convolution.1729, f32[1,14,14,16]{3,2,1,0} %convolution.1730, f32[1,14,14,16]{3,2,1,0} %convolution.1731, f32[1,14,14,16]{3,2,1,0} %convolution.1732, f32[1,14,14,16]{3,2,1,0} %convolution.1703, f32[1,14,14,16]{3,2,1,0} %convolution.1704, f32[1,14,14,16]{3,2,1,0} %convolution.1705, f32[1,14,14,16]{3,2,1,0} %convolution.1706, f32[1,14,14,16]{3,2,1,0} %convolution.1707, f32[1,14,14,16]{3,2,1,0} %convolution.1708, f32[1,14,14,16]{3,2,1,0} %convolution.1709, f32[1,14,14,16]{3,2,1,0} %convolution.1710, f32[1,14,14,16]{3,2,1,0} %convolution.1711, f32[1,14,14,16]{3,2,1,0} %convolution.1712, f32[1,14,14,16]{3,2,1,0} %convolution.1714, f32[1,14,14,16]{3,2,1,0} %convolution.1715, f32[1,14,14,16]{3,2,1,0} %convolution.1716, f32[1,14,14,16]{3,2,1,0} %convolution.1717, f32[1,14,14,16]{3,2,1,0} %convolution.1718, f32[1,14,14,16]{3,2,1,0} %convolution.1719, f32[1,14,14,16]{3,2,1,0} %convolution.1720, f32[1,14,14,16]{3,2,1,0} %convolution.1721, f32[1,14,14,16]{3,2,1,0} %convolution.1722, f32[1,14,14,16]{3,2,1,0} %convolution.1723, f32[1,14,14,16]{3,2,1,0} %convolution.1725, f32[1,14,14,16]{3,2,1,0} %convolution.1726), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_7"}
  %constant.1613 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit1_bn2/add"}
  %broadcast.1614 = f32[512]{0} broadcast(f32[] %constant.1613), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit1_bn2/add"}
  %arg47.48 = f32[512]{0} parameter(47), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.317 = f32[512]{0} reshape(f32[512]{0} %arg47.48)
  %add.1615 = f32[512]{0} add(f32[512]{0} %broadcast.1614, f32[512]{0} %reshape.317), metadata={op_type="AddV2" op_name="stage3_unit1_bn2/add"}
  %rsqrt.1616 = f32[512]{0} rsqrt(f32[512]{0} %add.1615), metadata={op_type="Rsqrt" op_name="stage3_unit1_bn2/Rsqrt"}
  %arg106.107 = f32[512]{0} parameter(106), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.376 = f32[512]{0} reshape(f32[512]{0} %arg106.107)
  %multiply.1617 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1616, f32[512]{0} %reshape.376), metadata={op_type="Mul" op_name="stage3_unit1_bn2/mul"}
  %broadcast.1734 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1617), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit1_bn2/mul_1"}
  %multiply.1735 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.1733, f32[1,14,14,512]{3,2,1,0} %broadcast.1734), metadata={op_type="Mul" op_name="stage3_unit1_bn2/mul_1"}
  %arg213.214 = f32[512]{0} parameter(213), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.483 = f32[512]{0} reshape(f32[512]{0} %arg213.214)
  %arg160.161 = f32[512]{0} parameter(160), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.430 = f32[512]{0} reshape(f32[512]{0} %arg160.161)
  %multiply.1618 = f32[512]{0} multiply(f32[512]{0} %multiply.1617, f32[512]{0} %reshape.430), metadata={op_type="Mul" op_name="stage3_unit1_bn2/mul_2"}
  %subtract.1619 = f32[512]{0} subtract(f32[512]{0} %reshape.483, f32[512]{0} %multiply.1618), metadata={op_type="Sub" op_name="stage3_unit1_bn2/sub"}
  %broadcast.1736 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1619), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit1_bn2/add_1"}
  %add.1737 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.1735, f32[1,14,14,512]{3,2,1,0} %broadcast.1736), metadata={op_type="AddV2" op_name="stage3_unit1_bn2/add_1"}
  %maximum.1740 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.1739, f32[1,14,14,512]{3,2,1,0} %add.1737), metadata={op_type="Relu" op_name="stage3_unit1_relu2"}
  %arg250.251 = f32[1,1,512,1024]{3,2,1,0} parameter(250), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.520 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg250.251)
  %convolution.1741 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.1740, f32[1,1,512,1024]{3,2,1,0} %reshape.520), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv3"}
  %multiply.1743 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.1742, f32[1,14,14,1024]{3,2,1,0} %convolution.1741), metadata={op_type="Mul" op_name="stage3_unit1_bn3/mul_1"}
  %arg214.215 = f32[1024]{0} parameter(214), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.484 = f32[1024]{0} reshape(f32[1024]{0} %arg214.215)
  %arg161.162 = f32[1024]{0} parameter(161), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.431 = f32[1024]{0} reshape(f32[1024]{0} %arg161.162)
  %multiply.1625 = f32[1024]{0} multiply(f32[1024]{0} %multiply.1624, f32[1024]{0} %reshape.431), metadata={op_type="Mul" op_name="stage3_unit1_bn3/mul_2"}
  %subtract.1626 = f32[1024]{0} subtract(f32[1024]{0} %reshape.484, f32[1024]{0} %multiply.1625), metadata={op_type="Sub" op_name="stage3_unit1_bn3/sub"}
  %broadcast.1744 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.1626), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit1_bn3/add_1"}
  %add.1745 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.1743, f32[1,14,14,1024]{3,2,1,0} %broadcast.1744), metadata={op_type="AddV2" op_name="stage3_unit1_bn3/add_1"}
  %arg249.250 = f32[1,1,512,1024]{3,2,1,0} parameter(249), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.519 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg249.250)
  %convolution.1753 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1605, f32[1,1,512,1024]{3,2,1,0} %reshape.519), window={size=1x1 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_sc"}
  %constant.1746 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit1_sc_bn/add"}
  %broadcast.1747 = f32[1024]{0} broadcast(f32[] %constant.1746), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit1_sc_bn/add"}
  %arg45.46 = f32[1024]{0} parameter(45), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.315 = f32[1024]{0} reshape(f32[1024]{0} %arg45.46)
  %add.1748 = f32[1024]{0} add(f32[1024]{0} %broadcast.1747, f32[1024]{0} %reshape.315), metadata={op_type="AddV2" op_name="stage3_unit1_sc_bn/add"}
  %rsqrt.1749 = f32[1024]{0} rsqrt(f32[1024]{0} %add.1748), metadata={op_type="Rsqrt" op_name="stage3_unit1_sc_bn/Rsqrt"}
  %arg105.106 = f32[1024]{0} parameter(105), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.375 = f32[1024]{0} reshape(f32[1024]{0} %arg105.106)
  %multiply.1750 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.1749, f32[1024]{0} %reshape.375), metadata={op_type="Mul" op_name="stage3_unit1_sc_bn/mul"}
  %broadcast.1754 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.1750), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit1_sc_bn/mul_1"}
  %multiply.1755 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %convolution.1753, f32[1,14,14,1024]{3,2,1,0} %broadcast.1754), metadata={op_type="Mul" op_name="stage3_unit1_sc_bn/mul_1"}
  %arg212.213 = f32[1024]{0} parameter(212), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.482 = f32[1024]{0} reshape(f32[1024]{0} %arg212.213)
  %arg159.160 = f32[1024]{0} parameter(159), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.429 = f32[1024]{0} reshape(f32[1024]{0} %arg159.160)
  %multiply.1751 = f32[1024]{0} multiply(f32[1024]{0} %multiply.1750, f32[1024]{0} %reshape.429), metadata={op_type="Mul" op_name="stage3_unit1_sc_bn/mul_2"}
  %subtract.1752 = f32[1024]{0} subtract(f32[1024]{0} %reshape.482, f32[1024]{0} %multiply.1751), metadata={op_type="Sub" op_name="stage3_unit1_sc_bn/sub"}
  %broadcast.1756 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.1752), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit1_sc_bn/add_1"}
  %add.1757 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.1755, f32[1,14,14,1024]{3,2,1,0} %broadcast.1756), metadata={op_type="AddV2" op_name="stage3_unit1_sc_bn/add_1"}
  %add.1758 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %add.1745, f32[1,14,14,1024]{3,2,1,0} %add.1757), metadata={op_type="AddV2" op_name="add_7"}
  %maximum.1761 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.1760, f32[1,14,14,1024]{3,2,1,0} %add.1758), metadata={op_type="Relu" op_name="stage3_unit1_relu"}
  %constant.1776 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit2_bn3/add"}
  %broadcast.1777 = f32[1024]{0} broadcast(f32[] %constant.1776), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit2_bn3/add"}
  %arg55.56 = f32[1024]{0} parameter(55), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.325 = f32[1024]{0} reshape(f32[1024]{0} %arg55.56)
  %add.1778 = f32[1024]{0} add(f32[1024]{0} %broadcast.1777, f32[1024]{0} %reshape.325), metadata={op_type="AddV2" op_name="stage3_unit2_bn3/add"}
  %rsqrt.1779 = f32[1024]{0} rsqrt(f32[1024]{0} %add.1778), metadata={op_type="Rsqrt" op_name="stage3_unit2_bn3/Rsqrt"}
  %arg112.113 = f32[1024]{0} parameter(112), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.382 = f32[1024]{0} reshape(f32[1024]{0} %arg112.113)
  %multiply.1780 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.1779, f32[1024]{0} %reshape.382), metadata={op_type="Mul" op_name="stage3_unit2_bn3/mul"}
  %broadcast.1898 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.1780), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit2_bn3/mul_1"}
  %constant.1894 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit2_relu2"}
  %broadcast.1895 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.1894), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit2_relu2"}
  %constant.1788 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit2_relu1"}
  %broadcast.1789 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.1788), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit2_relu1"}
  %constant.1762 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit2_bn1/add"}
  %broadcast.1763 = f32[512]{0} broadcast(f32[] %constant.1762), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit2_bn1/add"}
  %arg51.52 = f32[512]{0} parameter(51), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.321 = f32[512]{0} reshape(f32[512]{0} %arg51.52)
  %add.1764 = f32[512]{0} add(f32[512]{0} %broadcast.1763, f32[512]{0} %reshape.321), metadata={op_type="AddV2" op_name="stage3_unit2_bn1/add"}
  %rsqrt.1765 = f32[512]{0} rsqrt(f32[512]{0} %add.1764), metadata={op_type="Rsqrt" op_name="stage3_unit2_bn1/Rsqrt"}
  %arg110.111 = f32[512]{0} parameter(110), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.380 = f32[512]{0} reshape(f32[512]{0} %arg110.111)
  %multiply.1766 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1765, f32[512]{0} %reshape.380), metadata={op_type="Mul" op_name="stage3_unit2_bn1/mul"}
  %broadcast.1784 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1766), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit2_bn1/mul_1"}
  %arg251.252 = f32[1,1,1024,512]{3,2,1,0} parameter(251), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.521 = f32[1,1,1024,512]{3,2,1,0} reshape(f32[1,1,1024,512]{3,2,1,0} %arg251.252)
  %convolution.1783 = f32[1,14,14,512]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.1761, f32[1,1,1024,512]{3,2,1,0} %reshape.521), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv1"}
  %multiply.1785 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %broadcast.1784, f32[1,14,14,512]{3,2,1,0} %convolution.1783), metadata={op_type="Mul" op_name="stage3_unit2_bn1/mul_1"}
  %arg217.218 = f32[512]{0} parameter(217), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.487 = f32[512]{0} reshape(f32[512]{0} %arg217.218)
  %arg164.165 = f32[512]{0} parameter(164), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.434 = f32[512]{0} reshape(f32[512]{0} %arg164.165)
  %multiply.1767 = f32[512]{0} multiply(f32[512]{0} %multiply.1766, f32[512]{0} %reshape.434), metadata={op_type="Mul" op_name="stage3_unit2_bn1/mul_2"}
  %subtract.1768 = f32[512]{0} subtract(f32[512]{0} %reshape.487, f32[512]{0} %multiply.1767), metadata={op_type="Sub" op_name="stage3_unit2_bn1/sub"}
  %broadcast.1786 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1768), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit2_bn1/add_1"}
  %add.1787 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.1785, f32[1,14,14,512]{3,2,1,0} %broadcast.1786), metadata={op_type="AddV2" op_name="stage3_unit2_bn1/add_1"}
  %maximum.1790 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.1789, f32[1,14,14,512]{3,2,1,0} %add.1787), metadata={op_type="Relu" op_name="stage3_unit2_relu1"}
  %constant.1791 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_9"}
  %pad.1792 = f32[1,16,16,512]{3,2,1,0} pad(f32[1,14,14,512]{3,2,1,0} %maximum.1790, f32[] %constant.1791), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_9"}
  %slice.1793 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_17"}
  %arg52.53 = f32[3,3,16,512]{3,2,1,0} parameter(52), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.322 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg52.53)
  %slice.1825 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1857 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1793, f32[3,3,16,16]{3,2,1,0} %slice.1825), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2"}
  %slice.1794 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1826 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1858 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1794, f32[3,3,16,16]{3,2,1,0} %slice.1826), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_1"}
  %slice.1795 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1827 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1869 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1795, f32[3,3,16,16]{3,2,1,0} %slice.1827), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_2"}
  %slice.1796 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1828 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1880 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1796, f32[3,3,16,16]{3,2,1,0} %slice.1828), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_3"}
  %slice.1797 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1829 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1883 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1797, f32[3,3,16,16]{3,2,1,0} %slice.1829), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_4"}
  %slice.1798 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1830 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1884 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1798, f32[3,3,16,16]{3,2,1,0} %slice.1830), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_5"}
  %slice.1799 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1831 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1885 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1799, f32[3,3,16,16]{3,2,1,0} %slice.1831), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_6"}
  %slice.1800 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1832 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1886 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1800, f32[3,3,16,16]{3,2,1,0} %slice.1832), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_7"}
  %slice.1801 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1833 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1887 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1801, f32[3,3,16,16]{3,2,1,0} %slice.1833), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_8"}
  %slice.1802 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1834 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1888 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1802, f32[3,3,16,16]{3,2,1,0} %slice.1834), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_9"}
  %slice.1803 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1835 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1859 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1803, f32[3,3,16,16]{3,2,1,0} %slice.1835), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_10"}
  %slice.1804 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1836 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1860 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1804, f32[3,3,16,16]{3,2,1,0} %slice.1836), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_11"}
  %slice.1805 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1837 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1861 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1805, f32[3,3,16,16]{3,2,1,0} %slice.1837), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_12"}
  %slice.1806 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1838 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1862 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1806, f32[3,3,16,16]{3,2,1,0} %slice.1838), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_13"}
  %slice.1807 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1839 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1863 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1807, f32[3,3,16,16]{3,2,1,0} %slice.1839), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_14"}
  %slice.1808 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1840 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1864 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1808, f32[3,3,16,16]{3,2,1,0} %slice.1840), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_15"}
  %slice.1809 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1841 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1865 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1809, f32[3,3,16,16]{3,2,1,0} %slice.1841), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_16"}
  %slice.1810 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1842 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1866 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1810, f32[3,3,16,16]{3,2,1,0} %slice.1842), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_17"}
  %slice.1811 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1843 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1867 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1811, f32[3,3,16,16]{3,2,1,0} %slice.1843), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_18"}
  %slice.1812 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1844 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1868 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1812, f32[3,3,16,16]{3,2,1,0} %slice.1844), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_19"}
  %slice.1813 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1845 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1870 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1813, f32[3,3,16,16]{3,2,1,0} %slice.1845), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_20"}
  %slice.1814 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1846 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1871 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1814, f32[3,3,16,16]{3,2,1,0} %slice.1846), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_21"}
  %slice.1815 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1847 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1872 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1815, f32[3,3,16,16]{3,2,1,0} %slice.1847), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_22"}
  %slice.1816 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1848 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1873 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1816, f32[3,3,16,16]{3,2,1,0} %slice.1848), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_23"}
  %slice.1817 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1849 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1874 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1817, f32[3,3,16,16]{3,2,1,0} %slice.1849), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_24"}
  %slice.1818 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1850 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1875 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1818, f32[3,3,16,16]{3,2,1,0} %slice.1850), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_25"}
  %slice.1819 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1851 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1876 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1819, f32[3,3,16,16]{3,2,1,0} %slice.1851), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_26"}
  %slice.1820 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1852 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1877 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1820, f32[3,3,16,16]{3,2,1,0} %slice.1852), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_27"}
  %slice.1821 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1853 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1878 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1821, f32[3,3,16,16]{3,2,1,0} %slice.1853), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_28"}
  %slice.1822 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1854 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1879 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1822, f32[3,3,16,16]{3,2,1,0} %slice.1854), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_29"}
  %slice.1823 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1855 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1881 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1823, f32[3,3,16,16]{3,2,1,0} %slice.1855), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_30"}
  %slice.1824 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1792), slice={[0:1], [0:16], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1856 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1882 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1824, f32[3,3,16,16]{3,2,1,0} %slice.1856), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_31"}
  %concatenate.1889 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.1857, f32[1,14,14,16]{3,2,1,0} %convolution.1858, f32[1,14,14,16]{3,2,1,0} %convolution.1869, f32[1,14,14,16]{3,2,1,0} %convolution.1880, f32[1,14,14,16]{3,2,1,0} %convolution.1883, f32[1,14,14,16]{3,2,1,0} %convolution.1884, f32[1,14,14,16]{3,2,1,0} %convolution.1885, f32[1,14,14,16]{3,2,1,0} %convolution.1886, f32[1,14,14,16]{3,2,1,0} %convolution.1887, f32[1,14,14,16]{3,2,1,0} %convolution.1888, f32[1,14,14,16]{3,2,1,0} %convolution.1859, f32[1,14,14,16]{3,2,1,0} %convolution.1860, f32[1,14,14,16]{3,2,1,0} %convolution.1861, f32[1,14,14,16]{3,2,1,0} %convolution.1862, f32[1,14,14,16]{3,2,1,0} %convolution.1863, f32[1,14,14,16]{3,2,1,0} %convolution.1864, f32[1,14,14,16]{3,2,1,0} %convolution.1865, f32[1,14,14,16]{3,2,1,0} %convolution.1866, f32[1,14,14,16]{3,2,1,0} %convolution.1867, f32[1,14,14,16]{3,2,1,0} %convolution.1868, f32[1,14,14,16]{3,2,1,0} %convolution.1870, f32[1,14,14,16]{3,2,1,0} %convolution.1871, f32[1,14,14,16]{3,2,1,0} %convolution.1872, f32[1,14,14,16]{3,2,1,0} %convolution.1873, f32[1,14,14,16]{3,2,1,0} %convolution.1874, f32[1,14,14,16]{3,2,1,0} %convolution.1875, f32[1,14,14,16]{3,2,1,0} %convolution.1876, f32[1,14,14,16]{3,2,1,0} %convolution.1877, f32[1,14,14,16]{3,2,1,0} %convolution.1878, f32[1,14,14,16]{3,2,1,0} %convolution.1879, f32[1,14,14,16]{3,2,1,0} %convolution.1881, f32[1,14,14,16]{3,2,1,0} %convolution.1882), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_8"}
  %constant.1769 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit2_bn2/add"}
  %broadcast.1770 = f32[512]{0} broadcast(f32[] %constant.1769), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit2_bn2/add"}
  %arg53.54 = f32[512]{0} parameter(53), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.323 = f32[512]{0} reshape(f32[512]{0} %arg53.54)
  %add.1771 = f32[512]{0} add(f32[512]{0} %broadcast.1770, f32[512]{0} %reshape.323), metadata={op_type="AddV2" op_name="stage3_unit2_bn2/add"}
  %rsqrt.1772 = f32[512]{0} rsqrt(f32[512]{0} %add.1771), metadata={op_type="Rsqrt" op_name="stage3_unit2_bn2/Rsqrt"}
  %arg111.112 = f32[512]{0} parameter(111), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.381 = f32[512]{0} reshape(f32[512]{0} %arg111.112)
  %multiply.1773 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1772, f32[512]{0} %reshape.381), metadata={op_type="Mul" op_name="stage3_unit2_bn2/mul"}
  %broadcast.1890 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1773), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit2_bn2/mul_1"}
  %multiply.1891 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.1889, f32[1,14,14,512]{3,2,1,0} %broadcast.1890), metadata={op_type="Mul" op_name="stage3_unit2_bn2/mul_1"}
  %arg218.219 = f32[512]{0} parameter(218), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.488 = f32[512]{0} reshape(f32[512]{0} %arg218.219)
  %arg165.166 = f32[512]{0} parameter(165), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.435 = f32[512]{0} reshape(f32[512]{0} %arg165.166)
  %multiply.1774 = f32[512]{0} multiply(f32[512]{0} %multiply.1773, f32[512]{0} %reshape.435), metadata={op_type="Mul" op_name="stage3_unit2_bn2/mul_2"}
  %subtract.1775 = f32[512]{0} subtract(f32[512]{0} %reshape.488, f32[512]{0} %multiply.1774), metadata={op_type="Sub" op_name="stage3_unit2_bn2/sub"}
  %broadcast.1892 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1775), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit2_bn2/add_1"}
  %add.1893 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.1891, f32[1,14,14,512]{3,2,1,0} %broadcast.1892), metadata={op_type="AddV2" op_name="stage3_unit2_bn2/add_1"}
  %maximum.1896 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.1895, f32[1,14,14,512]{3,2,1,0} %add.1893), metadata={op_type="Relu" op_name="stage3_unit2_relu2"}
  %arg252.253 = f32[1,1,512,1024]{3,2,1,0} parameter(252), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.522 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg252.253)
  %convolution.1897 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.1896, f32[1,1,512,1024]{3,2,1,0} %reshape.522), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv3"}
  %multiply.1899 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.1898, f32[1,14,14,1024]{3,2,1,0} %convolution.1897), metadata={op_type="Mul" op_name="stage3_unit2_bn3/mul_1"}
  %arg219.220 = f32[1024]{0} parameter(219), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.489 = f32[1024]{0} reshape(f32[1024]{0} %arg219.220)
  %arg166.167 = f32[1024]{0} parameter(166), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.436 = f32[1024]{0} reshape(f32[1024]{0} %arg166.167)
  %multiply.1781 = f32[1024]{0} multiply(f32[1024]{0} %multiply.1780, f32[1024]{0} %reshape.436), metadata={op_type="Mul" op_name="stage3_unit2_bn3/mul_2"}
  %subtract.1782 = f32[1024]{0} subtract(f32[1024]{0} %reshape.489, f32[1024]{0} %multiply.1781), metadata={op_type="Sub" op_name="stage3_unit2_bn3/sub"}
  %broadcast.1900 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.1782), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit2_bn3/add_1"}
  %add.1901 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.1899, f32[1,14,14,1024]{3,2,1,0} %broadcast.1900), metadata={op_type="AddV2" op_name="stage3_unit2_bn3/add_1"}
  %add.1902 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %maximum.1761, f32[1,14,14,1024]{3,2,1,0} %add.1901), metadata={op_type="AddV2" op_name="add_8"}
  %maximum.1905 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.1904, f32[1,14,14,1024]{3,2,1,0} %add.1902), metadata={op_type="Relu" op_name="stage3_unit2_relu"}
  %constant.1920 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit3_bn3/add"}
  %broadcast.1921 = f32[1024]{0} broadcast(f32[] %constant.1920), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit3_bn3/add"}
  %arg59.60 = f32[1024]{0} parameter(59), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.329 = f32[1024]{0} reshape(f32[1024]{0} %arg59.60)
  %add.1922 = f32[1024]{0} add(f32[1024]{0} %broadcast.1921, f32[1024]{0} %reshape.329), metadata={op_type="AddV2" op_name="stage3_unit3_bn3/add"}
  %rsqrt.1923 = f32[1024]{0} rsqrt(f32[1024]{0} %add.1922), metadata={op_type="Rsqrt" op_name="stage3_unit3_bn3/Rsqrt"}
  %arg115.116 = f32[1024]{0} parameter(115), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.385 = f32[1024]{0} reshape(f32[1024]{0} %arg115.116)
  %multiply.1924 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.1923, f32[1024]{0} %reshape.385), metadata={op_type="Mul" op_name="stage3_unit3_bn3/mul"}
  %broadcast.2042 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.1924), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit3_bn3/mul_1"}
  %constant.2038 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit3_relu2"}
  %broadcast.2039 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2038), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit3_relu2"}
  %constant.1932 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit3_relu1"}
  %broadcast.1933 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.1932), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit3_relu1"}
  %constant.1906 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit3_bn1/add"}
  %broadcast.1907 = f32[512]{0} broadcast(f32[] %constant.1906), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit3_bn1/add"}
  %arg56.57 = f32[512]{0} parameter(56), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.326 = f32[512]{0} reshape(f32[512]{0} %arg56.57)
  %add.1908 = f32[512]{0} add(f32[512]{0} %broadcast.1907, f32[512]{0} %reshape.326), metadata={op_type="AddV2" op_name="stage3_unit3_bn1/add"}
  %rsqrt.1909 = f32[512]{0} rsqrt(f32[512]{0} %add.1908), metadata={op_type="Rsqrt" op_name="stage3_unit3_bn1/Rsqrt"}
  %arg113.114 = f32[512]{0} parameter(113), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.383 = f32[512]{0} reshape(f32[512]{0} %arg113.114)
  %multiply.1910 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1909, f32[512]{0} %reshape.383), metadata={op_type="Mul" op_name="stage3_unit3_bn1/mul"}
  %broadcast.1928 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1910), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit3_bn1/mul_1"}
  %arg253.254 = f32[1,1,1024,512]{3,2,1,0} parameter(253), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.523 = f32[1,1,1024,512]{3,2,1,0} reshape(f32[1,1,1024,512]{3,2,1,0} %arg253.254)
  %convolution.1927 = f32[1,14,14,512]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.1905, f32[1,1,1024,512]{3,2,1,0} %reshape.523), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv1"}
  %multiply.1929 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %broadcast.1928, f32[1,14,14,512]{3,2,1,0} %convolution.1927), metadata={op_type="Mul" op_name="stage3_unit3_bn1/mul_1"}
  %arg220.221 = f32[512]{0} parameter(220), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.490 = f32[512]{0} reshape(f32[512]{0} %arg220.221)
  %arg167.168 = f32[512]{0} parameter(167), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.437 = f32[512]{0} reshape(f32[512]{0} %arg167.168)
  %multiply.1911 = f32[512]{0} multiply(f32[512]{0} %multiply.1910, f32[512]{0} %reshape.437), metadata={op_type="Mul" op_name="stage3_unit3_bn1/mul_2"}
  %subtract.1912 = f32[512]{0} subtract(f32[512]{0} %reshape.490, f32[512]{0} %multiply.1911), metadata={op_type="Sub" op_name="stage3_unit3_bn1/sub"}
  %broadcast.1930 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1912), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit3_bn1/add_1"}
  %add.1931 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.1929, f32[1,14,14,512]{3,2,1,0} %broadcast.1930), metadata={op_type="AddV2" op_name="stage3_unit3_bn1/add_1"}
  %maximum.1934 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.1933, f32[1,14,14,512]{3,2,1,0} %add.1931), metadata={op_type="Relu" op_name="stage3_unit3_relu1"}
  %constant.1935 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_10"}
  %pad.1936 = f32[1,16,16,512]{3,2,1,0} pad(f32[1,14,14,512]{3,2,1,0} %maximum.1934, f32[] %constant.1935), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_10"}
  %slice.1937 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_19"}
  %arg57.58 = f32[3,3,16,512]{3,2,1,0} parameter(57), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.327 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg57.58)
  %slice.1969 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2001 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1937, f32[3,3,16,16]{3,2,1,0} %slice.1969), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2"}
  %slice.1938 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1970 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2002 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1938, f32[3,3,16,16]{3,2,1,0} %slice.1970), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_1"}
  %slice.1939 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1971 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2013 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1939, f32[3,3,16,16]{3,2,1,0} %slice.1971), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_2"}
  %slice.1940 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1972 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2024 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1940, f32[3,3,16,16]{3,2,1,0} %slice.1972), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_3"}
  %slice.1941 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1973 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2027 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1941, f32[3,3,16,16]{3,2,1,0} %slice.1973), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_4"}
  %slice.1942 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1974 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2028 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1942, f32[3,3,16,16]{3,2,1,0} %slice.1974), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_5"}
  %slice.1943 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1975 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2029 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1943, f32[3,3,16,16]{3,2,1,0} %slice.1975), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_6"}
  %slice.1944 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1976 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2030 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1944, f32[3,3,16,16]{3,2,1,0} %slice.1976), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_7"}
  %slice.1945 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1977 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2031 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1945, f32[3,3,16,16]{3,2,1,0} %slice.1977), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_8"}
  %slice.1946 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1978 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2032 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1946, f32[3,3,16,16]{3,2,1,0} %slice.1978), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_9"}
  %slice.1947 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1979 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2003 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1947, f32[3,3,16,16]{3,2,1,0} %slice.1979), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_10"}
  %slice.1948 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1980 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2004 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1948, f32[3,3,16,16]{3,2,1,0} %slice.1980), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_11"}
  %slice.1949 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1981 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2005 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1949, f32[3,3,16,16]{3,2,1,0} %slice.1981), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_12"}
  %slice.1950 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1982 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2006 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1950, f32[3,3,16,16]{3,2,1,0} %slice.1982), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_13"}
  %slice.1951 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1983 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2007 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1951, f32[3,3,16,16]{3,2,1,0} %slice.1983), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_14"}
  %slice.1952 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1984 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2008 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1952, f32[3,3,16,16]{3,2,1,0} %slice.1984), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_15"}
  %slice.1953 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1985 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2009 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1953, f32[3,3,16,16]{3,2,1,0} %slice.1985), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_16"}
  %slice.1954 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1986 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2010 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1954, f32[3,3,16,16]{3,2,1,0} %slice.1986), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_17"}
  %slice.1955 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1987 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2011 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1955, f32[3,3,16,16]{3,2,1,0} %slice.1987), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_18"}
  %slice.1956 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1988 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2012 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1956, f32[3,3,16,16]{3,2,1,0} %slice.1988), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_19"}
  %slice.1957 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1989 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2014 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1957, f32[3,3,16,16]{3,2,1,0} %slice.1989), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_20"}
  %slice.1958 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1990 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2015 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1958, f32[3,3,16,16]{3,2,1,0} %slice.1990), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_21"}
  %slice.1959 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1991 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2016 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1959, f32[3,3,16,16]{3,2,1,0} %slice.1991), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_22"}
  %slice.1960 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1992 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2017 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1960, f32[3,3,16,16]{3,2,1,0} %slice.1992), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_23"}
  %slice.1961 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1993 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2018 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1961, f32[3,3,16,16]{3,2,1,0} %slice.1993), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_24"}
  %slice.1962 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1994 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2019 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1962, f32[3,3,16,16]{3,2,1,0} %slice.1994), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_25"}
  %slice.1963 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1995 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2020 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1963, f32[3,3,16,16]{3,2,1,0} %slice.1995), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_26"}
  %slice.1964 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1996 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2021 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1964, f32[3,3,16,16]{3,2,1,0} %slice.1996), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_27"}
  %slice.1965 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1997 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2022 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1965, f32[3,3,16,16]{3,2,1,0} %slice.1997), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_28"}
  %slice.1966 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1998 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2023 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1966, f32[3,3,16,16]{3,2,1,0} %slice.1998), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_29"}
  %slice.1967 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1999 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2025 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1967, f32[3,3,16,16]{3,2,1,0} %slice.1999), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_30"}
  %slice.1968 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1936), slice={[0:1], [0:16], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_19"}
  %slice.2000 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.327), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2026 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1968, f32[3,3,16,16]{3,2,1,0} %slice.2000), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_31"}
  %concatenate.2033 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.2001, f32[1,14,14,16]{3,2,1,0} %convolution.2002, f32[1,14,14,16]{3,2,1,0} %convolution.2013, f32[1,14,14,16]{3,2,1,0} %convolution.2024, f32[1,14,14,16]{3,2,1,0} %convolution.2027, f32[1,14,14,16]{3,2,1,0} %convolution.2028, f32[1,14,14,16]{3,2,1,0} %convolution.2029, f32[1,14,14,16]{3,2,1,0} %convolution.2030, f32[1,14,14,16]{3,2,1,0} %convolution.2031, f32[1,14,14,16]{3,2,1,0} %convolution.2032, f32[1,14,14,16]{3,2,1,0} %convolution.2003, f32[1,14,14,16]{3,2,1,0} %convolution.2004, f32[1,14,14,16]{3,2,1,0} %convolution.2005, f32[1,14,14,16]{3,2,1,0} %convolution.2006, f32[1,14,14,16]{3,2,1,0} %convolution.2007, f32[1,14,14,16]{3,2,1,0} %convolution.2008, f32[1,14,14,16]{3,2,1,0} %convolution.2009, f32[1,14,14,16]{3,2,1,0} %convolution.2010, f32[1,14,14,16]{3,2,1,0} %convolution.2011, f32[1,14,14,16]{3,2,1,0} %convolution.2012, f32[1,14,14,16]{3,2,1,0} %convolution.2014, f32[1,14,14,16]{3,2,1,0} %convolution.2015, f32[1,14,14,16]{3,2,1,0} %convolution.2016, f32[1,14,14,16]{3,2,1,0} %convolution.2017, f32[1,14,14,16]{3,2,1,0} %convolution.2018, f32[1,14,14,16]{3,2,1,0} %convolution.2019, f32[1,14,14,16]{3,2,1,0} %convolution.2020, f32[1,14,14,16]{3,2,1,0} %convolution.2021, f32[1,14,14,16]{3,2,1,0} %convolution.2022, f32[1,14,14,16]{3,2,1,0} %convolution.2023, f32[1,14,14,16]{3,2,1,0} %convolution.2025, f32[1,14,14,16]{3,2,1,0} %convolution.2026), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_9"}
  %constant.1913 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit3_bn2/add"}
  %broadcast.1914 = f32[512]{0} broadcast(f32[] %constant.1913), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit3_bn2/add"}
  %arg58.59 = f32[512]{0} parameter(58), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.328 = f32[512]{0} reshape(f32[512]{0} %arg58.59)
  %add.1915 = f32[512]{0} add(f32[512]{0} %broadcast.1914, f32[512]{0} %reshape.328), metadata={op_type="AddV2" op_name="stage3_unit3_bn2/add"}
  %rsqrt.1916 = f32[512]{0} rsqrt(f32[512]{0} %add.1915), metadata={op_type="Rsqrt" op_name="stage3_unit3_bn2/Rsqrt"}
  %arg114.115 = f32[512]{0} parameter(114), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.384 = f32[512]{0} reshape(f32[512]{0} %arg114.115)
  %multiply.1917 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1916, f32[512]{0} %reshape.384), metadata={op_type="Mul" op_name="stage3_unit3_bn2/mul"}
  %broadcast.2034 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1917), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit3_bn2/mul_1"}
  %multiply.2035 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.2033, f32[1,14,14,512]{3,2,1,0} %broadcast.2034), metadata={op_type="Mul" op_name="stage3_unit3_bn2/mul_1"}
  %arg221.222 = f32[512]{0} parameter(221), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.491 = f32[512]{0} reshape(f32[512]{0} %arg221.222)
  %arg168.169 = f32[512]{0} parameter(168), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.438 = f32[512]{0} reshape(f32[512]{0} %arg168.169)
  %multiply.1918 = f32[512]{0} multiply(f32[512]{0} %multiply.1917, f32[512]{0} %reshape.438), metadata={op_type="Mul" op_name="stage3_unit3_bn2/mul_2"}
  %subtract.1919 = f32[512]{0} subtract(f32[512]{0} %reshape.491, f32[512]{0} %multiply.1918), metadata={op_type="Sub" op_name="stage3_unit3_bn2/sub"}
  %broadcast.2036 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1919), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit3_bn2/add_1"}
  %add.2037 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2035, f32[1,14,14,512]{3,2,1,0} %broadcast.2036), metadata={op_type="AddV2" op_name="stage3_unit3_bn2/add_1"}
  %maximum.2040 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2039, f32[1,14,14,512]{3,2,1,0} %add.2037), metadata={op_type="Relu" op_name="stage3_unit3_relu2"}
  %arg254.255 = f32[1,1,512,1024]{3,2,1,0} parameter(254), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.524 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg254.255)
  %convolution.2041 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.2040, f32[1,1,512,1024]{3,2,1,0} %reshape.524), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv3"}
  %multiply.2043 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.2042, f32[1,14,14,1024]{3,2,1,0} %convolution.2041), metadata={op_type="Mul" op_name="stage3_unit3_bn3/mul_1"}
  %arg222.223 = f32[1024]{0} parameter(222), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.492 = f32[1024]{0} reshape(f32[1024]{0} %arg222.223)
  %arg169.170 = f32[1024]{0} parameter(169), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.439 = f32[1024]{0} reshape(f32[1024]{0} %arg169.170)
  %multiply.1925 = f32[1024]{0} multiply(f32[1024]{0} %multiply.1924, f32[1024]{0} %reshape.439), metadata={op_type="Mul" op_name="stage3_unit3_bn3/mul_2"}
  %subtract.1926 = f32[1024]{0} subtract(f32[1024]{0} %reshape.492, f32[1024]{0} %multiply.1925), metadata={op_type="Sub" op_name="stage3_unit3_bn3/sub"}
  %broadcast.2044 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.1926), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit3_bn3/add_1"}
  %add.2045 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.2043, f32[1,14,14,1024]{3,2,1,0} %broadcast.2044), metadata={op_type="AddV2" op_name="stage3_unit3_bn3/add_1"}
  %add.2046 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %maximum.1905, f32[1,14,14,1024]{3,2,1,0} %add.2045), metadata={op_type="AddV2" op_name="add_9"}
  %maximum.2049 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.2048, f32[1,14,14,1024]{3,2,1,0} %add.2046), metadata={op_type="Relu" op_name="stage3_unit3_relu"}
  %constant.2064 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit4_bn3/add"}
  %broadcast.2065 = f32[1024]{0} broadcast(f32[] %constant.2064), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit4_bn3/add"}
  %arg64.65 = f32[1024]{0} parameter(64), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.334 = f32[1024]{0} reshape(f32[1024]{0} %arg64.65)
  %add.2066 = f32[1024]{0} add(f32[1024]{0} %broadcast.2065, f32[1024]{0} %reshape.334), metadata={op_type="AddV2" op_name="stage3_unit4_bn3/add"}
  %rsqrt.2067 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2066), metadata={op_type="Rsqrt" op_name="stage3_unit4_bn3/Rsqrt"}
  %arg119.120 = f32[1024]{0} parameter(119), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.389 = f32[1024]{0} reshape(f32[1024]{0} %arg119.120)
  %multiply.2068 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2067, f32[1024]{0} %reshape.389), metadata={op_type="Mul" op_name="stage3_unit4_bn3/mul"}
  %broadcast.2186 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2068), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit4_bn3/mul_1"}
  %constant.2182 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit4_relu2"}
  %broadcast.2183 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2182), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit4_relu2"}
  %constant.2076 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit4_relu1"}
  %broadcast.2077 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2076), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit4_relu1"}
  %constant.2050 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit4_bn1/add"}
  %broadcast.2051 = f32[512]{0} broadcast(f32[] %constant.2050), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit4_bn1/add"}
  %arg60.61 = f32[512]{0} parameter(60), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.330 = f32[512]{0} reshape(f32[512]{0} %arg60.61)
  %add.2052 = f32[512]{0} add(f32[512]{0} %broadcast.2051, f32[512]{0} %reshape.330), metadata={op_type="AddV2" op_name="stage3_unit4_bn1/add"}
  %rsqrt.2053 = f32[512]{0} rsqrt(f32[512]{0} %add.2052), metadata={op_type="Rsqrt" op_name="stage3_unit4_bn1/Rsqrt"}
  %arg116.117 = f32[512]{0} parameter(116), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.386 = f32[512]{0} reshape(f32[512]{0} %arg116.117)
  %multiply.2054 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2053, f32[512]{0} %reshape.386), metadata={op_type="Mul" op_name="stage3_unit4_bn1/mul"}
  %broadcast.2072 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2054), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit4_bn1/mul_1"}
  %arg255.256 = f32[1,1,1024,512]{3,2,1,0} parameter(255), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.525 = f32[1,1,1024,512]{3,2,1,0} reshape(f32[1,1,1024,512]{3,2,1,0} %arg255.256)
  %convolution.2071 = f32[1,14,14,512]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.2049, f32[1,1,1024,512]{3,2,1,0} %reshape.525), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv1"}
  %multiply.2073 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %broadcast.2072, f32[1,14,14,512]{3,2,1,0} %convolution.2071), metadata={op_type="Mul" op_name="stage3_unit4_bn1/mul_1"}
  %arg223.224 = f32[512]{0} parameter(223), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.493 = f32[512]{0} reshape(f32[512]{0} %arg223.224)
  %arg170.171 = f32[512]{0} parameter(170), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.440 = f32[512]{0} reshape(f32[512]{0} %arg170.171)
  %multiply.2055 = f32[512]{0} multiply(f32[512]{0} %multiply.2054, f32[512]{0} %reshape.440), metadata={op_type="Mul" op_name="stage3_unit4_bn1/mul_2"}
  %subtract.2056 = f32[512]{0} subtract(f32[512]{0} %reshape.493, f32[512]{0} %multiply.2055), metadata={op_type="Sub" op_name="stage3_unit4_bn1/sub"}
  %broadcast.2074 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2056), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit4_bn1/add_1"}
  %add.2075 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2073, f32[1,14,14,512]{3,2,1,0} %broadcast.2074), metadata={op_type="AddV2" op_name="stage3_unit4_bn1/add_1"}
  %maximum.2078 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2077, f32[1,14,14,512]{3,2,1,0} %add.2075), metadata={op_type="Relu" op_name="stage3_unit4_relu1"}
  %constant.2079 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_11"}
  %pad.2080 = f32[1,16,16,512]{3,2,1,0} pad(f32[1,14,14,512]{3,2,1,0} %maximum.2078, f32[] %constant.2079), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_11"}
  %slice.2081 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_21"}
  %arg62.63 = f32[3,3,16,512]{3,2,1,0} parameter(62), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.332 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg62.63)
  %slice.2113 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2145 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2081, f32[3,3,16,16]{3,2,1,0} %slice.2113), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2"}
  %slice.2082 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2114 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2146 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2082, f32[3,3,16,16]{3,2,1,0} %slice.2114), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_1"}
  %slice.2083 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2115 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2157 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2083, f32[3,3,16,16]{3,2,1,0} %slice.2115), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_2"}
  %slice.2084 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2116 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2168 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2084, f32[3,3,16,16]{3,2,1,0} %slice.2116), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_3"}
  %slice.2085 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2117 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2171 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2085, f32[3,3,16,16]{3,2,1,0} %slice.2117), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_4"}
  %slice.2086 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2118 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2172 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2086, f32[3,3,16,16]{3,2,1,0} %slice.2118), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_5"}
  %slice.2087 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2119 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2173 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2087, f32[3,3,16,16]{3,2,1,0} %slice.2119), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_6"}
  %slice.2088 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2120 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2174 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2088, f32[3,3,16,16]{3,2,1,0} %slice.2120), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_7"}
  %slice.2089 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2121 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2175 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2089, f32[3,3,16,16]{3,2,1,0} %slice.2121), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_8"}
  %slice.2090 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2122 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2176 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2090, f32[3,3,16,16]{3,2,1,0} %slice.2122), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_9"}
  %slice.2091 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2123 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2147 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2091, f32[3,3,16,16]{3,2,1,0} %slice.2123), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_10"}
  %slice.2092 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2124 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2148 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2092, f32[3,3,16,16]{3,2,1,0} %slice.2124), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_11"}
  %slice.2093 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2125 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2149 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2093, f32[3,3,16,16]{3,2,1,0} %slice.2125), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_12"}
  %slice.2094 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2126 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2150 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2094, f32[3,3,16,16]{3,2,1,0} %slice.2126), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_13"}
  %slice.2095 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2127 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2151 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2095, f32[3,3,16,16]{3,2,1,0} %slice.2127), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_14"}
  %slice.2096 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2128 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2152 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2096, f32[3,3,16,16]{3,2,1,0} %slice.2128), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_15"}
  %slice.2097 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2129 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2153 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2097, f32[3,3,16,16]{3,2,1,0} %slice.2129), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_16"}
  %slice.2098 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2130 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2154 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2098, f32[3,3,16,16]{3,2,1,0} %slice.2130), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_17"}
  %slice.2099 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2131 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2155 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2099, f32[3,3,16,16]{3,2,1,0} %slice.2131), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_18"}
  %slice.2100 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2132 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2156 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2100, f32[3,3,16,16]{3,2,1,0} %slice.2132), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_19"}
  %slice.2101 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2133 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2158 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2101, f32[3,3,16,16]{3,2,1,0} %slice.2133), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_20"}
  %slice.2102 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2134 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2159 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2102, f32[3,3,16,16]{3,2,1,0} %slice.2134), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_21"}
  %slice.2103 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2135 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2160 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2103, f32[3,3,16,16]{3,2,1,0} %slice.2135), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_22"}
  %slice.2104 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2136 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2161 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2104, f32[3,3,16,16]{3,2,1,0} %slice.2136), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_23"}
  %slice.2105 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2137 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2162 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2105, f32[3,3,16,16]{3,2,1,0} %slice.2137), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_24"}
  %slice.2106 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2138 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2163 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2106, f32[3,3,16,16]{3,2,1,0} %slice.2138), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_25"}
  %slice.2107 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2139 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2164 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2107, f32[3,3,16,16]{3,2,1,0} %slice.2139), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_26"}
  %slice.2108 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2140 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2165 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2108, f32[3,3,16,16]{3,2,1,0} %slice.2140), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_27"}
  %slice.2109 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2141 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2166 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2109, f32[3,3,16,16]{3,2,1,0} %slice.2141), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_28"}
  %slice.2110 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2142 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2167 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2110, f32[3,3,16,16]{3,2,1,0} %slice.2142), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_29"}
  %slice.2111 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2143 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2169 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2111, f32[3,3,16,16]{3,2,1,0} %slice.2143), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_30"}
  %slice.2112 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2080), slice={[0:1], [0:16], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2144 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.332), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2170 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2112, f32[3,3,16,16]{3,2,1,0} %slice.2144), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_31"}
  %concatenate.2177 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.2145, f32[1,14,14,16]{3,2,1,0} %convolution.2146, f32[1,14,14,16]{3,2,1,0} %convolution.2157, f32[1,14,14,16]{3,2,1,0} %convolution.2168, f32[1,14,14,16]{3,2,1,0} %convolution.2171, f32[1,14,14,16]{3,2,1,0} %convolution.2172, f32[1,14,14,16]{3,2,1,0} %convolution.2173, f32[1,14,14,16]{3,2,1,0} %convolution.2174, f32[1,14,14,16]{3,2,1,0} %convolution.2175, f32[1,14,14,16]{3,2,1,0} %convolution.2176, f32[1,14,14,16]{3,2,1,0} %convolution.2147, f32[1,14,14,16]{3,2,1,0} %convolution.2148, f32[1,14,14,16]{3,2,1,0} %convolution.2149, f32[1,14,14,16]{3,2,1,0} %convolution.2150, f32[1,14,14,16]{3,2,1,0} %convolution.2151, f32[1,14,14,16]{3,2,1,0} %convolution.2152, f32[1,14,14,16]{3,2,1,0} %convolution.2153, f32[1,14,14,16]{3,2,1,0} %convolution.2154, f32[1,14,14,16]{3,2,1,0} %convolution.2155, f32[1,14,14,16]{3,2,1,0} %convolution.2156, f32[1,14,14,16]{3,2,1,0} %convolution.2158, f32[1,14,14,16]{3,2,1,0} %convolution.2159, f32[1,14,14,16]{3,2,1,0} %convolution.2160, f32[1,14,14,16]{3,2,1,0} %convolution.2161, f32[1,14,14,16]{3,2,1,0} %convolution.2162, f32[1,14,14,16]{3,2,1,0} %convolution.2163, f32[1,14,14,16]{3,2,1,0} %convolution.2164, f32[1,14,14,16]{3,2,1,0} %convolution.2165, f32[1,14,14,16]{3,2,1,0} %convolution.2166, f32[1,14,14,16]{3,2,1,0} %convolution.2167, f32[1,14,14,16]{3,2,1,0} %convolution.2169, f32[1,14,14,16]{3,2,1,0} %convolution.2170), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_10"}
  %constant.2057 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit4_bn2/add"}
  %broadcast.2058 = f32[512]{0} broadcast(f32[] %constant.2057), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit4_bn2/add"}
  %arg63.64 = f32[512]{0} parameter(63), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.333 = f32[512]{0} reshape(f32[512]{0} %arg63.64)
  %add.2059 = f32[512]{0} add(f32[512]{0} %broadcast.2058, f32[512]{0} %reshape.333), metadata={op_type="AddV2" op_name="stage3_unit4_bn2/add"}
  %rsqrt.2060 = f32[512]{0} rsqrt(f32[512]{0} %add.2059), metadata={op_type="Rsqrt" op_name="stage3_unit4_bn2/Rsqrt"}
  %arg118.119 = f32[512]{0} parameter(118), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.388 = f32[512]{0} reshape(f32[512]{0} %arg118.119)
  %multiply.2061 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2060, f32[512]{0} %reshape.388), metadata={op_type="Mul" op_name="stage3_unit4_bn2/mul"}
  %broadcast.2178 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2061), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit4_bn2/mul_1"}
  %multiply.2179 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.2177, f32[1,14,14,512]{3,2,1,0} %broadcast.2178), metadata={op_type="Mul" op_name="stage3_unit4_bn2/mul_1"}
  %arg225.226 = f32[512]{0} parameter(225), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.495 = f32[512]{0} reshape(f32[512]{0} %arg225.226)
  %arg172.173 = f32[512]{0} parameter(172), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.442 = f32[512]{0} reshape(f32[512]{0} %arg172.173)
  %multiply.2062 = f32[512]{0} multiply(f32[512]{0} %multiply.2061, f32[512]{0} %reshape.442), metadata={op_type="Mul" op_name="stage3_unit4_bn2/mul_2"}
  %subtract.2063 = f32[512]{0} subtract(f32[512]{0} %reshape.495, f32[512]{0} %multiply.2062), metadata={op_type="Sub" op_name="stage3_unit4_bn2/sub"}
  %broadcast.2180 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2063), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit4_bn2/add_1"}
  %add.2181 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2179, f32[1,14,14,512]{3,2,1,0} %broadcast.2180), metadata={op_type="AddV2" op_name="stage3_unit4_bn2/add_1"}
  %maximum.2184 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2183, f32[1,14,14,512]{3,2,1,0} %add.2181), metadata={op_type="Relu" op_name="stage3_unit4_relu2"}
  %arg256.257 = f32[1,1,512,1024]{3,2,1,0} parameter(256), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.526 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg256.257)
  %convolution.2185 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.2184, f32[1,1,512,1024]{3,2,1,0} %reshape.526), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv3"}
  %multiply.2187 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.2186, f32[1,14,14,1024]{3,2,1,0} %convolution.2185), metadata={op_type="Mul" op_name="stage3_unit4_bn3/mul_1"}
  %arg226.227 = f32[1024]{0} parameter(226), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.496 = f32[1024]{0} reshape(f32[1024]{0} %arg226.227)
  %arg173.174 = f32[1024]{0} parameter(173), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.443 = f32[1024]{0} reshape(f32[1024]{0} %arg173.174)
  %multiply.2069 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2068, f32[1024]{0} %reshape.443), metadata={op_type="Mul" op_name="stage3_unit4_bn3/mul_2"}
  %subtract.2070 = f32[1024]{0} subtract(f32[1024]{0} %reshape.496, f32[1024]{0} %multiply.2069), metadata={op_type="Sub" op_name="stage3_unit4_bn3/sub"}
  %broadcast.2188 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2070), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit4_bn3/add_1"}
  %add.2189 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.2187, f32[1,14,14,1024]{3,2,1,0} %broadcast.2188), metadata={op_type="AddV2" op_name="stage3_unit4_bn3/add_1"}
  %add.2190 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %maximum.2049, f32[1,14,14,1024]{3,2,1,0} %add.2189), metadata={op_type="AddV2" op_name="add_10"}
  %maximum.2193 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.2192, f32[1,14,14,1024]{3,2,1,0} %add.2190), metadata={op_type="Relu" op_name="stage3_unit4_relu"}
  %constant.2208 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit5_bn3/add"}
  %broadcast.2209 = f32[1024]{0} broadcast(f32[] %constant.2208), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit5_bn3/add"}
  %arg69.70 = f32[1024]{0} parameter(69), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.339 = f32[1024]{0} reshape(f32[1024]{0} %arg69.70)
  %add.2210 = f32[1024]{0} add(f32[1024]{0} %broadcast.2209, f32[1024]{0} %reshape.339), metadata={op_type="AddV2" op_name="stage3_unit5_bn3/add"}
  %rsqrt.2211 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2210), metadata={op_type="Rsqrt" op_name="stage3_unit5_bn3/Rsqrt"}
  %arg123.124 = f32[1024]{0} parameter(123), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.393 = f32[1024]{0} reshape(f32[1024]{0} %arg123.124)
  %multiply.2212 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2211, f32[1024]{0} %reshape.393), metadata={op_type="Mul" op_name="stage3_unit5_bn3/mul"}
  %broadcast.2330 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2212), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit5_bn3/mul_1"}
  %constant.2326 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit5_relu2"}
  %broadcast.2327 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2326), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit5_relu2"}
  %constant.2220 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit5_relu1"}
  %broadcast.2221 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2220), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit5_relu1"}
  %constant.2194 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit5_bn1/add"}
  %broadcast.2195 = f32[512]{0} broadcast(f32[] %constant.2194), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit5_bn1/add"}
  %arg65.66 = f32[512]{0} parameter(65), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.335 = f32[512]{0} reshape(f32[512]{0} %arg65.66)
  %add.2196 = f32[512]{0} add(f32[512]{0} %broadcast.2195, f32[512]{0} %reshape.335), metadata={op_type="AddV2" op_name="stage3_unit5_bn1/add"}
  %rsqrt.2197 = f32[512]{0} rsqrt(f32[512]{0} %add.2196), metadata={op_type="Rsqrt" op_name="stage3_unit5_bn1/Rsqrt"}
  %arg120.121 = f32[512]{0} parameter(120), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.390 = f32[512]{0} reshape(f32[512]{0} %arg120.121)
  %multiply.2198 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2197, f32[512]{0} %reshape.390), metadata={op_type="Mul" op_name="stage3_unit5_bn1/mul"}
  %broadcast.2216 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2198), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit5_bn1/mul_1"}
  %arg257.258 = f32[1,1,1024,512]{3,2,1,0} parameter(257), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.527 = f32[1,1,1024,512]{3,2,1,0} reshape(f32[1,1,1024,512]{3,2,1,0} %arg257.258)
  %convolution.2215 = f32[1,14,14,512]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.2193, f32[1,1,1024,512]{3,2,1,0} %reshape.527), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv1"}
  %multiply.2217 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %broadcast.2216, f32[1,14,14,512]{3,2,1,0} %convolution.2215), metadata={op_type="Mul" op_name="stage3_unit5_bn1/mul_1"}
  %arg227.228 = f32[512]{0} parameter(227), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.497 = f32[512]{0} reshape(f32[512]{0} %arg227.228)
  %arg174.175 = f32[512]{0} parameter(174), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.444 = f32[512]{0} reshape(f32[512]{0} %arg174.175)
  %multiply.2199 = f32[512]{0} multiply(f32[512]{0} %multiply.2198, f32[512]{0} %reshape.444), metadata={op_type="Mul" op_name="stage3_unit5_bn1/mul_2"}
  %subtract.2200 = f32[512]{0} subtract(f32[512]{0} %reshape.497, f32[512]{0} %multiply.2199), metadata={op_type="Sub" op_name="stage3_unit5_bn1/sub"}
  %broadcast.2218 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2200), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit5_bn1/add_1"}
  %add.2219 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2217, f32[1,14,14,512]{3,2,1,0} %broadcast.2218), metadata={op_type="AddV2" op_name="stage3_unit5_bn1/add_1"}
  %maximum.2222 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2221, f32[1,14,14,512]{3,2,1,0} %add.2219), metadata={op_type="Relu" op_name="stage3_unit5_relu1"}
  %constant.2223 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_12"}
  %pad.2224 = f32[1,16,16,512]{3,2,1,0} pad(f32[1,14,14,512]{3,2,1,0} %maximum.2222, f32[] %constant.2223), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_12"}
  %slice.2225 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_23"}
  %arg67.68 = f32[3,3,16,512]{3,2,1,0} parameter(67), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.337 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg67.68)
  %slice.2257 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2289 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2225, f32[3,3,16,16]{3,2,1,0} %slice.2257), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2"}
  %slice.2226 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2258 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2290 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2226, f32[3,3,16,16]{3,2,1,0} %slice.2258), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_1"}
  %slice.2227 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2259 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2301 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2227, f32[3,3,16,16]{3,2,1,0} %slice.2259), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_2"}
  %slice.2228 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2260 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2312 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2228, f32[3,3,16,16]{3,2,1,0} %slice.2260), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_3"}
  %slice.2229 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2261 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2315 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2229, f32[3,3,16,16]{3,2,1,0} %slice.2261), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_4"}
  %slice.2230 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2262 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2316 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2230, f32[3,3,16,16]{3,2,1,0} %slice.2262), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_5"}
  %slice.2231 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2263 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2317 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2231, f32[3,3,16,16]{3,2,1,0} %slice.2263), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_6"}
  %slice.2232 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2264 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2318 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2232, f32[3,3,16,16]{3,2,1,0} %slice.2264), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_7"}
  %slice.2233 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2265 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2319 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2233, f32[3,3,16,16]{3,2,1,0} %slice.2265), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_8"}
  %slice.2234 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2266 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2320 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2234, f32[3,3,16,16]{3,2,1,0} %slice.2266), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_9"}
  %slice.2235 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2267 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2291 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2235, f32[3,3,16,16]{3,2,1,0} %slice.2267), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_10"}
  %slice.2236 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2268 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2292 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2236, f32[3,3,16,16]{3,2,1,0} %slice.2268), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_11"}
  %slice.2237 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2269 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2293 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2237, f32[3,3,16,16]{3,2,1,0} %slice.2269), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_12"}
  %slice.2238 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2270 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2294 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2238, f32[3,3,16,16]{3,2,1,0} %slice.2270), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_13"}
  %slice.2239 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2271 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2295 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2239, f32[3,3,16,16]{3,2,1,0} %slice.2271), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_14"}
  %slice.2240 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2272 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2296 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2240, f32[3,3,16,16]{3,2,1,0} %slice.2272), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_15"}
  %slice.2241 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2273 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2297 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2241, f32[3,3,16,16]{3,2,1,0} %slice.2273), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_16"}
  %slice.2242 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2274 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2298 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2242, f32[3,3,16,16]{3,2,1,0} %slice.2274), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_17"}
  %slice.2243 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2275 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2299 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2243, f32[3,3,16,16]{3,2,1,0} %slice.2275), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_18"}
  %slice.2244 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2276 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2300 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2244, f32[3,3,16,16]{3,2,1,0} %slice.2276), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_19"}
  %slice.2245 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2277 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2302 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2245, f32[3,3,16,16]{3,2,1,0} %slice.2277), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_20"}
  %slice.2246 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2278 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2303 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2246, f32[3,3,16,16]{3,2,1,0} %slice.2278), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_21"}
  %slice.2247 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2279 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2304 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2247, f32[3,3,16,16]{3,2,1,0} %slice.2279), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_22"}
  %slice.2248 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2280 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2305 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2248, f32[3,3,16,16]{3,2,1,0} %slice.2280), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_23"}
  %slice.2249 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2281 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2306 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2249, f32[3,3,16,16]{3,2,1,0} %slice.2281), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_24"}
  %slice.2250 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2282 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2307 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2250, f32[3,3,16,16]{3,2,1,0} %slice.2282), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_25"}
  %slice.2251 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2283 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2308 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2251, f32[3,3,16,16]{3,2,1,0} %slice.2283), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_26"}
  %slice.2252 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2284 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2309 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2252, f32[3,3,16,16]{3,2,1,0} %slice.2284), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_27"}
  %slice.2253 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2285 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2310 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2253, f32[3,3,16,16]{3,2,1,0} %slice.2285), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_28"}
  %slice.2254 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2286 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2311 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2254, f32[3,3,16,16]{3,2,1,0} %slice.2286), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_29"}
  %slice.2255 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2287 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2313 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2255, f32[3,3,16,16]{3,2,1,0} %slice.2287), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_30"}
  %slice.2256 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2224), slice={[0:1], [0:16], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2288 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.337), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2314 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2256, f32[3,3,16,16]{3,2,1,0} %slice.2288), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_31"}
  %concatenate.2321 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.2289, f32[1,14,14,16]{3,2,1,0} %convolution.2290, f32[1,14,14,16]{3,2,1,0} %convolution.2301, f32[1,14,14,16]{3,2,1,0} %convolution.2312, f32[1,14,14,16]{3,2,1,0} %convolution.2315, f32[1,14,14,16]{3,2,1,0} %convolution.2316, f32[1,14,14,16]{3,2,1,0} %convolution.2317, f32[1,14,14,16]{3,2,1,0} %convolution.2318, f32[1,14,14,16]{3,2,1,0} %convolution.2319, f32[1,14,14,16]{3,2,1,0} %convolution.2320, f32[1,14,14,16]{3,2,1,0} %convolution.2291, f32[1,14,14,16]{3,2,1,0} %convolution.2292, f32[1,14,14,16]{3,2,1,0} %convolution.2293, f32[1,14,14,16]{3,2,1,0} %convolution.2294, f32[1,14,14,16]{3,2,1,0} %convolution.2295, f32[1,14,14,16]{3,2,1,0} %convolution.2296, f32[1,14,14,16]{3,2,1,0} %convolution.2297, f32[1,14,14,16]{3,2,1,0} %convolution.2298, f32[1,14,14,16]{3,2,1,0} %convolution.2299, f32[1,14,14,16]{3,2,1,0} %convolution.2300, f32[1,14,14,16]{3,2,1,0} %convolution.2302, f32[1,14,14,16]{3,2,1,0} %convolution.2303, f32[1,14,14,16]{3,2,1,0} %convolution.2304, f32[1,14,14,16]{3,2,1,0} %convolution.2305, f32[1,14,14,16]{3,2,1,0} %convolution.2306, f32[1,14,14,16]{3,2,1,0} %convolution.2307, f32[1,14,14,16]{3,2,1,0} %convolution.2308, f32[1,14,14,16]{3,2,1,0} %convolution.2309, f32[1,14,14,16]{3,2,1,0} %convolution.2310, f32[1,14,14,16]{3,2,1,0} %convolution.2311, f32[1,14,14,16]{3,2,1,0} %convolution.2313, f32[1,14,14,16]{3,2,1,0} %convolution.2314), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_11"}
  %constant.2201 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit5_bn2/add"}
  %broadcast.2202 = f32[512]{0} broadcast(f32[] %constant.2201), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit5_bn2/add"}
  %arg68.69 = f32[512]{0} parameter(68), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.338 = f32[512]{0} reshape(f32[512]{0} %arg68.69)
  %add.2203 = f32[512]{0} add(f32[512]{0} %broadcast.2202, f32[512]{0} %reshape.338), metadata={op_type="AddV2" op_name="stage3_unit5_bn2/add"}
  %rsqrt.2204 = f32[512]{0} rsqrt(f32[512]{0} %add.2203), metadata={op_type="Rsqrt" op_name="stage3_unit5_bn2/Rsqrt"}
  %arg122.123 = f32[512]{0} parameter(122), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.392 = f32[512]{0} reshape(f32[512]{0} %arg122.123)
  %multiply.2205 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2204, f32[512]{0} %reshape.392), metadata={op_type="Mul" op_name="stage3_unit5_bn2/mul"}
  %broadcast.2322 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2205), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit5_bn2/mul_1"}
  %multiply.2323 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.2321, f32[1,14,14,512]{3,2,1,0} %broadcast.2322), metadata={op_type="Mul" op_name="stage3_unit5_bn2/mul_1"}
  %arg229.230 = f32[512]{0} parameter(229), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.499 = f32[512]{0} reshape(f32[512]{0} %arg229.230)
  %arg176.177 = f32[512]{0} parameter(176), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.446 = f32[512]{0} reshape(f32[512]{0} %arg176.177)
  %multiply.2206 = f32[512]{0} multiply(f32[512]{0} %multiply.2205, f32[512]{0} %reshape.446), metadata={op_type="Mul" op_name="stage3_unit5_bn2/mul_2"}
  %subtract.2207 = f32[512]{0} subtract(f32[512]{0} %reshape.499, f32[512]{0} %multiply.2206), metadata={op_type="Sub" op_name="stage3_unit5_bn2/sub"}
  %broadcast.2324 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2207), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit5_bn2/add_1"}
  %add.2325 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2323, f32[1,14,14,512]{3,2,1,0} %broadcast.2324), metadata={op_type="AddV2" op_name="stage3_unit5_bn2/add_1"}
  %maximum.2328 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2327, f32[1,14,14,512]{3,2,1,0} %add.2325), metadata={op_type="Relu" op_name="stage3_unit5_relu2"}
  %arg258.259 = f32[1,1,512,1024]{3,2,1,0} parameter(258), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.528 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg258.259)
  %convolution.2329 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.2328, f32[1,1,512,1024]{3,2,1,0} %reshape.528), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv3"}
  %multiply.2331 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.2330, f32[1,14,14,1024]{3,2,1,0} %convolution.2329), metadata={op_type="Mul" op_name="stage3_unit5_bn3/mul_1"}
  %arg230.231 = f32[1024]{0} parameter(230), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.500 = f32[1024]{0} reshape(f32[1024]{0} %arg230.231)
  %arg177.178 = f32[1024]{0} parameter(177), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.447 = f32[1024]{0} reshape(f32[1024]{0} %arg177.178)
  %multiply.2213 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2212, f32[1024]{0} %reshape.447), metadata={op_type="Mul" op_name="stage3_unit5_bn3/mul_2"}
  %subtract.2214 = f32[1024]{0} subtract(f32[1024]{0} %reshape.500, f32[1024]{0} %multiply.2213), metadata={op_type="Sub" op_name="stage3_unit5_bn3/sub"}
  %broadcast.2332 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2214), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit5_bn3/add_1"}
  %add.2333 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.2331, f32[1,14,14,1024]{3,2,1,0} %broadcast.2332), metadata={op_type="AddV2" op_name="stage3_unit5_bn3/add_1"}
  %add.2334 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %maximum.2193, f32[1,14,14,1024]{3,2,1,0} %add.2333), metadata={op_type="AddV2" op_name="add_11"}
  %maximum.2337 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.2336, f32[1,14,14,1024]{3,2,1,0} %add.2334), metadata={op_type="Relu" op_name="stage3_unit5_relu"}
  %constant.2352 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit6_bn3/add"}
  %broadcast.2353 = f32[1024]{0} broadcast(f32[] %constant.2352), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit6_bn3/add"}
  %arg19.20 = f32[1024]{0} parameter(19), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.289 = f32[1024]{0} reshape(f32[1024]{0} %arg19.20)
  %add.2354 = f32[1024]{0} add(f32[1024]{0} %broadcast.2353, f32[1024]{0} %reshape.289), metadata={op_type="AddV2" op_name="stage3_unit6_bn3/add"}
  %rsqrt.2355 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2354), metadata={op_type="Rsqrt" op_name="stage3_unit6_bn3/Rsqrt"}
  %arg85.86 = f32[1024]{0} parameter(85), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.355 = f32[1024]{0} reshape(f32[1024]{0} %arg85.86)
  %multiply.2356 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2355, f32[1024]{0} %reshape.355), metadata={op_type="Mul" op_name="stage3_unit6_bn3/mul"}
  %broadcast.2474 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2356), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit6_bn3/mul_1"}
  %constant.2470 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit6_relu2"}
  %broadcast.2471 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2470), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit6_relu2"}
  %constant.2364 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit6_relu1"}
  %broadcast.2365 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2364), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit6_relu1"}
  %constant.2338 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit6_bn1/add"}
  %broadcast.2339 = f32[512]{0} broadcast(f32[] %constant.2338), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit6_bn1/add"}
  %arg3.4 = f32[512]{0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.273 = f32[512]{0} reshape(f32[512]{0} %arg3.4)
  %add.2340 = f32[512]{0} add(f32[512]{0} %broadcast.2339, f32[512]{0} %reshape.273), metadata={op_type="AddV2" op_name="stage3_unit6_bn1/add"}
  %rsqrt.2341 = f32[512]{0} rsqrt(f32[512]{0} %add.2340), metadata={op_type="Rsqrt" op_name="stage3_unit6_bn1/Rsqrt"}
  %arg73.74 = f32[512]{0} parameter(73), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.343 = f32[512]{0} reshape(f32[512]{0} %arg73.74)
  %multiply.2342 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2341, f32[512]{0} %reshape.343), metadata={op_type="Mul" op_name="stage3_unit6_bn1/mul"}
  %broadcast.2360 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2342), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit6_bn1/mul_1"}
  %arg259.260 = f32[1,1,1024,512]{3,2,1,0} parameter(259), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.529 = f32[1,1,1024,512]{3,2,1,0} reshape(f32[1,1,1024,512]{3,2,1,0} %arg259.260)
  %convolution.2359 = f32[1,14,14,512]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.2337, f32[1,1,1024,512]{3,2,1,0} %reshape.529), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv1"}
  %multiply.2361 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %broadcast.2360, f32[1,14,14,512]{3,2,1,0} %convolution.2359), metadata={op_type="Mul" op_name="stage3_unit6_bn1/mul_1"}
  %arg180.181 = f32[512]{0} parameter(180), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.450 = f32[512]{0} reshape(f32[512]{0} %arg180.181)
  %arg127.128 = f32[512]{0} parameter(127), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.397 = f32[512]{0} reshape(f32[512]{0} %arg127.128)
  %multiply.2343 = f32[512]{0} multiply(f32[512]{0} %multiply.2342, f32[512]{0} %reshape.397), metadata={op_type="Mul" op_name="stage3_unit6_bn1/mul_2"}
  %subtract.2344 = f32[512]{0} subtract(f32[512]{0} %reshape.450, f32[512]{0} %multiply.2343), metadata={op_type="Sub" op_name="stage3_unit6_bn1/sub"}
  %broadcast.2362 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2344), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit6_bn1/add_1"}
  %add.2363 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2361, f32[1,14,14,512]{3,2,1,0} %broadcast.2362), metadata={op_type="AddV2" op_name="stage3_unit6_bn1/add_1"}
  %maximum.2366 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2365, f32[1,14,14,512]{3,2,1,0} %add.2363), metadata={op_type="Relu" op_name="stage3_unit6_relu1"}
  %constant.2367 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_13"}
  %pad.2368 = f32[1,16,16,512]{3,2,1,0} pad(f32[1,14,14,512]{3,2,1,0} %maximum.2366, f32[] %constant.2367), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_13"}
  %slice.2369 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_25"}
  %arg7.8 = f32[3,3,16,512]{3,2,1,0} parameter(7), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.277 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg7.8)
  %slice.2401 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2433 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2369, f32[3,3,16,16]{3,2,1,0} %slice.2401), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2"}
  %slice.2370 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2402 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2434 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2370, f32[3,3,16,16]{3,2,1,0} %slice.2402), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_1"}
  %slice.2371 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2403 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2445 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2371, f32[3,3,16,16]{3,2,1,0} %slice.2403), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_2"}
  %slice.2372 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2404 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2456 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2372, f32[3,3,16,16]{3,2,1,0} %slice.2404), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_3"}
  %slice.2373 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2405 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2459 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2373, f32[3,3,16,16]{3,2,1,0} %slice.2405), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_4"}
  %slice.2374 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2406 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2460 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2374, f32[3,3,16,16]{3,2,1,0} %slice.2406), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_5"}
  %slice.2375 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2407 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2461 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2375, f32[3,3,16,16]{3,2,1,0} %slice.2407), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_6"}
  %slice.2376 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2408 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2462 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2376, f32[3,3,16,16]{3,2,1,0} %slice.2408), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_7"}
  %slice.2377 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2409 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2463 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2377, f32[3,3,16,16]{3,2,1,0} %slice.2409), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_8"}
  %slice.2378 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2410 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2464 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2378, f32[3,3,16,16]{3,2,1,0} %slice.2410), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_9"}
  %slice.2379 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2411 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2435 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2379, f32[3,3,16,16]{3,2,1,0} %slice.2411), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_10"}
  %slice.2380 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2412 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2436 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2380, f32[3,3,16,16]{3,2,1,0} %slice.2412), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_11"}
  %slice.2381 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2413 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2437 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2381, f32[3,3,16,16]{3,2,1,0} %slice.2413), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_12"}
  %slice.2382 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2414 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2438 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2382, f32[3,3,16,16]{3,2,1,0} %slice.2414), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_13"}
  %slice.2383 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2415 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2439 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2383, f32[3,3,16,16]{3,2,1,0} %slice.2415), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_14"}
  %slice.2384 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2416 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2440 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2384, f32[3,3,16,16]{3,2,1,0} %slice.2416), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_15"}
  %slice.2385 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2417 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2441 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2385, f32[3,3,16,16]{3,2,1,0} %slice.2417), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_16"}
  %slice.2386 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2418 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2442 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2386, f32[3,3,16,16]{3,2,1,0} %slice.2418), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_17"}
  %slice.2387 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2419 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2443 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2387, f32[3,3,16,16]{3,2,1,0} %slice.2419), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_18"}
  %slice.2388 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2420 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2444 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2388, f32[3,3,16,16]{3,2,1,0} %slice.2420), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_19"}
  %slice.2389 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2421 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2446 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2389, f32[3,3,16,16]{3,2,1,0} %slice.2421), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_20"}
  %slice.2390 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2422 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2447 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2390, f32[3,3,16,16]{3,2,1,0} %slice.2422), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_21"}
  %slice.2391 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2423 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2448 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2391, f32[3,3,16,16]{3,2,1,0} %slice.2423), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_22"}
  %slice.2392 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2424 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2449 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2392, f32[3,3,16,16]{3,2,1,0} %slice.2424), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_23"}
  %slice.2393 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2425 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2450 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2393, f32[3,3,16,16]{3,2,1,0} %slice.2425), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_24"}
  %slice.2394 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2426 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2451 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2394, f32[3,3,16,16]{3,2,1,0} %slice.2426), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_25"}
  %slice.2395 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2427 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2452 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2395, f32[3,3,16,16]{3,2,1,0} %slice.2427), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_26"}
  %slice.2396 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2428 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2453 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2396, f32[3,3,16,16]{3,2,1,0} %slice.2428), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_27"}
  %slice.2397 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2429 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2454 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2397, f32[3,3,16,16]{3,2,1,0} %slice.2429), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_28"}
  %slice.2398 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2430 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2455 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2398, f32[3,3,16,16]{3,2,1,0} %slice.2430), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_29"}
  %slice.2399 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2431 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2457 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2399, f32[3,3,16,16]{3,2,1,0} %slice.2431), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_30"}
  %slice.2400 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2368), slice={[0:1], [0:16], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2432 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.277), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2458 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2400, f32[3,3,16,16]{3,2,1,0} %slice.2432), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_31"}
  %concatenate.2465 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.2433, f32[1,14,14,16]{3,2,1,0} %convolution.2434, f32[1,14,14,16]{3,2,1,0} %convolution.2445, f32[1,14,14,16]{3,2,1,0} %convolution.2456, f32[1,14,14,16]{3,2,1,0} %convolution.2459, f32[1,14,14,16]{3,2,1,0} %convolution.2460, f32[1,14,14,16]{3,2,1,0} %convolution.2461, f32[1,14,14,16]{3,2,1,0} %convolution.2462, f32[1,14,14,16]{3,2,1,0} %convolution.2463, f32[1,14,14,16]{3,2,1,0} %convolution.2464, f32[1,14,14,16]{3,2,1,0} %convolution.2435, f32[1,14,14,16]{3,2,1,0} %convolution.2436, f32[1,14,14,16]{3,2,1,0} %convolution.2437, f32[1,14,14,16]{3,2,1,0} %convolution.2438, f32[1,14,14,16]{3,2,1,0} %convolution.2439, f32[1,14,14,16]{3,2,1,0} %convolution.2440, f32[1,14,14,16]{3,2,1,0} %convolution.2441, f32[1,14,14,16]{3,2,1,0} %convolution.2442, f32[1,14,14,16]{3,2,1,0} %convolution.2443, f32[1,14,14,16]{3,2,1,0} %convolution.2444, f32[1,14,14,16]{3,2,1,0} %convolution.2446, f32[1,14,14,16]{3,2,1,0} %convolution.2447, f32[1,14,14,16]{3,2,1,0} %convolution.2448, f32[1,14,14,16]{3,2,1,0} %convolution.2449, f32[1,14,14,16]{3,2,1,0} %convolution.2450, f32[1,14,14,16]{3,2,1,0} %convolution.2451, f32[1,14,14,16]{3,2,1,0} %convolution.2452, f32[1,14,14,16]{3,2,1,0} %convolution.2453, f32[1,14,14,16]{3,2,1,0} %convolution.2454, f32[1,14,14,16]{3,2,1,0} %convolution.2455, f32[1,14,14,16]{3,2,1,0} %convolution.2457, f32[1,14,14,16]{3,2,1,0} %convolution.2458), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_12"}
  %constant.2345 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit6_bn2/add"}
  %broadcast.2346 = f32[512]{0} broadcast(f32[] %constant.2345), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit6_bn2/add"}
  %arg14.15 = f32[512]{0} parameter(14), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.284 = f32[512]{0} reshape(f32[512]{0} %arg14.15)
  %add.2347 = f32[512]{0} add(f32[512]{0} %broadcast.2346, f32[512]{0} %reshape.284), metadata={op_type="AddV2" op_name="stage3_unit6_bn2/add"}
  %rsqrt.2348 = f32[512]{0} rsqrt(f32[512]{0} %add.2347), metadata={op_type="Rsqrt" op_name="stage3_unit6_bn2/Rsqrt"}
  %arg81.82 = f32[512]{0} parameter(81), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.351 = f32[512]{0} reshape(f32[512]{0} %arg81.82)
  %multiply.2349 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2348, f32[512]{0} %reshape.351), metadata={op_type="Mul" op_name="stage3_unit6_bn2/mul"}
  %broadcast.2466 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2349), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit6_bn2/mul_1"}
  %multiply.2467 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.2465, f32[1,14,14,512]{3,2,1,0} %broadcast.2466), metadata={op_type="Mul" op_name="stage3_unit6_bn2/mul_1"}
  %arg188.189 = f32[512]{0} parameter(188), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.458 = f32[512]{0} reshape(f32[512]{0} %arg188.189)
  %arg135.136 = f32[512]{0} parameter(135), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.405 = f32[512]{0} reshape(f32[512]{0} %arg135.136)
  %multiply.2350 = f32[512]{0} multiply(f32[512]{0} %multiply.2349, f32[512]{0} %reshape.405), metadata={op_type="Mul" op_name="stage3_unit6_bn2/mul_2"}
  %subtract.2351 = f32[512]{0} subtract(f32[512]{0} %reshape.458, f32[512]{0} %multiply.2350), metadata={op_type="Sub" op_name="stage3_unit6_bn2/sub"}
  %broadcast.2468 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2351), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit6_bn2/add_1"}
  %add.2469 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2467, f32[1,14,14,512]{3,2,1,0} %broadcast.2468), metadata={op_type="AddV2" op_name="stage3_unit6_bn2/add_1"}
  %maximum.2472 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2471, f32[1,14,14,512]{3,2,1,0} %add.2469), metadata={op_type="Relu" op_name="stage3_unit6_relu2"}
  %arg260.261 = f32[1,1,512,1024]{3,2,1,0} parameter(260), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.530 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg260.261)
  %convolution.2473 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.2472, f32[1,1,512,1024]{3,2,1,0} %reshape.530), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv3"}
  %multiply.2475 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.2474, f32[1,14,14,1024]{3,2,1,0} %convolution.2473), metadata={op_type="Mul" op_name="stage3_unit6_bn3/mul_1"}
  %arg192.193 = f32[1024]{0} parameter(192), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.462 = f32[1024]{0} reshape(f32[1024]{0} %arg192.193)
  %arg139.140 = f32[1024]{0} parameter(139), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.409 = f32[1024]{0} reshape(f32[1024]{0} %arg139.140)
  %multiply.2357 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2356, f32[1024]{0} %reshape.409), metadata={op_type="Mul" op_name="stage3_unit6_bn3/mul_2"}
  %subtract.2358 = f32[1024]{0} subtract(f32[1024]{0} %reshape.462, f32[1024]{0} %multiply.2357), metadata={op_type="Sub" op_name="stage3_unit6_bn3/sub"}
  %broadcast.2476 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2358), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit6_bn3/add_1"}
  %add.2477 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.2475, f32[1,14,14,1024]{3,2,1,0} %broadcast.2476), metadata={op_type="AddV2" op_name="stage3_unit6_bn3/add_1"}
  %add.2478 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %maximum.2337, f32[1,14,14,1024]{3,2,1,0} %add.2477), metadata={op_type="AddV2" op_name="add_12"}
  %maximum.2481 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.2480, f32[1,14,14,1024]{3,2,1,0} %add.2478), metadata={op_type="Relu" op_name="stage3_unit6_relu"}
  %arg261.262 = f32[1,1,1024,1024]{3,2,1,0} parameter(261), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.531 = f32[1,1,1024,1024]{3,2,1,0} reshape(f32[1,1,1024,1024]{3,2,1,0} %arg261.262)
  %convolution.2503 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.2481, f32[1,1,1024,1024]{3,2,1,0} %reshape.531), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv1"}
  %multiply.2505 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.2504, f32[1,14,14,1024]{3,2,1,0} %convolution.2503), metadata={op_type="Mul" op_name="stage4_unit1_bn1/mul_1"}
  %arg197.198 = f32[1024]{0} parameter(197), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.467 = f32[1024]{0} reshape(f32[1024]{0} %arg197.198)
  %arg144.145 = f32[1024]{0} parameter(144), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.414 = f32[1024]{0} reshape(f32[1024]{0} %arg144.145)
  %multiply.2487 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2486, f32[1024]{0} %reshape.414), metadata={op_type="Mul" op_name="stage4_unit1_bn1/mul_2"}
  %subtract.2488 = f32[1024]{0} subtract(f32[1024]{0} %reshape.467, f32[1024]{0} %multiply.2487), metadata={op_type="Sub" op_name="stage4_unit1_bn1/sub"}
  %broadcast.2506 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2488), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit1_bn1/add_1"}
  %add.2507 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.2505, f32[1,14,14,1024]{3,2,1,0} %broadcast.2506), metadata={op_type="AddV2" op_name="stage4_unit1_bn1/add_1"}
  %maximum.2510 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.2509, f32[1,14,14,1024]{3,2,1,0} %add.2507), metadata={op_type="Relu" op_name="stage4_unit1_relu1"}
  %constant.2511 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_14"}
  %pad.2512 = f32[1,16,16,1024]{3,2,1,0} pad(f32[1,14,14,1024]{3,2,1,0} %maximum.2510, f32[] %constant.2511), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_14"}
  %slice.2513 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [0:32]}, metadata={op_type="Split" op_name="split_27"}
  %arg33.34 = f32[3,3,32,1024]{3,2,1,0} parameter(33), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.303 = f32[3,3,32,1024]{3,2,1,0} reshape(f32[3,3,32,1024]{3,2,1,0} %arg33.34)
  %slice.2545 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [0:32]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2577 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2513, f32[3,3,32,32]{3,2,1,0} %slice.2545), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2"}
  %slice.2514 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [32:64]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2546 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [32:64]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2578 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2514, f32[3,3,32,32]{3,2,1,0} %slice.2546), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_1"}
  %slice.2515 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [64:96]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2547 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [64:96]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2589 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2515, f32[3,3,32,32]{3,2,1,0} %slice.2547), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_2"}
  %slice.2516 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [96:128]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2548 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [96:128]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2600 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2516, f32[3,3,32,32]{3,2,1,0} %slice.2548), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_3"}
  %slice.2517 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [128:160]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2549 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [128:160]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2603 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2517, f32[3,3,32,32]{3,2,1,0} %slice.2549), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_4"}
  %slice.2518 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [160:192]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2550 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [160:192]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2604 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2518, f32[3,3,32,32]{3,2,1,0} %slice.2550), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_5"}
  %slice.2519 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [192:224]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2551 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [192:224]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2605 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2519, f32[3,3,32,32]{3,2,1,0} %slice.2551), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_6"}
  %slice.2520 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [224:256]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2552 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [224:256]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2606 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2520, f32[3,3,32,32]{3,2,1,0} %slice.2552), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_7"}
  %slice.2521 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [256:288]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2553 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [256:288]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2607 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2521, f32[3,3,32,32]{3,2,1,0} %slice.2553), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_8"}
  %slice.2522 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [288:320]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2554 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [288:320]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2608 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2522, f32[3,3,32,32]{3,2,1,0} %slice.2554), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_9"}
  %slice.2523 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [320:352]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2555 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [320:352]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2579 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2523, f32[3,3,32,32]{3,2,1,0} %slice.2555), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_10"}
  %slice.2524 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [352:384]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2556 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [352:384]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2580 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2524, f32[3,3,32,32]{3,2,1,0} %slice.2556), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_11"}
  %slice.2525 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [384:416]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2557 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [384:416]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2581 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2525, f32[3,3,32,32]{3,2,1,0} %slice.2557), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_12"}
  %slice.2526 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [416:448]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2558 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [416:448]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2582 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2526, f32[3,3,32,32]{3,2,1,0} %slice.2558), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_13"}
  %slice.2527 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [448:480]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2559 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [448:480]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2583 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2527, f32[3,3,32,32]{3,2,1,0} %slice.2559), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_14"}
  %slice.2528 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [480:512]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2560 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [480:512]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2584 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2528, f32[3,3,32,32]{3,2,1,0} %slice.2560), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_15"}
  %slice.2529 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [512:544]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2561 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [512:544]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2585 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2529, f32[3,3,32,32]{3,2,1,0} %slice.2561), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_16"}
  %slice.2530 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [544:576]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2562 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [544:576]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2586 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2530, f32[3,3,32,32]{3,2,1,0} %slice.2562), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_17"}
  %slice.2531 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [576:608]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2563 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [576:608]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2587 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2531, f32[3,3,32,32]{3,2,1,0} %slice.2563), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_18"}
  %slice.2532 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [608:640]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2564 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [608:640]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2588 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2532, f32[3,3,32,32]{3,2,1,0} %slice.2564), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_19"}
  %slice.2533 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [640:672]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2565 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [640:672]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2590 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2533, f32[3,3,32,32]{3,2,1,0} %slice.2565), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_20"}
  %slice.2534 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [672:704]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2566 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [672:704]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2591 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2534, f32[3,3,32,32]{3,2,1,0} %slice.2566), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_21"}
  %slice.2535 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [704:736]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2567 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [704:736]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2592 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2535, f32[3,3,32,32]{3,2,1,0} %slice.2567), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_22"}
  %slice.2536 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [736:768]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2568 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [736:768]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2593 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2536, f32[3,3,32,32]{3,2,1,0} %slice.2568), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_23"}
  %slice.2537 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [768:800]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2569 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [768:800]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2594 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2537, f32[3,3,32,32]{3,2,1,0} %slice.2569), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_24"}
  %slice.2538 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [800:832]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2570 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [800:832]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2595 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2538, f32[3,3,32,32]{3,2,1,0} %slice.2570), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_25"}
  %slice.2539 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [832:864]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2571 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [832:864]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2596 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2539, f32[3,3,32,32]{3,2,1,0} %slice.2571), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_26"}
  %slice.2540 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [864:896]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2572 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [864:896]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2597 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2540, f32[3,3,32,32]{3,2,1,0} %slice.2572), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_27"}
  %slice.2541 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [896:928]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2573 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [896:928]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2598 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2541, f32[3,3,32,32]{3,2,1,0} %slice.2573), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_28"}
  %slice.2542 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [928:960]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2574 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [928:960]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2599 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2542, f32[3,3,32,32]{3,2,1,0} %slice.2574), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_29"}
  %slice.2543 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [960:992]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2575 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [960:992]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2601 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2543, f32[3,3,32,32]{3,2,1,0} %slice.2575), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_30"}
  %slice.2544 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2512), slice={[0:1], [0:16], [0:16], [992:1024]}, metadata={op_type="Split" op_name="split_27"}
  %slice.2576 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:32], [992:1024]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2602 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2544, f32[3,3,32,32]{3,2,1,0} %slice.2576), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_31"}
  %concatenate.2609 = f32[1,7,7,1024]{3,2,1,0} concatenate(f32[1,7,7,32]{3,2,1,0} %convolution.2577, f32[1,7,7,32]{3,2,1,0} %convolution.2578, f32[1,7,7,32]{3,2,1,0} %convolution.2589, f32[1,7,7,32]{3,2,1,0} %convolution.2600, f32[1,7,7,32]{3,2,1,0} %convolution.2603, f32[1,7,7,32]{3,2,1,0} %convolution.2604, f32[1,7,7,32]{3,2,1,0} %convolution.2605, f32[1,7,7,32]{3,2,1,0} %convolution.2606, f32[1,7,7,32]{3,2,1,0} %convolution.2607, f32[1,7,7,32]{3,2,1,0} %convolution.2608, f32[1,7,7,32]{3,2,1,0} %convolution.2579, f32[1,7,7,32]{3,2,1,0} %convolution.2580, f32[1,7,7,32]{3,2,1,0} %convolution.2581, f32[1,7,7,32]{3,2,1,0} %convolution.2582, f32[1,7,7,32]{3,2,1,0} %convolution.2583, f32[1,7,7,32]{3,2,1,0} %convolution.2584, f32[1,7,7,32]{3,2,1,0} %convolution.2585, f32[1,7,7,32]{3,2,1,0} %convolution.2586, f32[1,7,7,32]{3,2,1,0} %convolution.2587, f32[1,7,7,32]{3,2,1,0} %convolution.2588, f32[1,7,7,32]{3,2,1,0} %convolution.2590, f32[1,7,7,32]{3,2,1,0} %convolution.2591, f32[1,7,7,32]{3,2,1,0} %convolution.2592, f32[1,7,7,32]{3,2,1,0} %convolution.2593, f32[1,7,7,32]{3,2,1,0} %convolution.2594, f32[1,7,7,32]{3,2,1,0} %convolution.2595, f32[1,7,7,32]{3,2,1,0} %convolution.2596, f32[1,7,7,32]{3,2,1,0} %convolution.2597, f32[1,7,7,32]{3,2,1,0} %convolution.2598, f32[1,7,7,32]{3,2,1,0} %convolution.2599, f32[1,7,7,32]{3,2,1,0} %convolution.2601, f32[1,7,7,32]{3,2,1,0} %convolution.2602), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_13"}
  %constant.2489 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit1_bn2/add"}
  %broadcast.2490 = f32[1024]{0} broadcast(f32[] %constant.2489), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit1_bn2/add"}
  %arg40.41 = f32[1024]{0} parameter(40), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.310 = f32[1024]{0} reshape(f32[1024]{0} %arg40.41)
  %add.2491 = f32[1024]{0} add(f32[1024]{0} %broadcast.2490, f32[1024]{0} %reshape.310), metadata={op_type="AddV2" op_name="stage4_unit1_bn2/add"}
  %rsqrt.2492 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2491), metadata={op_type="Rsqrt" op_name="stage4_unit1_bn2/Rsqrt"}
  %arg100.101 = f32[1024]{0} parameter(100), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.370 = f32[1024]{0} reshape(f32[1024]{0} %arg100.101)
  %multiply.2493 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2492, f32[1024]{0} %reshape.370), metadata={op_type="Mul" op_name="stage4_unit1_bn2/mul"}
  %broadcast.2610 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2493), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit1_bn2/mul_1"}
  %multiply.2611 = f32[1,7,7,1024]{3,2,1,0} multiply(f32[1,7,7,1024]{3,2,1,0} %concatenate.2609, f32[1,7,7,1024]{3,2,1,0} %broadcast.2610), metadata={op_type="Mul" op_name="stage4_unit1_bn2/mul_1"}
  %arg207.208 = f32[1024]{0} parameter(207), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.477 = f32[1024]{0} reshape(f32[1024]{0} %arg207.208)
  %arg154.155 = f32[1024]{0} parameter(154), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.424 = f32[1024]{0} reshape(f32[1024]{0} %arg154.155)
  %multiply.2494 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2493, f32[1024]{0} %reshape.424), metadata={op_type="Mul" op_name="stage4_unit1_bn2/mul_2"}
  %subtract.2495 = f32[1024]{0} subtract(f32[1024]{0} %reshape.477, f32[1024]{0} %multiply.2494), metadata={op_type="Sub" op_name="stage4_unit1_bn2/sub"}
  %broadcast.2612 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2495), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit1_bn2/add_1"}
  %add.2613 = f32[1,7,7,1024]{3,2,1,0} add(f32[1,7,7,1024]{3,2,1,0} %multiply.2611, f32[1,7,7,1024]{3,2,1,0} %broadcast.2612), metadata={op_type="AddV2" op_name="stage4_unit1_bn2/add_1"}
  %maximum.2616 = f32[1,7,7,1024]{3,2,1,0} maximum(f32[1,7,7,1024]{3,2,1,0} %broadcast.2615, f32[1,7,7,1024]{3,2,1,0} %add.2613), metadata={op_type="Relu" op_name="stage4_unit1_relu2"}
  %arg263.264 = f32[1,1,1024,2048]{3,2,1,0} parameter(263), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.533 = f32[1,1,1024,2048]{3,2,1,0} reshape(f32[1,1,1024,2048]{3,2,1,0} %arg263.264)
  %convolution.2617 = f32[1,7,7,2048]{3,2,1,0} convolution(f32[1,7,7,1024]{3,2,1,0} %maximum.2616, f32[1,1,1024,2048]{3,2,1,0} %reshape.533), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv3"}
  %multiply.2619 = f32[1,7,7,2048]{3,2,1,0} multiply(f32[1,7,7,2048]{3,2,1,0} %broadcast.2618, f32[1,7,7,2048]{3,2,1,0} %convolution.2617), metadata={op_type="Mul" op_name="stage4_unit1_bn3/mul_1"}
  %arg211.212 = f32[2048]{0} parameter(211), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.481 = f32[2048]{0} reshape(f32[2048]{0} %arg211.212)
  %arg158.159 = f32[2048]{0} parameter(158), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.428 = f32[2048]{0} reshape(f32[2048]{0} %arg158.159)
  %multiply.2501 = f32[2048]{0} multiply(f32[2048]{0} %multiply.2500, f32[2048]{0} %reshape.428), metadata={op_type="Mul" op_name="stage4_unit1_bn3/mul_2"}
  %subtract.2502 = f32[2048]{0} subtract(f32[2048]{0} %reshape.481, f32[2048]{0} %multiply.2501), metadata={op_type="Sub" op_name="stage4_unit1_bn3/sub"}
  %broadcast.2620 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %subtract.2502), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit1_bn3/add_1"}
  %add.2621 = f32[1,7,7,2048]{3,2,1,0} add(f32[1,7,7,2048]{3,2,1,0} %multiply.2619, f32[1,7,7,2048]{3,2,1,0} %broadcast.2620), metadata={op_type="AddV2" op_name="stage4_unit1_bn3/add_1"}
  %arg262.263 = f32[1,1,1024,2048]{3,2,1,0} parameter(262), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.532 = f32[1,1,1024,2048]{3,2,1,0} reshape(f32[1,1,1024,2048]{3,2,1,0} %arg262.263)
  %convolution.2629 = f32[1,7,7,2048]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.2481, f32[1,1,1024,2048]{3,2,1,0} %reshape.532), window={size=1x1 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_sc"}
  %constant.2622 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit1_sc_bn/add"}
  %broadcast.2623 = f32[2048]{0} broadcast(f32[] %constant.2622), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit1_sc_bn/add"}
  %arg29.30 = f32[2048]{0} parameter(29), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.299 = f32[2048]{0} reshape(f32[2048]{0} %arg29.30)
  %add.2624 = f32[2048]{0} add(f32[2048]{0} %broadcast.2623, f32[2048]{0} %reshape.299), metadata={op_type="AddV2" op_name="stage4_unit1_sc_bn/add"}
  %rsqrt.2625 = f32[2048]{0} rsqrt(f32[2048]{0} %add.2624), metadata={op_type="Rsqrt" op_name="stage4_unit1_sc_bn/Rsqrt"}
  %arg93.94 = f32[2048]{0} parameter(93), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.363 = f32[2048]{0} reshape(f32[2048]{0} %arg93.94)
  %multiply.2626 = f32[2048]{0} multiply(f32[2048]{0} %rsqrt.2625, f32[2048]{0} %reshape.363), metadata={op_type="Mul" op_name="stage4_unit1_sc_bn/mul"}
  %broadcast.2630 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %multiply.2626), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit1_sc_bn/mul_1"}
  %multiply.2631 = f32[1,7,7,2048]{3,2,1,0} multiply(f32[1,7,7,2048]{3,2,1,0} %convolution.2629, f32[1,7,7,2048]{3,2,1,0} %broadcast.2630), metadata={op_type="Mul" op_name="stage4_unit1_sc_bn/mul_1"}
  %arg200.201 = f32[2048]{0} parameter(200), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.470 = f32[2048]{0} reshape(f32[2048]{0} %arg200.201)
  %arg147.148 = f32[2048]{0} parameter(147), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.417 = f32[2048]{0} reshape(f32[2048]{0} %arg147.148)
  %multiply.2627 = f32[2048]{0} multiply(f32[2048]{0} %multiply.2626, f32[2048]{0} %reshape.417), metadata={op_type="Mul" op_name="stage4_unit1_sc_bn/mul_2"}
  %subtract.2628 = f32[2048]{0} subtract(f32[2048]{0} %reshape.470, f32[2048]{0} %multiply.2627), metadata={op_type="Sub" op_name="stage4_unit1_sc_bn/sub"}
  %broadcast.2632 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %subtract.2628), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit1_sc_bn/add_1"}
  %add.2633 = f32[1,7,7,2048]{3,2,1,0} add(f32[1,7,7,2048]{3,2,1,0} %multiply.2631, f32[1,7,7,2048]{3,2,1,0} %broadcast.2632), metadata={op_type="AddV2" op_name="stage4_unit1_sc_bn/add_1"}
  %add.2634 = f32[1,7,7,2048]{3,2,1,0} add(f32[1,7,7,2048]{3,2,1,0} %add.2621, f32[1,7,7,2048]{3,2,1,0} %add.2633), metadata={op_type="AddV2" op_name="add_13"}
  %maximum.2637 = f32[1,7,7,2048]{3,2,1,0} maximum(f32[1,7,7,2048]{3,2,1,0} %broadcast.2636, f32[1,7,7,2048]{3,2,1,0} %add.2634), metadata={op_type="Relu" op_name="stage4_unit1_relu"}
  %constant.2652 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit2_bn3/add"}
  %broadcast.2653 = f32[2048]{0} broadcast(f32[] %constant.2652), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit2_bn3/add"}
  %arg66.67 = f32[2048]{0} parameter(66), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.336 = f32[2048]{0} reshape(f32[2048]{0} %arg66.67)
  %add.2654 = f32[2048]{0} add(f32[2048]{0} %broadcast.2653, f32[2048]{0} %reshape.336), metadata={op_type="AddV2" op_name="stage4_unit2_bn3/add"}
  %rsqrt.2655 = f32[2048]{0} rsqrt(f32[2048]{0} %add.2654), metadata={op_type="Rsqrt" op_name="stage4_unit2_bn3/Rsqrt"}
  %arg121.122 = f32[2048]{0} parameter(121), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.391 = f32[2048]{0} reshape(f32[2048]{0} %arg121.122)
  %multiply.2656 = f32[2048]{0} multiply(f32[2048]{0} %rsqrt.2655, f32[2048]{0} %reshape.391), metadata={op_type="Mul" op_name="stage4_unit2_bn3/mul"}
  %broadcast.2774 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %multiply.2656), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit2_bn3/mul_1"}
  %constant.2770 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit2_relu2"}
  %broadcast.2771 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[] %constant.2770), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit2_relu2"}
  %constant.2664 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit2_relu1"}
  %broadcast.2665 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[] %constant.2664), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit2_relu1"}
  %constant.2638 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit2_bn1/add"}
  %broadcast.2639 = f32[1024]{0} broadcast(f32[] %constant.2638), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit2_bn1/add"}
  %arg50.51 = f32[1024]{0} parameter(50), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.320 = f32[1024]{0} reshape(f32[1024]{0} %arg50.51)
  %add.2640 = f32[1024]{0} add(f32[1024]{0} %broadcast.2639, f32[1024]{0} %reshape.320), metadata={op_type="AddV2" op_name="stage4_unit2_bn1/add"}
  %rsqrt.2641 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2640), metadata={op_type="Rsqrt" op_name="stage4_unit2_bn1/Rsqrt"}
  %arg109.110 = f32[1024]{0} parameter(109), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.379 = f32[1024]{0} reshape(f32[1024]{0} %arg109.110)
  %multiply.2642 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2641, f32[1024]{0} %reshape.379), metadata={op_type="Mul" op_name="stage4_unit2_bn1/mul"}
  %broadcast.2660 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2642), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit2_bn1/mul_1"}
  %arg264.265 = f32[1,1,2048,1024]{3,2,1,0} parameter(264), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.534 = f32[1,1,2048,1024]{3,2,1,0} reshape(f32[1,1,2048,1024]{3,2,1,0} %arg264.265)
  %convolution.2659 = f32[1,7,7,1024]{3,2,1,0} convolution(f32[1,7,7,2048]{3,2,1,0} %maximum.2637, f32[1,1,2048,1024]{3,2,1,0} %reshape.534), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv1"}
  %multiply.2661 = f32[1,7,7,1024]{3,2,1,0} multiply(f32[1,7,7,1024]{3,2,1,0} %broadcast.2660, f32[1,7,7,1024]{3,2,1,0} %convolution.2659), metadata={op_type="Mul" op_name="stage4_unit2_bn1/mul_1"}
  %arg216.217 = f32[1024]{0} parameter(216), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.486 = f32[1024]{0} reshape(f32[1024]{0} %arg216.217)
  %arg163.164 = f32[1024]{0} parameter(163), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.433 = f32[1024]{0} reshape(f32[1024]{0} %arg163.164)
  %multiply.2643 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2642, f32[1024]{0} %reshape.433), metadata={op_type="Mul" op_name="stage4_unit2_bn1/mul_2"}
  %subtract.2644 = f32[1024]{0} subtract(f32[1024]{0} %reshape.486, f32[1024]{0} %multiply.2643), metadata={op_type="Sub" op_name="stage4_unit2_bn1/sub"}
  %broadcast.2662 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2644), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit2_bn1/add_1"}
  %add.2663 = f32[1,7,7,1024]{3,2,1,0} add(f32[1,7,7,1024]{3,2,1,0} %multiply.2661, f32[1,7,7,1024]{3,2,1,0} %broadcast.2662), metadata={op_type="AddV2" op_name="stage4_unit2_bn1/add_1"}
  %maximum.2666 = f32[1,7,7,1024]{3,2,1,0} maximum(f32[1,7,7,1024]{3,2,1,0} %broadcast.2665, f32[1,7,7,1024]{3,2,1,0} %add.2663), metadata={op_type="Relu" op_name="stage4_unit2_relu1"}
  %constant.2667 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_15"}
  %pad.2668 = f32[1,9,9,1024]{3,2,1,0} pad(f32[1,7,7,1024]{3,2,1,0} %maximum.2666, f32[] %constant.2667), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_15"}
  %slice.2669 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [0:32]}, metadata={op_type="Split" op_name="split_29"}
  %arg54.55 = f32[3,3,32,1024]{3,2,1,0} parameter(54), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.324 = f32[3,3,32,1024]{3,2,1,0} reshape(f32[3,3,32,1024]{3,2,1,0} %arg54.55)
  %slice.2701 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [0:32]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2733 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2669, f32[3,3,32,32]{3,2,1,0} %slice.2701), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2"}
  %slice.2670 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [32:64]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2702 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [32:64]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2734 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2670, f32[3,3,32,32]{3,2,1,0} %slice.2702), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_1"}
  %slice.2671 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [64:96]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2703 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [64:96]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2745 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2671, f32[3,3,32,32]{3,2,1,0} %slice.2703), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_2"}
  %slice.2672 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [96:128]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2704 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [96:128]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2756 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2672, f32[3,3,32,32]{3,2,1,0} %slice.2704), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_3"}
  %slice.2673 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [128:160]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2705 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [128:160]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2759 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2673, f32[3,3,32,32]{3,2,1,0} %slice.2705), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_4"}
  %slice.2674 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [160:192]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2706 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [160:192]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2760 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2674, f32[3,3,32,32]{3,2,1,0} %slice.2706), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_5"}
  %slice.2675 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [192:224]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2707 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [192:224]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2761 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2675, f32[3,3,32,32]{3,2,1,0} %slice.2707), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_6"}
  %slice.2676 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [224:256]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2708 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [224:256]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2762 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2676, f32[3,3,32,32]{3,2,1,0} %slice.2708), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_7"}
  %slice.2677 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [256:288]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2709 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [256:288]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2763 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2677, f32[3,3,32,32]{3,2,1,0} %slice.2709), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_8"}
  %slice.2678 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [288:320]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2710 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [288:320]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2764 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2678, f32[3,3,32,32]{3,2,1,0} %slice.2710), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_9"}
  %slice.2679 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [320:352]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2711 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [320:352]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2735 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2679, f32[3,3,32,32]{3,2,1,0} %slice.2711), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_10"}
  %slice.2680 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [352:384]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2712 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [352:384]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2736 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2680, f32[3,3,32,32]{3,2,1,0} %slice.2712), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_11"}
  %slice.2681 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [384:416]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2713 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [384:416]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2737 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2681, f32[3,3,32,32]{3,2,1,0} %slice.2713), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_12"}
  %slice.2682 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [416:448]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2714 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [416:448]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2738 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2682, f32[3,3,32,32]{3,2,1,0} %slice.2714), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_13"}
  %slice.2683 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [448:480]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2715 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [448:480]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2739 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2683, f32[3,3,32,32]{3,2,1,0} %slice.2715), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_14"}
  %slice.2684 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [480:512]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2716 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [480:512]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2740 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2684, f32[3,3,32,32]{3,2,1,0} %slice.2716), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_15"}
  %slice.2685 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [512:544]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2717 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [512:544]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2741 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2685, f32[3,3,32,32]{3,2,1,0} %slice.2717), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_16"}
  %slice.2686 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [544:576]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2718 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [544:576]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2742 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2686, f32[3,3,32,32]{3,2,1,0} %slice.2718), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_17"}
  %slice.2687 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [576:608]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2719 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [576:608]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2743 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2687, f32[3,3,32,32]{3,2,1,0} %slice.2719), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_18"}
  %slice.2688 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [608:640]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2720 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [608:640]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2744 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2688, f32[3,3,32,32]{3,2,1,0} %slice.2720), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_19"}
  %slice.2689 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [640:672]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2721 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [640:672]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2746 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2689, f32[3,3,32,32]{3,2,1,0} %slice.2721), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_20"}
  %slice.2690 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [672:704]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2722 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [672:704]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2747 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2690, f32[3,3,32,32]{3,2,1,0} %slice.2722), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_21"}
  %slice.2691 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [704:736]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2723 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [704:736]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2748 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2691, f32[3,3,32,32]{3,2,1,0} %slice.2723), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_22"}
  %slice.2692 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [736:768]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2724 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [736:768]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2749 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2692, f32[3,3,32,32]{3,2,1,0} %slice.2724), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_23"}
  %slice.2693 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [768:800]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2725 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [768:800]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2750 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2693, f32[3,3,32,32]{3,2,1,0} %slice.2725), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_24"}
  %slice.2694 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [800:832]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2726 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [800:832]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2751 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2694, f32[3,3,32,32]{3,2,1,0} %slice.2726), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_25"}
  %slice.2695 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [832:864]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2727 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [832:864]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2752 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2695, f32[3,3,32,32]{3,2,1,0} %slice.2727), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_26"}
  %slice.2696 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [864:896]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2728 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [864:896]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2753 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2696, f32[3,3,32,32]{3,2,1,0} %slice.2728), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_27"}
  %slice.2697 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [896:928]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2729 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [896:928]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2754 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2697, f32[3,3,32,32]{3,2,1,0} %slice.2729), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_28"}
  %slice.2698 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [928:960]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2730 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [928:960]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2755 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2698, f32[3,3,32,32]{3,2,1,0} %slice.2730), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_29"}
  %slice.2699 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [960:992]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2731 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [960:992]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2757 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2699, f32[3,3,32,32]{3,2,1,0} %slice.2731), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_30"}
  %slice.2700 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2668), slice={[0:1], [0:9], [0:9], [992:1024]}, metadata={op_type="Split" op_name="split_29"}
  %slice.2732 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.324), slice={[0:3], [0:3], [0:32], [992:1024]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2758 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2700, f32[3,3,32,32]{3,2,1,0} %slice.2732), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_31"}
  %concatenate.2765 = f32[1,7,7,1024]{3,2,1,0} concatenate(f32[1,7,7,32]{3,2,1,0} %convolution.2733, f32[1,7,7,32]{3,2,1,0} %convolution.2734, f32[1,7,7,32]{3,2,1,0} %convolution.2745, f32[1,7,7,32]{3,2,1,0} %convolution.2756, f32[1,7,7,32]{3,2,1,0} %convolution.2759, f32[1,7,7,32]{3,2,1,0} %convolution.2760, f32[1,7,7,32]{3,2,1,0} %convolution.2761, f32[1,7,7,32]{3,2,1,0} %convolution.2762, f32[1,7,7,32]{3,2,1,0} %convolution.2763, f32[1,7,7,32]{3,2,1,0} %convolution.2764, f32[1,7,7,32]{3,2,1,0} %convolution.2735, f32[1,7,7,32]{3,2,1,0} %convolution.2736, f32[1,7,7,32]{3,2,1,0} %convolution.2737, f32[1,7,7,32]{3,2,1,0} %convolution.2738, f32[1,7,7,32]{3,2,1,0} %convolution.2739, f32[1,7,7,32]{3,2,1,0} %convolution.2740, f32[1,7,7,32]{3,2,1,0} %convolution.2741, f32[1,7,7,32]{3,2,1,0} %convolution.2742, f32[1,7,7,32]{3,2,1,0} %convolution.2743, f32[1,7,7,32]{3,2,1,0} %convolution.2744, f32[1,7,7,32]{3,2,1,0} %convolution.2746, f32[1,7,7,32]{3,2,1,0} %convolution.2747, f32[1,7,7,32]{3,2,1,0} %convolution.2748, f32[1,7,7,32]{3,2,1,0} %convolution.2749, f32[1,7,7,32]{3,2,1,0} %convolution.2750, f32[1,7,7,32]{3,2,1,0} %convolution.2751, f32[1,7,7,32]{3,2,1,0} %convolution.2752, f32[1,7,7,32]{3,2,1,0} %convolution.2753, f32[1,7,7,32]{3,2,1,0} %convolution.2754, f32[1,7,7,32]{3,2,1,0} %convolution.2755, f32[1,7,7,32]{3,2,1,0} %convolution.2757, f32[1,7,7,32]{3,2,1,0} %convolution.2758), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_14"}
  %constant.2645 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit2_bn2/add"}
  %broadcast.2646 = f32[1024]{0} broadcast(f32[] %constant.2645), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit2_bn2/add"}
  %arg61.62 = f32[1024]{0} parameter(61), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.331 = f32[1024]{0} reshape(f32[1024]{0} %arg61.62)
  %add.2647 = f32[1024]{0} add(f32[1024]{0} %broadcast.2646, f32[1024]{0} %reshape.331), metadata={op_type="AddV2" op_name="stage4_unit2_bn2/add"}
  %rsqrt.2648 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2647), metadata={op_type="Rsqrt" op_name="stage4_unit2_bn2/Rsqrt"}
  %arg117.118 = f32[1024]{0} parameter(117), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.387 = f32[1024]{0} reshape(f32[1024]{0} %arg117.118)
  %multiply.2649 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2648, f32[1024]{0} %reshape.387), metadata={op_type="Mul" op_name="stage4_unit2_bn2/mul"}
  %broadcast.2766 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2649), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit2_bn2/mul_1"}
  %multiply.2767 = f32[1,7,7,1024]{3,2,1,0} multiply(f32[1,7,7,1024]{3,2,1,0} %concatenate.2765, f32[1,7,7,1024]{3,2,1,0} %broadcast.2766), metadata={op_type="Mul" op_name="stage4_unit2_bn2/mul_1"}
  %arg224.225 = f32[1024]{0} parameter(224), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.494 = f32[1024]{0} reshape(f32[1024]{0} %arg224.225)
  %arg171.172 = f32[1024]{0} parameter(171), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.441 = f32[1024]{0} reshape(f32[1024]{0} %arg171.172)
  %multiply.2650 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2649, f32[1024]{0} %reshape.441), metadata={op_type="Mul" op_name="stage4_unit2_bn2/mul_2"}
  %subtract.2651 = f32[1024]{0} subtract(f32[1024]{0} %reshape.494, f32[1024]{0} %multiply.2650), metadata={op_type="Sub" op_name="stage4_unit2_bn2/sub"}
  %broadcast.2768 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2651), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit2_bn2/add_1"}
  %add.2769 = f32[1,7,7,1024]{3,2,1,0} add(f32[1,7,7,1024]{3,2,1,0} %multiply.2767, f32[1,7,7,1024]{3,2,1,0} %broadcast.2768), metadata={op_type="AddV2" op_name="stage4_unit2_bn2/add_1"}
  %maximum.2772 = f32[1,7,7,1024]{3,2,1,0} maximum(f32[1,7,7,1024]{3,2,1,0} %broadcast.2771, f32[1,7,7,1024]{3,2,1,0} %add.2769), metadata={op_type="Relu" op_name="stage4_unit2_relu2"}
  %arg265.266 = f32[1,1,1024,2048]{3,2,1,0} parameter(265), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.535 = f32[1,1,1024,2048]{3,2,1,0} reshape(f32[1,1,1024,2048]{3,2,1,0} %arg265.266)
  %convolution.2773 = f32[1,7,7,2048]{3,2,1,0} convolution(f32[1,7,7,1024]{3,2,1,0} %maximum.2772, f32[1,1,1024,2048]{3,2,1,0} %reshape.535), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv3"}
  %multiply.2775 = f32[1,7,7,2048]{3,2,1,0} multiply(f32[1,7,7,2048]{3,2,1,0} %broadcast.2774, f32[1,7,7,2048]{3,2,1,0} %convolution.2773), metadata={op_type="Mul" op_name="stage4_unit2_bn3/mul_1"}
  %arg228.229 = f32[2048]{0} parameter(228), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.498 = f32[2048]{0} reshape(f32[2048]{0} %arg228.229)
  %arg175.176 = f32[2048]{0} parameter(175), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.445 = f32[2048]{0} reshape(f32[2048]{0} %arg175.176)
  %multiply.2657 = f32[2048]{0} multiply(f32[2048]{0} %multiply.2656, f32[2048]{0} %reshape.445), metadata={op_type="Mul" op_name="stage4_unit2_bn3/mul_2"}
  %subtract.2658 = f32[2048]{0} subtract(f32[2048]{0} %reshape.498, f32[2048]{0} %multiply.2657), metadata={op_type="Sub" op_name="stage4_unit2_bn3/sub"}
  %broadcast.2776 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %subtract.2658), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit2_bn3/add_1"}
  %add.2777 = f32[1,7,7,2048]{3,2,1,0} add(f32[1,7,7,2048]{3,2,1,0} %multiply.2775, f32[1,7,7,2048]{3,2,1,0} %broadcast.2776), metadata={op_type="AddV2" op_name="stage4_unit2_bn3/add_1"}
  %add.2778 = f32[1,7,7,2048]{3,2,1,0} add(f32[1,7,7,2048]{3,2,1,0} %maximum.2637, f32[1,7,7,2048]{3,2,1,0} %add.2777), metadata={op_type="AddV2" op_name="add_14"}
  %maximum.2781 = f32[1,7,7,2048]{3,2,1,0} maximum(f32[1,7,7,2048]{3,2,1,0} %broadcast.2780, f32[1,7,7,2048]{3,2,1,0} %add.2778), metadata={op_type="Relu" op_name="stage4_unit2_relu"}
  %constant.2796 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit3_bn3/add"}
  %broadcast.2797 = f32[2048]{0} broadcast(f32[] %constant.2796), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit3_bn3/add"}
  %arg1.2 = f32[2048]{0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.271 = f32[2048]{0} reshape(f32[2048]{0} %arg1.2)
  %add.2798 = f32[2048]{0} add(f32[2048]{0} %broadcast.2797, f32[2048]{0} %reshape.271), metadata={op_type="AddV2" op_name="stage4_unit3_bn3/add"}
  %rsqrt.2799 = f32[2048]{0} rsqrt(f32[2048]{0} %add.2798), metadata={op_type="Rsqrt" op_name="stage4_unit3_bn3/Rsqrt"}
  %arg71.72 = f32[2048]{0} parameter(71), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.341 = f32[2048]{0} reshape(f32[2048]{0} %arg71.72)
  %multiply.2800 = f32[2048]{0} multiply(f32[2048]{0} %rsqrt.2799, f32[2048]{0} %reshape.341), metadata={op_type="Mul" op_name="stage4_unit3_bn3/mul"}
  %broadcast.2918 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %multiply.2800), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit3_bn3/mul_1"}
  %constant.2914 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit3_relu2"}
  %broadcast.2915 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[] %constant.2914), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit3_relu2"}
  %constant.2808 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit3_relu1"}
  %broadcast.2809 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[] %constant.2808), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit3_relu1"}
  %constant.2782 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit3_bn1/add"}
  %broadcast.2783 = f32[1024]{0} broadcast(f32[] %constant.2782), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit3_bn1/add"}
  %arg4.5 = f32[1024]{0} parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.274 = f32[1024]{0} reshape(f32[1024]{0} %arg4.5)
  %add.2784 = f32[1024]{0} add(f32[1024]{0} %broadcast.2783, f32[1024]{0} %reshape.274), metadata={op_type="AddV2" op_name="stage4_unit3_bn1/add"}
  %rsqrt.2785 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2784), metadata={op_type="Rsqrt" op_name="stage4_unit3_bn1/Rsqrt"}
  %arg74.75 = f32[1024]{0} parameter(74), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.344 = f32[1024]{0} reshape(f32[1024]{0} %arg74.75)
  %multiply.2786 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2785, f32[1024]{0} %reshape.344), metadata={op_type="Mul" op_name="stage4_unit3_bn1/mul"}
  %broadcast.2804 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2786), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit3_bn1/mul_1"}
  %arg266.267 = f32[1,1,2048,1024]{3,2,1,0} parameter(266), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.536 = f32[1,1,2048,1024]{3,2,1,0} reshape(f32[1,1,2048,1024]{3,2,1,0} %arg266.267)
  %convolution.2803 = f32[1,7,7,1024]{3,2,1,0} convolution(f32[1,7,7,2048]{3,2,1,0} %maximum.2781, f32[1,1,2048,1024]{3,2,1,0} %reshape.536), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv1"}
  %multiply.2805 = f32[1,7,7,1024]{3,2,1,0} multiply(f32[1,7,7,1024]{3,2,1,0} %broadcast.2804, f32[1,7,7,1024]{3,2,1,0} %convolution.2803), metadata={op_type="Mul" op_name="stage4_unit3_bn1/mul_1"}
  %arg181.182 = f32[1024]{0} parameter(181), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.451 = f32[1024]{0} reshape(f32[1024]{0} %arg181.182)
  %arg128.129 = f32[1024]{0} parameter(128), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.398 = f32[1024]{0} reshape(f32[1024]{0} %arg128.129)
  %multiply.2787 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2786, f32[1024]{0} %reshape.398), metadata={op_type="Mul" op_name="stage4_unit3_bn1/mul_2"}
  %subtract.2788 = f32[1024]{0} subtract(f32[1024]{0} %reshape.451, f32[1024]{0} %multiply.2787), metadata={op_type="Sub" op_name="stage4_unit3_bn1/sub"}
  %broadcast.2806 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2788), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit3_bn1/add_1"}
  %add.2807 = f32[1,7,7,1024]{3,2,1,0} add(f32[1,7,7,1024]{3,2,1,0} %multiply.2805, f32[1,7,7,1024]{3,2,1,0} %broadcast.2806), metadata={op_type="AddV2" op_name="stage4_unit3_bn1/add_1"}
  %maximum.2810 = f32[1,7,7,1024]{3,2,1,0} maximum(f32[1,7,7,1024]{3,2,1,0} %broadcast.2809, f32[1,7,7,1024]{3,2,1,0} %add.2807), metadata={op_type="Relu" op_name="stage4_unit3_relu1"}
  %constant.2811 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_16"}
  %pad.2812 = f32[1,9,9,1024]{3,2,1,0} pad(f32[1,7,7,1024]{3,2,1,0} %maximum.2810, f32[] %constant.2811), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_16"}
  %slice.2813 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [0:32]}, metadata={op_type="Split" op_name="split_31"}
  %arg21.22 = f32[3,3,32,1024]{3,2,1,0} parameter(21), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.291 = f32[3,3,32,1024]{3,2,1,0} reshape(f32[3,3,32,1024]{3,2,1,0} %arg21.22)
  %slice.2845 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [0:32]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2877 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2813, f32[3,3,32,32]{3,2,1,0} %slice.2845), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2"}
  %slice.2814 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [32:64]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2846 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [32:64]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2878 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2814, f32[3,3,32,32]{3,2,1,0} %slice.2846), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_1"}
  %slice.2815 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [64:96]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2847 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [64:96]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2889 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2815, f32[3,3,32,32]{3,2,1,0} %slice.2847), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_2"}
  %slice.2816 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [96:128]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2848 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [96:128]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2900 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2816, f32[3,3,32,32]{3,2,1,0} %slice.2848), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_3"}
  %slice.2817 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [128:160]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2849 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [128:160]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2903 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2817, f32[3,3,32,32]{3,2,1,0} %slice.2849), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_4"}
  %slice.2818 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [160:192]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2850 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [160:192]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2904 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2818, f32[3,3,32,32]{3,2,1,0} %slice.2850), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_5"}
  %slice.2819 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [192:224]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2851 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [192:224]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2905 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2819, f32[3,3,32,32]{3,2,1,0} %slice.2851), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_6"}
  %slice.2820 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [224:256]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2852 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [224:256]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2906 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2820, f32[3,3,32,32]{3,2,1,0} %slice.2852), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_7"}
  %slice.2821 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [256:288]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2853 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [256:288]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2907 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2821, f32[3,3,32,32]{3,2,1,0} %slice.2853), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_8"}
  %slice.2822 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [288:320]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2854 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [288:320]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2908 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2822, f32[3,3,32,32]{3,2,1,0} %slice.2854), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_9"}
  %slice.2823 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [320:352]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2855 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [320:352]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2879 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2823, f32[3,3,32,32]{3,2,1,0} %slice.2855), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_10"}
  %slice.2824 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [352:384]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2856 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [352:384]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2880 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2824, f32[3,3,32,32]{3,2,1,0} %slice.2856), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_11"}
  %slice.2825 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [384:416]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2857 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [384:416]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2881 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2825, f32[3,3,32,32]{3,2,1,0} %slice.2857), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_12"}
  %slice.2826 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [416:448]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2858 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [416:448]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2882 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2826, f32[3,3,32,32]{3,2,1,0} %slice.2858), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_13"}
  %slice.2827 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [448:480]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2859 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [448:480]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2883 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2827, f32[3,3,32,32]{3,2,1,0} %slice.2859), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_14"}
  %slice.2828 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [480:512]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2860 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [480:512]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2884 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2828, f32[3,3,32,32]{3,2,1,0} %slice.2860), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_15"}
  %slice.2829 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [512:544]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2861 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [512:544]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2885 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2829, f32[3,3,32,32]{3,2,1,0} %slice.2861), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_16"}
  %slice.2830 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [544:576]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2862 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [544:576]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2886 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2830, f32[3,3,32,32]{3,2,1,0} %slice.2862), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_17"}
  %slice.2831 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [576:608]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2863 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [576:608]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2887 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2831, f32[3,3,32,32]{3,2,1,0} %slice.2863), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_18"}
  %slice.2832 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [608:640]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2864 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [608:640]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2888 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2832, f32[3,3,32,32]{3,2,1,0} %slice.2864), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_19"}
  %slice.2833 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [640:672]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2865 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [640:672]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2890 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2833, f32[3,3,32,32]{3,2,1,0} %slice.2865), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_20"}
  %slice.2834 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [672:704]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2866 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [672:704]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2891 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2834, f32[3,3,32,32]{3,2,1,0} %slice.2866), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_21"}
  %slice.2835 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [704:736]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2867 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [704:736]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2892 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2835, f32[3,3,32,32]{3,2,1,0} %slice.2867), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_22"}
  %slice.2836 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [736:768]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2868 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [736:768]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2893 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2836, f32[3,3,32,32]{3,2,1,0} %slice.2868), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_23"}
  %slice.2837 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [768:800]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2869 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [768:800]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2894 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2837, f32[3,3,32,32]{3,2,1,0} %slice.2869), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_24"}
  %slice.2838 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [800:832]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2870 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [800:832]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2895 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2838, f32[3,3,32,32]{3,2,1,0} %slice.2870), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_25"}
  %slice.2839 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [832:864]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2871 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [832:864]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2896 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2839, f32[3,3,32,32]{3,2,1,0} %slice.2871), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_26"}
  %slice.2840 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [864:896]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2872 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [864:896]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2897 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2840, f32[3,3,32,32]{3,2,1,0} %slice.2872), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_27"}
  %slice.2841 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [896:928]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2873 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [896:928]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2898 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2841, f32[3,3,32,32]{3,2,1,0} %slice.2873), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_28"}
  %slice.2842 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [928:960]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2874 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [928:960]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2899 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2842, f32[3,3,32,32]{3,2,1,0} %slice.2874), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_29"}
  %slice.2843 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [960:992]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2875 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [960:992]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2901 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2843, f32[3,3,32,32]{3,2,1,0} %slice.2875), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_30"}
  %slice.2844 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [992:1024]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2876 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.291), slice={[0:3], [0:3], [0:32], [992:1024]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2902 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2844, f32[3,3,32,32]{3,2,1,0} %slice.2876), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_31"}
  %concatenate.2909 = f32[1,7,7,1024]{3,2,1,0} concatenate(f32[1,7,7,32]{3,2,1,0} %convolution.2877, f32[1,7,7,32]{3,2,1,0} %convolution.2878, f32[1,7,7,32]{3,2,1,0} %convolution.2889, f32[1,7,7,32]{3,2,1,0} %convolution.2900, f32[1,7,7,32]{3,2,1,0} %convolution.2903, f32[1,7,7,32]{3,2,1,0} %convolution.2904, f32[1,7,7,32]{3,2,1,0} %convolution.2905, f32[1,7,7,32]{3,2,1,0} %convolution.2906, f32[1,7,7,32]{3,2,1,0} %convolution.2907, f32[1,7,7,32]{3,2,1,0} %convolution.2908, f32[1,7,7,32]{3,2,1,0} %convolution.2879, f32[1,7,7,32]{3,2,1,0} %convolution.2880, f32[1,7,7,32]{3,2,1,0} %convolution.2881, f32[1,7,7,32]{3,2,1,0} %convolution.2882, f32[1,7,7,32]{3,2,1,0} %convolution.2883, f32[1,7,7,32]{3,2,1,0} %convolution.2884, f32[1,7,7,32]{3,2,1,0} %convolution.2885, f32[1,7,7,32]{3,2,1,0} %convolution.2886, f32[1,7,7,32]{3,2,1,0} %convolution.2887, f32[1,7,7,32]{3,2,1,0} %convolution.2888, f32[1,7,7,32]{3,2,1,0} %convolution.2890, f32[1,7,7,32]{3,2,1,0} %convolution.2891, f32[1,7,7,32]{3,2,1,0} %convolution.2892, f32[1,7,7,32]{3,2,1,0} %convolution.2893, f32[1,7,7,32]{3,2,1,0} %convolution.2894, f32[1,7,7,32]{3,2,1,0} %convolution.2895, f32[1,7,7,32]{3,2,1,0} %convolution.2896, f32[1,7,7,32]{3,2,1,0} %convolution.2897, f32[1,7,7,32]{3,2,1,0} %convolution.2898, f32[1,7,7,32]{3,2,1,0} %convolution.2899, f32[1,7,7,32]{3,2,1,0} %convolution.2901, f32[1,7,7,32]{3,2,1,0} %convolution.2902), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_15"}
  %constant.2789 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit3_bn2/add"}
  %broadcast.2790 = f32[1024]{0} broadcast(f32[] %constant.2789), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit3_bn2/add"}
  %arg49.50 = f32[1024]{0} parameter(49), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.319 = f32[1024]{0} reshape(f32[1024]{0} %arg49.50)
  %add.2791 = f32[1024]{0} add(f32[1024]{0} %broadcast.2790, f32[1024]{0} %reshape.319), metadata={op_type="AddV2" op_name="stage4_unit3_bn2/add"}
  %rsqrt.2792 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2791), metadata={op_type="Rsqrt" op_name="stage4_unit3_bn2/Rsqrt"}
  %arg108.109 = f32[1024]{0} parameter(108), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.378 = f32[1024]{0} reshape(f32[1024]{0} %arg108.109)
  %multiply.2793 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2792, f32[1024]{0} %reshape.378), metadata={op_type="Mul" op_name="stage4_unit3_bn2/mul"}
  %broadcast.2910 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2793), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit3_bn2/mul_1"}
  %multiply.2911 = f32[1,7,7,1024]{3,2,1,0} multiply(f32[1,7,7,1024]{3,2,1,0} %concatenate.2909, f32[1,7,7,1024]{3,2,1,0} %broadcast.2910), metadata={op_type="Mul" op_name="stage4_unit3_bn2/mul_1"}
  %arg215.216 = f32[1024]{0} parameter(215), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.485 = f32[1024]{0} reshape(f32[1024]{0} %arg215.216)
  %arg162.163 = f32[1024]{0} parameter(162), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.432 = f32[1024]{0} reshape(f32[1024]{0} %arg162.163)
  %multiply.2794 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2793, f32[1024]{0} %reshape.432), metadata={op_type="Mul" op_name="stage4_unit3_bn2/mul_2"}
  %subtract.2795 = f32[1024]{0} subtract(f32[1024]{0} %reshape.485, f32[1024]{0} %multiply.2794), metadata={op_type="Sub" op_name="stage4_unit3_bn2/sub"}
  %broadcast.2912 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2795), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit3_bn2/add_1"}
  %add.2913 = f32[1,7,7,1024]{3,2,1,0} add(f32[1,7,7,1024]{3,2,1,0} %multiply.2911, f32[1,7,7,1024]{3,2,1,0} %broadcast.2912), metadata={op_type="AddV2" op_name="stage4_unit3_bn2/add_1"}
  %maximum.2916 = f32[1,7,7,1024]{3,2,1,0} maximum(f32[1,7,7,1024]{3,2,1,0} %broadcast.2915, f32[1,7,7,1024]{3,2,1,0} %add.2913), metadata={op_type="Relu" op_name="stage4_unit3_relu2"}
  %arg267.268 = f32[1,1,1024,2048]{3,2,1,0} parameter(267), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.537 = f32[1,1,1024,2048]{3,2,1,0} reshape(f32[1,1,1024,2048]{3,2,1,0} %arg267.268)
  %convolution.2917 = f32[1,7,7,2048]{3,2,1,0} convolution(f32[1,7,7,1024]{3,2,1,0} %maximum.2916, f32[1,1,1024,2048]{3,2,1,0} %reshape.537), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv3"}
  %multiply.2919 = f32[1,7,7,2048]{3,2,1,0} multiply(f32[1,7,7,2048]{3,2,1,0} %broadcast.2918, f32[1,7,7,2048]{3,2,1,0} %convolution.2917), metadata={op_type="Mul" op_name="stage4_unit3_bn3/mul_1"}
  %arg178.179 = f32[2048]{0} parameter(178), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.448 = f32[2048]{0} reshape(f32[2048]{0} %arg178.179)
  %arg125.126 = f32[2048]{0} parameter(125), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.395 = f32[2048]{0} reshape(f32[2048]{0} %arg125.126)
  %multiply.2801 = f32[2048]{0} multiply(f32[2048]{0} %multiply.2800, f32[2048]{0} %reshape.395), metadata={op_type="Mul" op_name="stage4_unit3_bn3/mul_2"}
  %subtract.2802 = f32[2048]{0} subtract(f32[2048]{0} %reshape.448, f32[2048]{0} %multiply.2801), metadata={op_type="Sub" op_name="stage4_unit3_bn3/sub"}
  %broadcast.2920 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %subtract.2802), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit3_bn3/add_1"}
  %add.2921 = f32[1,7,7,2048]{3,2,1,0} add(f32[1,7,7,2048]{3,2,1,0} %multiply.2919, f32[1,7,7,2048]{3,2,1,0} %broadcast.2920), metadata={op_type="AddV2" op_name="stage4_unit3_bn3/add_1"}
  %add.2922 = f32[1,7,7,2048]{3,2,1,0} add(f32[1,7,7,2048]{3,2,1,0} %maximum.2781, f32[1,7,7,2048]{3,2,1,0} %add.2921), metadata={op_type="AddV2" op_name="add_15"}
  %maximum.2925 = f32[1,7,7,2048]{3,2,1,0} maximum(f32[1,7,7,2048]{3,2,1,0} %broadcast.2924, f32[1,7,7,2048]{3,2,1,0} %add.2922), metadata={op_type="Relu" op_name="stage4_unit3_relu"}
  %convert.2926 = f32[1,7,7,2048]{3,2,1,0} convert(f32[1,7,7,2048]{3,2,1,0} %maximum.2925), metadata={op_type="AvgPool" op_name="pool1"}
  %constant.2928 = f32[] constant(0), metadata={op_type="AvgPool" op_name="pool1"}
  %pad.2929 = f32[1,7,7,2048]{3,2,1,0} pad(f32[1,7,7,2048]{3,2,1,0} %convert.2926, f32[] %constant.2928), padding=0_0x0_0x0_0x0_0, metadata={op_type="AvgPool" op_name="pool1"}
  %constant.2927 = f32[] constant(0), metadata={op_type="AvgPool" op_name="pool1"}
  %reduce-window.2934 = f32[1,1,1,2048]{3,2,1,0} reduce-window(f32[1,7,7,2048]{3,2,1,0} %pad.2929, f32[] %constant.2927), window={size=1x7x7x1}, to_apply=%add_F32.2930, metadata={op_type="AvgPool" op_name="pool1"}
  %constant.2935 = f32[] constant(49), metadata={op_type="AvgPool" op_name="pool1"}
  %broadcast.2936 = f32[1,1,1,2048]{3,2,1,0} broadcast(f32[] %constant.2935), dimensions={}, metadata={op_type="AvgPool" op_name="pool1"}
  %divide.2937 = f32[1,1,1,2048]{3,2,1,0} divide(f32[1,1,1,2048]{3,2,1,0} %reduce-window.2934, f32[1,1,1,2048]{3,2,1,0} %broadcast.2936), metadata={op_type="AvgPool" op_name="pool1"}
  %convert.2938 = f32[1,1,1,2048]{3,2,1,0} convert(f32[1,1,1,2048]{3,2,1,0} %divide.2937), metadata={op_type="AvgPool" op_name="pool1"}
  %get-dimension-size.2939 = s32[] get-dimension-size(f32[1,1,1,2048]{3,2,1,0} %convert.2938), dimensions={0}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %convert.2940 = s32[] convert(s32[] %get-dimension-size.2939), metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %broadcast.2941 = s32[1]{0} broadcast(s32[] %convert.2940), dimensions={}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %get-dimension-size.2942 = s32[] get-dimension-size(f32[1,1,1,2048]{3,2,1,0} %convert.2938), dimensions={1}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %convert.2943 = s32[] convert(s32[] %get-dimension-size.2942), metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %broadcast.2944 = s32[1]{0} broadcast(s32[] %convert.2943), dimensions={}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %get-dimension-size.2945 = s32[] get-dimension-size(f32[1,1,1,2048]{3,2,1,0} %convert.2938), dimensions={2}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %convert.2946 = s32[] convert(s32[] %get-dimension-size.2945), metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %broadcast.2947 = s32[1]{0} broadcast(s32[] %convert.2946), dimensions={}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %get-dimension-size.2948 = s32[] get-dimension-size(f32[1,1,1,2048]{3,2,1,0} %convert.2938), dimensions={3}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %convert.2949 = s32[] convert(s32[] %get-dimension-size.2948), metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %broadcast.2950 = s32[1]{0} broadcast(s32[] %convert.2949), dimensions={}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %concatenate.2951 = s32[4]{0} concatenate(s32[1]{0} %broadcast.2941, s32[1]{0} %broadcast.2944, s32[1]{0} %broadcast.2947, s32[1]{0} %broadcast.2950), dimensions={0}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %slice.2952 = s32[1]{0} slice(s32[4]{0} %concatenate.2951), slice={[0:1]}, metadata={op_type="StridedSlice" op_name="Flatten/flatten/strided_slice"}
  %reshape.2953 = s32[] reshape(s32[1]{0} %slice.2952), metadata={op_type="StridedSlice" op_name="Flatten/flatten/strided_slice"}
  %reshape.2955 = s32[1]{0} reshape(s32[] %reshape.2953), metadata={op_type="Pack" op_name="Flatten/flatten/Reshape/shape"}
  %constant.2954 = s32[] constant(-1), metadata={op_type="Pack" op_name="Flatten/flatten/Reshape/shape"}
  %reshape.2956 = s32[1]{0} reshape(s32[] %constant.2954), metadata={op_type="Pack" op_name="Flatten/flatten/Reshape/shape"}
  %concatenate.2957 = s32[2]{0} concatenate(s32[1]{0} %reshape.2955, s32[1]{0} %reshape.2956), dimensions={0}, metadata={op_type="Pack" op_name="Flatten/flatten/Reshape/shape"}
  %reshape.2959 = f32[1,2048]{1,0} reshape(f32[1,1,1,2048]{3,2,1,0} %convert.2938), inferred_dimension=1, metadata={op_type="Reshape" op_name="Flatten/flatten/Reshape"}
  %reshape.2960 = f32[1,2048]{1,0} reshape(f32[1,2048]{1,0} %reshape.2959), metadata={op_name="XLA_Retvals"}
  %tuple.2961 = (f32[1,2048]{1,0}) tuple(f32[1,2048]{1,0} %reshape.2960), metadata={op_name="XLA_Retvals"}
  ROOT %get-tuple-element.2962 = f32[1,2048]{1,0} get-tuple-element((f32[1,2048]{1,0}) %tuple.2961), index=0, metadata={op_name="XLA_Retvals"}
}
  )";

  HloModuleConfig cfg;
  auto hlo_module = std::make_unique<VerifiedHloModule>("module", cfg, false, false, nullptr);
  hlo_module->ParseHloStringAndVerifyModule(hlo_text);
  CompileAndCheck(std::move(hlo_module), {{args, outputs}});
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
