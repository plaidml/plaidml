// Tests that show HLO Module conversion to PlaidML Program.

#include <gtest/gtest.h>

#include <fstream>

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"

#include "plaidml/bridge/tensorflow/tests/archive_generated.h"
#include "plaidml/bridge/tensorflow/tests/codegen_test.h"
#include "plaidml/bridge/tensorflow/tests/utils.h"

using plaidml::edsl::MultiBuffer;
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

  std::vector<MultiBuffer> args = {std::vector<float>{0},  // stage4_unit3_relu
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
                                   std::vector<float>{0},  // Pad_16
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
                                   std::vector<float>{0},  // Pad_15
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
                                   std::vector<float>{0},  // Pad_13
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
                                   std::vector<float>{2e-05},  // stage3_unit5_bn2/add
                                   lookup("stage3_unit5_bn2_var"),
                                   lookup("stage3_unit5_bn2_bias"),
                                   lookup("stage3_unit5_conv2_weight"),
                                   std::vector<float>{0},  // Pad_12
                                   std::vector<float>{0},  // stage3_unit5_relu1
                                   lookup("stage3_unit5_bn1_mean"),
                                   lookup("stage3_unit5_bn1_scale"),
                                   std::vector<float>{2e-05},  // stage3_unit5_bn1/add
                                   lookup("stage3_unit5_bn1_var"),
                                   lookup("stage3_unit5_bn1_bias"),
                                   lookup("stage3_unit5_conv1_weight"),
                                   std::vector<float>{0},  // stage3_unit4_relu
                                   lookup("stage3_unit4_bn3_mean"),
                                   lookup("stage3_unit4_bn3_scale"),
                                   std::vector<float>{2e-05},  // stage3_unit4_bn3/add
                                   lookup("stage3_unit4_bn3_var"),
                                   lookup("stage3_unit4_bn3_bias"),
                                   lookup("stage3_unit4_conv3_weight"),
                                   std::vector<float>{0},  // stage3_unit4_relu2
                                   lookup("stage3_unit4_bn2_mean"),
                                   lookup("stage3_unit4_bn2_scale"),
                                   std::vector<float>{2e-05},  // stage3_unit4_bn2/add
                                   lookup("stage3_unit4_bn2_var"),
                                   lookup("stage3_unit4_bn2_bias"),
                                   lookup("stage3_unit4_conv2_weight"),
                                   std::vector<float>{0},  // Pad_11
                                   std::vector<float>{0},  // stage3_unit4_relu1
                                   lookup("stage3_unit4_bn1_mean"),
                                   lookup("stage3_unit4_bn1_scale"),
                                   std::vector<float>{2e-05},  // stage3_unit4_bn1/add
                                   lookup("stage3_unit4_bn1_var"),
                                   lookup("stage3_unit4_bn1_bias"),
                                   lookup("stage3_unit4_conv1_weight"),
                                   std::vector<float>{0},  // stage3_unit3_relu
                                   lookup("stage3_unit3_bn3_mean"),
                                   lookup("stage3_unit3_bn3_scale"),
                                   std::vector<float>{2e-05},  // stage3_unit3_bn3/add
                                   lookup("stage3_unit3_bn3_var"),
                                   lookup("stage3_unit3_bn3_bias"),
                                   lookup("stage3_unit3_conv3_weight"),
                                   std::vector<float>{0},  // stage3_unit3_relu2
                                   lookup("stage3_unit3_bn2_mean"),
                                   lookup("stage3_unit3_bn2_scale"),
                                   std::vector<float>{2e-05},  // stage3_unit3_bn2/add
                                   lookup("stage3_unit3_bn2_var"),
                                   lookup("stage3_unit3_bn2_bias"),
                                   lookup("stage3_unit3_conv2_weight"),
                                   std::vector<float>{0},  // Pad_10
                                   std::vector<float>{0},  // stage3_unit3_relu1
                                   lookup("stage3_unit3_bn1_mean"),
                                   lookup("stage3_unit3_bn1_scale"),
                                   std::vector<float>{2e-05},  // stage3_unit3_bn1/add
                                   lookup("stage3_unit3_bn1_var"),
                                   lookup("stage3_unit3_bn1_bias"),
                                   lookup("stage3_unit3_conv1_weight"),
                                   std::vector<float>{0},  // stage3_unit2_relu
                                   lookup("stage3_unit2_bn3_mean"),
                                   lookup("stage3_unit2_bn3_scale"),
                                   std::vector<float>{2e-05},  // stage3_unit2_bn3/add
                                   lookup("stage3_unit2_bn3_var"),
                                   lookup("stage3_unit2_bn3_bias"),
                                   lookup("stage3_unit2_conv3_weight"),
                                   std::vector<float>{0},  // stage3_unit2_relu2
                                   lookup("stage3_unit2_bn2_mean"),
                                   lookup("stage3_unit2_bn2_scale"),
                                   std::vector<float>{2e-05},  // stage3_unit2_bn2/add
                                   lookup("stage3_unit2_bn2_var"),
                                   lookup("stage3_unit2_bn2_bias"),
                                   lookup("stage3_unit2_conv2_weight"),
                                   std::vector<float>{0},  // Pad_9
                                   std::vector<float>{0},  // stage3_unit2_relu1
                                   lookup("stage3_unit2_bn1_mean"),
                                   lookup("stage3_unit2_bn1_scale"),
                                   std::vector<float>{2e-05},  // stage3_unit2_bn1/add
                                   lookup("stage3_unit2_bn1_var"),
                                   lookup("stage3_unit2_bn1_bias"),
                                   lookup("stage3_unit2_conv1_weight"),
                                   std::vector<float>{0},  // stage3_unit1_relu
                                   lookup("stage3_unit1_sc_bn_mean"),
                                   lookup("stage3_unit1_sc_bn_scale"),
                                   std::vector<float>{2e-05},  // stage3_unit1_sc_bn/add
                                   lookup("stage3_unit1_sc_bn_var"),
                                   lookup("stage3_unit1_sc_bn_bias"),
                                   lookup("stage3_unit1_sc_weight"),
                                   std::vector<float>{0},  // stage2_unit4_relu
                                   lookup("stage2_unit4_bn3_mean"),
                                   lookup("stage2_unit4_bn3_scale"),
                                   std::vector<float>{2e-05},  // stage2_unit4_bn3/add
                                   lookup("stage2_unit4_bn3_var"),
                                   lookup("stage2_unit4_bn3_bias"),
                                   lookup("stage2_unit4_conv3_weight"),
                                   std::vector<float>{0},  // stage2_unit4_relu2
                                   lookup("stage2_unit4_bn2_mean"),
                                   lookup("stage2_unit4_bn2_scale"),
                                   std::vector<float>{2e-05},  // stage2_unit4_bn2/add
                                   lookup("stage2_unit4_bn2_var"),
                                   lookup("stage2_unit4_bn2_bias"),
                                   lookup("stage2_unit4_conv2_weight"),
                                   std::vector<float>{0},  // Pad_7
                                   std::vector<float>{0},  // stage2_unit4_relu1
                                   lookup("stage2_unit4_bn1_mean"),
                                   lookup("stage2_unit4_bn1_scale"),
                                   std::vector<float>{2e-05},  // stage2_unit4_bn1/add
                                   lookup("stage2_unit4_bn1_var"),
                                   lookup("stage2_unit4_bn1_bias"),
                                   lookup("stage2_unit4_conv1_weight"),
                                   std::vector<float>{0},  // stage2_unit3_relu
                                   lookup("stage2_unit3_bn3_mean"),
                                   lookup("stage2_unit3_bn3_scale"),
                                   std::vector<float>{2e-05},  // stage2_unit3_bn3/add
                                   lookup("stage2_unit3_bn3_var"),
                                   lookup("stage2_unit3_bn3_bias"),
                                   lookup("stage2_unit3_conv3_weight"),
                                   std::vector<float>{0},  // stage2_unit3_relu2
                                   lookup("stage2_unit3_bn2_mean"),
                                   lookup("stage2_unit3_bn2_scale"),
                                   std::vector<float>{2e-05},  // stage2_unit3_bn2/add
                                   lookup("stage2_unit3_bn2_var"),
                                   lookup("stage2_unit3_bn2_bias"),
                                   lookup("stage2_unit3_conv2_weight"),
                                   std::vector<float>{0},  // Pad_6
                                   std::vector<float>{0},  // stage2_unit3_relu1
                                   lookup("stage2_unit3_bn1_mean"),
                                   lookup("stage2_unit3_bn1_scale"),
                                   std::vector<float>{2e-05},  // stage2_unit3_bn1/add
                                   lookup("stage2_unit3_bn1_var"),
                                   lookup("stage2_unit3_bn1_bias"),
                                   lookup("stage2_unit3_conv1_weight"),
                                   std::vector<float>{0},  // stage2_unit2_relu
                                   lookup("stage2_unit2_bn3_mean"),
                                   lookup("stage2_unit2_bn3_scale"),
                                   std::vector<float>{2e-05},  // stage2_unit2_bn3/add
                                   lookup("stage2_unit2_bn3_var"),
                                   lookup("stage2_unit2_bn3_bias"),
                                   lookup("stage2_unit2_conv3_weight"),
                                   std::vector<float>{0},  // stage2_unit2_relu2
                                   lookup("stage2_unit2_bn2_mean"),
                                   lookup("stage2_unit2_bn2_scale"),
                                   std::vector<float>{2e-05},  // stage2_unit2_bn2/add
                                   lookup("stage2_unit2_bn2_var"),
                                   lookup("stage2_unit2_bn2_bias"),
                                   lookup("stage2_unit2_conv2_weight"),
                                   std::vector<float>{0},  // Pad_5
                                   std::vector<float>{0},  // stage2_unit2_relu1
                                   lookup("stage2_unit2_bn1_mean"),
                                   lookup("stage2_unit2_bn1_scale"),
                                   std::vector<float>{2e-05},  // stage2_unit2_bn1/add
                                   lookup("stage2_unit2_bn1_var"),
                                   lookup("stage2_unit2_bn1_bias"),
                                   lookup("stage2_unit2_conv1_weight"),
                                   std::vector<float>{0},  // stage2_unit1_relu
                                   lookup("stage2_unit1_sc_bn_mean"),
                                   lookup("stage2_unit1_sc_bn_scale"),
                                   std::vector<float>{2e-05},  // stage2_unit1_sc_bn/add
                                   lookup("stage2_unit1_sc_bn_var"),
                                   lookup("stage2_unit1_sc_bn_bias"),
                                   lookup("stage2_unit1_sc_weight"),
                                   std::vector<float>{0},  // stage1_unit3_relu
                                   lookup("stage1_unit3_bn3_mean"),
                                   lookup("stage1_unit3_bn3_scale"),
                                   std::vector<float>{2e-05},  // stage1_unit3_bn3/add
                                   lookup("stage1_unit3_bn3_var"),
                                   lookup("stage1_unit3_bn3_bias"),
                                   lookup("stage1_unit3_conv3_weight"),
                                   std::vector<float>{0},  // stage1_unit3_relu2
                                   lookup("stage1_unit3_bn2_mean"),
                                   lookup("stage1_unit3_bn2_scale"),
                                   std::vector<float>{2e-05},  // stage1_unit3_bn2/add
                                   lookup("stage1_unit3_bn2_var"),
                                   lookup("stage1_unit3_bn2_bias"),
                                   lookup("stage1_unit3_conv2_weight"),
                                   std::vector<float>{0},  // Pad_3
                                   std::vector<float>{0},  // stage1_unit3_relu1
                                   lookup("stage1_unit3_bn1_mean"),
                                   lookup("stage1_unit3_bn1_scale"),
                                   std::vector<float>{2e-05},  // stage1_unit3_bn1/add
                                   lookup("stage1_unit3_bn1_var"),
                                   lookup("stage1_unit3_bn1_bias"),
                                   lookup("stage1_unit3_conv1_weight"),
                                   std::vector<float>{0},  // stage1_unit2_relu
                                   lookup("stage1_unit2_bn3_mean"),
                                   lookup("stage1_unit2_bn3_scale"),
                                   std::vector<float>{2e-05},  // stage1_unit2_bn3/add
                                   lookup("stage1_unit2_bn3_var"),
                                   lookup("stage1_unit2_bn3_bias"),
                                   lookup("stage1_unit2_conv3_weight"),
                                   std::vector<float>{0},  // stage1_unit2_relu2
                                   lookup("stage1_unit2_bn2_mean"),
                                   lookup("stage1_unit2_bn2_scale"),
                                   std::vector<float>{2e-05},  // stage1_unit2_bn2/add
                                   lookup("stage1_unit2_bn2_var"),
                                   lookup("stage1_unit2_bn2_bias"),
                                   lookup("stage1_unit2_conv2_weight"),
                                   std::vector<float>{0},  // Pad_2
                                   std::vector<float>{0},  // stage1_unit2_relu1
                                   lookup("stage1_unit2_bn1_mean"),
                                   lookup("stage1_unit2_bn1_scale"),
                                   std::vector<float>{2e-05},  // stage1_unit2_bn1/add
                                   lookup("stage1_unit2_bn1_var"),
                                   lookup("stage1_unit2_bn1_bias"),
                                   lookup("stage1_unit2_conv1_weight"),
                                   std::vector<float>{0},  // stage1_unit1_relu
                                   lookup("stage1_unit1_sc_bn_mean"),
                                   lookup("stage1_unit1_sc_bn_scale"),
                                   std::vector<float>{2e-05},  // stage1_unit1_sc_bn/add
                                   lookup("stage1_unit1_sc_bn_var"),
                                   lookup("stage1_unit1_sc_bn_bias"),
                                   lookup("stage1_unit1_sc_weight"),
                                   std::vector<float>{0},  // PadV2
                                   std::vector<float>{0},  // relu0
                                   lookup("bn0_mean"),
                                   lookup("bn0_scale"),
                                   std::vector<float>{2e-05},  // bn0/add
                                   lookup("bn0_var"),
                                   lookup("bn0_bias"),
                                   lookup("conv0_weight"),
                                   std::vector<float>{0},  // Pad
                                   lookup("bn_data_mean"),
                                   std::vector<float>{2e-05},  // bn_data/add
                                   lookup("bn_data_var"),
                                   lookup("bn_data_bias"),
                                   inputs[0],
                                   lookup("stage1_unit1_bn3_mean"),
                                   lookup("stage1_unit1_bn3_scale"),
                                   std::vector<float>{2e-05},  // stage1_unit1_bn3/add
                                   lookup("stage1_unit1_bn3_var"),
                                   lookup("stage1_unit1_bn3_bias"),
                                   lookup("stage1_unit1_conv3_weight"),
                                   std::vector<float>{0},  // stage1_unit1_relu2
                                   lookup("stage1_unit1_bn2_mean"),
                                   lookup("stage1_unit1_bn2_scale"),
                                   std::vector<float>{2e-05},  // stage1_unit1_bn2/add
                                   lookup("stage1_unit1_bn2_var"),
                                   lookup("stage1_unit1_bn2_bias"),
                                   lookup("stage1_unit1_conv2_weight"),
                                   std::vector<float>{0},  // Pad_1
                                   std::vector<float>{0},  // stage1_unit1_relu1
                                   lookup("stage1_unit1_bn1_mean"),
                                   lookup("stage1_unit1_bn1_scale"),
                                   std::vector<float>{2e-05},  // stage1_unit1_bn1/add
                                   lookup("stage1_unit1_bn1_var"),
                                   lookup("stage1_unit1_bn1_bias"),
                                   lookup("stage1_unit1_conv1_weight"),
                                   lookup("stage2_unit1_bn3_mean"),
                                   lookup("stage2_unit1_bn3_scale"),
                                   std::vector<float>{2e-05},  // stage2_unit1_bn3/add
                                   lookup("stage2_unit1_bn3_var"),
                                   lookup("stage2_unit1_bn3_bias"),
                                   lookup("stage2_unit1_conv3_weight"),
                                   std::vector<float>{0},  // stage2_unit1_relu2
                                   lookup("stage2_unit1_bn2_mean"),
                                   lookup("stage2_unit1_bn2_scale"),
                                   std::vector<float>{2e-05},  // stage2_unit1_bn2/add
                                   lookup("stage2_unit1_bn2_var"),
                                   lookup("stage2_unit1_bn2_bias"),
                                   lookup("stage2_unit1_conv2_weight"),
                                   std::vector<float>{0},  // Pad_4
                                   std::vector<float>{0},  // stage2_unit1_relu1
                                   lookup("stage2_unit1_bn1_mean"),
                                   lookup("stage2_unit1_bn1_scale"),
                                   std::vector<float>{2e-05},  // stage2_unit1_bn1/add
                                   lookup("stage2_unit1_bn1_var"),
                                   lookup("stage2_unit1_bn1_bias"),
                                   lookup("stage2_unit1_conv1_weight"),
                                   lookup("stage3_unit1_bn3_mean"),
                                   lookup("stage3_unit1_bn3_scale"),
                                   std::vector<float>{2e-05},  // stage3_unit1_bn3/add
                                   lookup("stage3_unit1_bn3_var"),
                                   lookup("stage3_unit1_bn3_bias"),
                                   lookup("stage3_unit1_conv3_weight"),
                                   std::vector<float>{0},  // stage3_unit1_relu2
                                   lookup("stage3_unit1_bn2_mean"),
                                   lookup("stage3_unit1_bn2_scale"),
                                   std::vector<float>{2e-05},  // stage3_unit1_bn2/add
                                   lookup("stage3_unit1_bn2_var"),
                                   lookup("stage3_unit1_bn2_bias"),
                                   lookup("stage3_unit1_conv2_weight"),
                                   std::vector<float>{0},  // Pad_8
                                   std::vector<float>{0},  // stage3_unit1_relu1
                                   lookup("stage3_unit1_bn1_mean"),
                                   lookup("stage3_unit1_bn1_scale"),
                                   std::vector<float>{2e-05},  // stage3_unit1_bn1/add
                                   lookup("stage3_unit1_bn1_var"),
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
                                   std::vector<float>{0},  // Pad_14
                                   std::vector<float>{0},  // stage4_unit1_relu1
                                   lookup("stage4_unit1_bn1_mean"),
                                   lookup("stage4_unit1_bn1_scale"),
                                   lookup("stage4_unit1_bn1_var"),
                                   std::vector<float>{2e-05},  // stage4_unit1_bn1/add
                                   lookup("stage4_unit1_bn1_bias"),
                                   lookup("stage4_unit1_conv1_weight")};

  auto hlo_text = archive.model;

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
