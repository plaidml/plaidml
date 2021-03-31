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

  std::vector<MultiBuffer> args = {
      std::vector<float>{0},                                         // %arg0: RGB/inception_i3d/Mean
      lookup("RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/w:0"),  // %arg1
      std::vector<float>{98},                                        // %arg2: RGB/inception_i3d/Logits/AvgPool3D
      std::vector<float>{0},                                         // %arg3: RGB/inception_i3d/Logits/AvgPool3D_0
      std::vector<float>{0},  // %arg4: RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/Relu
      lookup("RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/beta:0"),  // %arg5
      std::vector<float>{0.001},  // %arg6: RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean:0"),      // %arg7
      lookup("RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance:0"),  // %arg8
      lookup("RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/conv_3d/w:0"),                   // %arg9
      std::vector<float>{0},      // %arg10: RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/Relu
      std::vector<float>{0.001},  // %arg11: RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/beta:0"),             // %arg12
      lookup("RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean:0"),      // %arg13
      lookup("RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance:0"),  // %arg14
      lookup("RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/conv_3d/w:0"),                   // %arg15
      std::vector<float>{0},      // %arg16: RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/Relu
      std::vector<float>{0.001},  // %arg17: RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/beta:0"),             // %arg18
      lookup("RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean:0"),      // %arg19
      lookup("RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance:0"),  // %arg20
      lookup("RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/conv_3d/w:0"),                   // %arg21
      std::vector<float>{0},      // %arg22: RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/Relu
      std::vector<float>{0.001},  // %arg23: RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/beta:0"),             // %arg24
      lookup("RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean:0"),      // %arg25
      lookup("RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance:0"),  // %arg26
      lookup("RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/conv_3d/w:0"),                   // %arg27
      std::vector<float>{0},      // %arg28: RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/Relu
      std::vector<float>{0.001},  // %arg29: RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/beta:0"),             // %arg30
      lookup("RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean:0"),      // %arg31
      lookup("RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance:0"),  // %arg32
      lookup("RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/conv_3d/w:0"),                   // %arg33
      std::vector<float>{0},      // %arg34: RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/Relu
      std::vector<float>{0.001},  // %arg35: RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/beta:0"),             // %arg36
      lookup("RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean:0"),      // %arg37
      lookup("RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance:0"),  // %arg38
      lookup("RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/conv_3d/w:0"),                   // %arg39
      std::vector<float>{0},      // %arg40: RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/Relu
      std::vector<float>{0.001},  // %arg41: RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/beta:0"),             // %arg42
      lookup("RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean:0"),      // %arg43
      lookup("RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance:0"),  // %arg44
      lookup("RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/conv_3d/w:0"),                   // %arg45
      std::vector<float>{0},      // %arg46: RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/Relu
      std::vector<float>{0.001},  // %arg47: RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/beta:0"),             // %arg48
      lookup("RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean:0"),      // %arg49
      lookup("RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance:0"),  // %arg50
      lookup("RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/conv_3d/w:0"),                   // %arg51
      std::vector<float>{0},      // %arg52: RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/Relu
      std::vector<float>{0.001},  // %arg53: RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/beta:0"),             // %arg54
      lookup("RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean:0"),      // %arg55
      lookup("RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance:0"),  // %arg56
      lookup("RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/conv_3d/w:0"),                   // %arg57
      std::vector<float>{0},      // %arg58: RGB/inception_i3d/Conv3d_2c_3x3/Relu
      std::vector<float>{0.001},  // %arg59: RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/beta:0"),             // %arg60
      lookup("RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/moving_mean:0"),      // %arg61
      lookup("RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/moving_variance:0"),  // %arg62
      lookup("RGB/inception_i3d/Conv3d_2c_3x3/conv_3d/w:0"),                   // %arg63
      std::vector<float>{0},      // %arg64: RGB/inception_i3d/Conv3d_2b_1x1/Relu
      std::vector<float>{0.001},  // %arg65: RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/beta:0"),             // %arg66
      lookup("RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/moving_mean:0"),      // %arg67
      lookup("RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/moving_variance:0"),  // %arg68
      lookup("RGB/inception_i3d/Conv3d_2b_1x1/conv_3d/w:0"),                   // %arg69
      std::vector<float>{0},      // %arg70: RGB/inception_i3d/Conv3d_1a_7x7/Relu
      std::vector<float>{0.001},  // %arg71: RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/beta:0"),             // %arg72
      lookup("RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/moving_mean:0"),      // %arg73
      lookup("RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/moving_variance:0"),  // %arg74
      lookup("RGB/inception_i3d/Conv3d_1a_7x7/conv_3d/w:0"),                   // %arg75
      inputs[0],                                                               // %arg76
      std::vector<float>{0.001},  // %arg77: RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg78
      lookup("RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg79
      lookup("RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg80
      lookup("RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg81
      std::vector<float>{0.001},  // %arg82: RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg83
      lookup("RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg84
      lookup("RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg85
      lookup("RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg86
      std::vector<float>{0},      // %arg87: RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/Relu
      std::vector<float>{0.001},  // %arg88: RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg89
      lookup("RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg90
      lookup("RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg91
      lookup("RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg92
      std::vector<float>{0.001},  // %arg93: RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg94
      lookup("RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg95
      lookup("RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg96
      lookup("RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg97
      std::vector<float>{0},      // %arg98: RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/Relu
      std::vector<float>{0.001},  // %arg99: RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg100
      lookup("RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg101
      lookup("RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg102
      lookup("RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg103
      std::vector<float>{
          0.001},  // %arg104: RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg105
      lookup("RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg106
      lookup("RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg107
      lookup("RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg108
      std::vector<float>{
          0.001},  // %arg109: RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg110
      lookup("RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg111
      lookup("RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg112
      lookup("RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg113
      std::vector<float>{0},  // %arg114: RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/Relu
      std::vector<float>{
          0.001},  // %arg115: RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg116
      lookup("RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg117
      lookup("RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg118
      lookup("RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg119
      std::vector<float>{
          0.001},  // %arg120: RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg121
      lookup("RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg122
      lookup("RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg123
      lookup("RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg124
      std::vector<float>{0},  // %arg125: RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/Relu
      std::vector<float>{
          0.001},  // %arg126: RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg127
      lookup("RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg128
      lookup("RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg129
      lookup("RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg130
      std::vector<float>{
          0.001},  // %arg131: RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg132
      lookup("RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg133
      lookup("RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg134
      lookup("RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg135
      std::vector<float>{
          0.001},  // %arg136: RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg137
      lookup("RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg138
      lookup("RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg139
      lookup("RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg140
      std::vector<float>{0},  // %arg141: RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/Relu
      std::vector<float>{
          0.001},  // %arg142: RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg143
      lookup("RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg144
      lookup("RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg145
      lookup("RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg146
      std::vector<float>{
          0.001},  // %arg147: RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg148
      lookup("RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg149
      lookup("RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg150
      lookup("RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg151
      std::vector<float>{0},  // %arg152: RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/Relu
      std::vector<float>{
          0.001},  // %arg153: RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg154
      lookup("RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg155
      lookup("RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg156
      lookup("RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg157
      std::vector<float>{
          0.001},  // %arg158: RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg159
      lookup("RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg160
      lookup("RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg161
      lookup("RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg162
      std::vector<float>{
          0.001},  // %arg163: RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg164
      lookup("RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg165
      lookup("RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg166
      lookup("RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg167
      std::vector<float>{0},  // %arg168: RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/Relu
      std::vector<float>{
          0.001},  // %arg169: RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg170
      lookup("RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg171
      lookup("RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg172
      lookup("RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg173
      std::vector<float>{
          0.001},  // %arg174: RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg175
      lookup("RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg176
      lookup("RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg177
      lookup("RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg178
      std::vector<float>{0},  // %arg179: RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/Relu
      std::vector<float>{
          0.001},  // %arg180: RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg181
      lookup("RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg182
      lookup("RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg183
      lookup("RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg184
      std::vector<float>{
          0.001},  // %arg185: RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg186
      lookup("RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg187
      lookup("RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg188
      lookup("RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg189
      std::vector<float>{
          0.001},  // %arg190: RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg191
      lookup("RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg192
      lookup("RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg193
      lookup("RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg194
      std::vector<float>{0},  // %arg195: RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/Relu
      std::vector<float>{
          0.001},  // %arg196: RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg197
      lookup("RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg198
      lookup("RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg199
      lookup("RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg200
      std::vector<float>{
          0.001},  // %arg201: RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg202
      lookup("RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg203
      lookup("RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg204
      lookup("RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg205
      std::vector<float>{0},  // %arg206: RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/Relu
      std::vector<float>{
          0.001},  // %arg207: RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg208
      lookup("RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg209
      lookup("RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg210
      lookup("RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg211
      std::vector<float>{
          0.001},  // %arg212: RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg213
      lookup("RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg214
      lookup("RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg215
      lookup("RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg216
      std::vector<float>{
          0.001},  // %arg217: RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg218
      lookup("RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg219
      lookup("RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg220
      lookup("RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg221
      std::vector<float>{0},  // %arg222: RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/Relu
      std::vector<float>{
          0.001},  // %arg223: RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg224
      lookup("RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg224
      lookup("RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg226
      lookup("RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg227
      std::vector<float>{
          0.001},  // %arg228: RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg229
      lookup("RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg230
      lookup("RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg231
      lookup("RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg232
      std::vector<float>{0},  // %arg233: RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/Relu
      std::vector<float>{
          0.001},  // %arg234: RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg235
      lookup("RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg236
      lookup("RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg237
      lookup("RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg238
      std::vector<float>{
          0.001},  // %arg239: RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg240
      lookup("RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg241
      lookup("RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg242
      lookup("RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg243
      std::vector<float>{
          0.001},  // %arg244: RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg245
      lookup("RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg246
      lookup("RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg247
      lookup("RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg248
      std::vector<float>{0},  // %arg249: RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/Relu
      std::vector<float>{
          0.001},  // %arg250: RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg251
      lookup("RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg252
      lookup("RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg253
      lookup("RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg254
      std::vector<float>{
          0.001},  // %arg255: RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg256
      lookup("RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg257
      lookup("RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg258
      lookup("RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg259
      std::vector<float>{0},  // %arg260: RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/Relu
      std::vector<float>{
          0.001},  // %arg261: RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg262
      lookup("RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg263
      lookup("RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg264
      lookup("RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg265
      std::vector<float>{
          0.001},  // %arg266: RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg267
      lookup("RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg268
      lookup("RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg269
      lookup("RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg270
      std::vector<float>{
          0.001},  // %arg271: RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg272
      lookup("RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg273
      lookup("RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg274
      lookup("RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg275
      std::vector<float>{0},  // %arg276: RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/Relu
      std::vector<float>{
          0.001},  // %arg277: RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg278
      lookup("RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg279
      lookup("RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg280
      lookup("RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg281
      std::vector<float>{
          0.001},  // %arg282: RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/beta:0"),             // %arg283
      lookup("RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/moving_mean:0"),      // %arg284
      lookup("RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/moving_variance:0"),  // %arg285
      lookup("RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/conv_3d/w:0"),                   // %arg286
      std::vector<float>{0},  // %arg287: RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/Relu
      std::vector<float>{
          0.001},  // %arg288: RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg289
      lookup("RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg290
      lookup("RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg291
      lookup("RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg292
      std::vector<float>{
          0.001},  // %arg293: RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/beta:0"),             // %arg294
      lookup("RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg295
      lookup("RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg296
      lookup("RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg297
      lookup("RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg298
      std::vector<float>{
          0.001},  // %arg299: RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg300
      lookup("RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg301
      lookup("RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg302
      std::vector<float>{0},  // %arg303: RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/Relu
      lookup("RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/beta:0"),  // %arg304
      std::vector<float>{
          0.001},  // %arg305: RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg306
      lookup("RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg307
      lookup("RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg308
      lookup("RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/beta:0"),             // %arg309
      std::vector<float>{
          0.001},  // %arg310: RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean:0"),      // %arg311
      lookup("RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance:0"),  // %arg312
      lookup("RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/conv_3d/w:0"),                   // %arg313
      std::vector<float>{0},  // %arg314: RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/Relu
      lookup("RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/beta:0"),  // %arg315
      std::vector<float>{
          0.001},  // %arg316: RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
      lookup("RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean:0"),      // %arg317
      lookup("RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance:0"),  // %arg318
      lookup("RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/conv_3d/w:0"),                   // %arg319
      lookup("RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/b:0"),                              // %arg320
  };

  std::string hlo_text = R"(
HloModule cluster_1__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_11__XlaNumResourceArgs_230_.1186

%max_F32.247 (lhs.248: f32[], rhs.249: f32[]) -> f32[] {
  %lhs.248 = f32[] parameter(0)
  %rhs.249 = f32[] parameter(1)
  ROOT %maximum.250 = f32[] maximum(f32[] %lhs.248, f32[] %rhs.249)
}

%max_F32.285 (lhs.286: f32[], rhs.287: f32[]) -> f32[] {
  %lhs.286 = f32[] parameter(0)
  %rhs.287 = f32[] parameter(1)
  ROOT %maximum.288 = f32[] maximum(f32[] %lhs.286, f32[] %rhs.287)
}

%max_F32.294 (lhs.295: f32[], rhs.296: f32[]) -> f32[] {
  %lhs.295 = f32[] parameter(0)
  %rhs.296 = f32[] parameter(1)
  ROOT %maximum.297 = f32[] maximum(f32[] %lhs.295, f32[] %rhs.296)
}

%max_F32.389 (lhs.390: f32[], rhs.391: f32[]) -> f32[] {
  %lhs.390 = f32[] parameter(0)
  %rhs.391 = f32[] parameter(1)
  ROOT %maximum.392 = f32[] maximum(f32[] %lhs.390, f32[] %rhs.391)
}

%max_F32.480 (lhs.481: f32[], rhs.482: f32[]) -> f32[] {
  %lhs.481 = f32[] parameter(0)
  %rhs.482 = f32[] parameter(1)
  ROOT %maximum.483 = f32[] maximum(f32[] %lhs.481, f32[] %rhs.482)
}

%max_F32.489 (lhs.490: f32[], rhs.491: f32[]) -> f32[] {
  %lhs.490 = f32[] parameter(0)
  %rhs.491 = f32[] parameter(1)
  ROOT %maximum.492 = f32[] maximum(f32[] %lhs.490, f32[] %rhs.491)
}

%max_F32.583 (lhs.584: f32[], rhs.585: f32[]) -> f32[] {
  %lhs.584 = f32[] parameter(0)
  %rhs.585 = f32[] parameter(1)
  ROOT %maximum.586 = f32[] maximum(f32[] %lhs.584, f32[] %rhs.585)
}

%max_F32.677 (lhs.678: f32[], rhs.679: f32[]) -> f32[] {
  %lhs.678 = f32[] parameter(0)
  %rhs.679 = f32[] parameter(1)
  ROOT %maximum.680 = f32[] maximum(f32[] %lhs.678, f32[] %rhs.679)
}

%max_F32.771 (lhs.772: f32[], rhs.773: f32[]) -> f32[] {
  %lhs.772 = f32[] parameter(0)
  %rhs.773 = f32[] parameter(1)
  ROOT %maximum.774 = f32[] maximum(f32[] %lhs.772, f32[] %rhs.773)
}

%max_F32.865 (lhs.866: f32[], rhs.867: f32[]) -> f32[] {
  %lhs.866 = f32[] parameter(0)
  %rhs.867 = f32[] parameter(1)
  ROOT %maximum.868 = f32[] maximum(f32[] %lhs.866, f32[] %rhs.867)
}

%max_F32.956 (lhs.957: f32[], rhs.958: f32[]) -> f32[] {
  %lhs.957 = f32[] parameter(0)
  %rhs.958 = f32[] parameter(1)
  ROOT %maximum.959 = f32[] maximum(f32[] %lhs.957, f32[] %rhs.958)
}

%max_F32.965 (lhs.966: f32[], rhs.967: f32[]) -> f32[] {
  %lhs.966 = f32[] parameter(0)
  %rhs.967 = f32[] parameter(1)
  ROOT %maximum.968 = f32[] maximum(f32[] %lhs.966, f32[] %rhs.967)
}

%max_F32.1059 (lhs.1060: f32[], rhs.1061: f32[]) -> f32[] {
  %lhs.1060 = f32[] parameter(0)
  %rhs.1061 = f32[] parameter(1)
  ROOT %maximum.1062 = f32[] maximum(f32[] %lhs.1060, f32[] %rhs.1061)
}

%add_F32.1156 (lhs.1157: f32[], rhs.1158: f32[]) -> f32[] {
  %lhs.1157 = f32[] parameter(0)
  %rhs.1158 = f32[] parameter(1)
  ROOT %add.1159 = f32[] add(f32[] %lhs.1157, f32[] %rhs.1158)
}

%RGB_inception_i3d_Mean-reduction.1173 (x.1174: f32[], y.1175: f32[]) -> f32[] {
  %x.1174 = f32[] parameter(0)
  %y.1175 = f32[] parameter(1)
  ROOT %add.1176 = f32[] add(f32[] %x.1174, f32[] %y.1175)
}

ENTRY %cluster_1__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_11__XlaNumResourceArgs_230_.1186 (arg0.1: f32[1,32,224,224,3], arg1.2: f32[400], arg2.3: f32[1,1,1,832,384], arg3.4: f32[1,1,1,1,384], arg4.5: f32[1,1,1,1,384], arg5.6: f32[1,1,1,1,384], arg6.7: f32[1,1,1,832,192], arg7.8: f32[1,1,1,1,192], arg8.9: f32[1,1,1,1,192], arg9.10: f32[1,1,1,1,192], arg10.11: f32[3,3,3,192,384], arg11.12: f32[1,1,1,1,384], arg12.13: f32[1,1,1,1,384], arg13.14: f32[1,1,1,1,384], arg14.15: f32[1,1,1,832,48], arg15.16: f32[1,1,1,1,48], arg16.17: f32[1,1,1,1,48], arg17.18: f32[1,1,1,1,48], arg18.19: f32[3,3,3,48,128], arg19.20: f32[1,1,1,1,128], arg20.21: f32[1,1,1,1,128], arg21.22: f32[1,1,1,1,128], arg22.23: f32[1,1,1,832,256], arg23.24: f32[1,1,1,1,256], arg24.25: f32[1,1,1,1,256], arg25.26: f32[1,1,1,1,256], arg26.27: f32[1,1,1,832,160], arg27.28: f32[1,1,1,1,160], arg28.29: f32[1,1,1,1,160], arg29.30: f32[1,1,1,1,160], arg30.31: f32[3,3,3,160,320], arg31.32: f32[1,1,1,1,320], arg32.33: f32[1,1,1,1,320], arg33.34: f32[1,1,1,1,320], arg34.35: f32[1,1,1,832,32], arg35.36: f32[1,1,1,1,32], arg36.37: f32[1,1,1,1,32], arg37.38: f32[1,1,1,1,32], arg38.39: f32[3,3,3,32,128], arg39.40: f32[1,1,1,1,128], arg40.41: f32[1,1,1,1,128], arg41.42: f32[1,1,1,1,128], arg42.43: f32[1,1,1,528,256], arg43.44: f32[1,1,1,1,256], arg44.45: f32[1,1,1,1,256], arg45.46: f32[1,1,1,1,256], arg46.47: f32[1,1,1,528,160], arg47.48: f32[1,1,1,1,160], arg48.49: f32[1,1,1,1,160], arg49.50: f32[1,1,1,1,160], arg50.51: f32[3,3,3,160,320], arg51.52: f32[1,1,1,1,320], arg52.53: f32[1,1,1,1,320], arg53.54: f32[1,1,1,1,320], arg54.55: f32[1,1,1,528,32], arg55.56: f32[1,1,1,1,32], arg56.57: f32[1,1,1,1,32], arg57.58: f32[1,1,1,1,32], arg58.59: f32[3,3,3,32,128], arg59.60: f32[1,1,1,1,128], arg60.61: f32[1,1,1,1,128], arg61.62: f32[1,1,1,1,128], arg62.63: f32[1,1,1,512,112], arg63.64: f32[1,1,1,1,112], arg64.65: f32[1,1,1,1,112], arg65.66: f32[1,1,1,1,112], arg66.67: f32[1,1,1,512,144], arg67.68: f32[1,1,1,1,144], arg68.69: f32[1,1,1,1,144], arg69.70: f32[1,1,1,1,144], arg70.71: f32[3,3,3,144,288], arg71.72: f32[1,1,1,1,288], arg72.73: f32[1,1,1,1,288], arg73.74: f32[1,1,1,1,288], arg74.75: f32[1,1,1,512,32], arg75.76: f32[1,1,1,1,32], arg76.77: f32[1,1,1,1,32], arg77.78: f32[1,1,1,1,32], arg78.79: f32[3,3,3,32,64], arg79.80: f32[1,1,1,1,64], arg80.81: f32[1,1,1,1,64], arg81.82: f32[1,1,1,1,64], arg82.83: f32[1,1,1,512,128], arg83.84: f32[1,1,1,1,128], arg84.85: f32[1,1,1,1,128], arg85.86: f32[1,1,1,1,128], arg86.87: f32[1,1,1,512,128], arg87.88: f32[1,1,1,1,128], arg88.89: f32[1,1,1,1,128], arg89.90: f32[1,1,1,1,128], arg90.91: f32[3,3,3,128,256], arg91.92: f32[1,1,1,1,256], arg92.93: f32[1,1,1,1,256], arg93.94: f32[1,1,1,1,256], arg94.95: f32[1,1,1,512,24], arg95.96: f32[1,1,1,1,24], arg96.97: f32[1,1,1,1,24], arg97.98: f32[1,1,1,1,24], arg98.99: f32[3,3,3,24,64], arg99.100: f32[1,1,1,1,64], arg100.101: f32[1,1,1,1,64], arg101.102: f32[1,1,1,1,64], arg102.103: f32[1,1,1,512,160], arg103.104: f32[1,1,1,1,160], arg104.105: f32[1,1,1,1,160], arg105.106: f32[1,1,1,1,160], arg106.107: f32[1,1,1,512,112], arg107.108: f32[1,1,1,1,112], arg108.109: f32[1,1,1,1,112], arg109.110: f32[1,1,1,1,112], arg110.111: f32[3,3,3,112,224], arg111.112: f32[1,1,1,1,224], arg112.113: f32[1,1,1,1,224], arg113.114: f32[1,1,1,1,224], arg114.115: f32[1,1,1,512,24], arg115.116: f32[1,1,1,1,24], arg116.117: f32[1,1,1,1,24], arg117.118: f32[1,1,1,1,24], arg118.119: f32[3,3,3,24,64], arg119.120: f32[1,1,1,1,64], arg120.121: f32[1,1,1,1,64], arg121.122: f32[1,1,1,1,64], arg122.123: f32[1,1,1,480,192], arg123.124: f32[1,1,1,1,192], arg124.125: f32[1,1,1,1,192], arg125.126: f32[1,1,1,1,192], arg126.127: f32[1,1,1,480,96], arg127.128: f32[1,1,1,1,96], arg128.129: f32[1,1,1,1,96], arg129.130: f32[1,1,1,1,96], arg130.131: f32[3,3,3,96,208], arg131.132: f32[1,1,1,1,208], arg132.133: f32[1,1,1,1,208], arg133.134: f32[1,1,1,1,208], arg134.135: f32[1,1,1,480,16], arg135.136: f32[1,1,1,1,16], arg136.137: f32[1,1,1,1,16], arg137.138: f32[1,1,1,1,16], arg138.139: f32[3,3,3,16,48], arg139.140: f32[1,1,1,1,48], arg140.141: f32[1,1,1,1,48], arg141.142: f32[1,1,1,1,48], arg142.143: f32[1,1,1,256,128], arg143.144: f32[1,1,1,1,128], arg144.145: f32[1,1,1,1,128], arg145.146: f32[1,1,1,1,128], arg146.147: f32[1,1,1,256,128], arg147.148: f32[1,1,1,1,128], arg148.149: f32[1,1,1,1,128], arg149.150: f32[1,1,1,1,128], arg150.151: f32[3,3,3,128,192], arg151.152: f32[1,1,1,1,192], arg152.153: f32[1,1,1,1,192], arg153.154: f32[1,1,1,1,192], arg154.155: f32[1,1,1,256,32], arg155.156: f32[1,1,1,1,32], arg156.157: f32[1,1,1,1,32], arg157.158: f32[1,1,1,1,32], arg158.159: f32[3,3,3,32,96], arg159.160: f32[1,1,1,1,96], arg160.161: f32[1,1,1,1,96], arg161.162: f32[1,1,1,1,96], arg162.163: f32[1,1,1,192,64], arg163.164: f32[1,1,1,1,64], arg164.165: f32[1,1,1,1,64], arg165.166: f32[1,1,1,1,64], arg166.167: f32[1,1,1,192,96], arg167.168: f32[1,1,1,1,96], arg168.169: f32[1,1,1,1,96], arg169.170: f32[1,1,1,1,96], arg170.171: f32[3,3,3,96,128], arg171.172: f32[1,1,1,1,128], arg172.173: f32[1,1,1,1,128], arg173.174: f32[1,1,1,1,128], arg174.175: f32[1,1,1,192,16], arg175.176: f32[1,1,1,1,16], arg176.177: f32[1,1,1,1,16], arg177.178: f32[1,1,1,1,16], arg178.179: f32[3,3,3,16,32], arg179.180: f32[1,1,1,1,32], arg180.181: f32[1,1,1,1,32], arg181.182: f32[1,1,1,1,32], arg182.183: f32[7,7,7,3,64], arg183.184: f32[1,1,1,1,64], arg184.185: f32[1,1,1,1,64], arg185.186: f32[1,1,1,1,64], arg186.187: f32[1,1,1,64,64], arg187.188: f32[1,1,1,1,64], arg188.189: f32[1,1,1,1,64], arg189.190: f32[1,1,1,1,64], arg190.191: f32[3,3,3,64,192], arg191.192: f32[1,1,1,1,192], arg192.193: f32[1,1,1,1,192], arg193.194: f32[1,1,1,1,192], arg194.195: f32[1,1,1,192,32], arg195.196: f32[1,1,1,1,32], arg196.197: f32[1,1,1,1,32], arg197.198: f32[1,1,1,1,32], arg198.199: f32[1,1,1,256,64], arg199.200: f32[1,1,1,1,64], arg200.201: f32[1,1,1,1,64], arg201.202: f32[1,1,1,1,64], arg202.203: f32[1,1,1,480,64], arg203.204: f32[1,1,1,1,64], arg204.205: f32[1,1,1,1,64], arg205.206: f32[1,1,1,1,64], arg206.207: f32[1,1,1,512,64], arg207.208: f32[1,1,1,1,64], arg208.209: f32[1,1,1,1,64], arg209.210: f32[1,1,1,1,64], arg210.211: f32[1,1,1,512,64], arg211.212: f32[1,1,1,1,64], arg212.213: f32[1,1,1,1,64], arg213.214: f32[1,1,1,1,64], arg214.215: f32[1,1,1,512,64], arg215.216: f32[1,1,1,1,64], arg216.217: f32[1,1,1,1,64], arg217.218: f32[1,1,1,1,64], arg218.219: f32[1,1,1,528,128], arg219.220: f32[1,1,1,1,128], arg220.221: f32[1,1,1,1,128], arg221.222: f32[1,1,1,1,128], arg222.223: f32[1,1,1,832,128], arg223.224: f32[1,1,1,1,128], arg224.225: f32[1,1,1,1,128], arg225.226: f32[1,1,1,1,128], arg226.227: f32[1,1,1,832,128], arg227.228: f32[1,1,1,1,128], arg228.229: f32[1,1,1,1,128], arg229.230: f32[1,1,1,1,128], arg230.231: f32[1,1,1,1024,400]) -> f32[1,400] {
  %arg1.2 = f32[400]{0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.299 = f32[1,1,1,1,400]{4,3,2,1,0} reshape(f32[400]{0} %arg1.2), metadata={op_type="Reshape" op_name="RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/Reshape"}
  %reshape.1166 = f32[1,1,1,400]{3,2,1,0} reshape(f32[1,1,1,1,400]{4,3,2,1,0} %reshape.299), metadata={op_type="Add" op_name="RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/add"}
  %broadcast.1167 = f32[1,3,1,1,400]{4,3,2,1,0} broadcast(f32[1,1,1,400]{3,2,1,0} %reshape.1166), dimensions={0,2,3,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/add"}
  %constant.1149 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.1150 = f32[1,4,7,7,1024]{4,3,2,1,0} broadcast(f32[] %constant.1149), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg5.6 = f32[1,1,1,1,384]{4,3,2,1,0} parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.1064 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.1065 = f32[1,1,1,1,384]{4,3,2,1,0} broadcast(f32[] %constant.1064), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.1066 = f32[1,1,1,1,384]{4,3,2,1,0} add(f32[1,1,1,1,384]{4,3,2,1,0} %arg5.6, f32[1,1,1,1,384]{4,3,2,1,0} %broadcast.1065), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1067 = f32[1,1,1,1,384]{4,3,2,1,0} rsqrt(f32[1,1,1,1,384]{4,3,2,1,0} %add.1066), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1071 = f32[1,384]{1,0} reshape(f32[1,1,1,1,384]{4,3,2,1,0} %rsqrt.1067), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1072 = f32[1,4,7,7,384]{4,3,2,1,0} broadcast(f32[1,384]{1,0} %reshape.1071), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %constant.1055 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.1056 = f32[1,4,7,7,832]{4,3,2,1,0} broadcast(f32[] %constant.1055), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg25.26 = f32[1,1,1,1,256]{4,3,2,1,0} parameter(25), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.970 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.971 = f32[1,1,1,1,256]{4,3,2,1,0} broadcast(f32[] %constant.970), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.972 = f32[1,1,1,1,256]{4,3,2,1,0} add(f32[1,1,1,1,256]{4,3,2,1,0} %arg25.26, f32[1,1,1,1,256]{4,3,2,1,0} %broadcast.971), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.973 = f32[1,1,1,1,256]{4,3,2,1,0} rsqrt(f32[1,1,1,1,256]{4,3,2,1,0} %add.972), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.977 = f32[1,256]{1,0} reshape(f32[1,1,1,1,256]{4,3,2,1,0} %rsqrt.973), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.978 = f32[1,4,7,7,256]{4,3,2,1,0} broadcast(f32[1,256]{1,0} %reshape.977), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %constant.961 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.962 = f32[1,4,7,7,832]{4,3,2,1,0} broadcast(f32[] %constant.961), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg45.46 = f32[1,1,1,1,256]{4,3,2,1,0} parameter(45), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.870 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.871 = f32[1,1,1,1,256]{4,3,2,1,0} broadcast(f32[] %constant.870), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.872 = f32[1,1,1,1,256]{4,3,2,1,0} add(f32[1,1,1,1,256]{4,3,2,1,0} %arg45.46, f32[1,1,1,1,256]{4,3,2,1,0} %broadcast.871), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.873 = f32[1,1,1,1,256]{4,3,2,1,0} rsqrt(f32[1,1,1,1,256]{4,3,2,1,0} %add.872), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.877 = f32[1,256]{1,0} reshape(f32[1,1,1,1,256]{4,3,2,1,0} %rsqrt.873), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.878 = f32[1,8,14,14,256]{4,3,2,1,0} broadcast(f32[1,256]{1,0} %reshape.877), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %constant.861 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.862 = f32[1,8,14,14,528]{4,3,2,1,0} broadcast(f32[] %constant.861), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg65.66 = f32[1,1,1,1,112]{4,3,2,1,0} parameter(65), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.776 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.777 = f32[1,1,1,1,112]{4,3,2,1,0} broadcast(f32[] %constant.776), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.778 = f32[1,1,1,1,112]{4,3,2,1,0} add(f32[1,1,1,1,112]{4,3,2,1,0} %arg65.66, f32[1,1,1,1,112]{4,3,2,1,0} %broadcast.777), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.779 = f32[1,1,1,1,112]{4,3,2,1,0} rsqrt(f32[1,1,1,1,112]{4,3,2,1,0} %add.778), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.783 = f32[1,112]{1,0} reshape(f32[1,1,1,1,112]{4,3,2,1,0} %rsqrt.779), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.784 = f32[1,8,14,14,112]{4,3,2,1,0} broadcast(f32[1,112]{1,0} %reshape.783), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %constant.767 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.768 = f32[1,8,14,14,512]{4,3,2,1,0} broadcast(f32[] %constant.767), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg85.86 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(85), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.682 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.683 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.682), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.684 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %arg85.86, f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.683), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.685 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.684), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.689 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.685), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.690 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.689), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %constant.673 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.674 = f32[1,8,14,14,512]{4,3,2,1,0} broadcast(f32[] %constant.673), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg105.106 = f32[1,1,1,1,160]{4,3,2,1,0} parameter(105), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.588 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.589 = f32[1,1,1,1,160]{4,3,2,1,0} broadcast(f32[] %constant.588), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.590 = f32[1,1,1,1,160]{4,3,2,1,0} add(f32[1,1,1,1,160]{4,3,2,1,0} %arg105.106, f32[1,1,1,1,160]{4,3,2,1,0} %broadcast.589), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.591 = f32[1,1,1,1,160]{4,3,2,1,0} rsqrt(f32[1,1,1,1,160]{4,3,2,1,0} %add.590), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.595 = f32[1,160]{1,0} reshape(f32[1,1,1,1,160]{4,3,2,1,0} %rsqrt.591), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.596 = f32[1,8,14,14,160]{4,3,2,1,0} broadcast(f32[1,160]{1,0} %reshape.595), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %constant.579 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.580 = f32[1,8,14,14,512]{4,3,2,1,0} broadcast(f32[] %constant.579), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg125.126 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(125), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.494 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.495 = f32[1,1,1,1,192]{4,3,2,1,0} broadcast(f32[] %constant.494), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.496 = f32[1,1,1,1,192]{4,3,2,1,0} add(f32[1,1,1,1,192]{4,3,2,1,0} %arg125.126, f32[1,1,1,1,192]{4,3,2,1,0} %broadcast.495), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.497 = f32[1,1,1,1,192]{4,3,2,1,0} rsqrt(f32[1,1,1,1,192]{4,3,2,1,0} %add.496), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.501 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.497), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.502 = f32[1,8,14,14,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.501), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %constant.485 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.486 = f32[1,8,14,14,480]{4,3,2,1,0} broadcast(f32[] %constant.485), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg145.146 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(145), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.394 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.395 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.394), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.396 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %arg145.146, f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.395), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.397 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.396), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.401 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.397), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.402 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.401), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %constant.385 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.386 = f32[1,16,28,28,256]{4,3,2,1,0} broadcast(f32[] %constant.385), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg165.166 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(165), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.300 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.301 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.300), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.302 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %arg165.166, f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.301), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.303 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.302), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.307 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.303), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.308 = f32[1,16,28,28,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.307), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %constant.290 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2c_3x3/Relu"}
  %broadcast.291 = f32[1,16,28,28,192]{4,3,2,1,0} broadcast(f32[] %constant.290), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2c_3x3/Relu"}
  %arg193.194 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(193), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.271 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add"}
  %broadcast.272 = f32[1,1,1,1,192]{4,3,2,1,0} broadcast(f32[] %constant.271), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add"}
  %add.273 = f32[1,1,1,1,192]{4,3,2,1,0} add(f32[1,1,1,1,192]{4,3,2,1,0} %arg193.194, f32[1,1,1,1,192]{4,3,2,1,0} %broadcast.272), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add"}
  %rsqrt.274 = f32[1,1,1,1,192]{4,3,2,1,0} rsqrt(f32[1,1,1,1,192]{4,3,2,1,0} %add.273), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.278 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.274), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/mul"}
  %broadcast.279 = f32[1,16,56,56,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.278), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/mul"}
  %constant.268 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2b_1x1/Relu"}
  %broadcast.269 = f32[1,16,56,56,64]{4,3,2,1,0} broadcast(f32[] %constant.268), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2b_1x1/Relu"}
  %arg189.190 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(189), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.255 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add"}
  %broadcast.256 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.255), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add"}
  %add.257 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %arg189.190, f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.256), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.258 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.257), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.262 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.258), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.263 = f32[1,16,56,56,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.262), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/mul"}
  %constant.252 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_1a_7x7/Relu"}
  %broadcast.253 = f32[1,16,56,56,64]{4,3,2,1,0} broadcast(f32[] %constant.252), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_1a_7x7/Relu"}
  %arg185.186 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(185), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.233 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add"}
  %broadcast.234 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.233), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add"}
  %add.235 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %arg185.186, f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.234), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add"}
  %rsqrt.236 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.235), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/Rsqrt"}
  %reshape.240 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.236), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/mul"}
  %broadcast.241 = f32[1,16,112,112,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.240), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/mul"}
  %arg0.1 = f32[1,32,224,224,3]{4,3,2,1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.232 = f32[1,32,224,224,3]{4,3,2,1,0} reshape(f32[1,32,224,224,3]{4,3,2,1,0} %arg0.1)
  %arg182.183 = f32[7,7,7,3,64]{4,3,2,1,0} parameter(182), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.239 = f32[1,16,112,112,64]{4,3,2,1,0} convolution(f32[1,32,224,224,3]{4,3,2,1,0} %reshape.232, f32[7,7,7,3,64]{4,3,2,1,0} %arg182.183), window={size=7x7x7 stride=2x2x2 pad=2_3x2_3x2_3}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Conv3d_1a_7x7/conv_3d/convolution"}
  %multiply.242 = f32[1,16,112,112,64]{4,3,2,1,0} multiply(f32[1,16,112,112,64]{4,3,2,1,0} %broadcast.241, f32[1,16,112,112,64]{4,3,2,1,0} %convolution.239), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/mul"}
  %arg183.184 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(183), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg184.185 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(184), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.237 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %arg184.185, f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.236), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/mul_1"}
  %subtract.238 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg183.184, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.237), metadata={op_type="Sub" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/sub"}
  %reshape.243 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.238), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add_1"}
  %broadcast.244 = f32[1,16,112,112,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.243), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add_1"}
  %add.245 = f32[1,16,112,112,64]{4,3,2,1,0} add(f32[1,16,112,112,64]{4,3,2,1,0} %multiply.242, f32[1,16,112,112,64]{4,3,2,1,0} %broadcast.244), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add_1"}
  %constant.246 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_2a_3x3"}
  %reduce-window.251 = f32[1,16,56,56,64]{4,3,2,1,0} reduce-window(f32[1,16,112,112,64]{4,3,2,1,0} %add.245, f32[] %constant.246), window={size=1x1x3x3x1 stride=1x1x2x2x1 pad=0_0x0_0x0_1x0_1x0_0}, to_apply=%max_F32.247, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_2a_3x3"}
  %maximum.254 = f32[1,16,56,56,64]{4,3,2,1,0} maximum(f32[1,16,56,56,64]{4,3,2,1,0} %broadcast.253, f32[1,16,56,56,64]{4,3,2,1,0} %reduce-window.251), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_1a_7x7/Relu"}
  %arg186.187 = f32[1,1,1,64,64]{4,3,2,1,0} parameter(186), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.261 = f32[1,16,56,56,64]{4,3,2,1,0} convolution(f32[1,16,56,56,64]{4,3,2,1,0} %maximum.254, f32[1,1,1,64,64]{4,3,2,1,0} %arg186.187), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Conv3d_2b_1x1/conv_3d/convolution"}
  %multiply.264 = f32[1,16,56,56,64]{4,3,2,1,0} multiply(f32[1,16,56,56,64]{4,3,2,1,0} %broadcast.263, f32[1,16,56,56,64]{4,3,2,1,0} %convolution.261), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/mul"}
  %arg187.188 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(187), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg188.189 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(188), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.259 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %arg188.189, f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.258), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.260 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg187.188, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.259), metadata={op_type="Sub" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/sub"}
  %reshape.265 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.260), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.266 = f32[1,16,56,56,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.265), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add_1"}
  %add.267 = f32[1,16,56,56,64]{4,3,2,1,0} add(f32[1,16,56,56,64]{4,3,2,1,0} %multiply.264, f32[1,16,56,56,64]{4,3,2,1,0} %broadcast.266), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add_1"}
  %maximum.270 = f32[1,16,56,56,64]{4,3,2,1,0} maximum(f32[1,16,56,56,64]{4,3,2,1,0} %broadcast.269, f32[1,16,56,56,64]{4,3,2,1,0} %add.267), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2b_1x1/Relu"}
  %arg190.191 = f32[3,3,3,64,192]{4,3,2,1,0} parameter(190), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.277 = f32[1,16,56,56,192]{4,3,2,1,0} convolution(f32[1,16,56,56,64]{4,3,2,1,0} %maximum.270, f32[3,3,3,64,192]{4,3,2,1,0} %arg190.191), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Conv3d_2c_3x3/conv_3d/convolution"}
  %multiply.280 = f32[1,16,56,56,192]{4,3,2,1,0} multiply(f32[1,16,56,56,192]{4,3,2,1,0} %broadcast.279, f32[1,16,56,56,192]{4,3,2,1,0} %convolution.277), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/mul"}
  %arg191.192 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(191), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg192.193 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(192), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.275 = f32[1,1,1,1,192]{4,3,2,1,0} multiply(f32[1,1,1,1,192]{4,3,2,1,0} %arg192.193, f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.274), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.276 = f32[1,1,1,1,192]{4,3,2,1,0} subtract(f32[1,1,1,1,192]{4,3,2,1,0} %arg191.192, f32[1,1,1,1,192]{4,3,2,1,0} %multiply.275), metadata={op_type="Sub" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/sub"}
  %reshape.281 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %subtract.276), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.282 = f32[1,16,56,56,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.281), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add_1"}
  %add.283 = f32[1,16,56,56,192]{4,3,2,1,0} add(f32[1,16,56,56,192]{4,3,2,1,0} %multiply.280, f32[1,16,56,56,192]{4,3,2,1,0} %broadcast.282), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add_1"}
  %constant.284 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_3a_3x3"}
  %reduce-window.289 = f32[1,16,28,28,192]{4,3,2,1,0} reduce-window(f32[1,16,56,56,192]{4,3,2,1,0} %add.283, f32[] %constant.284), window={size=1x1x3x3x1 stride=1x1x2x2x1 pad=0_0x0_0x0_1x0_1x0_0}, to_apply=%max_F32.285, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_3a_3x3"}
  %maximum.292 = f32[1,16,28,28,192]{4,3,2,1,0} maximum(f32[1,16,28,28,192]{4,3,2,1,0} %broadcast.291, f32[1,16,28,28,192]{4,3,2,1,0} %reduce-window.289), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2c_3x3/Relu"}
  %arg162.163 = f32[1,1,1,192,64]{4,3,2,1,0} parameter(162), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.306 = f32[1,16,28,28,64]{4,3,2,1,0} convolution(f32[1,16,28,28,192]{4,3,2,1,0} %maximum.292, f32[1,1,1,192,64]{4,3,2,1,0} %arg162.163), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.309 = f32[1,16,28,28,64]{4,3,2,1,0} multiply(f32[1,16,28,28,64]{4,3,2,1,0} %broadcast.308, f32[1,16,28,28,64]{4,3,2,1,0} %convolution.306), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg163.164 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(163), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg164.165 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(164), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.304 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %arg164.165, f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.303), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.305 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg163.164, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.304), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.310 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.305), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.311 = f32[1,16,28,28,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.310), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.312 = f32[1,16,28,28,64]{4,3,2,1,0} add(f32[1,16,28,28,64]{4,3,2,1,0} %multiply.309, f32[1,16,28,28,64]{4,3,2,1,0} %broadcast.311), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %arg173.174 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(173), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.329 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.330 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.329), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %add.331 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %arg173.174, f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.330), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.332 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.331), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.336 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.332), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.337 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.336), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.326 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.327 = f32[1,16,28,28,96]{4,3,2,1,0} broadcast(f32[] %constant.326), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg169.170 = f32[1,1,1,1,96]{4,3,2,1,0} parameter(169), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.313 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.314 = f32[1,1,1,1,96]{4,3,2,1,0} broadcast(f32[] %constant.313), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.315 = f32[1,1,1,1,96]{4,3,2,1,0} add(f32[1,1,1,1,96]{4,3,2,1,0} %arg169.170, f32[1,1,1,1,96]{4,3,2,1,0} %broadcast.314), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.316 = f32[1,1,1,1,96]{4,3,2,1,0} rsqrt(f32[1,1,1,1,96]{4,3,2,1,0} %add.315), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.320 = f32[1,96]{1,0} reshape(f32[1,1,1,1,96]{4,3,2,1,0} %rsqrt.316), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.321 = f32[1,16,28,28,96]{4,3,2,1,0} broadcast(f32[1,96]{1,0} %reshape.320), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg166.167 = f32[1,1,1,192,96]{4,3,2,1,0} parameter(166), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.319 = f32[1,16,28,28,96]{4,3,2,1,0} convolution(f32[1,16,28,28,192]{4,3,2,1,0} %maximum.292, f32[1,1,1,192,96]{4,3,2,1,0} %arg166.167), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.322 = f32[1,16,28,28,96]{4,3,2,1,0} multiply(f32[1,16,28,28,96]{4,3,2,1,0} %broadcast.321, f32[1,16,28,28,96]{4,3,2,1,0} %convolution.319), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg167.168 = f32[1,1,1,1,96]{4,3,2,1,0} parameter(167), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg168.169 = f32[1,1,1,1,96]{4,3,2,1,0} parameter(168), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.317 = f32[1,1,1,1,96]{4,3,2,1,0} multiply(f32[1,1,1,1,96]{4,3,2,1,0} %arg168.169, f32[1,1,1,1,96]{4,3,2,1,0} %rsqrt.316), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.318 = f32[1,1,1,1,96]{4,3,2,1,0} subtract(f32[1,1,1,1,96]{4,3,2,1,0} %arg167.168, f32[1,1,1,1,96]{4,3,2,1,0} %multiply.317), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.323 = f32[1,96]{1,0} reshape(f32[1,1,1,1,96]{4,3,2,1,0} %subtract.318), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.324 = f32[1,16,28,28,96]{4,3,2,1,0} broadcast(f32[1,96]{1,0} %reshape.323), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.325 = f32[1,16,28,28,96]{4,3,2,1,0} add(f32[1,16,28,28,96]{4,3,2,1,0} %multiply.322, f32[1,16,28,28,96]{4,3,2,1,0} %broadcast.324), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.328 = f32[1,16,28,28,96]{4,3,2,1,0} maximum(f32[1,16,28,28,96]{4,3,2,1,0} %broadcast.327, f32[1,16,28,28,96]{4,3,2,1,0} %add.325), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg170.171 = f32[3,3,3,96,128]{4,3,2,1,0} parameter(170), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.335 = f32[1,16,28,28,128]{4,3,2,1,0} convolution(f32[1,16,28,28,96]{4,3,2,1,0} %maximum.328, f32[3,3,3,96,128]{4,3,2,1,0} %arg170.171), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.338 = f32[1,16,28,28,128]{4,3,2,1,0} multiply(f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.337, f32[1,16,28,28,128]{4,3,2,1,0} %convolution.335), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg171.172 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(171), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg172.173 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(172), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.333 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %arg172.173, f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.332), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.334 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg171.172, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.333), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.339 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.334), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.340 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.339), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.341 = f32[1,16,28,28,128]{4,3,2,1,0} add(f32[1,16,28,28,128]{4,3,2,1,0} %multiply.338, f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.340), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %arg181.182 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(181), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.358 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.359 = f32[1,1,1,1,32]{4,3,2,1,0} broadcast(f32[] %constant.358), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %add.360 = f32[1,1,1,1,32]{4,3,2,1,0} add(f32[1,1,1,1,32]{4,3,2,1,0} %arg181.182, f32[1,1,1,1,32]{4,3,2,1,0} %broadcast.359), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.361 = f32[1,1,1,1,32]{4,3,2,1,0} rsqrt(f32[1,1,1,1,32]{4,3,2,1,0} %add.360), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.365 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.361), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.366 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.365), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.355 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.356 = f32[1,16,28,28,16]{4,3,2,1,0} broadcast(f32[] %constant.355), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg177.178 = f32[1,1,1,1,16]{4,3,2,1,0} parameter(177), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.342 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.343 = f32[1,1,1,1,16]{4,3,2,1,0} broadcast(f32[] %constant.342), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.344 = f32[1,1,1,1,16]{4,3,2,1,0} add(f32[1,1,1,1,16]{4,3,2,1,0} %arg177.178, f32[1,1,1,1,16]{4,3,2,1,0} %broadcast.343), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.345 = f32[1,1,1,1,16]{4,3,2,1,0} rsqrt(f32[1,1,1,1,16]{4,3,2,1,0} %add.344), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.349 = f32[1,16]{1,0} reshape(f32[1,1,1,1,16]{4,3,2,1,0} %rsqrt.345), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.350 = f32[1,16,28,28,16]{4,3,2,1,0} broadcast(f32[1,16]{1,0} %reshape.349), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg174.175 = f32[1,1,1,192,16]{4,3,2,1,0} parameter(174), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.348 = f32[1,16,28,28,16]{4,3,2,1,0} convolution(f32[1,16,28,28,192]{4,3,2,1,0} %maximum.292, f32[1,1,1,192,16]{4,3,2,1,0} %arg174.175), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.351 = f32[1,16,28,28,16]{4,3,2,1,0} multiply(f32[1,16,28,28,16]{4,3,2,1,0} %broadcast.350, f32[1,16,28,28,16]{4,3,2,1,0} %convolution.348), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg175.176 = f32[1,1,1,1,16]{4,3,2,1,0} parameter(175), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg176.177 = f32[1,1,1,1,16]{4,3,2,1,0} parameter(176), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.346 = f32[1,1,1,1,16]{4,3,2,1,0} multiply(f32[1,1,1,1,16]{4,3,2,1,0} %arg176.177, f32[1,1,1,1,16]{4,3,2,1,0} %rsqrt.345), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.347 = f32[1,1,1,1,16]{4,3,2,1,0} subtract(f32[1,1,1,1,16]{4,3,2,1,0} %arg175.176, f32[1,1,1,1,16]{4,3,2,1,0} %multiply.346), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.352 = f32[1,16]{1,0} reshape(f32[1,1,1,1,16]{4,3,2,1,0} %subtract.347), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.353 = f32[1,16,28,28,16]{4,3,2,1,0} broadcast(f32[1,16]{1,0} %reshape.352), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.354 = f32[1,16,28,28,16]{4,3,2,1,0} add(f32[1,16,28,28,16]{4,3,2,1,0} %multiply.351, f32[1,16,28,28,16]{4,3,2,1,0} %broadcast.353), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.357 = f32[1,16,28,28,16]{4,3,2,1,0} maximum(f32[1,16,28,28,16]{4,3,2,1,0} %broadcast.356, f32[1,16,28,28,16]{4,3,2,1,0} %add.354), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg178.179 = f32[3,3,3,16,32]{4,3,2,1,0} parameter(178), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.364 = f32[1,16,28,28,32]{4,3,2,1,0} convolution(f32[1,16,28,28,16]{4,3,2,1,0} %maximum.357, f32[3,3,3,16,32]{4,3,2,1,0} %arg178.179), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.367 = f32[1,16,28,28,32]{4,3,2,1,0} multiply(f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.366, f32[1,16,28,28,32]{4,3,2,1,0} %convolution.364), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg179.180 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(179), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg180.181 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(180), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.362 = f32[1,1,1,1,32]{4,3,2,1,0} multiply(f32[1,1,1,1,32]{4,3,2,1,0} %arg180.181, f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.361), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.363 = f32[1,1,1,1,32]{4,3,2,1,0} subtract(f32[1,1,1,1,32]{4,3,2,1,0} %arg179.180, f32[1,1,1,1,32]{4,3,2,1,0} %multiply.362), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.368 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %subtract.363), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.369 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.368), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.370 = f32[1,16,28,28,32]{4,3,2,1,0} add(f32[1,16,28,28,32]{4,3,2,1,0} %multiply.367, f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.369), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %arg197.198 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(197), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.371 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.372 = f32[1,1,1,1,32]{4,3,2,1,0} broadcast(f32[] %constant.371), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %add.373 = f32[1,1,1,1,32]{4,3,2,1,0} add(f32[1,1,1,1,32]{4,3,2,1,0} %arg197.198, f32[1,1,1,1,32]{4,3,2,1,0} %broadcast.372), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.374 = f32[1,1,1,1,32]{4,3,2,1,0} rsqrt(f32[1,1,1,1,32]{4,3,2,1,0} %add.373), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.378 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.374), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.379 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.378), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %constant.293 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.298 = f32[1,16,28,28,192]{4,3,2,1,0} reduce-window(f32[1,16,28,28,192]{4,3,2,1,0} %maximum.292, f32[] %constant.293), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.294, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/MaxPool3d_0a_3x3"}
  %arg194.195 = f32[1,1,1,192,32]{4,3,2,1,0} parameter(194), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.377 = f32[1,16,28,28,32]{4,3,2,1,0} convolution(f32[1,16,28,28,192]{4,3,2,1,0} %reduce-window.298, f32[1,1,1,192,32]{4,3,2,1,0} %arg194.195), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %multiply.380 = f32[1,16,28,28,32]{4,3,2,1,0} multiply(f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.379, f32[1,16,28,28,32]{4,3,2,1,0} %convolution.377), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg195.196 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(195), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg196.197 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(196), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.375 = f32[1,1,1,1,32]{4,3,2,1,0} multiply(f32[1,1,1,1,32]{4,3,2,1,0} %arg196.197, f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.374), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.376 = f32[1,1,1,1,32]{4,3,2,1,0} subtract(f32[1,1,1,1,32]{4,3,2,1,0} %arg195.196, f32[1,1,1,1,32]{4,3,2,1,0} %multiply.375), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.381 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %subtract.376), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.382 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.381), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.383 = f32[1,16,28,28,32]{4,3,2,1,0} add(f32[1,16,28,28,32]{4,3,2,1,0} %multiply.380, f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.382), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.384 = f32[1,16,28,28,256]{4,3,2,1,0} concatenate(f32[1,16,28,28,64]{4,3,2,1,0} %add.312, f32[1,16,28,28,128]{4,3,2,1,0} %add.341, f32[1,16,28,28,32]{4,3,2,1,0} %add.370, f32[1,16,28,28,32]{4,3,2,1,0} %add.383), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_3b/concat"}
  %maximum.387 = f32[1,16,28,28,256]{4,3,2,1,0} maximum(f32[1,16,28,28,256]{4,3,2,1,0} %broadcast.386, f32[1,16,28,28,256]{4,3,2,1,0} %concatenate.384), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg142.143 = f32[1,1,1,256,128]{4,3,2,1,0} parameter(142), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.400 = f32[1,16,28,28,128]{4,3,2,1,0} convolution(f32[1,16,28,28,256]{4,3,2,1,0} %maximum.387, f32[1,1,1,256,128]{4,3,2,1,0} %arg142.143), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.403 = f32[1,16,28,28,128]{4,3,2,1,0} multiply(f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.402, f32[1,16,28,28,128]{4,3,2,1,0} %convolution.400), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg143.144 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(143), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg144.145 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(144), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.398 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %arg144.145, f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.397), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.399 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg143.144, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.398), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.404 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.399), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.405 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.404), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.406 = f32[1,16,28,28,128]{4,3,2,1,0} add(f32[1,16,28,28,128]{4,3,2,1,0} %multiply.403, f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.405), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %arg153.154 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(153), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.423 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.424 = f32[1,1,1,1,192]{4,3,2,1,0} broadcast(f32[] %constant.423), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %add.425 = f32[1,1,1,1,192]{4,3,2,1,0} add(f32[1,1,1,1,192]{4,3,2,1,0} %arg153.154, f32[1,1,1,1,192]{4,3,2,1,0} %broadcast.424), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.426 = f32[1,1,1,1,192]{4,3,2,1,0} rsqrt(f32[1,1,1,1,192]{4,3,2,1,0} %add.425), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.430 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.426), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.431 = f32[1,16,28,28,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.430), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.420 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.421 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[] %constant.420), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg149.150 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(149), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.407 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.408 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.407), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.409 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %arg149.150, f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.408), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.410 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.409), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.414 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.410), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.415 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.414), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg146.147 = f32[1,1,1,256,128]{4,3,2,1,0} parameter(146), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.413 = f32[1,16,28,28,128]{4,3,2,1,0} convolution(f32[1,16,28,28,256]{4,3,2,1,0} %maximum.387, f32[1,1,1,256,128]{4,3,2,1,0} %arg146.147), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.416 = f32[1,16,28,28,128]{4,3,2,1,0} multiply(f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.415, f32[1,16,28,28,128]{4,3,2,1,0} %convolution.413), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg147.148 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(147), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg148.149 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(148), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.411 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %arg148.149, f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.410), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.412 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg147.148, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.411), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.417 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.412), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.418 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.417), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.419 = f32[1,16,28,28,128]{4,3,2,1,0} add(f32[1,16,28,28,128]{4,3,2,1,0} %multiply.416, f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.418), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.422 = f32[1,16,28,28,128]{4,3,2,1,0} maximum(f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.421, f32[1,16,28,28,128]{4,3,2,1,0} %add.419), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg150.151 = f32[3,3,3,128,192]{4,3,2,1,0} parameter(150), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.429 = f32[1,16,28,28,192]{4,3,2,1,0} convolution(f32[1,16,28,28,128]{4,3,2,1,0} %maximum.422, f32[3,3,3,128,192]{4,3,2,1,0} %arg150.151), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.432 = f32[1,16,28,28,192]{4,3,2,1,0} multiply(f32[1,16,28,28,192]{4,3,2,1,0} %broadcast.431, f32[1,16,28,28,192]{4,3,2,1,0} %convolution.429), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg151.152 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(151), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg152.153 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(152), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.427 = f32[1,1,1,1,192]{4,3,2,1,0} multiply(f32[1,1,1,1,192]{4,3,2,1,0} %arg152.153, f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.426), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.428 = f32[1,1,1,1,192]{4,3,2,1,0} subtract(f32[1,1,1,1,192]{4,3,2,1,0} %arg151.152, f32[1,1,1,1,192]{4,3,2,1,0} %multiply.427), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.433 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %subtract.428), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.434 = f32[1,16,28,28,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.433), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.435 = f32[1,16,28,28,192]{4,3,2,1,0} add(f32[1,16,28,28,192]{4,3,2,1,0} %multiply.432, f32[1,16,28,28,192]{4,3,2,1,0} %broadcast.434), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %arg161.162 = f32[1,1,1,1,96]{4,3,2,1,0} parameter(161), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.452 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.453 = f32[1,1,1,1,96]{4,3,2,1,0} broadcast(f32[] %constant.452), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %add.454 = f32[1,1,1,1,96]{4,3,2,1,0} add(f32[1,1,1,1,96]{4,3,2,1,0} %arg161.162, f32[1,1,1,1,96]{4,3,2,1,0} %broadcast.453), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.455 = f32[1,1,1,1,96]{4,3,2,1,0} rsqrt(f32[1,1,1,1,96]{4,3,2,1,0} %add.454), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.459 = f32[1,96]{1,0} reshape(f32[1,1,1,1,96]{4,3,2,1,0} %rsqrt.455), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.460 = f32[1,16,28,28,96]{4,3,2,1,0} broadcast(f32[1,96]{1,0} %reshape.459), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.449 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.450 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[] %constant.449), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg157.158 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(157), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.436 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.437 = f32[1,1,1,1,32]{4,3,2,1,0} broadcast(f32[] %constant.436), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.438 = f32[1,1,1,1,32]{4,3,2,1,0} add(f32[1,1,1,1,32]{4,3,2,1,0} %arg157.158, f32[1,1,1,1,32]{4,3,2,1,0} %broadcast.437), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.439 = f32[1,1,1,1,32]{4,3,2,1,0} rsqrt(f32[1,1,1,1,32]{4,3,2,1,0} %add.438), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.443 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.439), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.444 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.443), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg154.155 = f32[1,1,1,256,32]{4,3,2,1,0} parameter(154), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.442 = f32[1,16,28,28,32]{4,3,2,1,0} convolution(f32[1,16,28,28,256]{4,3,2,1,0} %maximum.387, f32[1,1,1,256,32]{4,3,2,1,0} %arg154.155), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.445 = f32[1,16,28,28,32]{4,3,2,1,0} multiply(f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.444, f32[1,16,28,28,32]{4,3,2,1,0} %convolution.442), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg155.156 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(155), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg156.157 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(156), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.440 = f32[1,1,1,1,32]{4,3,2,1,0} multiply(f32[1,1,1,1,32]{4,3,2,1,0} %arg156.157, f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.439), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.441 = f32[1,1,1,1,32]{4,3,2,1,0} subtract(f32[1,1,1,1,32]{4,3,2,1,0} %arg155.156, f32[1,1,1,1,32]{4,3,2,1,0} %multiply.440), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.446 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %subtract.441), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.447 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.446), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.448 = f32[1,16,28,28,32]{4,3,2,1,0} add(f32[1,16,28,28,32]{4,3,2,1,0} %multiply.445, f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.447), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.451 = f32[1,16,28,28,32]{4,3,2,1,0} maximum(f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.450, f32[1,16,28,28,32]{4,3,2,1,0} %add.448), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg158.159 = f32[3,3,3,32,96]{4,3,2,1,0} parameter(158), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.458 = f32[1,16,28,28,96]{4,3,2,1,0} convolution(f32[1,16,28,28,32]{4,3,2,1,0} %maximum.451, f32[3,3,3,32,96]{4,3,2,1,0} %arg158.159), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.461 = f32[1,16,28,28,96]{4,3,2,1,0} multiply(f32[1,16,28,28,96]{4,3,2,1,0} %broadcast.460, f32[1,16,28,28,96]{4,3,2,1,0} %convolution.458), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg159.160 = f32[1,1,1,1,96]{4,3,2,1,0} parameter(159), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg160.161 = f32[1,1,1,1,96]{4,3,2,1,0} parameter(160), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.456 = f32[1,1,1,1,96]{4,3,2,1,0} multiply(f32[1,1,1,1,96]{4,3,2,1,0} %arg160.161, f32[1,1,1,1,96]{4,3,2,1,0} %rsqrt.455), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.457 = f32[1,1,1,1,96]{4,3,2,1,0} subtract(f32[1,1,1,1,96]{4,3,2,1,0} %arg159.160, f32[1,1,1,1,96]{4,3,2,1,0} %multiply.456), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.462 = f32[1,96]{1,0} reshape(f32[1,1,1,1,96]{4,3,2,1,0} %subtract.457), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.463 = f32[1,16,28,28,96]{4,3,2,1,0} broadcast(f32[1,96]{1,0} %reshape.462), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.464 = f32[1,16,28,28,96]{4,3,2,1,0} add(f32[1,16,28,28,96]{4,3,2,1,0} %multiply.461, f32[1,16,28,28,96]{4,3,2,1,0} %broadcast.463), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %arg201.202 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(201), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.465 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.466 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.465), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %add.467 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %arg201.202, f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.466), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.468 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.467), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.472 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.468), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.473 = f32[1,16,28,28,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.472), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %constant.388 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.393 = f32[1,16,28,28,256]{4,3,2,1,0} reduce-window(f32[1,16,28,28,256]{4,3,2,1,0} %maximum.387, f32[] %constant.388), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.389, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/MaxPool3d_0a_3x3"}
  %arg198.199 = f32[1,1,1,256,64]{4,3,2,1,0} parameter(198), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.471 = f32[1,16,28,28,64]{4,3,2,1,0} convolution(f32[1,16,28,28,256]{4,3,2,1,0} %reduce-window.393, f32[1,1,1,256,64]{4,3,2,1,0} %arg198.199), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %multiply.474 = f32[1,16,28,28,64]{4,3,2,1,0} multiply(f32[1,16,28,28,64]{4,3,2,1,0} %broadcast.473, f32[1,16,28,28,64]{4,3,2,1,0} %convolution.471), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg199.200 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(199), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg200.201 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(200), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.469 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %arg200.201, f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.468), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.470 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg199.200, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.469), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.475 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.470), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.476 = f32[1,16,28,28,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.475), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.477 = f32[1,16,28,28,64]{4,3,2,1,0} add(f32[1,16,28,28,64]{4,3,2,1,0} %multiply.474, f32[1,16,28,28,64]{4,3,2,1,0} %broadcast.476), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.478 = f32[1,16,28,28,480]{4,3,2,1,0} concatenate(f32[1,16,28,28,128]{4,3,2,1,0} %add.406, f32[1,16,28,28,192]{4,3,2,1,0} %add.435, f32[1,16,28,28,96]{4,3,2,1,0} %add.464, f32[1,16,28,28,64]{4,3,2,1,0} %add.477), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_3c/concat"}
  %constant.479 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_4a_3x3"}
  %reduce-window.484 = f32[1,8,14,14,480]{4,3,2,1,0} reduce-window(f32[1,16,28,28,480]{4,3,2,1,0} %concatenate.478, f32[] %constant.479), window={size=1x3x3x3x1 stride=1x2x2x2x1 pad=0_0x0_1x0_1x0_1x0_0}, to_apply=%max_F32.480, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_4a_3x3"}
  %maximum.487 = f32[1,8,14,14,480]{4,3,2,1,0} maximum(f32[1,8,14,14,480]{4,3,2,1,0} %broadcast.486, f32[1,8,14,14,480]{4,3,2,1,0} %reduce-window.484), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg122.123 = f32[1,1,1,480,192]{4,3,2,1,0} parameter(122), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.500 = f32[1,8,14,14,192]{4,3,2,1,0} convolution(f32[1,8,14,14,480]{4,3,2,1,0} %maximum.487, f32[1,1,1,480,192]{4,3,2,1,0} %arg122.123), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.503 = f32[1,8,14,14,192]{4,3,2,1,0} multiply(f32[1,8,14,14,192]{4,3,2,1,0} %broadcast.502, f32[1,8,14,14,192]{4,3,2,1,0} %convolution.500), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg123.124 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(123), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg124.125 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(124), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.498 = f32[1,1,1,1,192]{4,3,2,1,0} multiply(f32[1,1,1,1,192]{4,3,2,1,0} %arg124.125, f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.497), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.499 = f32[1,1,1,1,192]{4,3,2,1,0} subtract(f32[1,1,1,1,192]{4,3,2,1,0} %arg123.124, f32[1,1,1,1,192]{4,3,2,1,0} %multiply.498), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.504 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %subtract.499), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.505 = f32[1,8,14,14,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.504), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.506 = f32[1,8,14,14,192]{4,3,2,1,0} add(f32[1,8,14,14,192]{4,3,2,1,0} %multiply.503, f32[1,8,14,14,192]{4,3,2,1,0} %broadcast.505), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %arg133.134 = f32[1,1,1,1,208]{4,3,2,1,0} parameter(133), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.523 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.524 = f32[1,1,1,1,208]{4,3,2,1,0} broadcast(f32[] %constant.523), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %add.525 = f32[1,1,1,1,208]{4,3,2,1,0} add(f32[1,1,1,1,208]{4,3,2,1,0} %arg133.134, f32[1,1,1,1,208]{4,3,2,1,0} %broadcast.524), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.526 = f32[1,1,1,1,208]{4,3,2,1,0} rsqrt(f32[1,1,1,1,208]{4,3,2,1,0} %add.525), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.530 = f32[1,208]{1,0} reshape(f32[1,1,1,1,208]{4,3,2,1,0} %rsqrt.526), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.531 = f32[1,8,14,14,208]{4,3,2,1,0} broadcast(f32[1,208]{1,0} %reshape.530), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.520 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.521 = f32[1,8,14,14,96]{4,3,2,1,0} broadcast(f32[] %constant.520), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg129.130 = f32[1,1,1,1,96]{4,3,2,1,0} parameter(129), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.507 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.508 = f32[1,1,1,1,96]{4,3,2,1,0} broadcast(f32[] %constant.507), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.509 = f32[1,1,1,1,96]{4,3,2,1,0} add(f32[1,1,1,1,96]{4,3,2,1,0} %arg129.130, f32[1,1,1,1,96]{4,3,2,1,0} %broadcast.508), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.510 = f32[1,1,1,1,96]{4,3,2,1,0} rsqrt(f32[1,1,1,1,96]{4,3,2,1,0} %add.509), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.514 = f32[1,96]{1,0} reshape(f32[1,1,1,1,96]{4,3,2,1,0} %rsqrt.510), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.515 = f32[1,8,14,14,96]{4,3,2,1,0} broadcast(f32[1,96]{1,0} %reshape.514), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg126.127 = f32[1,1,1,480,96]{4,3,2,1,0} parameter(126), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.513 = f32[1,8,14,14,96]{4,3,2,1,0} convolution(f32[1,8,14,14,480]{4,3,2,1,0} %maximum.487, f32[1,1,1,480,96]{4,3,2,1,0} %arg126.127), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.516 = f32[1,8,14,14,96]{4,3,2,1,0} multiply(f32[1,8,14,14,96]{4,3,2,1,0} %broadcast.515, f32[1,8,14,14,96]{4,3,2,1,0} %convolution.513), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg127.128 = f32[1,1,1,1,96]{4,3,2,1,0} parameter(127), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg128.129 = f32[1,1,1,1,96]{4,3,2,1,0} parameter(128), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.511 = f32[1,1,1,1,96]{4,3,2,1,0} multiply(f32[1,1,1,1,96]{4,3,2,1,0} %arg128.129, f32[1,1,1,1,96]{4,3,2,1,0} %rsqrt.510), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.512 = f32[1,1,1,1,96]{4,3,2,1,0} subtract(f32[1,1,1,1,96]{4,3,2,1,0} %arg127.128, f32[1,1,1,1,96]{4,3,2,1,0} %multiply.511), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.517 = f32[1,96]{1,0} reshape(f32[1,1,1,1,96]{4,3,2,1,0} %subtract.512), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.518 = f32[1,8,14,14,96]{4,3,2,1,0} broadcast(f32[1,96]{1,0} %reshape.517), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.519 = f32[1,8,14,14,96]{4,3,2,1,0} add(f32[1,8,14,14,96]{4,3,2,1,0} %multiply.516, f32[1,8,14,14,96]{4,3,2,1,0} %broadcast.518), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.522 = f32[1,8,14,14,96]{4,3,2,1,0} maximum(f32[1,8,14,14,96]{4,3,2,1,0} %broadcast.521, f32[1,8,14,14,96]{4,3,2,1,0} %add.519), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg130.131 = f32[3,3,3,96,208]{4,3,2,1,0} parameter(130), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.529 = f32[1,8,14,14,208]{4,3,2,1,0} convolution(f32[1,8,14,14,96]{4,3,2,1,0} %maximum.522, f32[3,3,3,96,208]{4,3,2,1,0} %arg130.131), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.532 = f32[1,8,14,14,208]{4,3,2,1,0} multiply(f32[1,8,14,14,208]{4,3,2,1,0} %broadcast.531, f32[1,8,14,14,208]{4,3,2,1,0} %convolution.529), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg131.132 = f32[1,1,1,1,208]{4,3,2,1,0} parameter(131), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg132.133 = f32[1,1,1,1,208]{4,3,2,1,0} parameter(132), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.527 = f32[1,1,1,1,208]{4,3,2,1,0} multiply(f32[1,1,1,1,208]{4,3,2,1,0} %arg132.133, f32[1,1,1,1,208]{4,3,2,1,0} %rsqrt.526), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.528 = f32[1,1,1,1,208]{4,3,2,1,0} subtract(f32[1,1,1,1,208]{4,3,2,1,0} %arg131.132, f32[1,1,1,1,208]{4,3,2,1,0} %multiply.527), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.533 = f32[1,208]{1,0} reshape(f32[1,1,1,1,208]{4,3,2,1,0} %subtract.528), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.534 = f32[1,8,14,14,208]{4,3,2,1,0} broadcast(f32[1,208]{1,0} %reshape.533), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.535 = f32[1,8,14,14,208]{4,3,2,1,0} add(f32[1,8,14,14,208]{4,3,2,1,0} %multiply.532, f32[1,8,14,14,208]{4,3,2,1,0} %broadcast.534), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %arg141.142 = f32[1,1,1,1,48]{4,3,2,1,0} parameter(141), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.552 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.553 = f32[1,1,1,1,48]{4,3,2,1,0} broadcast(f32[] %constant.552), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %add.554 = f32[1,1,1,1,48]{4,3,2,1,0} add(f32[1,1,1,1,48]{4,3,2,1,0} %arg141.142, f32[1,1,1,1,48]{4,3,2,1,0} %broadcast.553), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.555 = f32[1,1,1,1,48]{4,3,2,1,0} rsqrt(f32[1,1,1,1,48]{4,3,2,1,0} %add.554), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.559 = f32[1,48]{1,0} reshape(f32[1,1,1,1,48]{4,3,2,1,0} %rsqrt.555), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.560 = f32[1,8,14,14,48]{4,3,2,1,0} broadcast(f32[1,48]{1,0} %reshape.559), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.549 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.550 = f32[1,8,14,14,16]{4,3,2,1,0} broadcast(f32[] %constant.549), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg137.138 = f32[1,1,1,1,16]{4,3,2,1,0} parameter(137), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.536 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.537 = f32[1,1,1,1,16]{4,3,2,1,0} broadcast(f32[] %constant.536), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.538 = f32[1,1,1,1,16]{4,3,2,1,0} add(f32[1,1,1,1,16]{4,3,2,1,0} %arg137.138, f32[1,1,1,1,16]{4,3,2,1,0} %broadcast.537), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.539 = f32[1,1,1,1,16]{4,3,2,1,0} rsqrt(f32[1,1,1,1,16]{4,3,2,1,0} %add.538), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.543 = f32[1,16]{1,0} reshape(f32[1,1,1,1,16]{4,3,2,1,0} %rsqrt.539), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.544 = f32[1,8,14,14,16]{4,3,2,1,0} broadcast(f32[1,16]{1,0} %reshape.543), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg134.135 = f32[1,1,1,480,16]{4,3,2,1,0} parameter(134), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.542 = f32[1,8,14,14,16]{4,3,2,1,0} convolution(f32[1,8,14,14,480]{4,3,2,1,0} %maximum.487, f32[1,1,1,480,16]{4,3,2,1,0} %arg134.135), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.545 = f32[1,8,14,14,16]{4,3,2,1,0} multiply(f32[1,8,14,14,16]{4,3,2,1,0} %broadcast.544, f32[1,8,14,14,16]{4,3,2,1,0} %convolution.542), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg135.136 = f32[1,1,1,1,16]{4,3,2,1,0} parameter(135), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg136.137 = f32[1,1,1,1,16]{4,3,2,1,0} parameter(136), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.540 = f32[1,1,1,1,16]{4,3,2,1,0} multiply(f32[1,1,1,1,16]{4,3,2,1,0} %arg136.137, f32[1,1,1,1,16]{4,3,2,1,0} %rsqrt.539), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.541 = f32[1,1,1,1,16]{4,3,2,1,0} subtract(f32[1,1,1,1,16]{4,3,2,1,0} %arg135.136, f32[1,1,1,1,16]{4,3,2,1,0} %multiply.540), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.546 = f32[1,16]{1,0} reshape(f32[1,1,1,1,16]{4,3,2,1,0} %subtract.541), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.547 = f32[1,8,14,14,16]{4,3,2,1,0} broadcast(f32[1,16]{1,0} %reshape.546), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.548 = f32[1,8,14,14,16]{4,3,2,1,0} add(f32[1,8,14,14,16]{4,3,2,1,0} %multiply.545, f32[1,8,14,14,16]{4,3,2,1,0} %broadcast.547), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.551 = f32[1,8,14,14,16]{4,3,2,1,0} maximum(f32[1,8,14,14,16]{4,3,2,1,0} %broadcast.550, f32[1,8,14,14,16]{4,3,2,1,0} %add.548), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg138.139 = f32[3,3,3,16,48]{4,3,2,1,0} parameter(138), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.558 = f32[1,8,14,14,48]{4,3,2,1,0} convolution(f32[1,8,14,14,16]{4,3,2,1,0} %maximum.551, f32[3,3,3,16,48]{4,3,2,1,0} %arg138.139), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.561 = f32[1,8,14,14,48]{4,3,2,1,0} multiply(f32[1,8,14,14,48]{4,3,2,1,0} %broadcast.560, f32[1,8,14,14,48]{4,3,2,1,0} %convolution.558), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg139.140 = f32[1,1,1,1,48]{4,3,2,1,0} parameter(139), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg140.141 = f32[1,1,1,1,48]{4,3,2,1,0} parameter(140), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.556 = f32[1,1,1,1,48]{4,3,2,1,0} multiply(f32[1,1,1,1,48]{4,3,2,1,0} %arg140.141, f32[1,1,1,1,48]{4,3,2,1,0} %rsqrt.555), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.557 = f32[1,1,1,1,48]{4,3,2,1,0} subtract(f32[1,1,1,1,48]{4,3,2,1,0} %arg139.140, f32[1,1,1,1,48]{4,3,2,1,0} %multiply.556), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.562 = f32[1,48]{1,0} reshape(f32[1,1,1,1,48]{4,3,2,1,0} %subtract.557), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.563 = f32[1,8,14,14,48]{4,3,2,1,0} broadcast(f32[1,48]{1,0} %reshape.562), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.564 = f32[1,8,14,14,48]{4,3,2,1,0} add(f32[1,8,14,14,48]{4,3,2,1,0} %multiply.561, f32[1,8,14,14,48]{4,3,2,1,0} %broadcast.563), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %arg205.206 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(205), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.565 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.566 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.565), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %add.567 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %arg205.206, f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.566), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.568 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.567), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.572 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.568), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.573 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.572), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %constant.488 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.493 = f32[1,8,14,14,480]{4,3,2,1,0} reduce-window(f32[1,8,14,14,480]{4,3,2,1,0} %maximum.487, f32[] %constant.488), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.489, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/MaxPool3d_0a_3x3"}
  %arg202.203 = f32[1,1,1,480,64]{4,3,2,1,0} parameter(202), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.571 = f32[1,8,14,14,64]{4,3,2,1,0} convolution(f32[1,8,14,14,480]{4,3,2,1,0} %reduce-window.493, f32[1,1,1,480,64]{4,3,2,1,0} %arg202.203), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %multiply.574 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.573, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.571), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg203.204 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(203), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg204.205 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(204), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.569 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %arg204.205, f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.568), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.570 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg203.204, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.569), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.575 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.570), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.576 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.575), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.577 = f32[1,8,14,14,64]{4,3,2,1,0} add(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.574, f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.576), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.578 = f32[1,8,14,14,512]{4,3,2,1,0} concatenate(f32[1,8,14,14,192]{4,3,2,1,0} %add.506, f32[1,8,14,14,208]{4,3,2,1,0} %add.535, f32[1,8,14,14,48]{4,3,2,1,0} %add.564, f32[1,8,14,14,64]{4,3,2,1,0} %add.577), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_4b/concat"}
  %maximum.581 = f32[1,8,14,14,512]{4,3,2,1,0} maximum(f32[1,8,14,14,512]{4,3,2,1,0} %broadcast.580, f32[1,8,14,14,512]{4,3,2,1,0} %concatenate.578), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg102.103 = f32[1,1,1,512,160]{4,3,2,1,0} parameter(102), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.594 = f32[1,8,14,14,160]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.581, f32[1,1,1,512,160]{4,3,2,1,0} %arg102.103), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.597 = f32[1,8,14,14,160]{4,3,2,1,0} multiply(f32[1,8,14,14,160]{4,3,2,1,0} %broadcast.596, f32[1,8,14,14,160]{4,3,2,1,0} %convolution.594), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg103.104 = f32[1,1,1,1,160]{4,3,2,1,0} parameter(103), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg104.105 = f32[1,1,1,1,160]{4,3,2,1,0} parameter(104), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.592 = f32[1,1,1,1,160]{4,3,2,1,0} multiply(f32[1,1,1,1,160]{4,3,2,1,0} %arg104.105, f32[1,1,1,1,160]{4,3,2,1,0} %rsqrt.591), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.593 = f32[1,1,1,1,160]{4,3,2,1,0} subtract(f32[1,1,1,1,160]{4,3,2,1,0} %arg103.104, f32[1,1,1,1,160]{4,3,2,1,0} %multiply.592), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.598 = f32[1,160]{1,0} reshape(f32[1,1,1,1,160]{4,3,2,1,0} %subtract.593), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.599 = f32[1,8,14,14,160]{4,3,2,1,0} broadcast(f32[1,160]{1,0} %reshape.598), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.600 = f32[1,8,14,14,160]{4,3,2,1,0} add(f32[1,8,14,14,160]{4,3,2,1,0} %multiply.597, f32[1,8,14,14,160]{4,3,2,1,0} %broadcast.599), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %arg113.114 = f32[1,1,1,1,224]{4,3,2,1,0} parameter(113), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.617 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.618 = f32[1,1,1,1,224]{4,3,2,1,0} broadcast(f32[] %constant.617), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %add.619 = f32[1,1,1,1,224]{4,3,2,1,0} add(f32[1,1,1,1,224]{4,3,2,1,0} %arg113.114, f32[1,1,1,1,224]{4,3,2,1,0} %broadcast.618), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.620 = f32[1,1,1,1,224]{4,3,2,1,0} rsqrt(f32[1,1,1,1,224]{4,3,2,1,0} %add.619), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.624 = f32[1,224]{1,0} reshape(f32[1,1,1,1,224]{4,3,2,1,0} %rsqrt.620), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.625 = f32[1,8,14,14,224]{4,3,2,1,0} broadcast(f32[1,224]{1,0} %reshape.624), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.614 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.615 = f32[1,8,14,14,112]{4,3,2,1,0} broadcast(f32[] %constant.614), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg109.110 = f32[1,1,1,1,112]{4,3,2,1,0} parameter(109), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.601 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.602 = f32[1,1,1,1,112]{4,3,2,1,0} broadcast(f32[] %constant.601), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.603 = f32[1,1,1,1,112]{4,3,2,1,0} add(f32[1,1,1,1,112]{4,3,2,1,0} %arg109.110, f32[1,1,1,1,112]{4,3,2,1,0} %broadcast.602), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.604 = f32[1,1,1,1,112]{4,3,2,1,0} rsqrt(f32[1,1,1,1,112]{4,3,2,1,0} %add.603), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.608 = f32[1,112]{1,0} reshape(f32[1,1,1,1,112]{4,3,2,1,0} %rsqrt.604), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.609 = f32[1,8,14,14,112]{4,3,2,1,0} broadcast(f32[1,112]{1,0} %reshape.608), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg106.107 = f32[1,1,1,512,112]{4,3,2,1,0} parameter(106), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.607 = f32[1,8,14,14,112]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.581, f32[1,1,1,512,112]{4,3,2,1,0} %arg106.107), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.610 = f32[1,8,14,14,112]{4,3,2,1,0} multiply(f32[1,8,14,14,112]{4,3,2,1,0} %broadcast.609, f32[1,8,14,14,112]{4,3,2,1,0} %convolution.607), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg107.108 = f32[1,1,1,1,112]{4,3,2,1,0} parameter(107), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg108.109 = f32[1,1,1,1,112]{4,3,2,1,0} parameter(108), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.605 = f32[1,1,1,1,112]{4,3,2,1,0} multiply(f32[1,1,1,1,112]{4,3,2,1,0} %arg108.109, f32[1,1,1,1,112]{4,3,2,1,0} %rsqrt.604), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.606 = f32[1,1,1,1,112]{4,3,2,1,0} subtract(f32[1,1,1,1,112]{4,3,2,1,0} %arg107.108, f32[1,1,1,1,112]{4,3,2,1,0} %multiply.605), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.611 = f32[1,112]{1,0} reshape(f32[1,1,1,1,112]{4,3,2,1,0} %subtract.606), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.612 = f32[1,8,14,14,112]{4,3,2,1,0} broadcast(f32[1,112]{1,0} %reshape.611), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.613 = f32[1,8,14,14,112]{4,3,2,1,0} add(f32[1,8,14,14,112]{4,3,2,1,0} %multiply.610, f32[1,8,14,14,112]{4,3,2,1,0} %broadcast.612), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.616 = f32[1,8,14,14,112]{4,3,2,1,0} maximum(f32[1,8,14,14,112]{4,3,2,1,0} %broadcast.615, f32[1,8,14,14,112]{4,3,2,1,0} %add.613), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg110.111 = f32[3,3,3,112,224]{4,3,2,1,0} parameter(110), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.623 = f32[1,8,14,14,224]{4,3,2,1,0} convolution(f32[1,8,14,14,112]{4,3,2,1,0} %maximum.616, f32[3,3,3,112,224]{4,3,2,1,0} %arg110.111), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.626 = f32[1,8,14,14,224]{4,3,2,1,0} multiply(f32[1,8,14,14,224]{4,3,2,1,0} %broadcast.625, f32[1,8,14,14,224]{4,3,2,1,0} %convolution.623), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg111.112 = f32[1,1,1,1,224]{4,3,2,1,0} parameter(111), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg112.113 = f32[1,1,1,1,224]{4,3,2,1,0} parameter(112), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.621 = f32[1,1,1,1,224]{4,3,2,1,0} multiply(f32[1,1,1,1,224]{4,3,2,1,0} %arg112.113, f32[1,1,1,1,224]{4,3,2,1,0} %rsqrt.620), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.622 = f32[1,1,1,1,224]{4,3,2,1,0} subtract(f32[1,1,1,1,224]{4,3,2,1,0} %arg111.112, f32[1,1,1,1,224]{4,3,2,1,0} %multiply.621), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.627 = f32[1,224]{1,0} reshape(f32[1,1,1,1,224]{4,3,2,1,0} %subtract.622), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.628 = f32[1,8,14,14,224]{4,3,2,1,0} broadcast(f32[1,224]{1,0} %reshape.627), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.629 = f32[1,8,14,14,224]{4,3,2,1,0} add(f32[1,8,14,14,224]{4,3,2,1,0} %multiply.626, f32[1,8,14,14,224]{4,3,2,1,0} %broadcast.628), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %arg121.122 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(121), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.646 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.647 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.646), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %add.648 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %arg121.122, f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.647), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.649 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.648), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.653 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.649), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.654 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.653), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.643 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.644 = f32[1,8,14,14,24]{4,3,2,1,0} broadcast(f32[] %constant.643), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg117.118 = f32[1,1,1,1,24]{4,3,2,1,0} parameter(117), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.630 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.631 = f32[1,1,1,1,24]{4,3,2,1,0} broadcast(f32[] %constant.630), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.632 = f32[1,1,1,1,24]{4,3,2,1,0} add(f32[1,1,1,1,24]{4,3,2,1,0} %arg117.118, f32[1,1,1,1,24]{4,3,2,1,0} %broadcast.631), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.633 = f32[1,1,1,1,24]{4,3,2,1,0} rsqrt(f32[1,1,1,1,24]{4,3,2,1,0} %add.632), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.637 = f32[1,24]{1,0} reshape(f32[1,1,1,1,24]{4,3,2,1,0} %rsqrt.633), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.638 = f32[1,8,14,14,24]{4,3,2,1,0} broadcast(f32[1,24]{1,0} %reshape.637), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg114.115 = f32[1,1,1,512,24]{4,3,2,1,0} parameter(114), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.636 = f32[1,8,14,14,24]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.581, f32[1,1,1,512,24]{4,3,2,1,0} %arg114.115), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.639 = f32[1,8,14,14,24]{4,3,2,1,0} multiply(f32[1,8,14,14,24]{4,3,2,1,0} %broadcast.638, f32[1,8,14,14,24]{4,3,2,1,0} %convolution.636), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg115.116 = f32[1,1,1,1,24]{4,3,2,1,0} parameter(115), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg116.117 = f32[1,1,1,1,24]{4,3,2,1,0} parameter(116), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.634 = f32[1,1,1,1,24]{4,3,2,1,0} multiply(f32[1,1,1,1,24]{4,3,2,1,0} %arg116.117, f32[1,1,1,1,24]{4,3,2,1,0} %rsqrt.633), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.635 = f32[1,1,1,1,24]{4,3,2,1,0} subtract(f32[1,1,1,1,24]{4,3,2,1,0} %arg115.116, f32[1,1,1,1,24]{4,3,2,1,0} %multiply.634), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.640 = f32[1,24]{1,0} reshape(f32[1,1,1,1,24]{4,3,2,1,0} %subtract.635), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.641 = f32[1,8,14,14,24]{4,3,2,1,0} broadcast(f32[1,24]{1,0} %reshape.640), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.642 = f32[1,8,14,14,24]{4,3,2,1,0} add(f32[1,8,14,14,24]{4,3,2,1,0} %multiply.639, f32[1,8,14,14,24]{4,3,2,1,0} %broadcast.641), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.645 = f32[1,8,14,14,24]{4,3,2,1,0} maximum(f32[1,8,14,14,24]{4,3,2,1,0} %broadcast.644, f32[1,8,14,14,24]{4,3,2,1,0} %add.642), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg118.119 = f32[3,3,3,24,64]{4,3,2,1,0} parameter(118), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.652 = f32[1,8,14,14,64]{4,3,2,1,0} convolution(f32[1,8,14,14,24]{4,3,2,1,0} %maximum.645, f32[3,3,3,24,64]{4,3,2,1,0} %arg118.119), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.655 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.654, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.652), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg119.120 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(119), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg120.121 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(120), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.650 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %arg120.121, f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.649), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.651 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg119.120, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.650), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.656 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.651), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.657 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.656), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.658 = f32[1,8,14,14,64]{4,3,2,1,0} add(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.655, f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.657), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %arg209.210 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(209), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.659 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.660 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.659), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %add.661 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %arg209.210, f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.660), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.662 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.661), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.666 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.662), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.667 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.666), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %constant.582 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.587 = f32[1,8,14,14,512]{4,3,2,1,0} reduce-window(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.581, f32[] %constant.582), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.583, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/MaxPool3d_0a_3x3"}
  %arg206.207 = f32[1,1,1,512,64]{4,3,2,1,0} parameter(206), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.665 = f32[1,8,14,14,64]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %reduce-window.587, f32[1,1,1,512,64]{4,3,2,1,0} %arg206.207), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %multiply.668 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.667, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.665), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg207.208 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(207), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg208.209 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(208), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.663 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %arg208.209, f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.662), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.664 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg207.208, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.663), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.669 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.664), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.670 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.669), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.671 = f32[1,8,14,14,64]{4,3,2,1,0} add(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.668, f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.670), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.672 = f32[1,8,14,14,512]{4,3,2,1,0} concatenate(f32[1,8,14,14,160]{4,3,2,1,0} %add.600, f32[1,8,14,14,224]{4,3,2,1,0} %add.629, f32[1,8,14,14,64]{4,3,2,1,0} %add.658, f32[1,8,14,14,64]{4,3,2,1,0} %add.671), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_4c/concat"}
  %maximum.675 = f32[1,8,14,14,512]{4,3,2,1,0} maximum(f32[1,8,14,14,512]{4,3,2,1,0} %broadcast.674, f32[1,8,14,14,512]{4,3,2,1,0} %concatenate.672), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg82.83 = f32[1,1,1,512,128]{4,3,2,1,0} parameter(82), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.688 = f32[1,8,14,14,128]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.675, f32[1,1,1,512,128]{4,3,2,1,0} %arg82.83), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.691 = f32[1,8,14,14,128]{4,3,2,1,0} multiply(f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.690, f32[1,8,14,14,128]{4,3,2,1,0} %convolution.688), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg83.84 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(83), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg84.85 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(84), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.686 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %arg84.85, f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.685), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.687 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg83.84, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.686), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.692 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.687), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.693 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.692), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.694 = f32[1,8,14,14,128]{4,3,2,1,0} add(f32[1,8,14,14,128]{4,3,2,1,0} %multiply.691, f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.693), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %arg93.94 = f32[1,1,1,1,256]{4,3,2,1,0} parameter(93), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.711 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.712 = f32[1,1,1,1,256]{4,3,2,1,0} broadcast(f32[] %constant.711), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %add.713 = f32[1,1,1,1,256]{4,3,2,1,0} add(f32[1,1,1,1,256]{4,3,2,1,0} %arg93.94, f32[1,1,1,1,256]{4,3,2,1,0} %broadcast.712), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.714 = f32[1,1,1,1,256]{4,3,2,1,0} rsqrt(f32[1,1,1,1,256]{4,3,2,1,0} %add.713), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.718 = f32[1,256]{1,0} reshape(f32[1,1,1,1,256]{4,3,2,1,0} %rsqrt.714), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.719 = f32[1,8,14,14,256]{4,3,2,1,0} broadcast(f32[1,256]{1,0} %reshape.718), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.708 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.709 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[] %constant.708), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg89.90 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(89), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.695 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.696 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.695), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.697 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %arg89.90, f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.696), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.698 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.697), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.702 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.698), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.703 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.702), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg86.87 = f32[1,1,1,512,128]{4,3,2,1,0} parameter(86), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.701 = f32[1,8,14,14,128]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.675, f32[1,1,1,512,128]{4,3,2,1,0} %arg86.87), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.704 = f32[1,8,14,14,128]{4,3,2,1,0} multiply(f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.703, f32[1,8,14,14,128]{4,3,2,1,0} %convolution.701), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg87.88 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(87), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg88.89 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(88), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.699 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %arg88.89, f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.698), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.700 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg87.88, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.699), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.705 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.700), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.706 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.705), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.707 = f32[1,8,14,14,128]{4,3,2,1,0} add(f32[1,8,14,14,128]{4,3,2,1,0} %multiply.704, f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.706), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.710 = f32[1,8,14,14,128]{4,3,2,1,0} maximum(f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.709, f32[1,8,14,14,128]{4,3,2,1,0} %add.707), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg90.91 = f32[3,3,3,128,256]{4,3,2,1,0} parameter(90), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.717 = f32[1,8,14,14,256]{4,3,2,1,0} convolution(f32[1,8,14,14,128]{4,3,2,1,0} %maximum.710, f32[3,3,3,128,256]{4,3,2,1,0} %arg90.91), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.720 = f32[1,8,14,14,256]{4,3,2,1,0} multiply(f32[1,8,14,14,256]{4,3,2,1,0} %broadcast.719, f32[1,8,14,14,256]{4,3,2,1,0} %convolution.717), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg91.92 = f32[1,1,1,1,256]{4,3,2,1,0} parameter(91), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg92.93 = f32[1,1,1,1,256]{4,3,2,1,0} parameter(92), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.715 = f32[1,1,1,1,256]{4,3,2,1,0} multiply(f32[1,1,1,1,256]{4,3,2,1,0} %arg92.93, f32[1,1,1,1,256]{4,3,2,1,0} %rsqrt.714), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.716 = f32[1,1,1,1,256]{4,3,2,1,0} subtract(f32[1,1,1,1,256]{4,3,2,1,0} %arg91.92, f32[1,1,1,1,256]{4,3,2,1,0} %multiply.715), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.721 = f32[1,256]{1,0} reshape(f32[1,1,1,1,256]{4,3,2,1,0} %subtract.716), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.722 = f32[1,8,14,14,256]{4,3,2,1,0} broadcast(f32[1,256]{1,0} %reshape.721), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.723 = f32[1,8,14,14,256]{4,3,2,1,0} add(f32[1,8,14,14,256]{4,3,2,1,0} %multiply.720, f32[1,8,14,14,256]{4,3,2,1,0} %broadcast.722), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %arg101.102 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(101), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.740 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.741 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.740), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %add.742 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %arg101.102, f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.741), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.743 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.742), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.747 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.743), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.748 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.747), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.737 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.738 = f32[1,8,14,14,24]{4,3,2,1,0} broadcast(f32[] %constant.737), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg97.98 = f32[1,1,1,1,24]{4,3,2,1,0} parameter(97), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.724 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.725 = f32[1,1,1,1,24]{4,3,2,1,0} broadcast(f32[] %constant.724), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.726 = f32[1,1,1,1,24]{4,3,2,1,0} add(f32[1,1,1,1,24]{4,3,2,1,0} %arg97.98, f32[1,1,1,1,24]{4,3,2,1,0} %broadcast.725), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.727 = f32[1,1,1,1,24]{4,3,2,1,0} rsqrt(f32[1,1,1,1,24]{4,3,2,1,0} %add.726), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.731 = f32[1,24]{1,0} reshape(f32[1,1,1,1,24]{4,3,2,1,0} %rsqrt.727), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.732 = f32[1,8,14,14,24]{4,3,2,1,0} broadcast(f32[1,24]{1,0} %reshape.731), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg94.95 = f32[1,1,1,512,24]{4,3,2,1,0} parameter(94), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.730 = f32[1,8,14,14,24]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.675, f32[1,1,1,512,24]{4,3,2,1,0} %arg94.95), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.733 = f32[1,8,14,14,24]{4,3,2,1,0} multiply(f32[1,8,14,14,24]{4,3,2,1,0} %broadcast.732, f32[1,8,14,14,24]{4,3,2,1,0} %convolution.730), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg95.96 = f32[1,1,1,1,24]{4,3,2,1,0} parameter(95), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg96.97 = f32[1,1,1,1,24]{4,3,2,1,0} parameter(96), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.728 = f32[1,1,1,1,24]{4,3,2,1,0} multiply(f32[1,1,1,1,24]{4,3,2,1,0} %arg96.97, f32[1,1,1,1,24]{4,3,2,1,0} %rsqrt.727), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.729 = f32[1,1,1,1,24]{4,3,2,1,0} subtract(f32[1,1,1,1,24]{4,3,2,1,0} %arg95.96, f32[1,1,1,1,24]{4,3,2,1,0} %multiply.728), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.734 = f32[1,24]{1,0} reshape(f32[1,1,1,1,24]{4,3,2,1,0} %subtract.729), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.735 = f32[1,8,14,14,24]{4,3,2,1,0} broadcast(f32[1,24]{1,0} %reshape.734), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.736 = f32[1,8,14,14,24]{4,3,2,1,0} add(f32[1,8,14,14,24]{4,3,2,1,0} %multiply.733, f32[1,8,14,14,24]{4,3,2,1,0} %broadcast.735), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.739 = f32[1,8,14,14,24]{4,3,2,1,0} maximum(f32[1,8,14,14,24]{4,3,2,1,0} %broadcast.738, f32[1,8,14,14,24]{4,3,2,1,0} %add.736), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg98.99 = f32[3,3,3,24,64]{4,3,2,1,0} parameter(98), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.746 = f32[1,8,14,14,64]{4,3,2,1,0} convolution(f32[1,8,14,14,24]{4,3,2,1,0} %maximum.739, f32[3,3,3,24,64]{4,3,2,1,0} %arg98.99), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.749 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.748, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.746), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg99.100 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(99), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg100.101 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(100), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.744 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %arg100.101, f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.743), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.745 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg99.100, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.744), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.750 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.745), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.751 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.750), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.752 = f32[1,8,14,14,64]{4,3,2,1,0} add(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.749, f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.751), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %arg213.214 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(213), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.753 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.754 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.753), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %add.755 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %arg213.214, f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.754), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.756 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.755), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.760 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.756), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.761 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.760), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %constant.676 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.681 = f32[1,8,14,14,512]{4,3,2,1,0} reduce-window(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.675, f32[] %constant.676), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.677, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/MaxPool3d_0a_3x3"}
  %arg210.211 = f32[1,1,1,512,64]{4,3,2,1,0} parameter(210), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.759 = f32[1,8,14,14,64]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %reduce-window.681, f32[1,1,1,512,64]{4,3,2,1,0} %arg210.211), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %multiply.762 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.761, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.759), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg211.212 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(211), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg212.213 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(212), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.757 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %arg212.213, f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.756), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.758 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg211.212, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.757), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.763 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.758), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.764 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.763), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.765 = f32[1,8,14,14,64]{4,3,2,1,0} add(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.762, f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.764), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.766 = f32[1,8,14,14,512]{4,3,2,1,0} concatenate(f32[1,8,14,14,128]{4,3,2,1,0} %add.694, f32[1,8,14,14,256]{4,3,2,1,0} %add.723, f32[1,8,14,14,64]{4,3,2,1,0} %add.752, f32[1,8,14,14,64]{4,3,2,1,0} %add.765), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_4d/concat"}
  %maximum.769 = f32[1,8,14,14,512]{4,3,2,1,0} maximum(f32[1,8,14,14,512]{4,3,2,1,0} %broadcast.768, f32[1,8,14,14,512]{4,3,2,1,0} %concatenate.766), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg62.63 = f32[1,1,1,512,112]{4,3,2,1,0} parameter(62), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.782 = f32[1,8,14,14,112]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.769, f32[1,1,1,512,112]{4,3,2,1,0} %arg62.63), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.785 = f32[1,8,14,14,112]{4,3,2,1,0} multiply(f32[1,8,14,14,112]{4,3,2,1,0} %broadcast.784, f32[1,8,14,14,112]{4,3,2,1,0} %convolution.782), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg63.64 = f32[1,1,1,1,112]{4,3,2,1,0} parameter(63), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg64.65 = f32[1,1,1,1,112]{4,3,2,1,0} parameter(64), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.780 = f32[1,1,1,1,112]{4,3,2,1,0} multiply(f32[1,1,1,1,112]{4,3,2,1,0} %arg64.65, f32[1,1,1,1,112]{4,3,2,1,0} %rsqrt.779), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.781 = f32[1,1,1,1,112]{4,3,2,1,0} subtract(f32[1,1,1,1,112]{4,3,2,1,0} %arg63.64, f32[1,1,1,1,112]{4,3,2,1,0} %multiply.780), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.786 = f32[1,112]{1,0} reshape(f32[1,1,1,1,112]{4,3,2,1,0} %subtract.781), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.787 = f32[1,8,14,14,112]{4,3,2,1,0} broadcast(f32[1,112]{1,0} %reshape.786), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.788 = f32[1,8,14,14,112]{4,3,2,1,0} add(f32[1,8,14,14,112]{4,3,2,1,0} %multiply.785, f32[1,8,14,14,112]{4,3,2,1,0} %broadcast.787), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %arg73.74 = f32[1,1,1,1,288]{4,3,2,1,0} parameter(73), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.805 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.806 = f32[1,1,1,1,288]{4,3,2,1,0} broadcast(f32[] %constant.805), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %add.807 = f32[1,1,1,1,288]{4,3,2,1,0} add(f32[1,1,1,1,288]{4,3,2,1,0} %arg73.74, f32[1,1,1,1,288]{4,3,2,1,0} %broadcast.806), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.808 = f32[1,1,1,1,288]{4,3,2,1,0} rsqrt(f32[1,1,1,1,288]{4,3,2,1,0} %add.807), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.812 = f32[1,288]{1,0} reshape(f32[1,1,1,1,288]{4,3,2,1,0} %rsqrt.808), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.813 = f32[1,8,14,14,288]{4,3,2,1,0} broadcast(f32[1,288]{1,0} %reshape.812), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.802 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.803 = f32[1,8,14,14,144]{4,3,2,1,0} broadcast(f32[] %constant.802), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg69.70 = f32[1,1,1,1,144]{4,3,2,1,0} parameter(69), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.789 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.790 = f32[1,1,1,1,144]{4,3,2,1,0} broadcast(f32[] %constant.789), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.791 = f32[1,1,1,1,144]{4,3,2,1,0} add(f32[1,1,1,1,144]{4,3,2,1,0} %arg69.70, f32[1,1,1,1,144]{4,3,2,1,0} %broadcast.790), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.792 = f32[1,1,1,1,144]{4,3,2,1,0} rsqrt(f32[1,1,1,1,144]{4,3,2,1,0} %add.791), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.796 = f32[1,144]{1,0} reshape(f32[1,1,1,1,144]{4,3,2,1,0} %rsqrt.792), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.797 = f32[1,8,14,14,144]{4,3,2,1,0} broadcast(f32[1,144]{1,0} %reshape.796), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg66.67 = f32[1,1,1,512,144]{4,3,2,1,0} parameter(66), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.795 = f32[1,8,14,14,144]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.769, f32[1,1,1,512,144]{4,3,2,1,0} %arg66.67), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.798 = f32[1,8,14,14,144]{4,3,2,1,0} multiply(f32[1,8,14,14,144]{4,3,2,1,0} %broadcast.797, f32[1,8,14,14,144]{4,3,2,1,0} %convolution.795), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg67.68 = f32[1,1,1,1,144]{4,3,2,1,0} parameter(67), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg68.69 = f32[1,1,1,1,144]{4,3,2,1,0} parameter(68), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.793 = f32[1,1,1,1,144]{4,3,2,1,0} multiply(f32[1,1,1,1,144]{4,3,2,1,0} %arg68.69, f32[1,1,1,1,144]{4,3,2,1,0} %rsqrt.792), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.794 = f32[1,1,1,1,144]{4,3,2,1,0} subtract(f32[1,1,1,1,144]{4,3,2,1,0} %arg67.68, f32[1,1,1,1,144]{4,3,2,1,0} %multiply.793), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.799 = f32[1,144]{1,0} reshape(f32[1,1,1,1,144]{4,3,2,1,0} %subtract.794), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.800 = f32[1,8,14,14,144]{4,3,2,1,0} broadcast(f32[1,144]{1,0} %reshape.799), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.801 = f32[1,8,14,14,144]{4,3,2,1,0} add(f32[1,8,14,14,144]{4,3,2,1,0} %multiply.798, f32[1,8,14,14,144]{4,3,2,1,0} %broadcast.800), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.804 = f32[1,8,14,14,144]{4,3,2,1,0} maximum(f32[1,8,14,14,144]{4,3,2,1,0} %broadcast.803, f32[1,8,14,14,144]{4,3,2,1,0} %add.801), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg70.71 = f32[3,3,3,144,288]{4,3,2,1,0} parameter(70), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.811 = f32[1,8,14,14,288]{4,3,2,1,0} convolution(f32[1,8,14,14,144]{4,3,2,1,0} %maximum.804, f32[3,3,3,144,288]{4,3,2,1,0} %arg70.71), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.814 = f32[1,8,14,14,288]{4,3,2,1,0} multiply(f32[1,8,14,14,288]{4,3,2,1,0} %broadcast.813, f32[1,8,14,14,288]{4,3,2,1,0} %convolution.811), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg71.72 = f32[1,1,1,1,288]{4,3,2,1,0} parameter(71), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg72.73 = f32[1,1,1,1,288]{4,3,2,1,0} parameter(72), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.809 = f32[1,1,1,1,288]{4,3,2,1,0} multiply(f32[1,1,1,1,288]{4,3,2,1,0} %arg72.73, f32[1,1,1,1,288]{4,3,2,1,0} %rsqrt.808), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.810 = f32[1,1,1,1,288]{4,3,2,1,0} subtract(f32[1,1,1,1,288]{4,3,2,1,0} %arg71.72, f32[1,1,1,1,288]{4,3,2,1,0} %multiply.809), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.815 = f32[1,288]{1,0} reshape(f32[1,1,1,1,288]{4,3,2,1,0} %subtract.810), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.816 = f32[1,8,14,14,288]{4,3,2,1,0} broadcast(f32[1,288]{1,0} %reshape.815), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.817 = f32[1,8,14,14,288]{4,3,2,1,0} add(f32[1,8,14,14,288]{4,3,2,1,0} %multiply.814, f32[1,8,14,14,288]{4,3,2,1,0} %broadcast.816), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %arg81.82 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(81), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.834 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.835 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.834), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %add.836 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %arg81.82, f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.835), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.837 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.836), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.841 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.837), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.842 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.841), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.831 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.832 = f32[1,8,14,14,32]{4,3,2,1,0} broadcast(f32[] %constant.831), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg77.78 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(77), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.818 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.819 = f32[1,1,1,1,32]{4,3,2,1,0} broadcast(f32[] %constant.818), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.820 = f32[1,1,1,1,32]{4,3,2,1,0} add(f32[1,1,1,1,32]{4,3,2,1,0} %arg77.78, f32[1,1,1,1,32]{4,3,2,1,0} %broadcast.819), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.821 = f32[1,1,1,1,32]{4,3,2,1,0} rsqrt(f32[1,1,1,1,32]{4,3,2,1,0} %add.820), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.825 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.821), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.826 = f32[1,8,14,14,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.825), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg74.75 = f32[1,1,1,512,32]{4,3,2,1,0} parameter(74), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.824 = f32[1,8,14,14,32]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.769, f32[1,1,1,512,32]{4,3,2,1,0} %arg74.75), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.827 = f32[1,8,14,14,32]{4,3,2,1,0} multiply(f32[1,8,14,14,32]{4,3,2,1,0} %broadcast.826, f32[1,8,14,14,32]{4,3,2,1,0} %convolution.824), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg75.76 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(75), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg76.77 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(76), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.822 = f32[1,1,1,1,32]{4,3,2,1,0} multiply(f32[1,1,1,1,32]{4,3,2,1,0} %arg76.77, f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.821), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.823 = f32[1,1,1,1,32]{4,3,2,1,0} subtract(f32[1,1,1,1,32]{4,3,2,1,0} %arg75.76, f32[1,1,1,1,32]{4,3,2,1,0} %multiply.822), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.828 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %subtract.823), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.829 = f32[1,8,14,14,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.828), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.830 = f32[1,8,14,14,32]{4,3,2,1,0} add(f32[1,8,14,14,32]{4,3,2,1,0} %multiply.827, f32[1,8,14,14,32]{4,3,2,1,0} %broadcast.829), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.833 = f32[1,8,14,14,32]{4,3,2,1,0} maximum(f32[1,8,14,14,32]{4,3,2,1,0} %broadcast.832, f32[1,8,14,14,32]{4,3,2,1,0} %add.830), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg78.79 = f32[3,3,3,32,64]{4,3,2,1,0} parameter(78), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.840 = f32[1,8,14,14,64]{4,3,2,1,0} convolution(f32[1,8,14,14,32]{4,3,2,1,0} %maximum.833, f32[3,3,3,32,64]{4,3,2,1,0} %arg78.79), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.843 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.842, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.840), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg79.80 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(79), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg80.81 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(80), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.838 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %arg80.81, f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.837), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.839 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg79.80, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.838), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.844 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.839), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.845 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.844), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.846 = f32[1,8,14,14,64]{4,3,2,1,0} add(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.843, f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.845), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %arg217.218 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(217), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.847 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.848 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.847), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %add.849 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %arg217.218, f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.848), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.850 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.849), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.854 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.850), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.855 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.854), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %constant.770 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.775 = f32[1,8,14,14,512]{4,3,2,1,0} reduce-window(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.769, f32[] %constant.770), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.771, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/MaxPool3d_0a_3x3"}
  %arg214.215 = f32[1,1,1,512,64]{4,3,2,1,0} parameter(214), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.853 = f32[1,8,14,14,64]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %reduce-window.775, f32[1,1,1,512,64]{4,3,2,1,0} %arg214.215), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %multiply.856 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.855, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.853), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg215.216 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(215), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg216.217 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(216), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.851 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %arg216.217, f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.850), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.852 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg215.216, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.851), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.857 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.852), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.858 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.857), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.859 = f32[1,8,14,14,64]{4,3,2,1,0} add(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.856, f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.858), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.860 = f32[1,8,14,14,528]{4,3,2,1,0} concatenate(f32[1,8,14,14,112]{4,3,2,1,0} %add.788, f32[1,8,14,14,288]{4,3,2,1,0} %add.817, f32[1,8,14,14,64]{4,3,2,1,0} %add.846, f32[1,8,14,14,64]{4,3,2,1,0} %add.859), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_4e/concat"}
  %maximum.863 = f32[1,8,14,14,528]{4,3,2,1,0} maximum(f32[1,8,14,14,528]{4,3,2,1,0} %broadcast.862, f32[1,8,14,14,528]{4,3,2,1,0} %concatenate.860), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg42.43 = f32[1,1,1,528,256]{4,3,2,1,0} parameter(42), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.876 = f32[1,8,14,14,256]{4,3,2,1,0} convolution(f32[1,8,14,14,528]{4,3,2,1,0} %maximum.863, f32[1,1,1,528,256]{4,3,2,1,0} %arg42.43), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.879 = f32[1,8,14,14,256]{4,3,2,1,0} multiply(f32[1,8,14,14,256]{4,3,2,1,0} %broadcast.878, f32[1,8,14,14,256]{4,3,2,1,0} %convolution.876), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg43.44 = f32[1,1,1,1,256]{4,3,2,1,0} parameter(43), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg44.45 = f32[1,1,1,1,256]{4,3,2,1,0} parameter(44), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.874 = f32[1,1,1,1,256]{4,3,2,1,0} multiply(f32[1,1,1,1,256]{4,3,2,1,0} %arg44.45, f32[1,1,1,1,256]{4,3,2,1,0} %rsqrt.873), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.875 = f32[1,1,1,1,256]{4,3,2,1,0} subtract(f32[1,1,1,1,256]{4,3,2,1,0} %arg43.44, f32[1,1,1,1,256]{4,3,2,1,0} %multiply.874), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.880 = f32[1,256]{1,0} reshape(f32[1,1,1,1,256]{4,3,2,1,0} %subtract.875), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.881 = f32[1,8,14,14,256]{4,3,2,1,0} broadcast(f32[1,256]{1,0} %reshape.880), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.882 = f32[1,8,14,14,256]{4,3,2,1,0} add(f32[1,8,14,14,256]{4,3,2,1,0} %multiply.879, f32[1,8,14,14,256]{4,3,2,1,0} %broadcast.881), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %arg53.54 = f32[1,1,1,1,320]{4,3,2,1,0} parameter(53), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.899 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.900 = f32[1,1,1,1,320]{4,3,2,1,0} broadcast(f32[] %constant.899), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %add.901 = f32[1,1,1,1,320]{4,3,2,1,0} add(f32[1,1,1,1,320]{4,3,2,1,0} %arg53.54, f32[1,1,1,1,320]{4,3,2,1,0} %broadcast.900), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.902 = f32[1,1,1,1,320]{4,3,2,1,0} rsqrt(f32[1,1,1,1,320]{4,3,2,1,0} %add.901), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.906 = f32[1,320]{1,0} reshape(f32[1,1,1,1,320]{4,3,2,1,0} %rsqrt.902), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.907 = f32[1,8,14,14,320]{4,3,2,1,0} broadcast(f32[1,320]{1,0} %reshape.906), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.896 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.897 = f32[1,8,14,14,160]{4,3,2,1,0} broadcast(f32[] %constant.896), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg49.50 = f32[1,1,1,1,160]{4,3,2,1,0} parameter(49), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.883 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.884 = f32[1,1,1,1,160]{4,3,2,1,0} broadcast(f32[] %constant.883), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.885 = f32[1,1,1,1,160]{4,3,2,1,0} add(f32[1,1,1,1,160]{4,3,2,1,0} %arg49.50, f32[1,1,1,1,160]{4,3,2,1,0} %broadcast.884), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.886 = f32[1,1,1,1,160]{4,3,2,1,0} rsqrt(f32[1,1,1,1,160]{4,3,2,1,0} %add.885), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.890 = f32[1,160]{1,0} reshape(f32[1,1,1,1,160]{4,3,2,1,0} %rsqrt.886), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.891 = f32[1,8,14,14,160]{4,3,2,1,0} broadcast(f32[1,160]{1,0} %reshape.890), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg46.47 = f32[1,1,1,528,160]{4,3,2,1,0} parameter(46), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.889 = f32[1,8,14,14,160]{4,3,2,1,0} convolution(f32[1,8,14,14,528]{4,3,2,1,0} %maximum.863, f32[1,1,1,528,160]{4,3,2,1,0} %arg46.47), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.892 = f32[1,8,14,14,160]{4,3,2,1,0} multiply(f32[1,8,14,14,160]{4,3,2,1,0} %broadcast.891, f32[1,8,14,14,160]{4,3,2,1,0} %convolution.889), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg47.48 = f32[1,1,1,1,160]{4,3,2,1,0} parameter(47), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg48.49 = f32[1,1,1,1,160]{4,3,2,1,0} parameter(48), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.887 = f32[1,1,1,1,160]{4,3,2,1,0} multiply(f32[1,1,1,1,160]{4,3,2,1,0} %arg48.49, f32[1,1,1,1,160]{4,3,2,1,0} %rsqrt.886), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.888 = f32[1,1,1,1,160]{4,3,2,1,0} subtract(f32[1,1,1,1,160]{4,3,2,1,0} %arg47.48, f32[1,1,1,1,160]{4,3,2,1,0} %multiply.887), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.893 = f32[1,160]{1,0} reshape(f32[1,1,1,1,160]{4,3,2,1,0} %subtract.888), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.894 = f32[1,8,14,14,160]{4,3,2,1,0} broadcast(f32[1,160]{1,0} %reshape.893), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.895 = f32[1,8,14,14,160]{4,3,2,1,0} add(f32[1,8,14,14,160]{4,3,2,1,0} %multiply.892, f32[1,8,14,14,160]{4,3,2,1,0} %broadcast.894), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.898 = f32[1,8,14,14,160]{4,3,2,1,0} maximum(f32[1,8,14,14,160]{4,3,2,1,0} %broadcast.897, f32[1,8,14,14,160]{4,3,2,1,0} %add.895), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg50.51 = f32[3,3,3,160,320]{4,3,2,1,0} parameter(50), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.905 = f32[1,8,14,14,320]{4,3,2,1,0} convolution(f32[1,8,14,14,160]{4,3,2,1,0} %maximum.898, f32[3,3,3,160,320]{4,3,2,1,0} %arg50.51), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.908 = f32[1,8,14,14,320]{4,3,2,1,0} multiply(f32[1,8,14,14,320]{4,3,2,1,0} %broadcast.907, f32[1,8,14,14,320]{4,3,2,1,0} %convolution.905), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg51.52 = f32[1,1,1,1,320]{4,3,2,1,0} parameter(51), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg52.53 = f32[1,1,1,1,320]{4,3,2,1,0} parameter(52), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.903 = f32[1,1,1,1,320]{4,3,2,1,0} multiply(f32[1,1,1,1,320]{4,3,2,1,0} %arg52.53, f32[1,1,1,1,320]{4,3,2,1,0} %rsqrt.902), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.904 = f32[1,1,1,1,320]{4,3,2,1,0} subtract(f32[1,1,1,1,320]{4,3,2,1,0} %arg51.52, f32[1,1,1,1,320]{4,3,2,1,0} %multiply.903), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.909 = f32[1,320]{1,0} reshape(f32[1,1,1,1,320]{4,3,2,1,0} %subtract.904), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.910 = f32[1,8,14,14,320]{4,3,2,1,0} broadcast(f32[1,320]{1,0} %reshape.909), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.911 = f32[1,8,14,14,320]{4,3,2,1,0} add(f32[1,8,14,14,320]{4,3,2,1,0} %multiply.908, f32[1,8,14,14,320]{4,3,2,1,0} %broadcast.910), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %arg61.62 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(61), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.928 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.929 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.928), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %add.930 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %arg61.62, f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.929), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.931 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.930), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.935 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.931), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.936 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.935), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.925 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.926 = f32[1,8,14,14,32]{4,3,2,1,0} broadcast(f32[] %constant.925), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg57.58 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(57), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.912 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.913 = f32[1,1,1,1,32]{4,3,2,1,0} broadcast(f32[] %constant.912), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.914 = f32[1,1,1,1,32]{4,3,2,1,0} add(f32[1,1,1,1,32]{4,3,2,1,0} %arg57.58, f32[1,1,1,1,32]{4,3,2,1,0} %broadcast.913), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.915 = f32[1,1,1,1,32]{4,3,2,1,0} rsqrt(f32[1,1,1,1,32]{4,3,2,1,0} %add.914), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.919 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.915), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.920 = f32[1,8,14,14,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.919), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg54.55 = f32[1,1,1,528,32]{4,3,2,1,0} parameter(54), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.918 = f32[1,8,14,14,32]{4,3,2,1,0} convolution(f32[1,8,14,14,528]{4,3,2,1,0} %maximum.863, f32[1,1,1,528,32]{4,3,2,1,0} %arg54.55), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.921 = f32[1,8,14,14,32]{4,3,2,1,0} multiply(f32[1,8,14,14,32]{4,3,2,1,0} %broadcast.920, f32[1,8,14,14,32]{4,3,2,1,0} %convolution.918), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg55.56 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(55), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg56.57 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(56), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.916 = f32[1,1,1,1,32]{4,3,2,1,0} multiply(f32[1,1,1,1,32]{4,3,2,1,0} %arg56.57, f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.915), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.917 = f32[1,1,1,1,32]{4,3,2,1,0} subtract(f32[1,1,1,1,32]{4,3,2,1,0} %arg55.56, f32[1,1,1,1,32]{4,3,2,1,0} %multiply.916), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.922 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %subtract.917), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.923 = f32[1,8,14,14,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.922), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.924 = f32[1,8,14,14,32]{4,3,2,1,0} add(f32[1,8,14,14,32]{4,3,2,1,0} %multiply.921, f32[1,8,14,14,32]{4,3,2,1,0} %broadcast.923), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.927 = f32[1,8,14,14,32]{4,3,2,1,0} maximum(f32[1,8,14,14,32]{4,3,2,1,0} %broadcast.926, f32[1,8,14,14,32]{4,3,2,1,0} %add.924), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg58.59 = f32[3,3,3,32,128]{4,3,2,1,0} parameter(58), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.934 = f32[1,8,14,14,128]{4,3,2,1,0} convolution(f32[1,8,14,14,32]{4,3,2,1,0} %maximum.927, f32[3,3,3,32,128]{4,3,2,1,0} %arg58.59), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.937 = f32[1,8,14,14,128]{4,3,2,1,0} multiply(f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.936, f32[1,8,14,14,128]{4,3,2,1,0} %convolution.934), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg59.60 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(59), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg60.61 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(60), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.932 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %arg60.61, f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.931), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.933 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg59.60, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.932), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.938 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.933), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.939 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.938), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.940 = f32[1,8,14,14,128]{4,3,2,1,0} add(f32[1,8,14,14,128]{4,3,2,1,0} %multiply.937, f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.939), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %arg221.222 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(221), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.941 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.942 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.941), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %add.943 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %arg221.222, f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.942), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.944 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.943), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.948 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.944), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.949 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.948), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %constant.864 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.869 = f32[1,8,14,14,528]{4,3,2,1,0} reduce-window(f32[1,8,14,14,528]{4,3,2,1,0} %maximum.863, f32[] %constant.864), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.865, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/MaxPool3d_0a_3x3"}
  %arg218.219 = f32[1,1,1,528,128]{4,3,2,1,0} parameter(218), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.947 = f32[1,8,14,14,128]{4,3,2,1,0} convolution(f32[1,8,14,14,528]{4,3,2,1,0} %reduce-window.869, f32[1,1,1,528,128]{4,3,2,1,0} %arg218.219), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %multiply.950 = f32[1,8,14,14,128]{4,3,2,1,0} multiply(f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.949, f32[1,8,14,14,128]{4,3,2,1,0} %convolution.947), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg219.220 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(219), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg220.221 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(220), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.945 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %arg220.221, f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.944), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.946 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg219.220, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.945), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.951 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.946), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.952 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.951), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.953 = f32[1,8,14,14,128]{4,3,2,1,0} add(f32[1,8,14,14,128]{4,3,2,1,0} %multiply.950, f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.952), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.954 = f32[1,8,14,14,832]{4,3,2,1,0} concatenate(f32[1,8,14,14,256]{4,3,2,1,0} %add.882, f32[1,8,14,14,320]{4,3,2,1,0} %add.911, f32[1,8,14,14,128]{4,3,2,1,0} %add.940, f32[1,8,14,14,128]{4,3,2,1,0} %add.953), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_4f/concat"}
  %constant.955 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_5a_2x2"}
  %reduce-window.960 = f32[1,4,7,7,832]{4,3,2,1,0} reduce-window(f32[1,8,14,14,832]{4,3,2,1,0} %concatenate.954, f32[] %constant.955), window={size=1x2x2x2x1 stride=1x2x2x2x1}, to_apply=%max_F32.956, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_5a_2x2"}
  %maximum.963 = f32[1,4,7,7,832]{4,3,2,1,0} maximum(f32[1,4,7,7,832]{4,3,2,1,0} %broadcast.962, f32[1,4,7,7,832]{4,3,2,1,0} %reduce-window.960), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg22.23 = f32[1,1,1,832,256]{4,3,2,1,0} parameter(22), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.976 = f32[1,4,7,7,256]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.963, f32[1,1,1,832,256]{4,3,2,1,0} %arg22.23), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.979 = f32[1,4,7,7,256]{4,3,2,1,0} multiply(f32[1,4,7,7,256]{4,3,2,1,0} %broadcast.978, f32[1,4,7,7,256]{4,3,2,1,0} %convolution.976), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg23.24 = f32[1,1,1,1,256]{4,3,2,1,0} parameter(23), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg24.25 = f32[1,1,1,1,256]{4,3,2,1,0} parameter(24), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.974 = f32[1,1,1,1,256]{4,3,2,1,0} multiply(f32[1,1,1,1,256]{4,3,2,1,0} %arg24.25, f32[1,1,1,1,256]{4,3,2,1,0} %rsqrt.973), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.975 = f32[1,1,1,1,256]{4,3,2,1,0} subtract(f32[1,1,1,1,256]{4,3,2,1,0} %arg23.24, f32[1,1,1,1,256]{4,3,2,1,0} %multiply.974), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.980 = f32[1,256]{1,0} reshape(f32[1,1,1,1,256]{4,3,2,1,0} %subtract.975), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.981 = f32[1,4,7,7,256]{4,3,2,1,0} broadcast(f32[1,256]{1,0} %reshape.980), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.982 = f32[1,4,7,7,256]{4,3,2,1,0} add(f32[1,4,7,7,256]{4,3,2,1,0} %multiply.979, f32[1,4,7,7,256]{4,3,2,1,0} %broadcast.981), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %arg33.34 = f32[1,1,1,1,320]{4,3,2,1,0} parameter(33), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.999 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.1000 = f32[1,1,1,1,320]{4,3,2,1,0} broadcast(f32[] %constant.999), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %add.1001 = f32[1,1,1,1,320]{4,3,2,1,0} add(f32[1,1,1,1,320]{4,3,2,1,0} %arg33.34, f32[1,1,1,1,320]{4,3,2,1,0} %broadcast.1000), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.1002 = f32[1,1,1,1,320]{4,3,2,1,0} rsqrt(f32[1,1,1,1,320]{4,3,2,1,0} %add.1001), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.1006 = f32[1,320]{1,0} reshape(f32[1,1,1,1,320]{4,3,2,1,0} %rsqrt.1002), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.1007 = f32[1,4,7,7,320]{4,3,2,1,0} broadcast(f32[1,320]{1,0} %reshape.1006), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.996 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.997 = f32[1,4,7,7,160]{4,3,2,1,0} broadcast(f32[] %constant.996), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg29.30 = f32[1,1,1,1,160]{4,3,2,1,0} parameter(29), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.983 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.984 = f32[1,1,1,1,160]{4,3,2,1,0} broadcast(f32[] %constant.983), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.985 = f32[1,1,1,1,160]{4,3,2,1,0} add(f32[1,1,1,1,160]{4,3,2,1,0} %arg29.30, f32[1,1,1,1,160]{4,3,2,1,0} %broadcast.984), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.986 = f32[1,1,1,1,160]{4,3,2,1,0} rsqrt(f32[1,1,1,1,160]{4,3,2,1,0} %add.985), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.990 = f32[1,160]{1,0} reshape(f32[1,1,1,1,160]{4,3,2,1,0} %rsqrt.986), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.991 = f32[1,4,7,7,160]{4,3,2,1,0} broadcast(f32[1,160]{1,0} %reshape.990), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg26.27 = f32[1,1,1,832,160]{4,3,2,1,0} parameter(26), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.989 = f32[1,4,7,7,160]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.963, f32[1,1,1,832,160]{4,3,2,1,0} %arg26.27), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.992 = f32[1,4,7,7,160]{4,3,2,1,0} multiply(f32[1,4,7,7,160]{4,3,2,1,0} %broadcast.991, f32[1,4,7,7,160]{4,3,2,1,0} %convolution.989), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg27.28 = f32[1,1,1,1,160]{4,3,2,1,0} parameter(27), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg28.29 = f32[1,1,1,1,160]{4,3,2,1,0} parameter(28), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.987 = f32[1,1,1,1,160]{4,3,2,1,0} multiply(f32[1,1,1,1,160]{4,3,2,1,0} %arg28.29, f32[1,1,1,1,160]{4,3,2,1,0} %rsqrt.986), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.988 = f32[1,1,1,1,160]{4,3,2,1,0} subtract(f32[1,1,1,1,160]{4,3,2,1,0} %arg27.28, f32[1,1,1,1,160]{4,3,2,1,0} %multiply.987), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.993 = f32[1,160]{1,0} reshape(f32[1,1,1,1,160]{4,3,2,1,0} %subtract.988), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.994 = f32[1,4,7,7,160]{4,3,2,1,0} broadcast(f32[1,160]{1,0} %reshape.993), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.995 = f32[1,4,7,7,160]{4,3,2,1,0} add(f32[1,4,7,7,160]{4,3,2,1,0} %multiply.992, f32[1,4,7,7,160]{4,3,2,1,0} %broadcast.994), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.998 = f32[1,4,7,7,160]{4,3,2,1,0} maximum(f32[1,4,7,7,160]{4,3,2,1,0} %broadcast.997, f32[1,4,7,7,160]{4,3,2,1,0} %add.995), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg30.31 = f32[3,3,3,160,320]{4,3,2,1,0} parameter(30), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1005 = f32[1,4,7,7,320]{4,3,2,1,0} convolution(f32[1,4,7,7,160]{4,3,2,1,0} %maximum.998, f32[3,3,3,160,320]{4,3,2,1,0} %arg30.31), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.1008 = f32[1,4,7,7,320]{4,3,2,1,0} multiply(f32[1,4,7,7,320]{4,3,2,1,0} %broadcast.1007, f32[1,4,7,7,320]{4,3,2,1,0} %convolution.1005), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg31.32 = f32[1,1,1,1,320]{4,3,2,1,0} parameter(31), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg32.33 = f32[1,1,1,1,320]{4,3,2,1,0} parameter(32), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1003 = f32[1,1,1,1,320]{4,3,2,1,0} multiply(f32[1,1,1,1,320]{4,3,2,1,0} %arg32.33, f32[1,1,1,1,320]{4,3,2,1,0} %rsqrt.1002), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.1004 = f32[1,1,1,1,320]{4,3,2,1,0} subtract(f32[1,1,1,1,320]{4,3,2,1,0} %arg31.32, f32[1,1,1,1,320]{4,3,2,1,0} %multiply.1003), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.1009 = f32[1,320]{1,0} reshape(f32[1,1,1,1,320]{4,3,2,1,0} %subtract.1004), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.1010 = f32[1,4,7,7,320]{4,3,2,1,0} broadcast(f32[1,320]{1,0} %reshape.1009), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.1011 = f32[1,4,7,7,320]{4,3,2,1,0} add(f32[1,4,7,7,320]{4,3,2,1,0} %multiply.1008, f32[1,4,7,7,320]{4,3,2,1,0} %broadcast.1010), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %arg41.42 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(41), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.1028 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/add"}
  %broadcast.1029 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.1028), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/add"}
  %add.1030 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %arg41.42, f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.1029), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/add"}
  %rsqrt.1031 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.1030), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.1035 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.1031), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/mul"}
  %broadcast.1036 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.1035), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/mul"}
  %constant.1025 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.1026 = f32[1,4,7,7,32]{4,3,2,1,0} broadcast(f32[] %constant.1025), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg37.38 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(37), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.1012 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.1013 = f32[1,1,1,1,32]{4,3,2,1,0} broadcast(f32[] %constant.1012), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %add.1014 = f32[1,1,1,1,32]{4,3,2,1,0} add(f32[1,1,1,1,32]{4,3,2,1,0} %arg37.38, f32[1,1,1,1,32]{4,3,2,1,0} %broadcast.1013), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1015 = f32[1,1,1,1,32]{4,3,2,1,0} rsqrt(f32[1,1,1,1,32]{4,3,2,1,0} %add.1014), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1019 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.1015), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1020 = f32[1,4,7,7,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.1019), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg34.35 = f32[1,1,1,832,32]{4,3,2,1,0} parameter(34), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1018 = f32[1,4,7,7,32]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.963, f32[1,1,1,832,32]{4,3,2,1,0} %arg34.35), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.1021 = f32[1,4,7,7,32]{4,3,2,1,0} multiply(f32[1,4,7,7,32]{4,3,2,1,0} %broadcast.1020, f32[1,4,7,7,32]{4,3,2,1,0} %convolution.1018), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg35.36 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(35), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg36.37 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(36), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1016 = f32[1,1,1,1,32]{4,3,2,1,0} multiply(f32[1,1,1,1,32]{4,3,2,1,0} %arg36.37, f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.1015), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1017 = f32[1,1,1,1,32]{4,3,2,1,0} subtract(f32[1,1,1,1,32]{4,3,2,1,0} %arg35.36, f32[1,1,1,1,32]{4,3,2,1,0} %multiply.1016), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.1022 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %subtract.1017), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1023 = f32[1,4,7,7,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.1022), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.1024 = f32[1,4,7,7,32]{4,3,2,1,0} add(f32[1,4,7,7,32]{4,3,2,1,0} %multiply.1021, f32[1,4,7,7,32]{4,3,2,1,0} %broadcast.1023), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.1027 = f32[1,4,7,7,32]{4,3,2,1,0} maximum(f32[1,4,7,7,32]{4,3,2,1,0} %broadcast.1026, f32[1,4,7,7,32]{4,3,2,1,0} %add.1024), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg38.39 = f32[3,3,3,32,128]{4,3,2,1,0} parameter(38), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1034 = f32[1,4,7,7,128]{4,3,2,1,0} convolution(f32[1,4,7,7,32]{4,3,2,1,0} %maximum.1027, f32[3,3,3,32,128]{4,3,2,1,0} %arg38.39), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/conv_3d/convolution"}
  %multiply.1037 = f32[1,4,7,7,128]{4,3,2,1,0} multiply(f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.1036, f32[1,4,7,7,128]{4,3,2,1,0} %convolution.1034), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/mul"}
  %arg39.40 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(39), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg40.41 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(40), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1032 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %arg40.41, f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.1031), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.1033 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg39.40, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.1032), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/sub"}
  %reshape.1038 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.1033), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.1039 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.1038), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/add_1"}
  %add.1040 = f32[1,4,7,7,128]{4,3,2,1,0} add(f32[1,4,7,7,128]{4,3,2,1,0} %multiply.1037, f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.1039), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/add_1"}
  %arg225.226 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(225), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %constant.1041 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.1042 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.1041), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %add.1043 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %arg225.226, f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.1042), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1044 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.1043), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1048 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.1044), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1049 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.1048), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %constant.964 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.969 = f32[1,4,7,7,832]{4,3,2,1,0} reduce-window(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.963, f32[] %constant.964), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.965, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/MaxPool3d_0a_3x3"}
  %arg222.223 = f32[1,1,1,832,128]{4,3,2,1,0} parameter(222), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1047 = f32[1,4,7,7,128]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %reduce-window.969, f32[1,1,1,832,128]{4,3,2,1,0} %arg222.223), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %multiply.1050 = f32[1,4,7,7,128]{4,3,2,1,0} multiply(f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.1049, f32[1,4,7,7,128]{4,3,2,1,0} %convolution.1047), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg223.224 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(223), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg224.225 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(224), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1045 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %arg224.225, f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.1044), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1046 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg223.224, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.1045), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.1051 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.1046), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1052 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.1051), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.1053 = f32[1,4,7,7,128]{4,3,2,1,0} add(f32[1,4,7,7,128]{4,3,2,1,0} %multiply.1050, f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.1052), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.1054 = f32[1,4,7,7,832]{4,3,2,1,0} concatenate(f32[1,4,7,7,256]{4,3,2,1,0} %add.982, f32[1,4,7,7,320]{4,3,2,1,0} %add.1011, f32[1,4,7,7,128]{4,3,2,1,0} %add.1040, f32[1,4,7,7,128]{4,3,2,1,0} %add.1053), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_5b/concat"}
  %maximum.1057 = f32[1,4,7,7,832]{4,3,2,1,0} maximum(f32[1,4,7,7,832]{4,3,2,1,0} %broadcast.1056, f32[1,4,7,7,832]{4,3,2,1,0} %concatenate.1054), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg2.3 = f32[1,1,1,832,384]{4,3,2,1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1070 = f32[1,4,7,7,384]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.1057, f32[1,1,1,832,384]{4,3,2,1,0} %arg2.3), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.1073 = f32[1,4,7,7,384]{4,3,2,1,0} multiply(f32[1,4,7,7,384]{4,3,2,1,0} %broadcast.1072, f32[1,4,7,7,384]{4,3,2,1,0} %convolution.1070), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg3.4 = f32[1,1,1,1,384]{4,3,2,1,0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg4.5 = f32[1,1,1,1,384]{4,3,2,1,0} parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1068 = f32[1,1,1,1,384]{4,3,2,1,0} multiply(f32[1,1,1,1,384]{4,3,2,1,0} %arg4.5, f32[1,1,1,1,384]{4,3,2,1,0} %rsqrt.1067), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1069 = f32[1,1,1,1,384]{4,3,2,1,0} subtract(f32[1,1,1,1,384]{4,3,2,1,0} %arg3.4, f32[1,1,1,1,384]{4,3,2,1,0} %multiply.1068), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.1074 = f32[1,384]{1,0} reshape(f32[1,1,1,1,384]{4,3,2,1,0} %subtract.1069), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1075 = f32[1,4,7,7,384]{4,3,2,1,0} broadcast(f32[1,384]{1,0} %reshape.1074), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.1076 = f32[1,4,7,7,384]{4,3,2,1,0} add(f32[1,4,7,7,384]{4,3,2,1,0} %multiply.1073, f32[1,4,7,7,384]{4,3,2,1,0} %broadcast.1075), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %constant.1093 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.1094 = f32[1,1,1,1,384]{4,3,2,1,0} broadcast(f32[] %constant.1093), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %arg13.14 = f32[1,1,1,1,384]{4,3,2,1,0} parameter(13), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %add.1095 = f32[1,1,1,1,384]{4,3,2,1,0} add(f32[1,1,1,1,384]{4,3,2,1,0} %broadcast.1094, f32[1,1,1,1,384]{4,3,2,1,0} %arg13.14), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.1096 = f32[1,1,1,1,384]{4,3,2,1,0} rsqrt(f32[1,1,1,1,384]{4,3,2,1,0} %add.1095), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.1100 = f32[1,384]{1,0} reshape(f32[1,1,1,1,384]{4,3,2,1,0} %rsqrt.1096), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.1101 = f32[1,4,7,7,384]{4,3,2,1,0} broadcast(f32[1,384]{1,0} %reshape.1100), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.1090 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.1091 = f32[1,4,7,7,192]{4,3,2,1,0} broadcast(f32[] %constant.1090), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/Relu"}
  %constant.1077 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.1078 = f32[1,1,1,1,192]{4,3,2,1,0} broadcast(f32[] %constant.1077), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg9.10 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(9), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %add.1079 = f32[1,1,1,1,192]{4,3,2,1,0} add(f32[1,1,1,1,192]{4,3,2,1,0} %broadcast.1078, f32[1,1,1,1,192]{4,3,2,1,0} %arg9.10), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1080 = f32[1,1,1,1,192]{4,3,2,1,0} rsqrt(f32[1,1,1,1,192]{4,3,2,1,0} %add.1079), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1084 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.1080), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1085 = f32[1,4,7,7,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.1084), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg6.7 = f32[1,1,1,832,192]{4,3,2,1,0} parameter(6), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1083 = f32[1,4,7,7,192]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.1057, f32[1,1,1,832,192]{4,3,2,1,0} %arg6.7), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.1086 = f32[1,4,7,7,192]{4,3,2,1,0} multiply(f32[1,4,7,7,192]{4,3,2,1,0} %broadcast.1085, f32[1,4,7,7,192]{4,3,2,1,0} %convolution.1083), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg7.8 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(7), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg8.9 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(8), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1081 = f32[1,1,1,1,192]{4,3,2,1,0} multiply(f32[1,1,1,1,192]{4,3,2,1,0} %arg8.9, f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.1080), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1082 = f32[1,1,1,1,192]{4,3,2,1,0} subtract(f32[1,1,1,1,192]{4,3,2,1,0} %arg7.8, f32[1,1,1,1,192]{4,3,2,1,0} %multiply.1081), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.1087 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %subtract.1082), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1088 = f32[1,4,7,7,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.1087), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.1089 = f32[1,4,7,7,192]{4,3,2,1,0} add(f32[1,4,7,7,192]{4,3,2,1,0} %multiply.1086, f32[1,4,7,7,192]{4,3,2,1,0} %broadcast.1088), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.1092 = f32[1,4,7,7,192]{4,3,2,1,0} maximum(f32[1,4,7,7,192]{4,3,2,1,0} %broadcast.1091, f32[1,4,7,7,192]{4,3,2,1,0} %add.1089), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg10.11 = f32[3,3,3,192,384]{4,3,2,1,0} parameter(10), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1099 = f32[1,4,7,7,384]{4,3,2,1,0} convolution(f32[1,4,7,7,192]{4,3,2,1,0} %maximum.1092, f32[3,3,3,192,384]{4,3,2,1,0} %arg10.11), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.1102 = f32[1,4,7,7,384]{4,3,2,1,0} multiply(f32[1,4,7,7,384]{4,3,2,1,0} %broadcast.1101, f32[1,4,7,7,384]{4,3,2,1,0} %convolution.1099), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg11.12 = f32[1,1,1,1,384]{4,3,2,1,0} parameter(11), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg12.13 = f32[1,1,1,1,384]{4,3,2,1,0} parameter(12), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1097 = f32[1,1,1,1,384]{4,3,2,1,0} multiply(f32[1,1,1,1,384]{4,3,2,1,0} %arg12.13, f32[1,1,1,1,384]{4,3,2,1,0} %rsqrt.1096), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.1098 = f32[1,1,1,1,384]{4,3,2,1,0} subtract(f32[1,1,1,1,384]{4,3,2,1,0} %arg11.12, f32[1,1,1,1,384]{4,3,2,1,0} %multiply.1097), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.1103 = f32[1,384]{1,0} reshape(f32[1,1,1,1,384]{4,3,2,1,0} %subtract.1098), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.1104 = f32[1,4,7,7,384]{4,3,2,1,0} broadcast(f32[1,384]{1,0} %reshape.1103), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.1105 = f32[1,4,7,7,384]{4,3,2,1,0} add(f32[1,4,7,7,384]{4,3,2,1,0} %multiply.1102, f32[1,4,7,7,384]{4,3,2,1,0} %broadcast.1104), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.1122 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.1123 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.1122), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %arg21.22 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(21), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %add.1124 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.1123, f32[1,1,1,1,128]{4,3,2,1,0} %arg21.22), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.1125 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.1124), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.1129 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.1125), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.1130 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.1129), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %constant.1119 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.1120 = f32[1,4,7,7,48]{4,3,2,1,0} broadcast(f32[] %constant.1119), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/Relu"}
  %constant.1106 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.1107 = f32[1,1,1,1,48]{4,3,2,1,0} broadcast(f32[] %constant.1106), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg17.18 = f32[1,1,1,1,48]{4,3,2,1,0} parameter(17), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %add.1108 = f32[1,1,1,1,48]{4,3,2,1,0} add(f32[1,1,1,1,48]{4,3,2,1,0} %broadcast.1107, f32[1,1,1,1,48]{4,3,2,1,0} %arg17.18), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1109 = f32[1,1,1,1,48]{4,3,2,1,0} rsqrt(f32[1,1,1,1,48]{4,3,2,1,0} %add.1108), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1113 = f32[1,48]{1,0} reshape(f32[1,1,1,1,48]{4,3,2,1,0} %rsqrt.1109), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1114 = f32[1,4,7,7,48]{4,3,2,1,0} broadcast(f32[1,48]{1,0} %reshape.1113), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg14.15 = f32[1,1,1,832,48]{4,3,2,1,0} parameter(14), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1112 = f32[1,4,7,7,48]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.1057, f32[1,1,1,832,48]{4,3,2,1,0} %arg14.15), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %multiply.1115 = f32[1,4,7,7,48]{4,3,2,1,0} multiply(f32[1,4,7,7,48]{4,3,2,1,0} %broadcast.1114, f32[1,4,7,7,48]{4,3,2,1,0} %convolution.1112), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg15.16 = f32[1,1,1,1,48]{4,3,2,1,0} parameter(15), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg16.17 = f32[1,1,1,1,48]{4,3,2,1,0} parameter(16), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1110 = f32[1,1,1,1,48]{4,3,2,1,0} multiply(f32[1,1,1,1,48]{4,3,2,1,0} %arg16.17, f32[1,1,1,1,48]{4,3,2,1,0} %rsqrt.1109), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1111 = f32[1,1,1,1,48]{4,3,2,1,0} subtract(f32[1,1,1,1,48]{4,3,2,1,0} %arg15.16, f32[1,1,1,1,48]{4,3,2,1,0} %multiply.1110), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.1116 = f32[1,48]{1,0} reshape(f32[1,1,1,1,48]{4,3,2,1,0} %subtract.1111), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1117 = f32[1,4,7,7,48]{4,3,2,1,0} broadcast(f32[1,48]{1,0} %reshape.1116), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.1118 = f32[1,4,7,7,48]{4,3,2,1,0} add(f32[1,4,7,7,48]{4,3,2,1,0} %multiply.1115, f32[1,4,7,7,48]{4,3,2,1,0} %broadcast.1117), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.1121 = f32[1,4,7,7,48]{4,3,2,1,0} maximum(f32[1,4,7,7,48]{4,3,2,1,0} %broadcast.1120, f32[1,4,7,7,48]{4,3,2,1,0} %add.1118), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg18.19 = f32[3,3,3,48,128]{4,3,2,1,0} parameter(18), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1128 = f32[1,4,7,7,128]{4,3,2,1,0} convolution(f32[1,4,7,7,48]{4,3,2,1,0} %maximum.1121, f32[3,3,3,48,128]{4,3,2,1,0} %arg18.19), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %multiply.1131 = f32[1,4,7,7,128]{4,3,2,1,0} multiply(f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.1130, f32[1,4,7,7,128]{4,3,2,1,0} %convolution.1128), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg19.20 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(19), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg20.21 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(20), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1126 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %arg20.21, f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.1125), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.1127 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg19.20, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.1126), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.1132 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.1127), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.1133 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.1132), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.1134 = f32[1,4,7,7,128]{4,3,2,1,0} add(f32[1,4,7,7,128]{4,3,2,1,0} %multiply.1131, f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.1133), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.1135 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.1136 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.1135), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %arg229.230 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(229), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %add.1137 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.1136, f32[1,1,1,1,128]{4,3,2,1,0} %arg229.230), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1138 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.1137), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1142 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.1138), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1143 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.1142), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %constant.1058 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.1063 = f32[1,4,7,7,832]{4,3,2,1,0} reduce-window(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.1057, f32[] %constant.1058), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.1059, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/MaxPool3d_0a_3x3"}
  %arg226.227 = f32[1,1,1,832,128]{4,3,2,1,0} parameter(226), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1141 = f32[1,4,7,7,128]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %reduce-window.1063, f32[1,1,1,832,128]{4,3,2,1,0} %arg226.227), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %multiply.1144 = f32[1,4,7,7,128]{4,3,2,1,0} multiply(f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.1143, f32[1,4,7,7,128]{4,3,2,1,0} %convolution.1141), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg227.228 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(227), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg228.229 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(228), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1139 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %arg228.229, f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.1138), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1140 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg227.228, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.1139), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.1145 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.1140), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1146 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.1145), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.1147 = f32[1,4,7,7,128]{4,3,2,1,0} add(f32[1,4,7,7,128]{4,3,2,1,0} %multiply.1144, f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.1146), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.1148 = f32[1,4,7,7,1024]{4,3,2,1,0} concatenate(f32[1,4,7,7,384]{4,3,2,1,0} %add.1076, f32[1,4,7,7,384]{4,3,2,1,0} %add.1105, f32[1,4,7,7,128]{4,3,2,1,0} %add.1134, f32[1,4,7,7,128]{4,3,2,1,0} %add.1147), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_5c/concat"}
  %maximum.1151 = f32[1,4,7,7,1024]{4,3,2,1,0} maximum(f32[1,4,7,7,1024]{4,3,2,1,0} %broadcast.1150, f32[1,4,7,7,1024]{4,3,2,1,0} %concatenate.1148), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/Relu"}
  %convert.1152 = f32[1,4,7,7,1024]{4,3,2,1,0} convert(f32[1,4,7,7,1024]{4,3,2,1,0} %maximum.1151), metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %constant.1154 = f32[] constant(0), metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %pad.1155 = f32[1,4,7,7,1024]{4,3,2,1,0} pad(f32[1,4,7,7,1024]{4,3,2,1,0} %convert.1152, f32[] %constant.1154), padding=0_0x0_0x0_0x0_0x0_0, metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %constant.1153 = f32[] constant(0), metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %reduce-window.1160 = f32[1,3,1,1,1024]{4,3,2,1,0} reduce-window(f32[1,4,7,7,1024]{4,3,2,1,0} %pad.1155, f32[] %constant.1153), window={size=1x2x7x7x1}, to_apply=%add_F32.1156, metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %constant.1161 = f32[] constant(98), metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %broadcast.1162 = f32[1,3,1,1,1024]{4,3,2,1,0} broadcast(f32[] %constant.1161), dimensions={}, metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %divide.1163 = f32[1,3,1,1,1024]{4,3,2,1,0} divide(f32[1,3,1,1,1024]{4,3,2,1,0} %reduce-window.1160, f32[1,3,1,1,1024]{4,3,2,1,0} %broadcast.1162), metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %convert.1164 = f32[1,3,1,1,1024]{4,3,2,1,0} convert(f32[1,3,1,1,1024]{4,3,2,1,0} %divide.1163), metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %arg230.231 = f32[1,1,1,1024,400]{4,3,2,1,0} parameter(230), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1165 = f32[1,3,1,1,400]{4,3,2,1,0} convolution(f32[1,3,1,1,1024]{4,3,2,1,0} %convert.1164, f32[1,1,1,1024,400]{4,3,2,1,0} %arg230.231), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/convolution"}
  %add.1168 = f32[1,3,1,1,400]{4,3,2,1,0} add(f32[1,3,1,1,400]{4,3,2,1,0} %broadcast.1167, f32[1,3,1,1,400]{4,3,2,1,0} %convolution.1165), metadata={op_type="Add" op_name="RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/add"}
  %reshape.1169 = f32[1,3,400]{2,1,0} reshape(f32[1,3,1,1,400]{4,3,2,1,0} %add.1168), metadata={op_type="Squeeze" op_name="RGB/inception_i3d/Logits/SpatialSqueeze"}
  %convert.1170 = f32[1,3,400]{2,1,0} convert(f32[1,3,400]{2,1,0} %reshape.1169), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %constant.1171 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %convert.1172 = f32[] convert(f32[] %constant.1171), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %reduce.1177 = f32[1,400]{1,0} reduce(f32[1,3,400]{2,1,0} %convert.1170, f32[] %convert.1172), dimensions={1}, to_apply=%RGB_inception_i3d_Mean-reduction.1173, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %constant.1178 = s32[] constant(3), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %convert.1179 = f32[] convert(s32[] %constant.1178), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %broadcast.1180 = f32[1,400]{1,0} broadcast(f32[] %convert.1179), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %divide.1181 = f32[1,400]{1,0} divide(f32[1,400]{1,0} %reduce.1177, f32[1,400]{1,0} %broadcast.1180), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %convert.1182 = f32[1,400]{1,0} convert(f32[1,400]{1,0} %divide.1181), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %reshape.1183 = f32[1,400]{1,0} reshape(f32[1,400]{1,0} %convert.1182), metadata={op_name="XLA_Retvals"}
  %tuple.1184 = (f32[1,400]{1,0}) tuple(f32[1,400]{1,0} %reshape.1183), metadata={op_name="XLA_Retvals"}
  ROOT %get-tuple-element.1185 = f32[1,400]{1,0} get-tuple-element((f32[1,400]{1,0}) %tuple.1184), index=0, metadata={op_name="XLA_Retvals"}
}
)";

  HloModuleConfig cfg;
  auto hlo_module = std::make_unique<VerifiedHloModule>("module", cfg, false, false, nullptr);
  hlo_module->ParseHloStringAndVerifyModule(hlo_text);
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
