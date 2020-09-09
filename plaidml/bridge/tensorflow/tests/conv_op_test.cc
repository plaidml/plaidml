// Tests that show HLO Module conversion to PlaidML Program.

// #include "plaidml/bridge/tensorflow/tests/conv_op_test.h.inc"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "tensorflow/compiler/xla/tests/test_utils.h"

#include "plaidml/bridge/tensorflow/tests/codegen_test.h"

namespace xla {
namespace plaidml {
namespace {

struct ConvTestSpec {
  PrimitiveType primitive_type;
};

string ConvTestSpecToString(const ::testing::TestParamInfo<ConvTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLConvOperationTest : public PlaidMLCodegenTest, public ::testing::WithParamInterface<ConvTestSpec> {};

// TODO: rewrite this test as an actual python-based unit tests, so we can use the tf python bindings directly.
// TEST_P(PlaidMLConvOperationTest, VariedInputConvTest) {
//   VLOG(0) << "Testing generated examples";

//   for (std::size_t i = 0; i < conv_modules.size(); ++i) {
//     std::string set_des = conv_descriptions[i];
//     VLOG(0) << "Testing set " << i << ": " << set_des;
//     std::vector<float> input_val = conv_is[i];
//     std::vector<float> kernel = conv_ks[i];
//     std::vector<float> expected_val = conv_os[i];
//     std::string module_text = conv_modules[i];

//     TestCaseVal inputs = {kernel, input_val};
//     TestCaseVal results = {expected_val};

//     TestCasePairs testcase_pairs = {{inputs, results}};

//     ConvTestSpec spec = GetParam();

//     HloModuleConfig cfg;

//     std::unique_ptr<VerifiedHloModule> hlo_module =
//         absl::make_unique<VerifiedHloModule>("module", cfg, false, false, nullptr);

//     hlo_module->ParseHloStringAndVerifyModule(module_text);

//     CompileAndCheck(std::move(hlo_module), testcases);
//   }
// }

TEST_P(PlaidMLConvOperationTest, SimpleConvOp) {
  std::vector<float> input_val = {0.05,  0.05,  0.05,  0.05,  0.05,   //
                                  0.025, 0.025, 0.025, 0.025, 0.025,  //
                                  0.01,  0.01,  0.01,  0.01,  0.01,   //
                                  0.025, 0.025, 0.025, 0.025, 0.025,  //
                                  0.05,  0.05,  0.05,  0.05,  0.05};
  std::vector<float> kernel_val = {1, 1, 1,  //
                                   1, 0, 1,  //
                                   1, 1, 1};

  std::vector<float> expected_val = {0.23, 0.23, 0.23,  //
                                     0.17, 0.17, 0.17,  //
                                     0.23, 0.23, 0.23};
  TestCases testcases = {
      TestCaseIO{{kernel_val, input_val}, {expected_val}},
  };

  HloComputation::Builder builder("SimpleConvOp");
  ConvTestSpec spec = GetParam();

  auto input_shape = ShapeUtil::MakeShape(spec.primitive_type, {1, 5, 5, 1});
  auto kernel_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3, 1, 1});

  HloInstruction* input = builder.AddInstruction(HloInstruction::CreateParameter(0, input_shape, "input"));
  HloInstruction* kernel = builder.AddInstruction(HloInstruction::CreateParameter(1, kernel_shape, "input"));

  auto conv_shape = Shape();
  conv_shape.set_element_type(spec.primitive_type);
  conv_shape.add_dimensions(1);
  conv_shape.add_dimensions(3);
  conv_shape.add_dimensions(3);
  conv_shape.add_dimensions(1);
  Window conv_window;
  WindowDimension* conv_dim_1 = conv_window.add_dimensions();
  conv_dim_1->set_size(3);
  conv_dim_1->set_padding_low(0);
  conv_dim_1->set_padding_high(0);
  conv_dim_1->set_stride(1);
  conv_dim_1->set_window_dilation(1);
  conv_dim_1->set_base_dilation(1);
  conv_dim_1->set_window_reversal(false);
  WindowDimension* conv_dim_2 = conv_window.add_dimensions();
  conv_dim_2->set_size(3);
  conv_dim_2->set_padding_low(0);
  conv_dim_2->set_padding_high(0);
  conv_dim_2->set_stride(1);
  conv_dim_2->set_window_dilation(1);
  conv_dim_2->set_base_dilation(1);
  conv_dim_2->set_window_reversal(false);
  ConvolutionDimensionNumbers conv_dnums;
  conv_dnums.set_input_batch_dimension(0);
  conv_dnums.add_input_spatial_dimensions(1);
  conv_dnums.add_input_spatial_dimensions(2);
  conv_dnums.set_input_feature_dimension(3);
  conv_dnums.add_kernel_spatial_dimensions(0);
  conv_dnums.add_kernel_spatial_dimensions(1);
  conv_dnums.set_kernel_input_feature_dimension(2);
  conv_dnums.set_kernel_output_feature_dimension(3);
  conv_dnums.set_output_batch_dimension(0);
  conv_dnums.add_output_spatial_dimensions(1);
  conv_dnums.add_output_spatial_dimensions(2);
  conv_dnums.set_output_feature_dimension(3);
  PrecisionConfig conv_pc;
  conv_pc.mutable_operand_precision()->Resize(
      /*new_size=*/2, PrecisionConfig::DEFAULT);
  auto convolution_19 = builder.AddInstruction(
      HloAllGatherInstruction::CreateConvolve(conv_shape, input, kernel, 1, 1, conv_window, conv_dnums, conv_pc));

  CompileAndCheck(builder.Build(), testcases);
}

std::vector<ConvTestSpec> GetConvTestCases() {
  std::vector<ConvTestSpec> result;
  result.push_back({F32});
  result.push_back({F64});
  return result;
}

INSTANTIATE_TEST_SUITE_P(All, PlaidMLConvOperationTest, ::testing::ValuesIn(GetConvTestCases()), ConvTestSpecToString);

}  // namespace
}  // namespace plaidml
}  // namespace xla
