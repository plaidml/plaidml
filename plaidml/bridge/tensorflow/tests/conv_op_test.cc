// Tests that show HLO Module conversion to PlaidML Program.

#include "plaidml/bridge/tensorflow/tests/conv_op_test.h.inc"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "absl/strings/str_cat.h"
#include "plaidml/bridge/tensorflow/service/compiler.h"
#include "plaidml/bridge/tensorflow/tests/codegen_test.h"
#include "plaidml/bridge/tensorflow/tests/filecheck.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

using ::plaidml::edsl::TensorBuffers;

namespace xla {
namespace plaidml {
namespace {

using TestCaseVal = std::vector<std::vector<float>>;
using TestCasePairs = std::map<TestCaseVal, TestCaseVal>;

struct ConvTestSpec {
  PrimitiveType primitive_type;
  string filecheck_lines;
};

string ConvTestSpecToString(const ::testing::TestParamInfo<ConvTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLConvOperationTest : public PlaidMLCodegenTest, public ::testing::WithParamInterface<ConvTestSpec> {
 protected:
  Status CompileAndCheck(std::unique_ptr<HloComputation> entry_computation, const string& filecheck_lines,
                         const TestCasePairs& testcase_pairs) {
    HloModuleConfig cfg;

    std::unique_ptr<HloModule> hlo_module = absl::make_unique<HloModule>("module", cfg);
    hlo_module->AddEntryComputation(std::move(entry_computation));

    auto program = CompileToProgram(std::move(hlo_module));

    VLOG(2) << "Program:\n" << program->str();

    StatusOr<bool> fc_result = RunFileCheck(program->str(), filecheck_lines);

    // TF_ASSERT_OK(fc_result.status());
    EXPECT_TRUE(fc_result.ValueOrDie());

    VLOG(2) << "Evaluating results";

    for (auto pair : testcase_pairs) {
      TensorBuffers inp;
      TensorBuffers exp;

      auto program_inputs = program->inputs();

      for (auto i = 0; i < program_inputs.size(); i++) {
        inp.insert(std::make_pair(program_inputs[i].tensor, pair.first[i]));
      }

      auto program_outputs = program->outputs();

      for (auto i = 0; i < program_outputs.size(); i++) {
        exp.insert(std::make_pair(program_outputs[i].tensor, pair.second[i]));
      }

      checkProgram(*program, inp, exp);
    }

    return Status::OK();
  }

  Status CompileAndCheck(std::unique_ptr<HloModule> hlo_module, const string& filecheck_lines,
                         const TestCasePairs& testcase_pairs) {
    auto program = CompileToProgram(std::move(hlo_module));

    StatusOr<bool> fc_result = RunFileCheck(program->str(), filecheck_lines);

    // TF_ASSERT_OK(fc_result.status());
    EXPECT_TRUE(fc_result.ValueOrDie());

    VLOG(2) << "Evaluating results";

    for (auto pair : testcase_pairs) {
      TensorBuffers inp;
      TensorBuffers exp;

      auto program_inputs = program->inputs();
      auto tcp_inputs = pair.first;

      if (tcp_inputs.size() != program_inputs.size()) {
        VLOG(1) << "Found mismatch in input sizes: tcp " << tcp_inputs.size() << " program " << program_inputs.size();
      }

      for (auto i = 0; i < program_inputs.size(); i++) {
        VLOG(1) << "Adding TestCaseInput " << i;
        inp.insert(std::make_pair(program_inputs[i].tensor, pair.first[i]));
      }

      auto program_outputs = program->outputs();
      auto tcp_outputs = pair.second;

      if (tcp_outputs.size() != program_outputs.size()) {
        VLOG(1) << "Found mismatch in output sizes: tcp " << tcp_outputs.size() << " program "
                << program_outputs.size();
      }

      for (auto i = 0; i < program_outputs.size(); i++) {
        VLOG(1) << "Adding TestCaseOutput " << i;
        exp.insert(std::make_pair(program_outputs[i].tensor, pair.second[i]));
      }

      VLOG(2) << "Calling checkProgram";

      checkProgram(*program, inp, exp);
    }
    return Status::OK();
  }
};

TEST_P(PlaidMLConvOperationTest, VariedInputConvTest) {
  VLOG(0) << "Testing generated examples";

  for (std::size_t i = 0; i < conv_modules.size(); ++i) {
    std::string set_des = conv_descriptions[i];
    VLOG(0) << "Testing set " << i << ": " << set_des;
    std::vector<float> input_val = conv_is[i];
    std::vector<float> kernel = conv_ks[i];
    std::vector<float> expected_val = conv_os[i];
    std::string module_text = conv_modules[i];

    TestCaseVal inputs = {kernel, input_val};
    TestCaseVal results = {expected_val};

    TestCasePairs testcase_pairs = {{inputs, results}};

    ConvTestSpec spec = GetParam();

    HloModuleConfig cfg;

    std::unique_ptr<VerifiedHloModule> hlo_module =
        absl::make_unique<VerifiedHloModule>("module", cfg, false, false, nullptr);

    hlo_module->ParseHloStringAndVerifyModule(module_text);

    CompileAndCheck(std::move(hlo_module), spec.filecheck_lines, testcase_pairs);
  }
}

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

  TestCaseVal inputs = {kernel_val, input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

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

  CompileAndCheck(builder.Build(), spec.filecheck_lines, testcase_pairs);
}

std::vector<ConvTestSpec> GetConvTestCases() {
  std::vector<ConvTestSpec> result;
  auto check_str = R"#(
                        CHECK: func @hlo_module{{.*}}%[[K1:.*]]: tensor<[[k1ss:.*]]x[[ic:.*]]x[[k1oc:.*]]x[[prec:.*]]>, %[[I:.*]]: tensor<[[b:.*]]x[[iss:.*]]x[[ic]]x[[prec]]>) -> tensor<[[b]]x[[oss:.*]]x[[oc:.*]]x[[prec]]> {
                        CHECK: %[[c0:.*]] = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> tensor<[[prec]]>
                        CHECK: %{{.*}} = tile.contract add, mul, %[[c0]], %[[I]], %[[K1]] {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map{{.*}}, srcs = [#map{{.*}}, #map{{.*}}]} : tensor<[[prec]]>, tensor<[[b]]x[[iss]]x[[ic]]x[[prec]]>, tensor<[[k1ss]]x[[ic]]x[[k1oc]]x[[prec]]> -> tensor<[[b]]x{{.*}}x[[k1oc]]x[[prec]]>
                        CHECK: return %{{.*}} : tensor<[[b]]x[[oss]]x[[oc]]x[[prec]]>
                    )#";
  result.push_back({F32, check_str});
  result.push_back({F64, check_str});
  return result;
}

/**/
// TODO: INSTANTIATE_TEST_CASE_P was deprecated in favor for INSTANTIATE_TEST_SUITE_P, but the version of gtest that
// bazel links in is looking for INSTANTIATE_TEST_CASE_P right now.
INSTANTIATE_TEST_CASE_P(All, PlaidMLConvOperationTest, ::testing::ValuesIn(GetConvTestCases()), ConvTestSpecToString);
/**/
}  // namespace
}  // namespace plaidml
}  // namespace xla
