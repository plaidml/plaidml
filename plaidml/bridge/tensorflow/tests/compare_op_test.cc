// Tests that show HLO Module conversion to PlaidML Program.

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <string>
#include <variant>

#include "absl/strings/str_cat.h"
#include "plaidml/bridge/tensorflow/service/compiler.h"
#include "plaidml/bridge/tensorflow/tests/codegen_test.h"
#include "plaidml/bridge/tensorflow/tests/filecheck.h"
#include "plaidml/testenv.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

using ::plaidml::edsl::TensorBuffers;

namespace xla {
namespace plaidml {
namespace {

using TestCaseVal = std::vector<std::vector<float>>;
using TestCaseVal_int = std::vector<std::vector<int8_t>>;
using TestCasePairs = std::map<TestCaseVal, TestCaseVal_int>;

struct CompareTestSpec {
  PrimitiveType primitive_type;
  string filecheck_lines;
};

string CompareTestSpecToString(const ::testing::TestParamInfo<CompareTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLCompareOperationTest : public PlaidMLCodegenTest, public ::testing::WithParamInterface<CompareTestSpec> {
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
};

TEST_P(PlaidMLCompareOperationTest, CompEqOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int8_t> expected_val = {0, 0, 0, 0, 1, 0, 0, 0, 0};
  auto comp_type = ComparisonDirection::kEq;

  TestCaseVal inputs = {B, A};
  TestCaseVal_int results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("CompOp");
  CompareTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4, "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xi1>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateCompare(param_shape, lhs, rhs, comp_type));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLCompareOperationTest, CompLtOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int8_t> expected_val = {1, 1, 1, 1, 0, 0, 0, 0, 0};
  auto comp_type = ComparisonDirection::kLt;

  TestCaseVal inputs = {B, A};
  TestCaseVal_int results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("CompOp");
  CompareTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4, "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xi1>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateCompare(param_shape, lhs, rhs, comp_type));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLCompareOperationTest, CompLeOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int8_t> expected_val = {1, 1, 1, 1, 1, 0, 0, 0, 0};
  auto comp_type = ComparisonDirection::kLe;

  TestCaseVal inputs = {B, A};
  TestCaseVal_int results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("CompOp");
  CompareTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4, "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xi1>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateCompare(param_shape, lhs, rhs, comp_type));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLCompareOperationTest, CompGtOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int8_t> expected_val = {0, 0, 0, 0, 0, 1, 1, 1, 1};
  auto comp_type = ComparisonDirection::kGt;

  TestCaseVal inputs = {B, A};
  TestCaseVal_int results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("CompOp");
  CompareTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4, "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xi1>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateCompare(param_shape, lhs, rhs, comp_type));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLCompareOperationTest, CompGeOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int8_t> expected_val = {0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto comp_type = ComparisonDirection::kGe;

  TestCaseVal inputs = {B, A};
  TestCaseVal_int results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("CompOp");
  CompareTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4, "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xi1>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateCompare(param_shape, lhs, rhs, comp_type));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

std::vector<CompareTestSpec> GetCompareTestCases() {
  std::vector<CompareTestSpec> result;
  // TODO: reenable F16 when it is ready
  //  result.push_back(
  //      {F16, R"(CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>)"});
  result.push_back({F32, R"#(
        CHECK: return %{{.*}} : tensor<3x3x{{.*}}>
        )#"});
  result.push_back({F64, R"#(
        CHECK: return %{{.*}} : tensor<3x3x{{.*}}>
        )#"});
  return result;
}

/**/
// TODO: INSTANTIATE_TEST_CASE_P was deprecated in favor for INSTANTIATE_TEST_SUITE_P, but the version of gtest that
// bazel links in is looking for INSTANTIATE_TEST_CASE_P right now.
INSTANTIATE_TEST_CASE_P(All, PlaidMLCompareOperationTest, ::testing::ValuesIn(GetCompareTestCases()),
                        CompareTestSpecToString);
/**/
}  // namespace
}  // namespace plaidml
}  // namespace xla
