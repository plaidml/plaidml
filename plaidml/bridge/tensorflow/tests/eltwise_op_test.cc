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
using TestCasePairs = std::map<TestCaseVal, TestCaseVal>;

struct EltwiseTestSpec {
  PrimitiveType primitive_type;
  string filecheck_lines;
};

string EltwiseTestSpecToString(const ::testing::TestParamInfo<EltwiseTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLEltwiseOperationTest : public PlaidMLCodegenTest, public ::testing::WithParamInterface<EltwiseTestSpec> {
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

// Unary Eltwise Ops
TEST_P(PlaidMLEltwiseOperationTest, EltwiseAbsOp) {
  std::vector<float> input_val = {-1, 2, -3, -4, 5, 6, -7, -8, -9};
  std::vector<float> expected_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  TestCaseVal inputs = {input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseAbsOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4, "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kAbs, lhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseCeilOp) {
  std::vector<float> input_val = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
  std::vector<float> expected_val = {2, 3, 4, 5, 6, 7, 8, 9, 10};

  TestCaseVal inputs = {input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseCeilOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4, "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kCeil, lhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseCosOp) {
  float PI = 3.141592653589;
  std::vector<float> input_val = {0, PI / 3, -PI / 3, 2 * PI / 3, -2 * PI / 3, 0, 0, PI, -PI};
  std::vector<float> expected_val = {1, 0.5, 0.5, -0.5, -0.5, 1, 1, -1, -1};

  TestCaseVal inputs = {input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseCosOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4, "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kCos, lhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseExpOp) {
  std::vector<float> input_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> expected_val = {2.7182818, 7.3890561, 20.085537, 54.598150, 148.41316,
                                     403.42879, 1096.6332, 2980.9580, 8103.0839};

  TestCaseVal inputs = {input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseExpOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4, "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kExp, lhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseFloorOp) {
  std::vector<float> input_val = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
  std::vector<float> expected_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  TestCaseVal inputs = {input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseFloorOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4, "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kFloor, lhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseLogOp) {
  std::vector<float> input_val = {2.7182818, 7.3890561, 20.085537, 54.598150, 148.41316,
                                  403.42879, 1096.6332, 2980.9580, 8103.0839};
  std::vector<float> expected_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  TestCaseVal inputs = {input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseLogOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4, "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kLog, lhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseNegOp) {
  std::vector<float> input_val = {-1, 2, -3, -4, 5, 6, -7, -8, -9};
  std::vector<float> expected_val = {1, -2, 3, 4, -5, -6, 7, 8, 9};

  TestCaseVal inputs = {input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseNegOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4, "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kNegate, lhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseRsqrtOp) {
  std::vector<float> input_val = {1, 4, 9, 16, 25, 36, 49, 64, 81};
  std::vector<float> expected_val = {1, 0.5, 1.0 / 3, 0.25, 0.2, 1.0 / 6, 1.0 / 7, .125, 1.0 / 9};

  TestCaseVal inputs = {input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseRsqrtOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4, "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kRsqrt, lhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseSinOp) {
  float PI = 3.141592653589;
  std::vector<float> input_val = {0, PI / 6, -PI / 6, 5 * PI / 6, -5 * PI / 6, PI / 2, -PI / 2, 0, 0};
  std::vector<float> expected_val = {0, 0.5, -0.5, 0.5, -0.5, 1, -1, 0, 0};

  TestCaseVal inputs = {input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseSinOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4, "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kSin, lhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseSqrtOp) {
  std::vector<float> input_val = {1, 4, 9, 16, 25, 36, 49, 64, 81};
  std::vector<float> expected_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  TestCaseVal inputs = {input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseSqrtOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4, "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kSqrt, lhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

// Binary Eltwise Ops
TEST_P(PlaidMLEltwiseOperationTest, EltwiseAddOp) {
  std::vector<float> input_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> expected_val = {2, 4, 6, 8, 10, 12, 14, 16, 18};

  TestCaseVal inputs = {input_val, input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseAddOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4,
                      "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kAdd, lhs, rhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseDivOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<float> expected_val = {1.0 / 9, 0.25, 3.0 / 7, 2.0 / 3, 1, 1.5, 7.0 / 3, 4, 9};

  TestCaseVal inputs = {B, A};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseDivOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4,
                      "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kDivide, lhs, rhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseMaxOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {0, 3, 2, 5, 4, 7, 6, 9, 8};
  std::vector<float> expected_val = {1, 3, 3, 5, 5, 7, 7, 9, 9};

  TestCaseVal inputs = {A, B};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseMaxOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4,
                      "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kMaximum, lhs, rhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseMinOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {0, 3, 2, 5, 4, 7, 6, 9, 8};
  std::vector<float> expected_val = {0, 2, 2, 4, 4, 6, 6, 8, 8};

  TestCaseVal inputs = {A, B};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseMinOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4,
                      "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kMinimum, lhs, rhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseMulOp) {
  std::vector<float> input_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> expected_val = {1, 4, 9, 16, 25, 36, 49, 64, 81};

  TestCaseVal inputs = {input_val, input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseMulOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4,
                      "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kMultiply, lhs, rhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwisePowOp) {
  std::vector<float> A = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  std::vector<float> B = {0, 0, 0, 1, 1, 1, 2, 2, 2};
  std::vector<float> expected_val = {1, 1, 1, 1, 2, 3, 1, 4, 9};

  TestCaseVal inputs = {B, A};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwisePowOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4,
                      "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kPower, lhs, rhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseRemOp) {
  std::vector<float> A = {10, 10, 10, 10, 10, 10, 10, 10, 10};
  std::vector<float> B = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> expected_val = {0, 0, 1, 2, 0, 4, 3, 2, 1};

  TestCaseVal inputs = {B, A};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseRemOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4,
                      "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kRemainder, lhs, rhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseSubOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<float> expected_val = {-8, -6, -4, -2, 0, 2, 4, 6, 8};

  TestCaseVal inputs = {B, A};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseSubOp");
  EltwiseTestSpec spec = GetParam();

  auto fcheck_lines = spec.filecheck_lines;
  fcheck_lines.insert(4,
                      "CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>\n");

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kSubtract, lhs, rhs));
  CompileAndCheck(builder.Build(), fcheck_lines, testcase_pairs);
}

std::vector<EltwiseTestSpec> GetEltwiseTestCases() {
  std::vector<EltwiseTestSpec> result;
  // TODO: reenable F16 when it is ready
  //  result.push_back(
  //      {F16, R"(CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>)"});
  result.push_back({F32, R"#(
        CHECK: return %{{.*}} : tensor<3x3xf32>
        )#"});
  result.push_back({F64, R"#(
        CHECK: return %{{.*}} : tensor<3x3xf32>
        )#"});
  return result;
}

/**/
// TODO: INSTANTIATE_TEST_CASE_P was deprecated in favor for INSTANTIATE_TEST_SUITE_P, but the version of gtest that
// bazel links in is looking for INSTANTIATE_TEST_CASE_P right now.
INSTANTIATE_TEST_CASE_P(All, PlaidMLEltwiseOperationTest, ::testing::ValuesIn(GetEltwiseTestCases()),
                        EltwiseTestSpecToString);
/**/
}  // namespace
}  // namespace plaidml
}  // namespace xla
