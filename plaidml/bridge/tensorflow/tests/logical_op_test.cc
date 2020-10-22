// Tests that show HLO Module conversion to PlaidML Program.

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <string>

#include "tensorflow/compiler/xla/tests/test_utils.h"

#include "plaidml/bridge/tensorflow/tests/codegen_test.h"

namespace xla {
namespace plaidml {
namespace {

struct LogicalTestSpec {
  PrimitiveType primitive_type;
};

string LogicalTestSpecToString(const ::testing::TestParamInfo<LogicalTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLLogicalOperationTest : public PlaidMLCodegenTest, public ::testing::WithParamInterface<LogicalTestSpec> {};

TEST_P(PlaidMLLogicalOperationTest, LogicalAndOp) {
  std::vector<int> input_A = {0, 0, 1, 1, 0, 0, 1, 1, 0};
  std::vector<int> input_B = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> output_C = {0, 0, 1, 0, 0, 0, 1, 0, 0};

  TestCases testcases = {
      TestCaseIO{{input_A, input_B}, {output_C}},
  };

  HloComputation::Builder builder("LogicalAndOp");
  LogicalTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kAnd, lhs, rhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLLogicalOperationTest, LogicalNotOp) {
  std::vector<int> input_A = {
      static_cast<int>(0x00000000), static_cast<int>(0x11111111), static_cast<int>(0x22222222),  //
      static_cast<int>(0x33333333), static_cast<int>(0x44444444), static_cast<int>(0x55555555),  //
      static_cast<int>(0xDDDDDDDD), static_cast<int>(0xEEEEEEEE), static_cast<int>(0xFFFFFFFF)};
  std::vector<int> output_C = {
      static_cast<int>(0xFFFFFFFF), static_cast<int>(0xEEEEEEEE), static_cast<int>(0xDDDDDDDD),  //
      static_cast<int>(0xCCCCCCCC), static_cast<int>(0xBBBBBBBB), static_cast<int>(0xAAAAAAAA),  //
      static_cast<int>(0x22222222), static_cast<int>(0x11111111), static_cast<int>(0x00000000)};

  TestCases testcases = {
      TestCaseIO{{input_A}, {output_C}},
  };

  HloComputation::Builder builder("LogicalNotOp");
  LogicalTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kNot, lhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLLogicalOperationTest, LogicalOrOp) {
  std::vector<int> input_A = {0, 0, 1, 1, 0, 0, 1, 1, 0};
  std::vector<int> input_B = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> output_C = {1, 0, 1, 1, 1, 0, 1, 1, 1};

  TestCases testcases = {
      TestCaseIO{{input_A, input_B}, {output_C}},
  };

  HloComputation::Builder builder("LogicalOrOp");
  LogicalTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kOr, lhs, rhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLLogicalOperationTest, LogicalXorOp) {
  std::vector<int> input_A = {0, 0, 1, 1, 0, 0, 1, 1, 0};
  std::vector<int> input_B = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> output_C = {1, 0, 0, 1, 1, 0, 0, 1, 1};

  TestCases testcases = {
      TestCaseIO{{input_A, input_B}, {output_C}},
  };

  HloComputation::Builder builder("LogicalXorOp");
  LogicalTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kXor, lhs, rhs));
  CompileAndCheck(builder.Build(), testcases);
}

std::vector<LogicalTestSpec> GetLogicalTestCases() {
  std::vector<LogicalTestSpec> result;
  result.push_back({S32});
  // TODO: Determine issue with si64 testing
  return result;
}

INSTANTIATE_TEST_SUITE_P(LogicalAndOp, PlaidMLLogicalOperationTest, ::testing::ValuesIn(GetLogicalTestCases()),
                         LogicalTestSpecToString);

}  // namespace
}  // namespace plaidml
}  // namespace xla
