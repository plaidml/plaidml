// Tests that show HLO Module conversion to PlaidML Program.

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "tensorflow/compiler/xla/tests/test_utils.h"

#include "plaidml/bridge/tensorflow/tests/codegen_test.h"

namespace xla {
namespace plaidml {
namespace {

struct DotTestSpec {
  PrimitiveType primitive_type;
};

string DotTestSpecToString(const ::testing::TestParamInfo<DotTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLDotOperationTest : public PlaidMLCodegenTest, public ::testing::WithParamInterface<DotTestSpec> {};

TEST_P(PlaidMLDotOperationTest, SimpleDotOp) {
  std::vector<float> input_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> expected_val = {30, 36, 42, 66, 81, 96, 102, 126, 150};
  TestCases testcases = {
      TestCaseIO{{input_val, input_val}, {expected_val}},
  };

  HloComputation::Builder builder("SimpleDotOp");
  DotTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(CreateCanonicalDot(param_shape, lhs, rhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLDotOperationTest, DotTransposeOp) {
  std::vector<float> input_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> expected_val = {66, 78, 90, 78, 93, 108, 90, 108, 126};
  TestCases testcases = {
      TestCaseIO{{input_val, input_val}, {expected_val}},
  };

  HloComputation::Builder builder("DotTransposeOp");
  DotTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));
  HloInstruction* lhs_transposed = builder.AddInstruction(HloInstruction::CreateTranspose(param_shape, lhs, {1, 0}));

  builder.AddInstruction(CreateCanonicalDot(param_shape, lhs_transposed, rhs));
  CompileAndCheck(builder.Build(), testcases);
}

std::vector<DotTestSpec> GetDotTestCases() {
  std::vector<DotTestSpec> result;
  // TODO: reenable F16 when it is ready
  // result.push_back({F16});
  result.push_back({F32});
  result.push_back({F64});
  return result;
}

INSTANTIATE_TEST_SUITE_P(All, PlaidMLDotOperationTest, ::testing::ValuesIn(GetDotTestCases()), DotTestSpecToString);

}  // namespace
}  // namespace plaidml
}  // namespace xla
