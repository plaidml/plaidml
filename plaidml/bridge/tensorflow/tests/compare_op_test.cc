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

struct CompareTestSpec {
  PrimitiveType primitive_type;
};

string CompareTestSpecToString(const ::testing::TestParamInfo<CompareTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLCompareOperationTest : public PlaidMLCodegenTest, public ::testing::WithParamInterface<CompareTestSpec> {};

TEST_P(PlaidMLCompareOperationTest, CompEqOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int8_t> expected_val = {0, 0, 0, 0, 1, 0, 0, 0, 0};
  auto comp_type = ComparisonDirection::kEq;

  TestCases testcases = {
      TestCaseIO{{B, A}, {expected_val}},
  };

  HloComputation::Builder builder("CompOp");
  CompareTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateCompare(param_shape, lhs, rhs, comp_type));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLCompareOperationTest, CompLtOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int8_t> expected_val = {1, 1, 1, 1, 0, 0, 0, 0, 0};
  auto comp_type = ComparisonDirection::kLt;

  TestCases testcases = {
      TestCaseIO{{B, A}, {expected_val}},
  };

  HloComputation::Builder builder("CompOp");
  CompareTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateCompare(param_shape, lhs, rhs, comp_type));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLCompareOperationTest, CompLeOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int8_t> expected_val = {1, 1, 1, 1, 1, 0, 0, 0, 0};
  auto comp_type = ComparisonDirection::kLe;

  TestCases testcases = {
      TestCaseIO{{B, A}, {expected_val}},
  };

  HloComputation::Builder builder("CompOp");
  CompareTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateCompare(param_shape, lhs, rhs, comp_type));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLCompareOperationTest, CompGtOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int8_t> expected_val = {0, 0, 0, 0, 0, 1, 1, 1, 1};
  auto comp_type = ComparisonDirection::kGt;

  TestCases testcases = {
      TestCaseIO{{B, A}, {expected_val}},
  };

  HloComputation::Builder builder("CompOp");
  CompareTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateCompare(param_shape, lhs, rhs, comp_type));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLCompareOperationTest, CompGeOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int8_t> expected_val = {0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto comp_type = ComparisonDirection::kGe;

  TestCases testcases = {
      TestCaseIO{{B, A}, {expected_val}},
  };

  HloComputation::Builder builder("CompOp");
  CompareTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateCompare(param_shape, lhs, rhs, comp_type));
  CompileAndCheck(builder.Build(), testcases);
}

std::vector<CompareTestSpec> GetCompareTestCases() {
  std::vector<CompareTestSpec> result;
  // TODO: reenable F16 when it is ready
  // result.push_back({F16});
  result.push_back({F32});
  result.push_back({F64});
  return result;
}

INSTANTIATE_TEST_SUITE_P(All, PlaidMLCompareOperationTest, ::testing::ValuesIn(GetCompareTestCases()),
                         CompareTestSpecToString);

}  // namespace
}  // namespace plaidml
}  // namespace xla
