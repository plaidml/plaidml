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

struct EltwiseTestSpec {
  PrimitiveType primitive_type;
};

string EltwiseTestSpecToString(const ::testing::TestParamInfo<EltwiseTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLEltwiseOperationTest : public PlaidMLCodegenTest, public ::testing::WithParamInterface<EltwiseTestSpec> {};

// Unary Eltwise Ops
TEST_P(PlaidMLEltwiseOperationTest, EltwiseAbsOp) {
  std::vector<float> input_val = {-1, 2, -3, -4, 5, 6, -7, -8, -9};
  std::vector<float> expected_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  TestCases testcases = {
      TestCaseIO{{input_val}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseAbsOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kAbs, lhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseCeilOp) {
  std::vector<float> input_val = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
  std::vector<float> expected_val = {2, 3, 4, 5, 6, 7, 8, 9, 10};

  TestCases testcases = {
      TestCaseIO{{input_val}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseCeilOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kCeil, lhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseCosOp) {
  float PI = 3.141592653589;
  std::vector<float> input_val = {0, PI / 3, -PI / 3, 2 * PI / 3, -2 * PI / 3, 0, 0, PI, -PI};
  std::vector<float> expected_val = {1, 0.5, 0.5, -0.5, -0.5, 1, 1, -1, -1};

  TestCases testcases = {
      TestCaseIO{{input_val}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseCosOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kCos, lhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseExpOp) {
  std::vector<float> input_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> expected_val = {2.7182818, 7.3890561, 20.085537, 54.598150, 148.41316,
                                     403.42879, 1096.6332, 2980.9580, 8103.0839};

  TestCases testcases = {
      TestCaseIO{{input_val}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseExpOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kExp, lhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseFloorOp) {
  std::vector<float> input_val = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
  std::vector<float> expected_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  TestCases testcases = {
      TestCaseIO{{input_val}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseFloorOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kFloor, lhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseLogOp) {
  std::vector<float> input_val = {2.7182818, 7.3890561, 20.085537, 54.598150, 148.41316,
                                  403.42879, 1096.6332, 2980.9580, 8103.0839};
  std::vector<float> expected_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  TestCases testcases = {
      TestCaseIO{{input_val}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseLogOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kLog, lhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseNegOp) {
  std::vector<float> input_val = {-1, 2, -3, -4, 5, 6, -7, -8, -9};
  std::vector<float> expected_val = {1, -2, 3, 4, -5, -6, 7, 8, 9};

  TestCases testcases = {
      TestCaseIO{{input_val}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseNegOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kNegate, lhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseRsqrtOp) {
  std::vector<float> input_val = {1, 4, 9, 16, 25, 36, 49, 64, 81};
  std::vector<float> expected_val = {1, 0.5, 1.0 / 3, 0.25, 0.2, 1.0 / 6, 1.0 / 7, .125, 1.0 / 9};

  TestCases testcases = {
      TestCaseIO{{input_val}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseRsqrtOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kRsqrt, lhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseSinOp) {
  float PI = 3.141592653589;
  std::vector<float> input_val = {0, PI / 6, -PI / 6, 5 * PI / 6, -5 * PI / 6, PI / 2, -PI / 2, 0, 0};
  std::vector<float> expected_val = {0, 0.5, -0.5, 0.5, -0.5, 1, -1, 0, 0};

  TestCases testcases = {
      TestCaseIO{{input_val}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseSinOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kSin, lhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseSqrtOp) {
  std::vector<float> input_val = {1, 4, 9, 16, 25, 36, 49, 64, 81};
  std::vector<float> expected_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  TestCases testcases = {
      TestCaseIO{{input_val}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseSqrtOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateUnary(param_shape, HloOpcode::kSqrt, lhs));
  CompileAndCheck(builder.Build(), testcases);
}

// Binary Eltwise Ops
TEST_P(PlaidMLEltwiseOperationTest, EltwiseAddOp) {
  std::vector<float> input_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> expected_val = {2, 4, 6, 8, 10, 12, 14, 16, 18};

  TestCases testcases = {
      TestCaseIO{{input_val, input_val}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseAddOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kAdd, lhs, rhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseDivOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<float> expected_val = {1.0 / 9, 0.25, 3.0 / 7, 2.0 / 3, 1, 1.5, 7.0 / 3, 4, 9};

  TestCases testcases = {
      TestCaseIO{{B, A}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseDivOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kDivide, lhs, rhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseMaxOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {0, 3, 2, 5, 4, 7, 6, 9, 8};
  std::vector<float> expected_val = {1, 3, 3, 5, 5, 7, 7, 9, 9};

  TestCases testcases = {
      TestCaseIO{{A, B}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseMaxOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kMaximum, lhs, rhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseMinOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {0, 3, 2, 5, 4, 7, 6, 9, 8};
  std::vector<float> expected_val = {0, 2, 2, 4, 4, 6, 6, 8, 8};

  TestCases testcases = {
      TestCaseIO{{A, B}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseMinOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kMinimum, lhs, rhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseMulOp) {
  std::vector<float> input_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> expected_val = {1, 4, 9, 16, 25, 36, 49, 64, 81};

  TestCases testcases = {
      TestCaseIO{{input_val, input_val}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseMulOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kMultiply, lhs, rhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwisePowOp) {
  std::vector<float> A = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  std::vector<float> B = {0, 0, 0, 1, 1, 1, 2, 2, 2};
  std::vector<float> expected_val = {1, 1, 1, 1, 2, 3, 1, 4, 9};

  TestCases testcases = {
      TestCaseIO{{B, A}, {expected_val}},
  };

  HloComputation::Builder builder("EltwisePowOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kPower, lhs, rhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseRemOp) {
  std::vector<float> A = {10, 10, 10, 10, 10, 10, 10, 10, 10};
  std::vector<float> B = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> expected_val = {0, 0, 1, 2, 0, 4, 3, 2, 1};

  TestCases testcases = {
      TestCaseIO{{B, A}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseRemOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kRemainder, lhs, rhs));
  CompileAndCheck(builder.Build(), testcases);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseSubOp) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<float> expected_val = {-8, -6, -4, -2, 0, 2, 4, 6, 8};

  TestCases testcases = {
      TestCaseIO{{B, A}, {expected_val}},
  };

  HloComputation::Builder builder("EltwiseSubOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kSubtract, lhs, rhs));
  CompileAndCheck(builder.Build(), testcases);
}

std::vector<EltwiseTestSpec> GetEltwiseTestCases() {
  std::vector<EltwiseTestSpec> result;
  // TODO: reenable F16 when it is ready
  // result.push_back({F16});
  result.push_back({F32});
  result.push_back({F64});
  return result;
}

INSTANTIATE_TEST_SUITE_P(All, PlaidMLEltwiseOperationTest, ::testing::ValuesIn(GetEltwiseTestCases()),
                         EltwiseTestSpecToString);

}  // namespace
}  // namespace plaidml
}  // namespace xla
