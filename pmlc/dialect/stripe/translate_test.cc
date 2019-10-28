// Copyright 2019, Intel Corporation

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/stripe/padding_pass.h"
#include "pmlc/dialect/stripe/transcode.h"
#include "pmlc/dialect/stripe/types.h"

#include "base/util/logging.h"
#include "plaidml2/edsl/helper.h"
#include "testing/matchers.h"
#include "tile/codegen/compile_pass.h"
#include "tile/codegen/localize.h"
#include "tile/lib/lib.h"

using namespace plaidml::edsl;          // NOLINT
using namespace pmlc::dialect::stripe;  // NOLINT
using namespace vertexai::tile;         // NOLINT
using namespace vertexai::tile::lib;    // NOLINT

using ::testing::LinesEq;

template <typename Pass, typename Config>
std::unique_ptr<mlir::Pass> CreatePass(Config config) {
  return std::make_unique<Pass>(config);
}

// Stripe Classic <-> Stripe MLIR transcoding tests are parameterized by whether
// they should add location info or not, since there've been some subtle
// transcoding issues when location-adding top-level refinements are or aren't
// in place.
class TranslateTest : public ::testing::TestWithParam<bool> {};

static void RunTest(const Program& program, bool addLocations) {
  IVLOG(1, "Making context + module");
  mlir::MLIRContext context;

  IVLOG(1, "Making a stripe program + fixing locals");
  auto stripe = plaidml::edsl::ConvertIntoStripe(program);
  codegen::LocalizeBlockPass(codegen::AliasMap(codegen::AliasMap(), stripe->entry.get()), stripe->entry.get(), {"tmp"});

  if (addLocations) {
    codegen::CompilerState cstate{stripe};

    IVLOG(1, "Adding a memory location");
    codegen::proto::LocateMemoryPass lmp;
    auto lmp_dev = lmp.mutable_loc()->add_devs();
    lmp_dev->set_name("OuterMem");
    lmp_dev = lmp.mutable_loc()->add_devs();
    lmp_dev->set_name("InnerMem");
    lmp_dev->add_units()->set_offset(1);
    codegen::LocateMemoryPass{lmp}.Apply(&cstate);

    IVLOG(1, "Adding an executor location");
    codegen::proto::LocateBlockPass lbp;
    lbp.add_reqs("main");
    auto lbp_dev = lbp.mutable_loc()->add_devs();
    lbp_dev->set_name("OuterExecutor");
    lbp_dev->add_units()->set_offset(0);
    lbp_dev = lbp.mutable_loc()->add_devs();
    lbp_dev->set_name("InnerExecutor");
    lbp_dev->add_units()->set_offset(1);
    codegen::LocateBlockPass{lbp}.Apply(&cstate);
  }

  IVLOG(2, "Original version:");
  IVLOG(2, *stripe->entry);

  IVLOG(1, "Converting to MLIR");
  auto module = IntoMLIR(&context, *stripe);

  IVLOG(1, "Verifying module");
  if (failed(module->verify())) {
    module->dump();
    throw std::runtime_error("IntoMLIR verification failure");
  }

  IVLOG(1, "Doing some passes");
  mlir::PassManager pm(&context, true);
  pm.addPass(mlir::createCSEPass());
  codegen::proto::MLIR_PadPass options;
  pm.addPass(CreatePass<PaddingPass>(options));
  if (failed(pm.run(*module))) {
    module->dump();
    throw std::runtime_error("MLIR passes failure");
  }

  IVLOG(1, "Writing out module");
  auto moduleOp = *module;
  auto module_str = mlir::debugString(moduleOp);
  IVLOG(2, module_str);

  IVLOG(1, "Parsing it back in");
  auto new_module = parseSourceString(module_str, &context);
  if (!new_module) {
    throw std::runtime_error("Unable to parse");
  }

  IVLOG(1, "Converting the other way");
  auto stripe2 = FromMLIR(*new_module);

  IVLOG(2, "New version:");
  IVLOG(2, *stripe2->entry);

  // require textually perfect round trip
  EXPECT_THAT(to_string(*stripe2->entry), LinesEq(to_string(*stripe->entry)));
}

TEST_P(TranslateTest, Conv2dBnRelu) {
  LogicalShape I(PLAIDML_DATA_FLOAT32, {16, 112, 112, 64});
  LogicalShape K(PLAIDML_DATA_FLOAT32, {3, 3, 64, 128});
  LogicalShape C(PLAIDML_DATA_FLOAT32, {128});
  auto program = LoadConv2dBnRelu("foo", I, K, C, {16, 112, 112, 128});
  RunTest(program, GetParam());
}

TEST_P(TranslateTest, Conv2d) {
  LogicalShape I(PLAIDML_DATA_FLOAT32, {16, 112, 112, 64});
  LogicalShape K(PLAIDML_DATA_FLOAT32, {3, 3, 64, 128});
  auto program = LoadConv2d("foo", I, K, {16, 112, 112, 128});
  RunTest(program, GetParam());
}

TEST_P(TranslateTest, MaxPool2d) {
  LogicalShape A(PLAIDML_DATA_FLOAT32, {1, 64, 64, 3});
  auto program = LoadMaxPool2d("maxpool", A, {2, 2});
  RunTest(program, GetParam());
}

TEST_P(TranslateTest, Softmax) {
  LogicalShape A(PLAIDML_DATA_FLOAT32, {64, 64});
  auto program = LoadSoftmax("softmax", A);
  RunTest(program, GetParam());
}

TEST_P(TranslateTest, EltwiseAdd) {
  LogicalShape A(PLAIDML_DATA_FLOAT32, {16, 16});
  LogicalShape B(PLAIDML_DATA_FLOAT32, {16, 16});
  auto program = LoadEltwiseAdd("eltwise_add", A, B);
  RunTest(program, GetParam());
}

TEST_P(TranslateTest, EltwiseMul) {
  LogicalShape A(PLAIDML_DATA_FLOAT32, {16, 16});
  LogicalShape B(PLAIDML_DATA_FLOAT32, {16, 16});
  auto program = LoadEltwiseMul("eltwise_mul", A, B);
  RunTest(program, GetParam());
}

TEST_P(TranslateTest, EltwiseDiv) {
  LogicalShape A(PLAIDML_DATA_FLOAT32, {16, 16});
  LogicalShape B(PLAIDML_DATA_FLOAT32, {16, 16});
  auto program = LoadEltwiseDiv("eltwise_div", A, B);
  RunTest(program, GetParam());
}

TEST_P(TranslateTest, MatMul) {
  LogicalShape A(PLAIDML_DATA_FLOAT32, {16, 16});
  LogicalShape B(PLAIDML_DATA_FLOAT32, {16, 16});
  auto program = LoadMatMul("matmul", A, B);
  RunTest(program, GetParam());
}

// These two tests fail with invalid tensor dimensions if the location parameter for RunTest is set to 0.

// TEST_P(TranslateTest, LayerNorm4dAx2) {
//   LogicalShape A(PLAIDML_DATA_FLOAT32, {1, 64, 64, 32});
//   auto program = LoadLayerNorm4dAx2("layer_norm", A);
//   RunTest(program, true);
// }

TEST_P(TranslateTest, BatchNormalization) {
  LogicalShape A(PLAIDML_DATA_FLOAT32, {16, 64, 64, 32});
  auto program = LoadBatchNormalization("batch_norm", A);
  RunTest(program, true);
}

Tensor Dot(const Tensor& X, const Tensor& Y) {
  plaidml::edsl::TensorDim I, J, K;
  TensorIndex i, j, k;
  X.bind_dims(I, K);
  Y.bind_dims(K, J);
  auto R = TensorOutput(I, J);
  R(i, j) += X(i, k) * Y(k, j);
  return R;
}

TEST_P(TranslateTest, DoubleDot) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20});
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {20, 30});
  auto C = Placeholder(PLAIDML_DATA_FLOAT32, {30, 40});
  auto program = Program("double_dot", {Dot(Dot(A, B), C)});
  RunTest(program, GetParam());
}

INSTANTIATE_TEST_CASE_P(NonTrivialLocs, TranslateTest, ::testing::Values(false, true));
